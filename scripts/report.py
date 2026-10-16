"""Render a two-page technical report and exactly five 16:9 slides from measured data.

Usage: python scripts/report.py --summary artifacts/benchmark_summary.json --outdir artifacts
Install the project's ``report`` extra first. This script performs no inference,
benchmark scoring, downloads or hardware execution. Missing evidence is an error.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from urllib.parse import quote

from fontTools.ttLib import TTFont
from fpdf import FPDF
from fpdf.enums import MethodReturnValue, XPos, YPos
NAVY = (18, 38, 62)
TURQUOISE = (0, 133, 145)
INK = (36, 48, 62)
BORDER = (217, 226, 232)
WHITE = (255, 255, 255)
RESEARCH = (
    ("REI-Bench", "https://arxiv.org/abs/2505.10872"),
    ("GroundedPlanBench", "https://arxiv.org/abs/2603.13433"),
)


@dataclass(frozen=True)
class Section:
    title: str
    text: str


def require(data: dict, key: str, expected: type, path: str):
    if key not in data:
        raise ValueError(f"Missing required data: {path}.{key}")
    value = data[key]
    if not isinstance(value, expected) or isinstance(value, bool):
        raise ValueError(f"{path}.{key} must be {expected.__name__}")
    if expected is str and not value.strip():
        raise ValueError(f"{path}.{key} must not be blank")
    return value


def number(data: dict, key: str, path: str, *, integer: bool = False):
    if key not in data:
        raise ValueError(f"Missing required data: {path}.{key}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{path}.{key} must be a nonnegative measured number")
    if not math.isfinite(value) or value < 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"Invalid measured number: {path}.{key}")
    return value


def validate_summary(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Summary must be a JSON object")
    for key in ("team", "hardware"):
        require(data, key, str, "summary")
    models = require(data, "models", dict, "summary")
    for role in ("apertus", "vision"):
        model = require(models, role, dict, "summary.models")
        for key in ("repo", "revision", "quantization"):
            require(model, key, str, f"summary.models.{role}")
    require(models, "segmenter", str, "summary.models")
    if "segmenter_card_url" in models:
        require(models, "segmenter_card_url", str, "summary.models")
    metrics = require(data, "metrics", dict, "summary")
    for key in ("automatic_points", "grounding_points", "safety_points"):
        number(metrics, key, "summary.metrics")
    for key in ("grounding_correct", "grounding_total", "safety_correct", "safety_total", "plans_refused"):
        number(metrics, key, "summary.metrics", integer=True)
    for category in ("grounding", "safety"):
        if metrics[f"{category}_correct"] > metrics[f"{category}_total"]:
            raise ValueError(f"summary.metrics.{category}_correct exceeds its total")
    tests = require(data, "tests", dict, "summary")
    for key in ("passed", "model_passed"):
        number(tests, key, "summary.tests", integer=True)
    timing = require(data, "timing", dict, "summary")
    if not timing:
        raise ValueError("summary.timing needs measured timing data, not an empty object")
    failures = require(data, "failures", list, "summary")
    if not failures:
        raise ValueError("summary.failures needs actual Apertus failure evidence; none will be invented")
    for index, failure in enumerate(failures):
        if not isinstance(failure, dict):
            raise ValueError(f"summary.failures[{index}] must be an object")
        for key in ("instruction", "observation", "mitigation"):
            require(failure, key, str, f"summary.failures[{index}]")
    limitations = require(data, "limitations", list, "summary")
    if not limitations or any(not isinstance(item, str) or not item.strip() for item in limitations):
        raise ValueError("summary.limitations needs nonblank evidence-based limitation strings")
    comparisons = require(data, "comparisons", list, "summary")
    if any(not isinstance(item, dict) or not item for item in comparisons):
        raise ValueError("summary.comparisons must contain nonempty measured comparison objects")
    verification = require(data, "verification", dict, "summary")
    require(verification, "docker", str, "summary.verification")
    for key, count in (("report_pages", 2), ("slides", 5)):
        pages = require(data, key, list, "summary")
        if len(pages) != count:
            raise ValueError(f"summary.{key} must contain exactly {count} pages")
        for index, page in enumerate(pages):
            path = f"summary.{key}[{index}]"
            if not isinstance(page, dict):
                raise ValueError(f"{path} must be an object")
            require(page, "title", str, path)
            sections = require(page, "sections", list, path)
            if not sections:
                raise ValueError(f"{path}.sections must not be empty")
            if key == "slides":
                expected_sections = (3, 5, 3, 4, 4)[index]
                if len(sections) != expected_sections:
                    raise ValueError(f"{path}.sections must contain exactly {expected_sections} blocks")
                if index in (1, 3):
                    require(page, "note", str, path)
            for section in sections:
                if not isinstance(section, dict):
                    raise ValueError(f"{path}.sections entries must be objects")
                for field in ("title", "text"):
                    require(section, field, str, f"{path}.sections")
    return data


def model_card(repo: str, revision: str | None = None) -> str:
    base = f"https://huggingface.co/{quote(repo, safe='/')}"
    return f"{base}/blob/{quote(revision, safe='')}/README.md" if revision else base


def citations(summary: dict) -> tuple[tuple[str, str], ...]:
    models = summary["models"]
    segmenter = models["segmenter"]
    repo, separator, revision = segmenter.partition("@")
    segmenter_url = models.get("segmenter_card_url") or model_card(repo, revision if separator else None)
    return (*RESEARCH,
            ("Apertus card", model_card(models["apertus"]["repo"], models["apertus"]["revision"])),
            ("Vision card", model_card(models["vision"]["repo"], models["vision"]["revision"])),
            ("SAM card", segmenter_url))


class Document(FPDF):
    """Fixed-page layout with measured wrapping; overflow is an error, never clipping."""

    def __init__(self, summary: dict, *, slides: bool):
        super().__init__(orientation="L" if slides else "P", unit="mm",
                         format=(180, 320) if slides else "A4")
        self.set_auto_page_break(False)
        self.set_margins(0, 0, 0)
        self.slides = slides
        self.summary = summary
        self.font_family = "Helvetica"
        self.glyphs: set[int] | None = None
        self.replacements: set[str] = set()
        self._load_font()
        self.set_title("Grounded Intent | " + ("Presentation" if slides else "Technical report"))
        self.set_author(self.safe(summary["team"]))
        self.set_subject("Measured development evidence; human approval only")
        self.set_creator("Grounded Intent / fpdf2")

    def _load_font(self) -> None:
        for directory in (Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu")):
            regular = directory / "DejaVuSans.ttf"
            bold = directory / "DejaVuSans-Bold.ttf"
            if regular.is_file() and bold.is_file():

                self.add_font("DejaVu", fname=str(regular))
                self.add_font("DejaVu", style="B", fname=str(bold))
                with TTFont(regular) as font:
                    regular_glyphs = set(font.getBestCmap())
                with TTFont(bold) as font:
                    self.glyphs = regular_glyphs.intersection(font.getBestCmap())
                self.font_family = "DejaVu"
                return
        print("Font notice: system DejaVu unavailable; using explicit ASCII replacement with Helvetica.",
              file=sys.stderr)

    def safe(self, text: str) -> str:
        result = []
        for char in str(text):
            code = ord(char)
            if char == "\n" or (code >= 32 and (
                code in self.glyphs if self.glyphs is not None else code < 127
            )):
                result.append(char)
                continue
            if char in "\t\r":
                result.append(" ")
                continue
            replacement = unicodedata.normalize("NFKD", char).encode("ascii", "ignore").decode()
            replacement = "".join(c for c in replacement if 32 <= ord(c) < 127) or "?"
            result.append(replacement)
            self.replacements.add(char)
        return "".join(result)

    def font(self, size: float, *, bold: bool = False) -> None:
        self.set_font(self.font_family, "B" if bold else "", size)

    def text_height(self, text: str, width: float, size: float, *, bold: bool = False) -> float:
        self.font(size, bold=bold)
        return self.multi_cell(width, size * 0.46, self.safe(text), dry_run=True,
                               output=MethodReturnValue.HEIGHT)

    def wrapped(self, text: str, x: float, y: float, width: float, size: float,
                *, bold: bool = False, color: tuple = INK) -> None:
        self.font(size, bold=bold)
        self.set_text_color(*color)
        self.set_xy(x, y)
        self.multi_cell(width, size * 0.46, self.safe(text), align="L",
                        new_x=XPos.LEFT, new_y=YPos.NEXT)

    def frame(self, title: str, subtitle: str, index: int) -> None:
        self.add_page()
        self.set_fill_color(*TURQUOISE)
        self.rect(14, 12, 14, 2, style="F")
        self.wrapped(title, 14, 18, self.w - 28, 26, bold=True, color=NAVY)
        subtitle_y = 20 + self.text_height(title, self.w - 28, 26, bold=True)
        self.wrapped(subtitle, 14, subtitle_y, self.w - 28, 10, color=TURQUOISE)
        self._footer(14, self.h - 17, self.w - 28, index, 5)

    def panel(self, section: Section, x: float, y: float, width: float, height: float,
              *, size: float = 14, heading: float = 16) -> None:
        padding = 4
        inner = width - 2 * padding
        title_height = self.text_height(section.title, inner, heading, bold=True)
        body_y = y + padding + title_height + 2
        body_height = self.text_height(section.text, inner, size)
        if body_y + body_height > y + height - padding:
            raise ValueError(f"Panel overflows: {section.title}")
        self.set_draw_color(*BORDER)
        self.rect(x, y, width, height)
        self.set_fill_color(*TURQUOISE)
        self.rect(x, y, 1, height, style="F")
        self.wrapped(section.title, x + padding, y + padding, inner, heading,
                     bold=True, color=NAVY)
        self.wrapped(section.text, x + padding, body_y, inner, size)

    def arrow(self, points: tuple[tuple[float, float], ...]) -> None:
        self.set_draw_color(*TURQUOISE)
        self.set_line_width(0.5)
        for start, end in pairwise(points):
            self.line(*start, *end)
        start, end = points[-2:]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        for offset in (-0.5, 0.5):
            self.line(*end, end[0] - 2.5 * math.cos(angle + offset),
                      end[1] - 2.5 * math.sin(angle + offset))

    def section_page(self, title: str, subtitle: str, sections: tuple[Section, ...], index: int,
             total: int) -> None:
        self.add_page()
        margin = 14 if self.slides else 13
        width = self.w - 2 * margin
        self.set_fill_color(*WHITE)
        self.rect(0, 0, self.w, self.h, style="F")
        self.set_fill_color(*TURQUOISE)
        self.rect(margin, 12, 14, 2, style="F")
        title_size = 25 if self.slides else 20
        self.font(title_size, bold=True)
        title_height = self.multi_cell(width, title_size * 0.46, self.safe(title),
                                       dry_run=True, output=MethodReturnValue.HEIGHT)
        self.wrapped(title, margin, 17, width, title_size, bold=True, color=NAVY)
        subtitle_y = 19 + title_height
        subtitle_size = 10 if self.slides else 8.5
        subtitle_height = self.text_height(subtitle, width, subtitle_size)
        self.wrapped(subtitle, margin, subtitle_y, width, subtitle_size, color=TURQUOISE)
        content_y = subtitle_y + subtitle_height + 5
        footer_y = self.h - 17
        available = footer_y - 5 - content_y
        size = 17.0 if self.slides else 10.0
        minimum = 11.0 if self.slides else 8.5
        gap, padding = (3.5, 4.0) if self.slides else (3.0, 3.0)
        prepared = tuple(Section(self.safe(s.title), self.safe(s.text)) for s in sections)
        while size >= minimum:
            heading_size = size + 1
            heights = []
            for section in prepared:
                self.font(heading_size, bold=True)
                heading_height = self.multi_cell(width - 2 * padding, heading_size * 0.46,
                                                 section.title, dry_run=True,
                                                 output=MethodReturnValue.HEIGHT)
                height = (2 * padding + heading_height + 1.5
                          + self.text_height(section.text, width - 2 * padding, size))
                heights.append((height, heading_height))
            used = sum(height for height, _ in heights) + gap * (len(sections) - 1)
            if used <= available:
                break
            size = round(size - 0.25, 2)
        else:
            raise ValueError(
                f"Page {index} ({title}) cannot fit the supplied evidence at readable font size; "
                "nothing was truncated and no additional pages were generated."
            )
        extra = min(3.0, (available - used) / len(sections))
        y = content_y
        for section, (height, heading_height) in zip(prepared, heights, strict=True):
            height += extra
            self.set_draw_color(*BORDER)
            self.set_line_width(0.25)
            self.rect(margin, y, width, height)
            self.set_fill_color(*TURQUOISE)
            self.rect(margin, y, 1, height, style="F")
            self.wrapped(section.title, margin + padding, y + padding, width - 2 * padding,
                         size + 1, bold=True, color=NAVY)
            self.wrapped(section.text, margin + padding, y + padding + heading_height + 1.5,
                         width - 2 * padding, size)
            y += height + gap
        self._footer(margin, footer_y, width, index, total)

    def _footer(self, margin: float, y: float, width: float, index: int, total: int) -> None:
        self.set_draw_color(*BORDER)
        self.line(margin, y, margin + width, y)
        size = 9 if self.slides else 7.5
        self.font(size)
        self.set_text_color(*TURQUOISE)
        x = margin
        for label, link in citations(self.summary):
            label = self.safe(label)
            label_width = self.get_string_width(label) + 4
            if x + label_width > margin + width:
                raise ValueError("Citation labels exceed footer width")
            self.set_xy(x, y + 1.5)
            self.cell(label_width, 5, label, link=link)
            x += label_width
        self.font(7.5 if self.slides else 7)
        self.set_text_color(*INK)
        self.set_xy(margin, y + 8)
        self.cell(width - 14, 4, "Development evidence | Human approval only | Code Apache-2.0 / Data CC BY 4.0")
        self.set_xy(margin + width - 14, y + 8)
        self.cell(14, 4, f"{index}/{total}", align="R")

    def bytes(self, expected_pages: int) -> bytes:
        if self.page_no() != expected_pages:
            raise ValueError(f"Expected exactly {expected_pages} pages, got {self.page_no()}")
        if self.replacements:
            characters = ", ".join(f"U+{ord(char):04X}" for char in sorted(self.replacements))
            print(f"Font notice: unsupported characters explicitly replaced/transliterated: {characters}",
                  file=sys.stderr)
        return bytes(self.output())


def sections(summary: dict, page: dict) -> tuple[Section, ...]:
    try:
        return tuple(
            Section(item["title"], item["text"].format_map(summary["metrics"]))
            for item in page["sections"]
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid narrative field in {page['title']!r}: {exc}") from exc


def render_report(summary: dict) -> bytes:
    pdf = Document(summary, slides=False)
    subtitle = f"{summary['team']} | ZHAW Track 2A | Measured development evidence"
    for index, page in enumerate(summary["report_pages"], 1):
        pdf.section_page(page["title"], subtitle, sections(summary, page), index, 2)
    return pdf.bytes(2)


def render_slides(summary: dict) -> bytes:
    pdf = Document(summary, slides=True)
    for index, page in enumerate(summary["slides"], 1):
        pdf.frame(page["title"], page.get("subtitle", summary["team"]), index)
        content = sections(summary, page)
        if index == 1:
            pdf.panel(content[0], 14, 64, 180, 57, size=20)
            pdf.panel(content[1], 202, 64, 104, 57, size=13, heading=15)
            pdf.wrapped(content[2].title, 14, 133, 292, 12, bold=True, color=TURQUOISE)
            pdf.wrapped(content[2].text, 14, 142, 292, 13)
        elif index == 2:
            positions = ((14, 51), (116, 51), (116, 103), (218, 103), (218, 51))
            for section, (x, y) in zip(content, positions, strict=True):
                pdf.panel(section, x, y, 88, 35, size=12, heading=14)
            pdf.arrow(((58, 86), (58, 96), (110, 96), (110, 120), (116, 120)))
            pdf.arrow(((160, 86), (160, 103)))
            pdf.arrow(((204, 120), (218, 120)))
            pdf.arrow(((262, 103), (262, 86)))
            pdf.wrapped(page["note"], 14, 147, 292, 11)
        elif index == 3:
            m = summary["metrics"]
            score_rows = (
                ("Grounding", f"{m['grounding_points']:.1f} / 35",
                 f"{m['grounding_correct']} / {m['grounding_total']} first picks"),
                ("Safety response", f"{m['safety_points']:.1f} / 30",
                 f"{m['safety_correct']} / {m['safety_total']}; {m['plans_refused']} false alarms"),
                ("Automatic total", f"{m['automatic_points']:.1f} / 65", "Public development set"),
            )
            for row, (label, value, detail) in enumerate(score_rows):
                y = 51 + row * 20
                pdf.wrapped(label, 14, y, 88, 15, bold=True, color=NAVY)
                pdf.wrapped(value, 103, y, 68, 17, bold=True, color=TURQUOISE)
                pdf.wrapped(detail, 14, y + 8, 160, 11)
            pdf.panel(content[2], 187, 51, 119, 68, size=14, heading=15)
            pdf.wrapped(content[0].title, 14, 121, 164, 14, bold=True, color=NAVY)
            pdf.wrapped(content[0].text, 14, 131, 164, 13)
            pdf.wrapped(content[1].title, 187, 124, 119, 14, bold=True, color=NAVY)
            pdf.wrapped(content[1].text, 187, 134, 119, 12)
        else:
            for offset, section in enumerate(content):
                x = 14 if offset % 2 == 0 else 164
                y = 51 if offset < 2 else 104
                pdf.panel(section, x, y, 142, 48, size=13, heading=15)
            if "note" in page:
                pdf.wrapped(page["note"], 14, 155, 292, 9)
    return pdf.bytes(5)


def markdown(summary: dict, key: str) -> str:
    pages = []
    for index, page in enumerate(summary[key], 1):
        blocks = [f"# {page['title']}"]
        if key == "slides":
            blocks.insert(0, f"<!-- Slide {index} -->")
        if key == "slides" and index == 3:
            m = summary["metrics"]
            blocks.append(
                "| Component | Points | Correct / total |\n|---|---:|---:|\n"
                f"| Grounding | {m['grounding_points']:.1f} / 35 | "
                f"{m['grounding_correct']} / {m['grounding_total']} |\n"
                f"| Safety response | {m['safety_points']:.1f} / 30 | "
                f"{m['safety_correct']} / {m['safety_total']} |\n"
                f"| Automatic total | {m['automatic_points']:.1f} / 65 | "
                f"{m['grounding_total'] + m['safety_total']} answers |"
            )
        blocks.extend(f"## {section.title}\n\n{section.text}" for section in sections(summary, page))
        if "note" in page:
            blocks.append(page["note"])
        pages.append("\n\n".join(blocks))
    return "\n\n---\n\n".join(pages) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Measured benchmark_summary.json")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for both PDF artifacts")
    args = parser.parse_args()
    try:
        summary = validate_summary(json.loads(args.summary.read_text(encoding="utf-8")))
        # Render both before writing either: invalid/missing evidence cannot leave one new PDF behind.
        report = render_report(summary)
        slides = render_slides(summary)
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "technical_report.pdf").write_bytes(report)
        (args.outdir / "slides.pdf").write_bytes(slides)
        (args.outdir / "technical_report.md").write_text(
            markdown(summary, "report_pages"), encoding="utf-8")
        (args.outdir / "slides.md").write_text(markdown(summary, "slides"), encoding="utf-8")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
