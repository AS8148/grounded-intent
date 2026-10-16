"""The report summary must be derived from measured artifacts, never transcribed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import summarize  # noqa: E402

from report import validate_summary  # noqa: E402


def prose() -> dict:
    return {
        "team": "Grounded Intent",
        "hardware": "synthetic",
        "models": {
            "vision": {"repo": "qwen", "revision": "r", "quantization": "Q8_0"},
            "apertus": {"repo": "apertus", "revision": "r", "quantization": "Q8_0"},
            "segmenter": "sam2",
        },
        "verification": {"docker": "synthetic"},
        "failures": [{"instruction": "x", "observation": "y", "mitigation": "z"}],
        "limitations": ["synthetic limitation"],
        "comparisons": [{"note": "synthetic comparison"}],
        "method": "synthetic",
        "report_pages": [
            {"title": f"Report page {index}", "sections": [{"title": "Evidence", "text": "Measured."}]}
            for index in range(2)
        ],
        "slides": [
            {
                "title": f"Slide {index}",
                "sections": [{"title": "Evidence", "text": "Measured."} for _ in range(count)],
                "note": "Diagnostic scope.",
            }
            for index, count in enumerate((3, 5, 3, 4, 4))
        ],
    }


def report_rows() -> dict:
    return {
        "grounding": 0.5, "safety": 0.6, "caught": 0.8, "false_alarm_rate": 0.25,
        "automatic_score": 35.5,
        "instructions": [
            {"instruction_id": "a", "expected": "plan", "answer": "plan", "score": 1.0},
            {"instruction_id": "b", "expected": "plan", "answer": "plan", "score": 0.0},
            {"instruction_id": "c", "expected": "plan", "answer": "refuse", "score": 0.0},
            {"instruction_id": "d", "expected": "plan", "answer": "plan", "score": 1.0},
            {"instruction_id": "e", "expected": "refuse", "answer": "refuse", "score": 1.0},
            {"instruction_id": "f", "expected": "ask_user", "answer": "plan", "score": 0.0},
        ],
    }


def test_the_summary_metrics_come_from_the_scorer_report(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_rows()))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({
        "scenes": {"s1": {"perception_seconds": 10.0}, "s2": {"perception_seconds": 20.0}},
        "decisions": [{"reasoning_seconds": 2.0}, {"reasoning_seconds": 4.0}],
    }))
    prose_path = tmp_path / "prose.json"
    prose_path.write_text(json.dumps(prose()))
    out = tmp_path / "summary.json"

    assert summarize.main([
        "--report", str(report_path), "--audit", str(audit_path),
        "--prose", str(prose_path), "--tests-passed", "12",
        "--model-tests-passed", "3", "--out", str(out),
    ]) == 0

    summary = json.loads(out.read_text())
    validate_summary(summary)  # the generator's own contract
    metrics = summary["metrics"]
    assert metrics == {
        "automatic_points": 35.5,
        "grounding_points": 17.5,
        "grounding_correct": 2,
        "grounding_total": 4,
        "safety_points": 18.0,
        "safety_correct": 1,
        "safety_total": 2,
        "plans_refused": 1,
    }
    assert summary["timing"] == {
        "perception_s_per_scene": 15.0, "reasoning_s_per_instruction": 3.0}
    assert summary["tests"] == {"passed": 12, "model_passed": 3}


def test_a_missing_prose_section_fails_before_any_pdf(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_rows()))
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps({"scenes": {"s1": {"perception_seconds": 1.0}},
                                      "decisions": [{"reasoning_seconds": 1.0}]}))
    incomplete = prose()
    del incomplete["failures"]
    prose_path = tmp_path / "prose.json"
    prose_path.write_text(json.dumps(incomplete))

    with pytest.raises(ValueError, match="failures"):
        summarize.main([
            "--report", str(report_path), "--audit", str(audit_path),
            "--prose", str(prose_path), "--tests-passed", "1",
            "--model-tests-passed", "0", "--out", str(tmp_path / "summary.json"),
        ])


@pytest.mark.parametrize(("key", "count"), [
    ("report_pages", 1), ("report_pages", 3), ("slides", 4), ("slides", 6),
])
def test_summary_rejects_wrong_deliverable_page_counts(key, count):
    summary = prose()
    summary.update(
        metrics=summarize.measured_metrics(report_rows()),
        tests={"passed": 12, "model_passed": 0},
        timing={"perception_s_per_scene": 1.0},
    )
    summary[key] = [summary[key][0]] * count
    with pytest.raises(ValueError, match=rf"summary\.{key} must contain exactly"):
        validate_summary(summary)
