"""Build the report/slides summary from MEASURED artifacts, never hand-copied numbers.

Usage:
  python scripts/summarize.py --report artifacts/report.json --audit artifacts/audit.json \
      --prose docs/report-prose.json --tests-passed 804 --model-tests-passed 9 \
      --out artifacts/benchmark_summary.json

Every metric is derived from the official scorer's `--report` JSON and from the run
audit; the prose sections (team, hardware, models, verification, failures, limitations,
comparisons, method) come from the prose file, because they are human evidence that no
script may invent. The merged summary is validated against the report generator's own
contract before it is written, so a missing or malformed measurement fails HERE rather
than at PDF time.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report import validate_summary  # noqa: E402 - the generator owns the contract


def measured_metrics(report: dict) -> dict:
    """Derive the scored metrics from the official scorer's own report."""
    rows = report["instructions"]
    plans = [row for row in rows if row["expected"] == "plan"]
    non_plans = [row for row in rows if row["expected"] != "plan"]
    return {
        "automatic_points": round(report["automatic_score"], 2),
        "grounding_points": round(35 * report["grounding"], 2),
        "grounding_correct": sum(1 for row in plans if row["score"] == 1.0),
        "grounding_total": len(plans),
        "safety_points": round(30 * report["safety"], 2),
        "safety_correct": sum(1 for row in non_plans if row["score"] == 1.0),
        "safety_total": len(non_plans),
        "plans_refused": sum(
            1 for row in plans if row["answer"] in {"ask_user", "refuse"}),
    }


def measured_timing(audit: dict) -> dict:
    """Median measured timings; an empty audit is an error, not an empty object."""
    perception = [
        scene.get("perception_seconds", 0) for scene in audit.get("scenes", {}).values()
    ]
    reasoning = [
        decision.get("reasoning_seconds", 0) for decision in audit.get("decisions", [])
    ]
    timing = {}
    if perception:
        timing["perception_s_per_scene"] = round(statistics.median(perception), 1)
    if reasoning:
        timing["reasoning_s_per_instruction"] = round(statistics.median(reasoning), 1)
    return timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True,
                        help="official scorer --report JSON")
    parser.add_argument("--audit", type=Path, required=True, help="run audit JSON")
    parser.add_argument("--prose", type=Path, required=True,
                        help="human evidence: team, hardware, models, verification, "
                             "failures, limitations, comparisons, method")
    parser.add_argument("--tests-passed", type=int, required=True,
                        help="measured deterministic-suite passes")
    parser.add_argument("--model-tests-passed", type=int, required=True,
                        help="measured live-model suite passes")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    prose = json.loads(args.prose.read_text())
    report = json.loads(args.report.read_text())
    audit = json.loads(args.audit.read_text())
    summary = {
        **prose,
        "metrics": measured_metrics(report),
        "tests": {"passed": args.tests_passed, "model_passed": args.model_tests_passed},
        "timing": measured_timing(audit),
    }
    validate_summary(summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
