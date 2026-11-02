#!/usr/bin/env python3
"""Summarize saved human/VLM label agreement without rerunning a model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ASSESSMENTS = {
    "grasp_success": "lift_assessments",
    "preinsert_reached": "preinsert_assessments",
    "insertion_success": "released_insertion_assessments",
}


def _latest_report(attempt: Path, directory: str) -> Path | None:
    reports = sorted((attempt / directory).glob("*/report.json"))
    return reports[-1] if reports else None


def summarize(attempt_root: Path) -> dict:
    """Compare labels already persisted by manual-mode live episodes."""
    root = attempt_root.expanduser().resolve()
    rows = []
    attempts = sorted(path for path in root.glob("attempt_*") if path.is_dir())
    for attempt in attempts:
        result_path = attempt / "result.json"
        result = (json.loads(result_path.read_text(encoding="utf-8"))
                  if result_path.is_file() else {})
        for stage, directory in ASSESSMENTS.items():
            report_path = _latest_report(attempt, directory)
            if report_path is None:
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manual = report.get("manual_annotation")
            vlm = report.get("vlm_label")
            manual_value = (manual.get("value")
                            if isinstance(manual, dict) else None)
            vlm_value = vlm.get("value") if isinstance(vlm, dict) else None
            if type(manual_value) is not bool and type(vlm_value) is not bool:
                continue
            rows.append({
                "attempt_id": attempt.name,
                "candidate_id": result.get("candidate_id"),
                "stage": stage,
                "manual_value": (manual_value
                                 if type(manual_value) is bool else None),
                "vlm_value": vlm_value if type(vlm_value) is bool else None,
                "vlm_class": (vlm.get("class")
                              if isinstance(vlm, dict) else None),
                "manual_note": (manual.get("note", "")
                                if isinstance(manual, dict) else ""),
                "report": str(report_path),
            })

    stages = {}
    for stage in ASSESSMENTS:
        selected = [row for row in rows if row["stage"] == stage]
        paired = [row for row in selected
                  if type(row["manual_value"]) is bool and
                  type(row["vlm_value"]) is bool]
        confusion = Counter(
            ("positive" if row["manual_value"] else "negative",
             "positive" if row["vlm_value"] else "negative")
            for row in paired)
        correct = sum(row["manual_value"] == row["vlm_value"]
                      for row in paired)
        stages[stage] = {
            "saved_assessments": len(selected),
            "manual_labels": sum(type(row["manual_value"]) is bool
                                 for row in selected),
            "vlm_labels": sum(type(row["vlm_value"]) is bool
                              for row in selected),
            "paired_labels": len(paired),
            "correct": correct,
            "accuracy": (correct / len(paired) if paired else None),
            "confusion": {
                "true_positive": confusion[("positive", "positive")],
                "true_negative": confusion[("negative", "negative")],
                "false_positive": confusion[("negative", "positive")],
                "false_negative": confusion[("positive", "negative")],
            },
        }
    return {
        "schema": "precision_insertion_saved_label_summary_v1",
        "attempt_root": str(root),
        "attempts_scanned": len(attempts),
        "stages": stages,
        "comparisons": rows,
        "scope": "saved_label_agreement_no_model_rerun_or_physical_truth_claim",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.attempt_root)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
