"""Saved manual/VLM labels remain auditable without model inference."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_live_labels import summarize  # noqa: E402


def _report(root: Path, attempt: str, directory: str, index: int,
            manual, vlm) -> None:
    path = root / attempt / directory / f"{index:03d}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "manual_annotation": manual,
        "vlm_label": vlm,
    }), encoding="utf-8")


def test_summary_uses_latest_assessment_and_counts_confusion(tmp_path):
    attempt = "attempt_a1234567"
    result = tmp_path / attempt / "result.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"candidate_id": "table/0/20195"}),
                      encoding="utf-8")
    _report(tmp_path, attempt, "lift_assessments", 0,
            {"value": True, "note": "old"},
            {"value": False, "class": "miss"})
    _report(tmp_path, attempt, "lift_assessments", 1,
            {"value": False, "note": "key fell"},
            {"value": True, "class": "held"})
    _report(tmp_path, attempt, "preinsert_assessments", 0,
            {"value": True, "note": ""},
            {"value": True, "class": "coarse_match"})

    summary = summarize(tmp_path)
    grasp = summary["stages"]["grasp_success"]
    assert grasp["paired_labels"] == 1
    assert grasp["correct"] == 0
    assert grasp["confusion"]["false_positive"] == 1
    assert summary["comparisons"][0]["candidate_id"] == "table/0/20195"
    assert summary["comparisons"][0]["manual_note"] == "key fell"
    preinsert = summary["stages"]["preinsert_reached"]
    assert preinsert["accuracy"] == 1.0


def test_unpaired_saved_label_is_counted_but_not_scored(tmp_path):
    _report(tmp_path, "attempt_b1234567", "released_insertion_assessments", 0,
            None, {"value": False, "class": "outside_socket"})
    stage = summarize(tmp_path)["stages"]["insertion_success"]
    assert stage["saved_assessments"] == 1
    assert stage["manual_labels"] == 0
    assert stage["vlm_labels"] == 1
    assert stage["paired_labels"] == 0
    assert stage["accuracy"] is None
