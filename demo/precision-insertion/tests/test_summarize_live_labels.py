"""Saved manual/VLM labels remain auditable without model inference."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_live_labels import summarize  # noqa: E402


def _report(root: Path, attempt: str, directory: str, index: int,
            manual, vlm, *, policy="test_visual_policy_v1",
            prompt="test prompt") -> None:
    path = root / attempt / directory / f"{index:03d}" / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "manual_annotation": manual,
        "vlm_label": vlm,
        "visual": {
            "prompt": prompt,
            "aggregation": {"schema": policy},
        },
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
    assert summary["comparisons"][0]["visual_policy"] == (
        "test_visual_policy_v1")
    assert len(grasp["policies"]) == 1
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


def test_summary_does_not_mix_historical_vlm_policies(tmp_path):
    _report(tmp_path, "attempt_old", "lift_assessments", 0,
            {"value": False, "note": "miss"},
            {"value": True, "class": "held"}, policy="visible_grip_v3",
            prompt="old one-view consensus")
    _report(tmp_path, "attempt_new", "lift_assessments", 0,
            {"value": False, "note": "miss"},
            {"value": False, "class": "miss"}, policy="visible_grip_v4",
            prompt="new multiview consensus")

    stage = summarize(tmp_path)["stages"]["grasp_success"]
    assert stage["paired_labels"] == 2
    assert stage["accuracy"] == 0.5
    assert len(stage["policies"]) == 2
    old = next(value for value in stage["policies"].values()
               if value["visual_policy"] == "visible_grip_v3")
    new = next(value for value in stage["policies"].values()
               if value["visual_policy"] == "visible_grip_v4")
    assert old["accuracy"] == 0.0
    assert old["confusion"]["false_positive"] == 1
    assert new["accuracy"] == 1.0
    assert new["confusion"]["true_negative"] == 1
