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


def test_dynamic_image_manifest_does_not_split_semantic_prompt(tmp_path):
    for attempt, timestamp in (("attempt_one", "100.0"),
                               ("attempt_two", "200.0")):
        path = (tmp_path / attempt / "lift_assessments/000/report.json")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "manual_annotation": {"value": True, "note": ""},
            "vlm_label": {"value": True, "class": "held"},
            "visual": {
                "schema": "visible_grip_v3",
                "manifest": {"prompt": (
                    "Images are supplied in this exact order:\n"
                    f"1: after_lift/camera@{timestamp}")},
                "views": [{"prompt": "Is the peg held? Reply YES or NO."}],
            },
        }), encoding="utf-8")

    stage = summarize(tmp_path)["stages"]["grasp_success"]
    assert len(stage["prompt_sha256_values"]) == 1
    assert len(stage["full_prompt_sha256_values"]) == 2


def test_current_lift_policy_replay_uses_primary_multiview_votes(tmp_path):
    cases = [
        ("attempt_two_yes", True, [True, True, False, False], "held"),
        ("attempt_all_no", False, [False, False, False, False], "miss"),
        ("attempt_one_yes", False, [True, False, False, False],
         "unobservable"),
    ]
    for attempt, manual, votes, _expected in cases:
        path = tmp_path / attempt / "lift_assessments/000/report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "manual_annotation": {"value": manual, "note": ""},
            # Historical saved aggregate may disagree with the new vote rule.
            "vlm_label": {"value": True, "class": "held"},
            "visual": {"aggregation": {
                "schema": "precision_insertion_visible_grip_v3",
                "per_view": [
                    {"camera_id": f"cam{index}", "yes": vote}
                    for index, vote in enumerate(votes)
                ],
                "finger_focus_retry": {"per_view": [
                    {"camera_id": "cam0", "yes": True},
                    {"camera_id": "cam1", "yes": True},
                ]},
            }},
        }), encoding="utf-8")

    summary = summarize(tmp_path)
    rows = {row["attempt_id"]: row for row in summary["comparisons"]}
    for attempt, _manual, _votes, expected in cases:
        assert rows[attempt]["current_policy_replay_class"] == expected
    replay = summary["stages"]["grasp_success"]["current_policy_replay"]
    assert replay["source_assessments"] == 3
    assert replay["abstentions"] == 1
    assert replay["paired_labels"] == 2
    assert replay["correct"] == 2
    assert replay["accuracy"] == 1.0
    assert replay["confusion"] == {
        "true_positive": 1,
        "true_negative": 1,
        "false_positive": 0,
        "false_negative": 0,
    }
