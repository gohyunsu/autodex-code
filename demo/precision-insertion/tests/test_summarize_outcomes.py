"""Outcome funnel and alignment branch use different denominators."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_outcomes import summarize  # noqa: E402


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _attempt(root: Path, name: str, *, metric: bool, inserted: bool,
             aligned: bool) -> None:
    attempt = root / name
    for filename in ("pickup_started.json", "pickup_execution.json",
                     "transfer_execution.json", "guarded_insertion_execution.json"):
        _write(attempt / filename, {})
    _write(attempt / "result.json", {
        "candidate_id": "table/0/20195",
        "labels": {"grasp_success": True, "preinsert_reached": True,
                   "insertion_success": inserted},
        "diagnostic_labels": {"tip_entry_success": inserted},
    })
    _write(attempt / "state_000.json", {
        "candidate_id": "table/0/20195", "gap_mm": 5.0,
        "session_calibration_sha256": "session-a"})
    first = {"alignment": {
        "status": ("diagnostic_metric_xy_correction" if metric else
                   "diagnostic_planar_xy_hypothesis"),
        "tip_socket_m": [0.003, 0.004, 0.085],
        "inlier_cameras": (["a", "b"] if metric else ["a"])},
        "choice": {"status": "shift"}}
    final = {"alignment": {"status": "diagnostic_metric_xy_correction",
                            "tip_socket_m": [0.001, 0.001, 0.085],
                            "inlier_cameras": ["a", "b"]},
             "choice": {"status": "aligned" if aligned else
                        "not_metric_multiview"}}
    _write(attempt / "same_trial_xy/observations/000.json", first)
    _write(attempt / "same_trial_xy/observations/001.json", final)


def test_outcomes_separate_alignment_from_execution_funnel(tmp_path):
    _attempt(tmp_path, "attempt_a", metric=False, inserted=True, aligned=True)
    _attempt(tmp_path, "attempt_b", metric=True, inserted=False, aligned=False)
    result = summarize(tmp_path, candidate_id="table/0/20195")
    funnel = {row["stage"]: row["count"]
              for row in result["execution_funnel"]}
    assert funnel["physical_pickup_started"] == 2
    assert funnel["guarded_insertion_execution_recorded"] == 2
    assert funnel["insertion_success"] == 1
    assert result["rates"]["end_to_end_observed_success"]["proportion"] == 0.5
    assert result["alignment_branch"]["first_metric_endpoint"] == 1
    assert result["alignment_branch"]["physical_correction_attempted"] == 2
    assert result["alignment_branch"]["alignment_target_reached"] == 1


def test_outcomes_count_missing_labels_as_coverage_not_success(tmp_path):
    attempt = tmp_path / "attempt_a"
    _write(attempt / "pickup_started.json", {})
    _write(attempt / "pickup_execution.json", {})
    _write(attempt / "result.json", {
        "candidate_id": "table/0/20195", "labels": {},
        "diagnostic_labels": {}})
    result = summarize(tmp_path)
    assert result["rates"]["conditional_grasp_all_executions"]["successes"] == 0
    assert result["rates"]["conditional_grasp_all_executions"]["trials"] == 1
    assert result["rates"]["conditional_grasp_labeled"]["trials"] == 0
    assert result["execution_funnel"][2]["annotated"] == 0
