"""Fixed-grasp repeatability summary keeps wrist and key evidence separate."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_repeatability import summarize  # noqa: E402


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _attempt(root: Path, name: str, *, wrist_mm: float,
             alignment: dict, target_x: float = 0.5) -> None:
    attempt = root / name
    _write(attempt / "result.json", {
        "candidate_id": "table/0/20195",
        "labels": {"grasp_success": True, "preinsert_reached": True,
                   "insertion_success": False},
        "diagnostic_labels": {"tip_entry_success": False},
    })
    target = [[1, 0, 0, target_x], [0, 1, 0, 0.1],
              [0, 0, 1, 0.3], [0, 0, 0, 1]]
    measured = [[1, 0, 0, target_x + wrist_mm / 1000.0],
                [0, 1, 0, 0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]]
    _write(attempt / "postlift_preflights/000/report.json", {
        "targets": {"T_robot_hand_preinsert": target}})
    _write(attempt / "preinsert_assessments/000/report.json", {
        "candidate_id": "table/0/20195",
        "session_calibration_sha256": "session-a",
        "hand_translation_error_m": wrist_mm / 1000.0,
        "hand_rotation_error_deg": 0.5,
        "comparison": {"prediction": {
            "T_robot_hand_measured_fk": measured}},
        "nominal_source_context": {"family": "cylinder", "gap_mm": 5.0},
    })
    _write(attempt / "same_trial_xy/observations/000.json", {
        "alignment": alignment})


def test_repeatability_excludes_single_view_planar_tip(tmp_path):
    _attempt(tmp_path, "attempt_a", wrist_mm=1.0, alignment={
        "status": "diagnostic_planar_xy_hypothesis",
        "reason": "single_view_nominal_tip_height_plane",
        "tip_socket_m": [0.08, 0.01, 0.085],
        "inlier_cameras": ["cam-a"],
    })
    _attempt(tmp_path, "attempt_b", wrist_mm=3.0, alignment={
        "status": "diagnostic_metric_xy_correction",
        "reason": "multiview_tip_xy_vertical_axis_assumption",
        "tip_socket_m": [0.003, 0.004, 0.085],
        "inlier_cameras": ["cam-a", "cam-b"],
        "axis_tilt_deg": 2.0,
    })

    result = summarize(tmp_path, "table/0/20195")
    assert result["attempts_with_preinsert_measurement"] == 2
    group = result["groups"][0]
    assert group["same_nominal_preinsert_target"] is True
    assert group["wrist_translation_error_mm"]["median"] == 2.0
    assert group["metric_key_tip_count"] == 1
    assert group["metric_tip_xy_residual_mm"]["median"] == 5.0
    assert group["first_tip_status_counts"] == {
        "diagnostic_metric_xy_correction": 1,
        "diagnostic_planar_xy_hypothesis": 1,
    }
    planar = result["attempts"][0]["first_preinsert_key_tip"]
    assert planar["metric_observation"] is False
    assert planar["tip_socket_m"] is None
    assert planar["diagnostic_tip_socket_m"] == [0.08, 0.01, 0.085]


def test_repeatability_detects_changed_nominal_target(tmp_path):
    alignment = {"status": "abstain", "reason": "no_tip"}
    _attempt(tmp_path, "attempt_a", wrist_mm=1.0, alignment=alignment,
             target_x=0.5)
    _attempt(tmp_path, "attempt_b", wrist_mm=1.0, alignment=alignment,
             target_x=0.51)
    group = summarize(tmp_path)["groups"][0]
    assert group["same_nominal_preinsert_target"] is False
    assert len(group["preinsert_target_sha256_values"]) == 2
