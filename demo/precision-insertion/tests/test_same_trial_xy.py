"""Same-episode XY correction uses measured multiview residuals and replans."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.guarded_axial_handoff import (  # noqa: E402
    verify_guarded_axial_handoff,
)
from precision_insertion.same_trial_xy import (  # noqa: E402
    _alignment_frame_inputs, _shifted_targets, choose_xy_move,
)
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.targets import InsertionTargets  # noqa: E402


def test_same_trial_move_requires_metric_multiview_and_caps_one_motion():
    metric = {
        "status": "diagnostic_metric_xy_correction",
        "inlier_cameras": ["cam1", "cam2"],
        "xy_correction_socket_m": [.024, -.018],
    }
    move = choose_xy_move(metric, tolerance_m=.0025)
    assert move["status"] == "shift"
    np.testing.assert_allclose(
        np.linalg.norm(move["increment_socket_m"]), .010)
    assert choose_xy_move({**metric, "inlier_cameras": ["cam1"]},
                          tolerance_m=.0025)["status"] == "not_metric_multiview"
    assert choose_xy_move({**metric,
                           "status": "diagnostic_planar_xy_hypothesis"},
                          tolerance_m=.0025)["status"] == "not_metric_multiview"
    assert choose_xy_move({**metric,
                           "xy_correction_socket_m": [.001, -.001]},
                          tolerance_m=.0025)["status"] == "aligned"


def test_live_xy_measurement_reuses_just_saved_pixels(tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    frames = {"100": np.zeros((3, 4, 3), dtype=np.uint8),
              "200": np.ones((3, 4, 3), dtype=np.uint8)}
    raw = RawCameraCapture(
        capture_id="current", request_id=17, images_bgr=frames,
        frame_ids={"100": 1, "200": 2}, acquisition_metadata={})
    (tmp_path / "manifest.json").write_text(json.dumps({
        "phase": "preinsert", "capture_id": "current", "request_id": 17,
        "frame_evidence": {"100": {"timestamp_s": 1.0},
                           "200": {"timestamp_s": 1.0}},
    }), encoding="utf-8")
    monkeypatch.setattr(same_trial, "verify_raw_camera_capture",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("live path reread saved PNGs")))
    rows, images = _alignment_frame_inputs(tmp_path, raw)
    assert set(rows) == set(images) == {"100", "200"}
    assert images["100"] is frames["100"]
    assert images["200"] is frames["200"]


def test_corrected_axial_keeps_key_at_observed_residual_and_moves_wrist():
    mode = select_mode("cylinder", 5)
    key = np.eye(4)
    key[2, 3] = .05
    hand = key.copy()
    hand[0, 3] = .02
    entry = key.copy()
    entry[2, 3] = .04
    verification = key.copy()
    verification[2, 3] = .02
    original = InsertionTargets(
        mode, "a" * 64, "b" * 64, np.linalg.inv(key) @ hand,
        key, entry, verification, hand,
        hand.copy(), hand.copy(), np.array([0., 0., -1.]),
        (0., 0.), .01)
    socket = np.eye(4)
    corrected = _shifted_targets(
        original, socket, np.array([.003, -.001]),
        np.array([-.002, .001]))
    np.testing.assert_allclose(
        corrected.T_robot_key_preinsert[:2, 3], [-.002, .001])
    np.testing.assert_allclose(
        corrected.T_robot_hand_preinsert[:2, 3], [.023, -.001])
    np.testing.assert_allclose(
        corrected.T_robot_key_preinsert @ corrected.T_key_hand,
        corrected.T_robot_hand_preinsert)
    np.testing.assert_allclose(corrected.xy_offset_socket_m, [-.002, .001])


def test_corrected_handoff_dispatch_binds_latest_snapshot(tmp_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source = tmp_path / "guarded_axial_handoffs" / "000"
    source.mkdir(parents=True)
    postlift = tmp_path / "postlift.json"
    preinsert = tmp_path / "preinsert.json"
    final = tmp_path / "last_preinsert"
    final.mkdir()
    manifest = final / "manifest.json"
    for path in (postlift, preinsert, manifest):
        path.write_text("{}", encoding="utf-8")
    axial = np.stack((np.zeros(13), np.array([.01] + [0.] * 12)))
    archive = source / "planned_trajectories.npz"
    np.savez_compressed(archive, axial=axial)
    report = source / "report.json"
    report.write_text(json.dumps({
        "schema": "precision_insertion_same_trial_axial_handoff_v1",
        "robot_ready": False,
        "scope": "same_trial_replanned_axial_path_not_physical_key_depth",
        "postlift_report_path": str(postlift),
        "postlift_report_sha256": digest(postlift),
        "preinsert_report_path": str(preinsert),
        "preinsert_report_sha256": digest(preinsert),
        "final_preinsert_bundle_path": str(final),
        "final_preinsert_manifest_sha256": digest(manifest),
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": digest(archive),
        "axial_sample_count": len(axial),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "measured_start": {"full_q": axial[0].tolist()},
        "planning": {"sampled_planning_pass": True,
                     "sampled_held_path_audit": {"sampled_clear": True}},
        "observations": [], "shifts": [],
    }), encoding="utf-8")
    assert verify_guarded_axial_handoff(report)["axial_sample_count"] == 2
    manifest.write_text('{"changed": true}', encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="capture changed"):
        verify_guarded_axial_handoff(report)


def test_corrected_axial_screens_measured_hand_endpoint_before_planning(
    tmp_path, monkeypatch,
):
    import precision_insertion.same_trial_xy as same_trial

    mode = select_mode("cylinder", 5)
    pose = np.eye(4)
    original = InsertionTargets(
        mode, "a" * 64, "b" * 64, pose, pose, pose, pose,
        pose, pose, pose, np.array([0., 0., -1.]), (0., 0.), .03)
    state = SimpleNamespace(
        full_q=np.zeros(13), to_record=lambda: {"full_q": [0.] * 13})
    runner = SimpleNamespace(
        mode=mode, shared_root=tmp_path, calibration=None,
        _attempt_dir=tmp_path / "attempt", _guarded_axial_index=0,
        _postlift_preflight=SimpleNamespace(targets=original),
        _preflight=SimpleNamespace(selected_candidate_key=("table", "0", "42")),
        active_attempt=SimpleNamespace(attempt_id="attempt_1", candidate_id="table/0/42"),
    )
    (tmp_path / "attempt").mkdir()
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: pose)
    monkeypatch.setattr(same_trial, "build_rigid_insertion_targets",
                        lambda **_kwargs: original)
    monkeypatch.setattr(same_trial, "screen_grasp_endpoint",
                        lambda **_kwargs: {
                            "endpoint_pass": False,
                            "T_socket_key_tested": pose.tolist(),
                            "hand_socket_clear_at_20mm": False,
                        })
    monkeypatch.setattr(same_trial, "plan_held_transfer_and_axial",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("collision endpoint reached axial planner")))
    with pytest.raises(RuntimeError, match="endpoint intersects"):
        same_trial.plan_corrected_axial_handoff(
            runner=runner,
            planner=SimpleNamespace(fk_wrist=lambda _q: pose),
            path_limits=SimpleNamespace(minimum_hand_clearance_m=0.),
            read_state=lambda: state, total_xy_m=(0., 0.),
            residual_xy_m=(0., 0.), residual_source="test",
            observations=[], shifts=[], final_preinsert_bundle=tmp_path)
    failure = json.loads((tmp_path / "attempt/guarded_axial_handoffs/000/plan_failure.json").read_text())
    assert failure["status"] == "corrected_20mm_endpoint_rejected"
    assert failure["endpoint"]["hand_socket_clear_at_20mm"] is False
