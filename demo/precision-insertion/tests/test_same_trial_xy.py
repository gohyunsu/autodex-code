"""Same-episode XY correction uses measured multiview residuals and replans."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.guarded_axial_handoff import (  # noqa: E402
    verify_guarded_axial_handoff,
)
from precision_insertion.same_trial_xy import (  # noqa: E402
    _shifted_targets, choose_xy_move,
)
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
