"""A metric VLM increment cannot bypass its saved physical source."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import grounded_lateral  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.lateral_preflight import LateralHoldPreflight  # noqa: E402
from precision_insertion.retry_session import (  # noqa: E402
    GroundedXYDiagnostic, write_retry_session_artifacts,
)
from test_retry_session import _inputs  # noqa: E402


def _setup(tmp_path, monkeypatch):
    inputs = _inputs(tmp_path, monkeypatch)
    mode = select_mode("cylinder", 15)
    fixture = AssetPaths(tmp_path, mode).task_geometry
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_text(json.dumps({
        "key_frame": {"tip_z_m": .08,
                      "insertion_axis": [0., 0., 1.]},
        "socket_entry_plane_z_m": .055,
    }), encoding="utf-8")
    calibration = inputs["calibration"]
    calibration.record["camera_calibration_sha256"] = "a" * 64
    start = np.eye(4)
    start[2, 3] = .30
    relation = np.diag([1., -1., -1., 1.])
    postlift_path = inputs["postlift_preflight_report_path"]
    postlift = {
        "schema": "precision_insertion_bounded_postlift_preflight_v1",
        "bounded_held_relation": {
            "T_key_hand": relation.tolist(),
            "surface_bounds": {"key_surface_m": .003,
                               "hand_surface_m": .001,
                               "source": "commissioned_future_trial_surface_bound"}},
        "targets": {"T_robot_hand_preinsert": start.tolist()},
    }
    postlift_path.write_text(json.dumps(postlift), encoding="utf-8")
    withdrawal = inputs["withdrawal_evidence_path"]
    mean = np.array([-.0015, 0.])
    increment = np.array([.001, 0.])
    covariance = np.eye(2) * 1e-10
    lower = (-2 * float(mean @ increment) - float(increment @ increment) -
             2 * math.sqrt(float(chi2.ppf(.95, df=2))) *
             math.sqrt(float(increment @ covariance @ increment)))
    uncertainty = math.sqrt(float(chi2.ppf(.95, df=2))) * 1e-5
    alignment = {
        "schema": "precision_insertion_grounded_alignment_v2",
        "status": "diagnostic_metric_xy_correction",
        "inlier_cameras": ["a", "b"],
        "estimator_limits": {"minimum_views": 2,
                             "max_lateral_uncertainty_95_m": .0005},
        "key_object": mode.key_object,
        "socket_object": mode.socket_object,
        "verification_depth_m": .02,
        "xy_correction_socket_m": [.0015, 0.],
        "bounded_xy_increment_socket_m": [.001, 0.],
        "rim_error_xy_m": mean.tolist(),
        "depth_error_xy_m": mean.tolist(),
        "mean_error_xy_m": mean.tolist(),
        "lateral_covariance_m2": covariance.tolist(),
        "lateral_uncertainty_95_m": uncertainty,
        "increment_squared_error_improvement_lower_95_m2": lower,
        "tip_socket_m": [-.0015, 0., .22],
        "insertion_axis_socket": [0., 0., -1.],
    }
    attempt = inputs["attempt"]
    diagnostic = GroundedXYDiagnostic(
        "diagnostic_metric_xy_correction", attempt.attempt_id,
        attempt.candidate_id, attempt.session_calibration_sha256,
        "a" * 64, hashlib.sha256(fixture.read_bytes()).hexdigest(),
        inputs["frame_request_id"], withdrawal,
        hashlib.sha256(withdrawal.read_bytes()).hexdigest(),
        postlift_path, hashlib.sha256(postlift_path.read_bytes()).hexdigest(),
        inputs["acquisition_metadata"]["frames"], inputs["joint_sample"],
        alignment, ())
    diagnostic_dir = write_retry_session_artifacts(
        diagnostic, inputs["frames"], tmp_path / "diagnostic")
    monkeypatch.setattr(
        grounded_lateral, "_validated_retry_trial_context",
        lambda **_kwargs: (
            next(event for event in attempt.events
                 if event["stage"] == "insertion_success"),
            postlift, postlift_path, tmp_path))
    monkeypatch.setattr(
        grounded_lateral, "validated_frozen_socket_pose",
        lambda **_kwargs: np.eye(4))
    seen = []

    def plan(**kwargs):
        seen.append(kwargs)
        goal = start.copy()
        goal[0, 3] += kwargs["increment_socket_xy_m"][0]
        trajectory = np.repeat(kwargs["start_q"][None, :], 2, axis=0)
        trajectory[1, 0] += kwargs["increment_socket_xy_m"][0]
        return LateralHoldPreflight(
            "sampled_lateral_hold_shift_pass",
            kwargs["increment_socket_xy_m"], kwargs["start_q"], start,
            goal, kwargs["T_key_hand"], trajectory,
            {"success": True}, {"sampled_clear": True})

    monkeypatch.setattr(grounded_lateral, "plan_lateral_hold_shift", plan)
    planner = SimpleNamespace(fk_wrist=lambda _q: start)
    call = dict(
        planner=planner, mode=mode, shared_root=tmp_path,
        calibration=calibration, catalog=inputs["catalog"],
        trial=inputs["trial"], attempt=attempt,
        diagnostic=diagnostic,
        diagnostic_report_path=diagnostic_dir / "report.json",
        joint_sample=inputs["joint_sample"], decision_timestamp_s=100.05,
        limits=inputs["limits"], max_hold_joint_drift_rad=.01,
        max_grounded_tip_error_m=.001,
        max_grounded_axis_error_deg=1.,
        max_path_deviation_m=.0001,
        max_hold_height_deviation_m=.0001,
        max_hold_rotation_deg=1.)
    return call, seen


def test_grounded_lateral_binds_diagnostic_and_medoid_without_retry(
        tmp_path, monkeypatch):
    call, seen = _setup(tmp_path, monkeypatch)
    result = grounded_lateral.plan_grounded_lateral_from_withdrawal(**call)
    assert result.to_record()["pending_retry"] is False
    assert result.to_record()["robot_ready"] is False
    assert result.predicted_observed_tip_error_m == pytest.approx(.0015)
    assert len(seen) == 1
    assert seen[0]["increment_socket_xy_m"] == (.001, 0.)
    assert np.allclose(seen[0]["T_key_hand"],
                       np.diag([1., -1., -1., 1.]))
    assert seen[0]["bounds"].key_surface_m == .003
    saved_dir = grounded_lateral.write_grounded_lateral_preflight(
        result, tmp_path / "grounded_plan")
    saved = json.loads((saved_dir / "report.json").read_text())
    assert saved["pending_retry"] is False
    assert len(saved["lateral_report_sha256"]) == 64
    assert (saved_dir / "lateral/lateral_trajectory.npy").is_file()


def test_grounded_lateral_rejects_changed_pixels_and_relation_mismatch(
        tmp_path, monkeypatch):
    call, seen = _setup(tmp_path, monkeypatch)
    path = call["diagnostic_report_path"].parent / "frames/a.png"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="image bytes changed"):
        grounded_lateral.plan_grounded_lateral_from_withdrawal(**call)
    assert seen == []


def test_grounded_lateral_rejects_stale_state_or_excess_tip_error(
        tmp_path, monkeypatch):
    call, seen = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="became stale"):
        grounded_lateral.plan_grounded_lateral_from_withdrawal(
            **{**call, "decision_timestamp_s": 101.0})
    changed = call["diagnostic"].alignment.copy()
    changed["tip_socket_m"] = [.02, 0., .22]
    from dataclasses import replace
    modified = replace(call["diagnostic"], alignment=changed)
    # A changed in-memory diagnostic is rejected before geometric reasoning.
    with pytest.raises(ValueError, match="differs from saved report"):
        grounded_lateral.plan_grounded_lateral_from_withdrawal(
            **{**call, "diagnostic": modified})
    assert seen == []


def test_grounded_lateral_rechecks_saved_xy_math_and_axis_direction(
        tmp_path, monkeypatch):
    from dataclasses import replace

    call, seen = _setup(tmp_path, monkeypatch)
    report = call["diagnostic_report_path"]
    original = json.loads(report.read_text(encoding="utf-8"))
    broken = call["diagnostic"].alignment.copy()
    broken["xy_correction_socket_m"] = [.0008, 0.]
    modified = replace(call["diagnostic"], alignment=broken)
    saved = dict(original, alignment=broken)
    report.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="not a valid 1 mm correction"):
        grounded_lateral.plan_grounded_lateral_from_withdrawal(
            **{**call, "diagnostic": modified})

    reversed_axis = call["diagnostic"].alignment.copy()
    reversed_axis["insertion_axis_socket"] = [0., 0., 1.]
    modified = replace(call["diagnostic"], alignment=reversed_axis)
    report.write_text(json.dumps(dict(original, alignment=reversed_axis)),
                      encoding="utf-8")
    with pytest.raises(ValueError, match="contradicts socket insertion"):
        grounded_lateral.plan_grounded_lateral_from_withdrawal(
            **{**call, "diagnostic": modified})
    assert seen == []
