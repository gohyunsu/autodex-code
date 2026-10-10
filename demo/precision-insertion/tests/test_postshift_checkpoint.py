"""A lateral plan is not an insertion retry until fresh frames follow motion."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import grounded_lateral, postshift_checkpoint  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from test_grounded_lateral import _setup as _lateral_setup  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path, monkeypatch):
    prior, _seen = _lateral_setup(tmp_path, monkeypatch)
    plan = grounded_lateral.plan_grounded_lateral_from_withdrawal(**prior)
    plan_dir = grounded_lateral.write_grounded_lateral_preflight(
        plan, tmp_path / "shift_plan")
    plan_report = plan_dir / "report.json"
    source_records = {}
    for name in ("trajectory_feedback", "safety", "grasp_state"):
        file = tmp_path / f"{name}.json"
        file.write_text('{"source":"test_claim_only"}', encoding="utf-8")
        source_records[name] = {"path": str(file), "sha256": _sha(file)}
    execution = tmp_path / "lateral_execution.json"
    execution.write_text(json.dumps({
        "schema": "precision_insertion_lateral_hold_execution_v1",
        "source": "commissioned_lateral_controller",
        "attempt_id": plan.attempt_id,
        "candidate_id": plan.candidate_id,
        "preflight_report_path": str(plan_report),
        "preflight_report_sha256": _sha(plan_report),
        "started_at_s": 100.1, "completed_at_s": 100.2,
        "measurement": {"trajectory_complete": True,
                        "safety_abort": False, "grasp_held": True},
        "source_records": source_records,
    }), encoding="utf-8")
    bgr = cv2.imread(str(plan.diagnostic_report_path.parent / "frames/a.png"))
    images = {camera: bgr.copy() for camera in ("a", "b")}
    frame_ids = {"a": 41, "b": 42}
    frames = {camera: {
        "frame_id": frame_ids[camera],
        "image_sha256": image_sha256(images[camera]),
        "timestamp_s": 100.3,
        "max_error_s": .001,
        "timestamp_method": "hardware_exposure",
        "clock_domain": "unix_utc",
    } for camera in images}
    capture = write_raw_camera_capture(RawCameraCapture(
        "post_xy_1", 16, images, frame_ids,
        {"request_id": 16, "source": "camera_acquisition", "frames": frames}),
        tmp_path / "postshift_images", phase="post_lateral_hold")
    old = prior["joint_sample"]
    q = old.full_q.copy()
    q[0] += .001
    joint = replace(old, full_q=q, sample_timestamp_s=100.3,
                    arm_timestamp_s=100.3, hand_timestamp_s=100.3,
                    arm_robot_uptime_s=10.3)
    def fk(q):
        wrist = np.eye(4)
        wrist[:3, 3] = q[:3]
        return wrist
    monkeypatch.setattr(postshift_checkpoint,
                        "validated_frozen_socket_pose",
                        lambda **_kwargs: np.eye(4))
    monkeypatch.setattr(postshift_checkpoint,
                        "validate_session_camera_calibration",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(postshift_checkpoint,
                        "verify_bounded_postlift_preflight",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(postshift_checkpoint,
                        "observe_grounded_cylinder_axis",
                        lambda *_args, **_kwargs: ([], ()))
    alignment = {
        "status": "abstain",
        "reason": "continuous_xy_correction_not_confident",
        "tip_socket_m": [.001, 0., .22],
        "rim_error_xy_m": [.001, 0.],
        "depth_error_xy_m": [.001, 0.],
        "lateral_uncertainty_95_m": .0001,
        "axis_tilt_deg": 0.,
    }
    monkeypatch.setattr(postshift_checkpoint,
                        "estimate_grounded_line_alignment",
                        lambda *_args, **_kwargs: alignment.copy())
    call = dict(
        plan=plan, plan_report_path=plan_report,
        execution_log_path=execution, capture_dir=capture,
        joint_sample=joint, decision_timestamp_s=100.35,
        mode=prior["mode"], shared_root=tmp_path,
        calibration=prior["calibration"],
        planner=SimpleNamespace(fk_wrist=fk),
        intrinsics_full={camera: {"K_undist": [[1000., 0., 500.],
                                                  [0., 1000., 400.],
                                                  [0., 0., 1.]]}
                         for camera in images},
        extrinsics_full={camera: np.eye(4) for camera in images},
        backend=object(),
        alignment_limits=SimpleNamespace(
            validate=lambda: None, minimum_views=2),
        max_capture_skew_s=.01,
        max_execution_observation_gap_s=.2,
        max_frame_age_s=.2,
        max_joint_frame_skew_s=.02,
        max_arm_hand_skew_s=.02,
        max_hand_command_error_raw=30.,
        max_arm_velocity_rad_s=.05,
        max_joint_goal_error_rad=.01,
        max_goal_translation_error_m=.002,
        max_goal_rotation_error_deg=1.,
        max_visual_lateral_error_m=.005,
        max_visual_axis_tilt_deg=2.,
        max_grounded_tip_error_m=.001,
    )
    return call, alignment


def test_postshift_requires_new_capture_and_returns_visual_only(
        tmp_path, monkeypatch):
    call, alignment = _case(tmp_path, monkeypatch)
    result = postshift_checkpoint.assess_postshift_alignment(**call)
    assert result.status == "visual_alignment_within_budget"
    assert result.to_record()["insertion_replan_allowed"] is False
    assert result.to_record()["robot_ready"] is False
    path = postshift_checkpoint.write_postshift_checkpoint(
        result, tmp_path / "postshift_checkpoint")
    assert json.loads(path.read_text())["status"] == result.status
    assert postshift_checkpoint.verify_postshift_checkpoint(
        result, path, plan=call["plan"])["status"] == result.status
    alignment["rim_error_xy_m"] = [.008, 0.]
    # The patched estimator returns a copy of the current synthetic result.
    assert postshift_checkpoint.assess_postshift_alignment(
        **call).status == "residual_requires_new_shift"


def test_postshift_rejects_old_frames_abort_and_changed_path(
        tmp_path, monkeypatch):
    call, _alignment = _case(tmp_path, monkeypatch)
    manifest = call["capture_dir"] / "manifest.json"
    old = json.loads(manifest.read_text())
    old["request_id"] = 15
    manifest.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(ValueError, match="new synchronized camera frames"):
        postshift_checkpoint.assess_postshift_alignment(**call)
    old["request_id"] = 16
    manifest.write_text(json.dumps(old), encoding="utf-8")
    log = json.loads(call["execution_log_path"].read_text())
    log["measurement"]["safety_abort"] = True
    call["execution_log_path"].write_text(json.dumps(log), encoding="utf-8")
    with pytest.raises(ValueError, match="does not confirm"):
        postshift_checkpoint.assess_postshift_alignment(**call)
    log["measurement"]["safety_abort"] = False
    call["execution_log_path"].write_text(json.dumps(log), encoding="utf-8")
    path = call["plan_report_path"].parent / "lateral/lateral_trajectory.npy"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        postshift_checkpoint.assess_postshift_alignment(**call)


def test_postshift_rechecks_original_diagnostic_pixels(
        tmp_path, monkeypatch):
    call, _alignment = _case(tmp_path, monkeypatch)
    old_image = (call["plan"].diagnostic_report_path.parent /
                 "frames/a.png")
    with old_image.open("ab") as stream:
        stream.write(b"changed after planning")
    with pytest.raises(ValueError, match="grounded source image bytes changed"):
        postshift_checkpoint.assess_postshift_alignment(**call)


def test_saved_postshift_checkpoint_rechecks_capture_pixels(
        tmp_path, monkeypatch):
    call, _alignment = _case(tmp_path, monkeypatch)
    result = postshift_checkpoint.assess_postshift_alignment(**call)
    path = postshift_checkpoint.write_postshift_checkpoint(
        result, tmp_path / "postshift_checkpoint")
    image = call["capture_dir"] / "images/a.png"
    with image.open("ab") as stream:
        stream.write(b"changed after checkpoint")
    with pytest.raises(ValueError, match="raw camera PNG changed"):
        postshift_checkpoint.verify_postshift_checkpoint(
            result, path, plan=call["plan"])
