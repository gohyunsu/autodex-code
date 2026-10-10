"""A transfer log alone cannot substitute for fresh multi-view arrival."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import postshift_arrival_checkpoint as arrival  # noqa: E402
from precision_insertion import postshift_transfer_execution as transfer  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.observer import VLMObservation  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, write_raw_camera_capture,
)
from test_postshift_transfer_execution import _setup  # noqa: E402


def _case(tmp_path, monkeypatch):
    executable = _setup(tmp_path, monkeypatch)
    transfer.execute_bound_postshift_transfer(
        **executable, enable_robot_motion=True)
    prior = executable["checkpoint"].capture_dir
    images = {camera: cv2.imread(str(prior / "images" / f"{camera}.png"))
              for camera in ("a", "b")}
    frame_ids = {"a": 51, "b": 52}
    frames = {camera: {
        "frame_id": frame_ids[camera],
        "image_sha256": image_sha256(images[camera]),
        "timestamp_s": 100.9, "max_error_s": .001,
        "timestamp_method": "hardware_exposure", "clock_domain": "unix_utc",
    } for camera in images}
    bundle = write_raw_camera_capture(RawCameraCapture(
        "retry_preinsert", 19, images, frame_ids,
        {"request_id": 19, "source": "camera_acquisition", "frames": frames}),
        tmp_path / "retry_preinsert", phase="preinsert")
    preflight = executable["expected"]
    q = np.asarray(json.loads(executable["handoff_report_path"].read_text())[
        "transfer_end_q"])
    joint = replace(executable["pre_state"], full_q=q,
                    sample_timestamp_s=100.9, arm_timestamp_s=100.9,
                    hand_timestamp_s=100.9)
    target = preflight.targets.T_robot_hand_preinsert
    tip_z = .08
    predicted_tip = (target @ np.linalg.inv(
        preflight.hypothesis.T_key_hand) @
        np.array([0., 0., tip_z, 1.]))[:3]
    visual = {
        "schema": "precision_insertion_grounded_alignment_v2",
        "status": "diagnostic_metric_xy_correction",
        "inlier_cameras": ["a", "b"],
        "tip_socket_m": predicted_tip.tolist(),
        "rim_error_xy_m": [.0002, 0.],
        "depth_error_xy_m": [.0002, 0.],
        "lateral_uncertainty_95_m": .0001,
        "axis_tilt_deg": .1,
    }
    monkeypatch.setattr(arrival, "validated_frozen_socket_pose",
                        lambda **_kwargs: np.eye(4))
    monkeypatch.setattr(arrival, "validate_session_camera_calibration",
                        lambda *_args, **_kwargs: None)
    def observe(_backend, views):
        records = tuple(VLMObservation(
            "cylinder_tip_axis_line_grounding",
            {"tip_px": [500., 400.],
             "axis_line_px": [[500., 350.], [500., 300.]],
             "evidence": "visible shaft"},
            '{"tip_px":[500,400],"axis_line_px":[[500,350],[500,300]],'
            '"evidence":"visible shaft"}',
            "synthetic unit-test prompt",
            (f"raw_preinsert_hold/{view.camera_id}@"
             f"{view.timestamp_s:.6f}",),
            None, "test-backend", .01) for view in views)
        return [], records

    monkeypatch.setattr(arrival, "observe_grounded_cylinder_axis", observe)
    monkeypatch.setattr(arrival, "estimate_grounded_line_alignment",
                        lambda *_args, **_kwargs: visual.copy())
    args = dict(
        preflight=preflight, checkpoint=executable["checkpoint"],
        shift_plan=executable["shift_plan"],
        handoff_report_path=executable["handoff_report_path"],
        transfer_execution_path=(executable["runner"]._attempt_dir /
            "postshift_transfer_executions/000/execution.json"),
        capture_dir=bundle, joint_sample=joint,
        decision_timestamp_s=100.95,
        planner=SimpleNamespace(fk_wrist=lambda _q: target),
        mode=executable["runner"].mode,
        shared_root=tmp_path, calibration=executable["runner"].calibration,
        intrinsics_full={camera: {"K_undist": [[1000., 0., 500.],
                                                   [0., 1000., 400.],
                                                   [0., 0., 1.]]}
                         for camera in images},
        extrinsics_full={camera: np.eye(4) for camera in images},
        backend=object(),
        alignment_limits=SimpleNamespace(validate=lambda: None,
                                         minimum_views=2),
        max_capture_skew_s=.01,
        max_execution_observation_gap_s=.4,
        max_frame_age_s=.2,
        max_joint_frame_skew_s=.02,
        max_arm_hand_skew_s=.02,
        max_hand_command_error_raw=30.,
        max_arm_velocity_rad_s=.05,
        max_joint_goal_error_rad=.01,
        max_goal_translation_error_m=.2,
        max_goal_rotation_error_deg=5.,
        max_visual_lateral_error_m=.005,
        max_visual_axis_tilt_deg=2.,
        max_grounded_tip_error_m=.001,
    )
    return args, visual


def test_fresh_transfer_arrival_is_observation_only(tmp_path, monkeypatch):
    args, visual = _case(tmp_path, monkeypatch)
    result = arrival.assess_postshift_arrival(**args)
    assert result.status == "visual_alignment_within_budget"
    assert result.to_record()["axial_retry_allowed"] is False
    path = arrival.write_postshift_arrival_checkpoint(
        result, tmp_path / "saved_arrival")
    assert arrival.verify_postshift_arrival_checkpoint(
        result, path, preflight=args["preflight"],
        checkpoint=args["checkpoint"], shift_plan=args["shift_plan"],
        mode=args["mode"], shared_root=tmp_path,
        calibration=args["calibration"])["status"] == result.status
    visual["rim_error_xy_m"] = [.008, 0.]
    assert arrival.assess_postshift_arrival(
        **args).status == "residual_requires_new_shift"


def test_centered_axis_need_not_have_beneficial_xy_correction(
        tmp_path, monkeypatch):
    args, visual = _case(tmp_path, monkeypatch)
    visual["status"] = "abstain"
    visual["reason"] = "continuous_xy_correction_not_confident"
    visual["rim_error_xy_m"] = [0., 0.]
    visual["depth_error_xy_m"] = [0., 0.]
    assert arrival.assess_postshift_arrival(
        **args).status == "visual_alignment_within_budget"


def test_arrival_rejects_reused_camera_frames(tmp_path, monkeypatch):
    args, _ = _case(tmp_path, monkeypatch)
    manifest = args["capture_dir"] / "manifest.json"
    saved = json.loads(manifest.read_text())
    saved["request_id"] = 16  # matches the previous post-lateral hold
    manifest.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="new synchronized camera frames"):
        arrival.assess_postshift_arrival(**args)


def test_arrival_rejects_changed_transfer_source(tmp_path, monkeypatch):
    args, _ = _case(tmp_path, monkeypatch)
    source = Path(args["transfer_execution_path"])
    source.write_text("{}")
    with pytest.raises(ValueError, match="retry transfer log differs"):
        arrival.assess_postshift_arrival(**args)
