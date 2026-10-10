"""A VLM lift verdict must refer to saved same-camera key exposures."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.key_perception import (  # noqa: E402
    admit_postlift_key_capture, write_key_capture_artifacts,
)
from precision_insertion.lift_checkpoint import (  # noqa: E402
    assess_lift_checkpoint, verify_lift_checkpoint, write_lift_checkpoint,
)
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from test_key_perception import (  # noqa: E402
    CAMERAS, SelectorStub, _admit, _capture, _session,
)


class FakeBackend:
    model = "fake-vlm"

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def infer(self, images, prompt):
        self.calls.append((images, prompt))
        return self.answer if isinstance(self.answer, str) else json.dumps(self.answer)


def _setup(tmp_path, *, rise_m=0.1):
    mode = select_mode("square", 1.5)
    raw_mesh = (tmp_path / "object_processing" / mode.key_object /
                "raw_mesh" / f"{mode.key_object}.obj")
    raw_mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.02, 0.02, 0.08)).export(raw_mesh)
    before_pose = np.eye(4)
    before_capture = _capture(
        {camera: before_pose for camera in CAMERAS}, tmp_path)
    before_observation = _admit(
        before_capture, mode, tmp_path, SelectorStub(mode.key_object))
    before_dir = write_key_capture_artifacts(
        before_capture, before_observation, tmp_path / "before")
    after_pose = np.eye(4)
    after_pose[2, 3] = rise_m
    after_base = _capture(
        {camera: after_pose for camera in CAMERAS}, tmp_path)
    later = {"cam_a": 101.0, "cam_b": 101.005}
    after_capture = replace(
        after_base, capture_id="held_2", request_id=52,
        frame_timestamps_s=later,
        frame_evidence={
            camera: {**after_base.frame_evidence[camera],
                     "timestamp_s": later[camera]}
            for camera in CAMERAS},
        capture_dir=tmp_path / "held_2_request_52")
    calibration = _session(tmp_path, mode)
    held_observation = admit_postlift_key_capture(
        capture=after_capture,
        init_orchestrator=SelectorStub(mode.key_object),
        mode=mode, shared_root=tmp_path, calibration=calibration,
        calibrated_camera_ids=set(CAMERAS),
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        maximum_multiview_center_error_mm=2.0,
        maximum_multiview_angle_error_deg=5.0,
        candidate_pose_prior_world=after_pose,
        measured_wrist_timestamp_s=101.002,
        maximum_candidate_prior_center_error_mm=20.0,
        maximum_candidate_prior_angle_error_deg=30.0,
        maximum_prior_time_skew_s=0.02,
        minimum_refinement_iou=0.5)
    after_dir = write_key_capture_artifacts(
        after_capture, held_observation, tmp_path / "after")
    raw = np.zeros(6)
    q = np.zeros(13)
    q[7:] = convert_inspire_raw(raw[None, :])[0]
    joint = LiveRobotState(
        q, np.zeros(7), 101.002, 101.0, 123.0, 101.004,
        raw, raw.copy(), 0.0, np.zeros(6))
    backend = FakeBackend({
        "class": "held", "evidence_views": list(CAMERAS),
        "evidence": "the key rose with the hand in both views"})
    return dict(
        mode=mode, shared_root=tmp_path, calibration=calibration,
        attempt_id="trial_1", candidate_id="table/0/3",
        attempt_started_at_s=100.5, lift_completed_at_s=100.8,
        decision_timestamp_s=101.02,
        before_capture_id=before_observation.capture_id,
        before_pose_world=before_observation.pose_world,
        before_bundle=before_dir,
        after_observation=held_observation, after_bundle=after_dir,
        joint_sample=joint, expected_candidate_prior_world=after_pose,
        backend=backend, max_state_skew_s=0.02,
        max_phase_skew_s=0.02, max_lift_observation_gap_s=2.0,
        min_center_rise_m=0.03, max_arm_hand_skew_s=0.02,
        max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05), backend


def test_saved_two_view_vlm_and_observed_key_rise_are_bound(tmp_path):
    args, backend = _setup(tmp_path)
    result = assess_lift_checkpoint(**args)
    assert result.grasp_success is True
    assert result.center_rise_m == pytest.approx(0.1)
    assert len(backend.calls) == 1
    assert len(backend.calls[0][0]) == 4
    assert result.visual.parsed["class"] == "held"
    output = write_lift_checkpoint(result, tmp_path / "lift_assessment")
    report = verify_lift_checkpoint(output / "report.json")
    assert report["grasp_success"] is True
    assert report["visual"]["raw_answer"]
    assert report["visual"]["prompt"]
    assert report["robot_ready"] is False
    report_file = output / "report.json"
    original_report = report_file.read_bytes()
    changed = dict(report)
    changed["grasp_success"] = False
    report_file.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="label conflicts"):
        verify_lift_checkpoint(report_file)
    report_file.write_bytes(original_report)
    image = args["after_bundle"] / "images" / "cam_a.png"
    image.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="key evidence changed"):
        verify_lift_checkpoint(output / "report.json")


def test_saved_local_vlm_json_fence_replays_at_lift_checkpoint(tmp_path):
    args, _backend = _setup(tmp_path)
    answer = {"class": "held", "evidence_views": list(CAMERAS),
              "evidence": "key rose in both views"}
    args["backend"] = FakeBackend(
        "```json\n" + json.dumps(answer) + "\n```")
    result = assess_lift_checkpoint(**args)
    assert result.visual.parsed == answer
    saved = write_lift_checkpoint(result, tmp_path / "local_lift")
    assert verify_lift_checkpoint(saved / "report.json")["grasp_success"] is True


def test_visual_key_motion_conflict_abstains_and_prior_mismatch_blocks_vlm(
        tmp_path):
    args, backend = _setup(tmp_path, rise_m=0.01)
    result = assess_lift_checkpoint(**args)
    assert result.grasp_success is None
    assert result.reason == "visual_and_key_motion_incomplete_or_conflicting"
    other = args["expected_candidate_prior_world"].copy()
    other[0, 3] += 0.02
    with pytest.raises(ValueError, match="another candidate/wrist prior"):
        assess_lift_checkpoint(**{**args,
                                  "expected_candidate_prior_world": other})
    assert len(backend.calls) == 1
