"""A visible held key can drive the existing observed post-lift v8 replan."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_observed_relation_handlers as module  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


def _sample():
    return LiveRobotState(
        np.zeros(13), np.zeros(7), 100.0, 100.0, 1.0, 100.0,
        np.zeros(6), np.zeros(6), 0.0, np.zeros(6))


def _parameters():
    return {
        "max_state_skew_s": .05, "max_arm_hand_skew_s": .02,
        "max_hand_command_error_raw": 30., "max_arm_velocity_rad_s": .1,
        "max_grasp_translation_drift_m": .03,
        "max_grasp_rotation_drift_deg": 20.,
        "axial_waypoint_step_m": .002,
    }


def _capture_parameters():
    return {
        "maximum_multiview_center_error_mm": 10.,
        "maximum_multiview_angle_error_deg": 10.,
        "maximum_candidate_prior_center_error_mm": 30.,
        "maximum_candidate_prior_angle_error_deg": 30.,
        "maximum_prior_time_skew_s": .05,
        "minimum_refinement_iou": .2,
        "capture_timeout_s": 5., "image_write_timeout_s": 5.,
    }


def test_observed_relation_uses_fresh_foundpose_and_measured_feedback(
        monkeypatch, tmp_path):
    decision = SessionDecision("held_relation_evidence_required", "test",
                               "table/0/1")
    received = {}
    observed = SimpleNamespace(capture_id="held_1")
    capture = SimpleNamespace(
        capture_id="held_1", request_id=17,
        images_bgr={camera: np.zeros((2, 2, 3), dtype=np.uint8)
                    for camera in ("cam_a", "cam_b")},
        frame_evidence={camera: {"frame_id": index}
                        for index, camera in enumerate(("cam_a", "cam_b"), 1)})
    monkeypatch.setattr(module, "collect_key_capture",
                        lambda **kwargs: received.setdefault("capture", kwargs)
                        and capture)
    monkeypatch.setattr(module, "admit_postlift_key_capture",
                        lambda **kwargs: received.setdefault("admit", kwargs)
                        and observed)
    monkeypatch.setattr(module, "write_key_capture_artifacts",
                        lambda capture, observation, output:
                        received.setdefault("write", (capture, observation, output)))
    monkeypatch.setattr(module, "verify_key_capture_artifacts",
                        lambda output: received.setdefault("verify", output))

    class Runner:
        mode = SimpleNamespace(key_object="key")
        shared_root = tmp_path
        output_dir = tmp_path
        calibration = object()

        def current_decision(self):
            return decision

        def postlift_candidate_pose_prior(self, **kwargs):
            received["prior"] = kwargs
            return np.eye(4)

        def prepare_postlift_transfer(self, **kwargs):
            received["plan"] = kwargs
            return SimpleNamespace(status="sampled_postlift_preflight_pass")

    runner = Runner()
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        cameras=SimpleNamespace(
            init_orchestrator=object(),
            acquisition_metadata_for_request=lambda request: None,
            launch=SimpleNamespace(
                camera_ids=("cam_a", "cam_b"), capture_root=tmp_path,
                view_limits=object())))
    handler = module.LiveObservedHeldRelationHandlers(
        live=live, planner=object(),
        state_for_capture=lambda raw: received.setdefault("raw", raw) and _sample(),
        key_prompt="white key handle", path_limits=object(),
        capture_parameters=_capture_parameters(),
        parameters=_parameters(), capture_id_factory=lambda: "held_1")
    handler.handlers()[decision.action](runner, decision)
    assert received["capture"]["init_orchestrator"] is \
        live.cameras.init_orchestrator
    assert received["admit"]["candidate_pose_prior_world"].shape == (4, 4)
    assert received["raw"].capture_id == "held_1"
    assert received["plan"]["key_observation"] is observed
    assert received["plan"]["joint_sample"].sample_timestamp_s == 100.
    assert received["verify"] == tmp_path / "held_key_captures" / "held_1"


def test_observed_relation_exports_sample_only_when_configured_and_planned(
        monkeypatch, tmp_path):
    exported = []
    monkeypatch.setattr(module, "write_observed_physical_pickup_sample",
                        lambda **kwargs: exported.append(kwargs))
    decision = SessionDecision("held_relation_evidence_required", "test",
                               "table/0/1")
    capture = SimpleNamespace(
        capture_id="held_1", request_id=17,
        images_bgr={"cam_a": np.zeros((2, 2, 3), dtype=np.uint8)},
        frame_evidence={"cam_a": {"frame_id": 1}})
    monkeypatch.setattr(module, "collect_key_capture", lambda **kw: capture)
    monkeypatch.setattr(module, "admit_postlift_key_capture",
                        lambda **kw: SimpleNamespace(capture_id="held_1"))
    monkeypatch.setattr(module, "write_key_capture_artifacts",
                        lambda *args: None)
    monkeypatch.setattr(module, "verify_key_capture_artifacts",
                        lambda *args: None)

    class Runner:
        mode = SimpleNamespace(key_object="key")
        shared_root = tmp_path
        output_dir = tmp_path
        calibration = object()
        active_attempt = SimpleNamespace(attempt_id="trial_1",
                                         candidate_id="table/0/1")

        def current_decision(self):
            return decision

        def postlift_candidate_pose_prior(self, **kwargs):
            return np.eye(4)

        def prepare_postlift_transfer(self, **kwargs):
            return SimpleNamespace(status="sampled_postlift_preflight_pass")

    runner = Runner()
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        cameras=SimpleNamespace(
            init_orchestrator=object(),
            acquisition_metadata_for_request=lambda request: None,
            launch=SimpleNamespace(camera_ids=("cam_a",),
                                   capture_root=tmp_path,
                                   view_limits=object())))
    bounds = {
        "key_translation_error_bound_m": .001,
        "key_rotation_error_bound_deg": 1.,
        "wrist_translation_error_bound_m": .001,
        "wrist_rotation_error_bound_deg": 1.,
    }
    handler = module.LiveObservedHeldRelationHandlers(
        live=live, planner=object(),
        state_for_capture=lambda raw: _sample(),
        key_prompt="white key", path_limits=object(),
        capture_parameters=_capture_parameters(), parameters=_parameters(),
        calibration_sample_bounds=bounds,
        capture_id_factory=lambda: "held_1")
    handler.observe_and_plan(runner, decision)
    assert len(exported) == 1
    assert exported[0]["candidate_id"] == "table/0/1"
    assert exported[0]["output_path"] == (
        tmp_path / "physical_pickup_samples/trial_1.json")


def test_occluded_held_key_does_not_advance_to_transfer(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "collect_key_capture",
                        lambda **kwargs: (_ for _ in ()).throw(
                            ValueError("capture missing")))
    planned = []
    decision = SessionDecision("held_relation_evidence_required", "test",
                               "table/0/1")
    runner = SimpleNamespace(
        mode=SimpleNamespace(key_object="key"), shared_root=tmp_path,
        output_dir=tmp_path, calibration=object(),
        current_decision=lambda: decision,
        postlift_candidate_pose_prior=lambda **kwargs: np.eye(4),
        prepare_postlift_transfer=lambda **kwargs: planned.append(kwargs))
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        cameras=SimpleNamespace(
            init_orchestrator=object(),
            acquisition_metadata_for_request=lambda request: None,
            launch=SimpleNamespace(
                camera_ids=("cam_a", "cam_b"), capture_root=tmp_path,
                view_limits=object())))
    handler = module.LiveObservedHeldRelationHandlers(
        live=live, planner=object(), state_for_capture=lambda raw: _sample(),
        key_prompt="white key handle", path_limits=object(),
        capture_parameters=_capture_parameters(),
        parameters=_parameters(), capture_id_factory=lambda: "held_1")
    with pytest.raises(ValueError, match="capture missing"):
        handler.observe_and_plan(runner, decision)
    assert planned == []
