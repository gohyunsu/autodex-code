"""A measured lift is assessed from fresh cameras, not a nominal plan."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_lift_handlers  # noqa: E402
from precision_insertion.lift_execution import LiftExecutionLimits  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


ASSESS = {
    "max_phase_skew_s": .02,
    "max_lift_observation_gap_s": 2.,
    "max_arm_hand_skew_s": .02,
    "max_hand_command_error_raw": 20.,
    "max_arm_velocity_rad_s": .05,
}


def _state(stamp):
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=stamp, arm_timestamp_s=stamp,
        arm_robot_uptime_s=1., hand_timestamp_s=stamp,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


def _capture(cameras=("front", "side")):
    return RawCameraCapture(
        capture_id="lift_001", request_id=10,
        images_bgr={camera: np.zeros((16, 16, 3), dtype=np.uint8)
                    for camera in cameras},
        frame_ids={camera: 10 for camera in cameras},
        acquisition_metadata={})


class Runner:
    def __init__(self, root):
        self.output_dir = root
        self.active_attempt = SimpleNamespace(
            attempt_id="attempt_001", candidate_id="table/0/3")
        self.decision = SessionDecision(
            "await_lift_observation", "no_physical_stage_recorded", "table/0/3")
        self.events = []

    def current_decision(self):
        return self.decision

    def prepare_raw_lift_label(self, **kwargs):
        self.events.append(("assess", kwargs))
        self.decision = SessionDecision(
            "held_relation_evidence_required", "lifted_key_observed", "table/0/3")
        return SimpleNamespace(grasp_success=True)


def _handler(tmp_path, *, enabled=True, cameras=("front", "side")):
    runner = Runner(tmp_path)
    states = iter([_state(10.), _state(11.), _state(11.1)])
    capture = _capture(cameras)
    adapter = object()
    backend = SimpleNamespace(infer=lambda _images, _prompt: "{}")
    live = SimpleNamespace(
        opened=SimpleNamespace(runner=runner),
        cameras=SimpleNamespace(launch=SimpleNamespace(
            camera_ids={"front", "side"})))
    bound = live_lift_handlers.LiveLiftHandlers(
        live=live, adapter=adapter, read_state=lambda: next(states),
        state_for_capture=lambda value: (
            runner.events.append(("exposure", value)) or _state(11.2)),
        capture_after_lift=lambda *_args: (
            runner.events.append(("capture", capture)) or capture),
        backend=backend, motion_interlock=lambda: True,
        limits=LiftExecutionLimits(*([1.] * 12)),
        commissioning_record_path=tmp_path / "commissioned.json",
        assessment_parameters=ASSESS, enable_robot_motion=enabled,
        decision_time=lambda: 11.3)
    return bound, runner, capture, adapter, backend


def test_lift_then_same_session_raw_capture_then_qwen_label(
        tmp_path, monkeypatch):
    bound, runner, capture, adapter, backend = _handler(tmp_path)

    def execute(**kwargs):
        runner.events.append(("lift", kwargs))
        assert kwargs["adapter"] is adapter
        assert kwargs["pre_state"].sample_timestamp_s == 10.
        assert kwargs["enable_robot_motion"] is True
        kwargs["read_post_state"]()

    def write(value, output, *, phase):
        runner.events.append(("write", value, output, phase))

    def verify(output, *, phase):
        runner.events.append(("verify", output, phase))

    monkeypatch.setattr(live_lift_handlers, "execute_bound_measured_lift",
                        execute)
    monkeypatch.setattr(live_lift_handlers, "write_raw_camera_capture", write)
    monkeypatch.setattr(live_lift_handlers, "verify_raw_camera_capture", verify)
    bound.handlers()["await_lift_observation"](
        runner, runner.current_decision())
    assert [row[0] for row in runner.events] == [
        "lift", "capture", "write", "verify", "exposure", "assess"]
    assert runner.events[2][1] is capture
    assert runner.events[2][3] == "after_lift"
    assessed = runner.events[-1][1]
    assert assessed["backend"] is backend
    assert assessed["joint_sample"].sample_timestamp_s == 11.2
    assert assessed["lift_execution_log_path"] == (
        tmp_path / "attempts/attempt_001/lift_execution.json")
    assert assessed["after_raw_evidence_dir"] == runner.events[2][2]
    assert runner.current_decision().action == "held_relation_evidence_required"


def test_disabled_lift_does_not_capture_or_change_trial(tmp_path):
    bound, runner, *_ = _handler(tmp_path, enabled=False)
    with pytest.raises(PermissionError, match="not enabled"):
        bound.lift_and_observe(runner, runner.current_decision())
    assert runner.events == []
    assert runner.current_decision().action == "await_lift_observation"


def test_acknowledged_lift_abort_routes_unknown_grasp_to_manual_recovery(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    stages = []
    runner.observe_stage = lambda *args, **kwargs: stages.append((args, kwargs))
    runner.record_failure = lambda *args, **kwargs: stages.append((args, kwargs))

    def abort(**_kwargs):
        path = tmp_path / "attempts/attempt_001/lift_failure.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "schema": "precision_insertion_lift_execution_v1",
            "attempt_id": "attempt_001", "candidate_id": "table/0/3",
            "status": "execution_or_feedback_failed",
            "stop_acknowledgement": {"acknowledged": True},
        }), encoding="utf-8")
        raise RuntimeError("lift stopped")

    monkeypatch.setattr(live_lift_handlers, "execute_bound_measured_lift",
                        abort)
    bound.lift_and_observe(runner, runner.current_decision())
    assert stages[0][0] == ("grasp_success", None)
    assert stages[1][0] == ("lift_execution_failed",)


def test_vlm_abstention_is_recorded_once_for_supervised_recovery(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path)
    runner._lift_assessment_index = 0
    stages = []
    runner.observe_stage = lambda *args, **kwargs: stages.append((args, kwargs))
    monkeypatch.setattr(live_lift_handlers, "execute_bound_measured_lift",
                        lambda **_kwargs: None)
    monkeypatch.setattr(live_lift_handlers, "write_raw_camera_capture",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live_lift_handlers, "verify_raw_camera_capture",
                        lambda *_args, **_kwargs: None)

    def abstain(**_kwargs):
        report = tmp_path / "attempts/attempt_001/lift_assessments/000/report.json"
        report.parent.mkdir(parents=True)
        report.write_text("{}", encoding="utf-8")
        runner._lift_assessment_index += 1
        return SimpleNamespace(grasp_success=None)

    runner.prepare_raw_lift_label = abstain
    bound.lift_and_observe(runner, runner.current_decision())
    assert stages[0][0] == ("grasp_success", None)
    assert stages[0][1]["evidence_refs"]["vlm_observation"].endswith(
        "lift_assessments/000/report.json")


def test_uncalibrated_camera_cannot_be_used_for_lift_label(
        tmp_path, monkeypatch):
    bound, runner, *_ = _handler(
        tmp_path, cameras=("front", "unknown_camera"))
    monkeypatch.setattr(live_lift_handlers, "execute_bound_measured_lift",
                        lambda **_kwargs: None)
    with pytest.raises(ValueError, match="differ from session cameras"):
        bound.lift_and_observe(runner, runner.current_decision())
    assert [row[0] for row in runner.events] == ["capture"]


def test_wrong_session_or_action_does_not_lift(tmp_path):
    bound, runner, *_ = _handler(tmp_path)
    with pytest.raises(ValueError, match="another session or decision"):
        bound.lift_and_observe(object(), runner.current_decision())
    with pytest.raises(ValueError, match="another session or decision"):
        bound.lift_and_observe(runner, SessionDecision(
            "capture_fresh_key", "ready", None))
    assert runner.events == []
