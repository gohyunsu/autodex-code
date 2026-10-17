"""The unmeasured v8 relation is an explicit cylinder stage choice."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.live_nominal_relation_handlers import (  # noqa: E402
    LiveNominalRelationHandlers,
)
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


PARAMETERS = {
    "max_arm_hand_skew_s": .02,
    "max_hand_command_error_raw": 20., "max_arm_velocity_rad_s": .05,
    "axial_waypoint_step_m": .005,
}


def _state():
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=10., arm_timestamp_s=10.,
        arm_robot_uptime_s=1., hand_timestamp_s=10.,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


class _Runner:
    def __init__(self):
        self.decision = SessionDecision(
            "held_relation_evidence_required",
            "visible_grasp_but_key_hand_pose_unknown", "table/0/3")
        self.calls = []

    def current_decision(self):
        return self.decision

    def prepare_nominal_postlift_transfer(self, **kwargs):
        self.calls.append(kwargs)
        self.decision = SessionDecision(
            "transfer_execution_gate_required",
            "nominal_postlift_path_planned_relation_unmeasured", "table/0/3")
        return SimpleNamespace(status="sampled_postlift_preflight_pass")


def test_nominal_handler_uses_measured_feedback_without_calibration_file():
    runner = _Runner()
    planner = object()
    limits = object()
    handler = LiveNominalRelationHandlers(
        live=SimpleNamespace(opened=SimpleNamespace(runner=runner)),
        planner=planner, read_state=_state,
        limits=limits, parameters=PARAMETERS)
    handler.handlers()["held_relation_evidence_required"](
        runner, runner.current_decision())
    assert len(runner.calls) == 1
    assert runner.calls[0]["joint_sample"].source == "robot_joint_feedback"
    assert runner.calls[0]["planner"] is planner
    assert runner.calls[0]["limits"] is limits
    assert "physical_calibration_path" not in runner.calls[0]


def test_nominal_handler_rejects_wrong_session_and_missing_limits():
    runner = _Runner()
    handler = LiveNominalRelationHandlers(
        live=SimpleNamespace(opened=SimpleNamespace(runner=runner)),
        planner=object(), read_state=_state,
        limits=object(), parameters=PARAMETERS)
    with pytest.raises(ValueError, match="another held attempt"):
        handler.plan_nominal_transfer(object(), runner.current_decision())
    with pytest.raises(ValueError, match="another held attempt"):
        handler.plan_nominal_transfer(runner, SessionDecision(
            "capture_fresh_key", "ready", None))
    with pytest.raises(ValueError, match="limits missing"):
        LiveNominalRelationHandlers(
            live=handler.live, planner=object(), read_state=_state,
            limits=object(), parameters={})
