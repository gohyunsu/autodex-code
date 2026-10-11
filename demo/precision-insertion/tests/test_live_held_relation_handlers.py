"""Occluded-key transfer cannot start from a raw VLM label alone."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.live_held_relation_handlers import (  # noqa: E402
    LiveHeldRelationHandlers,
)
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


PARAMETERS = {
    "max_state_age_s": 1.,
    "max_arm_hand_skew_s": .02,
    "max_hand_command_error_raw": 20.,
    "max_arm_velocity_rad_s": .05,
    "max_postlift_arm_drift_rad": .05,
    "max_postlift_hand_drift_raw": 20.,
    "max_calibration_hand_excess_rad": .1,
    "axial_waypoint_step_m": .005,
}


def _state():
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=10., arm_timestamp_s=10.,
        arm_robot_uptime_s=1., hand_timestamp_s=10.,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


class Runner:
    def __init__(self):
        self.decision = SessionDecision(
            "held_relation_evidence_required",
            "visible_grasp_but_key_hand_pose_unknown", "table/0/3")
        self.calls = []

    def current_decision(self):
        return self.decision

    def prepare_bounded_postlift_transfer(self, **kwargs):
        self.calls.append(kwargs)
        self.decision = SessionDecision(
            "transfer_execution_gate_required", "bounded_postlift_path_planned",
            "table/0/3")
        return SimpleNamespace(status="sampled_postlift_preflight_pass")


def _handler(calibration_path):
    runner = Runner()
    planner = object()
    bounds = object()
    limits = object()
    bound = LiveHeldRelationHandlers(
        live=SimpleNamespace(opened=SimpleNamespace(runner=runner)),
        planner=planner, read_state=_state,
        calibration_for_candidate=lambda candidate: (
            calibration_path if candidate == "table/0/3" else None),
        bounds=bounds, path_limits=limits, parameters=PARAMETERS)
    return bound, runner, planner, bounds, limits


def test_calibrated_relation_plans_from_measured_hold(tmp_path):
    calibration = tmp_path / "physical_relation.json"
    calibration.write_text("{}", encoding="utf-8")
    bound, runner, planner, bounds, limits = _handler(calibration)
    bound.handlers()["held_relation_evidence_required"](
        runner, runner.current_decision())
    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["physical_calibration_path"] == calibration
    assert call["joint_sample"].source == "robot_joint_feedback"
    assert call["planner"] is planner
    assert call["bounds"] is bounds and call["limits"] is limits
    assert runner.current_decision().action == "transfer_execution_gate_required"


def test_missing_physical_calibration_stops_before_planning(tmp_path):
    bound, runner, *_ = _handler(tmp_path / "absent.json")
    with pytest.raises(FileNotFoundError, match="physical key/hand calibration"):
        bound.plan_bounded_transfer(runner, runner.current_decision())
    assert runner.calls == []


def test_wrong_session_or_action_does_not_read_calibration(tmp_path):
    bound, runner, *_ = _handler(tmp_path / "absent.json")
    with pytest.raises(ValueError, match="another session or decision"):
        bound.plan_bounded_transfer(object(), runner.current_decision())
    with pytest.raises(ValueError, match="another session or decision"):
        bound.plan_bounded_transfer(runner, SessionDecision(
            "capture_fresh_key", "ready", None))
    assert runner.calls == []
