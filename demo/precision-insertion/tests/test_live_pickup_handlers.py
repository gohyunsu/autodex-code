"""The live action loop uses the existing pickup and measured-lift boundary."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion import live_pickup_handlers  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.pickup_execution import PickupExecutionLimits  # noqa: E402
from precision_insertion.session_policy import SessionDecision  # noqa: E402


LIMITS = {
    "max_state_age_s": 1.,
    "max_post_squeeze_arm_drift_rad": .02,
    "max_post_squeeze_hand_drift_raw": 20.,
    "max_arm_hand_skew_s": .02,
    "max_hand_command_error_raw": 20.,
    "max_arm_velocity_rad_s": .05,
    "axial_waypoint_step_m": .005,
}


def _state(stamp):
    return LiveRobotState(
        full_q=np.zeros(13), arm_qvel=np.zeros(7),
        sample_timestamp_s=stamp, arm_timestamp_s=stamp,
        arm_robot_uptime_s=1., hand_timestamp_s=stamp,
        hand_raw_measured=np.zeros(6), hand_raw_commanded=np.zeros(6),
        max_hand_command_error_raw=0., wrench=np.zeros(6))


class Runner:
    def __init__(self, root):
        self.output_dir = root
        self.decision = SessionDecision(
            "execution_gate_required", "nominal_pickup_and_insertion_paths_only",
            "table/0/3")
        self.events = []

    def current_decision(self):
        return self.decision

    def begin_selected_attempt(self, **kwargs):
        self.events.append(("begin", kwargs))
        self.decision = SessionDecision("await_lift_observation", "selected",
                                        "table/0/3")
        return SimpleNamespace(candidate_id="table/0/3",
                               attempt_id=kwargs["attempt_id"])

    def prepare_measured_lift_chain(self, **kwargs):
        self.events.append(("chain", kwargs))
        return SimpleNamespace(status="sampled_measured_chain_pass")


def _handler(tmp_path, *, enabled=True):
    runner = Runner(tmp_path)
    states = iter([_state(10.1), _state(10.2)])
    executor = object()
    planner = object()
    bounds = object()
    path_limits = object()
    bound = live_pickup_handlers.LivePickupHandlers(
        live=SimpleNamespace(opened=SimpleNamespace(runner=runner)),
        planner=planner, executor=executor, read_state=lambda: next(states),
        motion_interlock=lambda: True,
        pickup_limits=PickupExecutionLimits(.5, .05, .05, .02, 20., .05),
        lift_bounds=bounds, path_limits=path_limits,
        measured_lift_parameters=LIMITS, enable_robot_motion=enabled,
        attempt_id_factory=lambda: "attempt_001",
        started_at_factory=lambda: 10.)
    return bound, runner, planner, executor, bounds, path_limits


def test_selected_v8_pickup_then_measured_squeeze_lift_preflight(
        tmp_path, monkeypatch):
    bound, runner, planner, executor, bounds, path_limits = _handler(tmp_path)

    def pickup(**kwargs):
        runner.events.append(("pickup", kwargs))
        assert runner.current_decision().action == "await_lift_observation"
        assert kwargs["runner"] is runner
        assert kwargs["executor"] is executor
        assert kwargs["planner"] is planner
        assert kwargs["pre_state"].sample_timestamp_s == 10.1
        assert kwargs["enable_robot_motion"] is True
        kwargs["read_post_state"]()

    monkeypatch.setattr(live_pickup_handlers, "execute_bound_pickup", pickup)
    decision = runner.current_decision()
    bound.handlers()["execution_gate_required"](runner, decision)
    assert [row[0] for row in runner.events] == ["begin", "pickup", "chain"]
    assert runner.events[0][1] == {
        "attempt_id": "attempt_001", "started_at_s": 10.}
    chain = runner.events[2][1]
    assert chain["joint_sample"].sample_timestamp_s == 10.2
    assert chain["pickup_execution_log"] == (
        tmp_path / "attempts/attempt_001/pickup_execution.json")
    assert chain["bounds"] is bounds and chain["limits"] is path_limits
    assert chain["planner"] is planner
    assert runner.current_decision().action == "await_lift_observation"


def test_disabled_motion_does_not_open_an_attempt(tmp_path, monkeypatch):
    bound, runner, *_ = _handler(tmp_path, enabled=False)
    monkeypatch.setattr(live_pickup_handlers, "execute_bound_pickup",
                        lambda **_kwargs: pytest.fail("motion began"))
    with pytest.raises(PermissionError, match="not enabled"):
        bound.begin_and_squeeze(runner, runner.current_decision())
    assert runner.events == []
    assert runner.current_decision().action == "execution_gate_required"


def test_other_session_or_decision_never_starts_pickup(tmp_path):
    bound, runner, *_ = _handler(tmp_path)
    with pytest.raises(ValueError, match="another session or decision"):
        bound.begin_and_squeeze(object(), runner.current_decision())
    with pytest.raises(ValueError, match="another session or decision"):
        bound.begin_and_squeeze(runner, SessionDecision(
            "capture_fresh_key", "ready", None))
    assert runner.events == []
