"""The demo FR3 wrapper never lands or ramps after an incomplete stream."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = DEMO_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.franka_completion_checked import (  # noqa: E402
    CompletionCheckedFrankaMixin,
)
from precision_insertion import franka_completion_checked  # noqa: E402
from precision_insertion.pickup_execution import (  # noqa: E402
    _reject_known_stock_follower,
)


class _Arm:
    def __init__(self, q, velocity=None, *, response=None):
        self.q = np.asarray(q, dtype=float)
        self.velocity = np.zeros(7) if velocity is None else np.asarray(
            velocity, dtype=float)
        self.moves = []
        self.stops = 0
        self.response = {"type": "success"} if response is None else response

    def get_data(self):
        return {"qpos": self.q.copy(), "qvel": self.velocity.copy()}

    def is_error(self):
        return False

    def move(self, target, **kwargs):
        self.moves.append(np.asarray(target).copy())

    def stop_streaming(self):
        self.stops += 1
        return self.response


class _Stock:
    _follow = None


class _Executor(CompletionCheckedFrankaMixin, _Stock):
    pass


def _wrapper(arm):
    executor = object.__new__(_Executor)
    executor.arm = arm
    executor.land_tol = .02
    executor.follow_tol = .04
    executor._last_stop_on_contact = False
    executor.ramps = []
    executor._ramp_hand = lambda q: executor.ramps.append(np.asarray(q).copy())
    return executor


def _stock_follow_stub(self, path, hand_traj, **kwargs):
    # Model only the dangerous stock terminal steps: final hand ramp and
    # blocking endpoint land after a stalled/early-ended stream.
    if hand_traj is not None:
        self._ramp_hand(hand_traj[-1])
    assert self.follow_tol == .02
    if np.linalg.norm(self.arm.q - path[-1, :7]) > self.land_tol:
        self.arm.move(path[-1, :7])


def test_completed_stream_permits_deferred_hand_ramp_without_blocking_land(
        monkeypatch):
    monkeypatch.setattr(_Stock, "_follow", _stock_follow_stub)
    goal = np.full(7, .2)
    arm = _Arm(goal)
    executor = _wrapper(arm)
    hand = np.array([[100.] * 6, [200.] * 6])
    executor._follow(np.vstack([np.zeros(7), goal]), hand)
    assert arm.moves == []
    assert len(executor.ramps) == 1
    np.testing.assert_array_equal(executor.ramps[0], hand[-1])
    assert executor.land_tol == .02
    assert executor.follow_tol == .04
    assert not executor._precision_follow_active


def test_incomplete_stream_never_lands_or_ramps(monkeypatch):
    monkeypatch.setattr(_Stock, "_follow", _stock_follow_stub)
    goal = np.full(7, .2)
    arm = _Arm(np.zeros(7))
    executor = _wrapper(arm)
    with pytest.raises(RuntimeError, match="stationary endpoint"):
        executor._follow(np.vstack([np.zeros(7), goal]),
                         np.full((2, 6), 500.))
    assert arm.moves == []
    assert executor.ramps == []
    assert executor.land_tol == .02
    assert executor.follow_tol == .04
    assert not executor._precision_follow_active


def test_stock_exception_restores_instance_state_without_hand_ramp(monkeypatch):
    def failing(self, _path, hand_traj, **_kwargs):
        self._ramp_hand(hand_traj[-1])
        raise RuntimeError("contact reflex")

    monkeypatch.setattr(_Stock, "_follow", failing)
    executor = _wrapper(_Arm(np.zeros(7)))
    with pytest.raises(RuntimeError, match="contact reflex"):
        executor._follow(np.zeros((2, 7)), np.full((2, 6), 100.))
    assert executor.ramps == []
    assert executor.land_tol == .02
    assert executor.follow_tol == .04
    assert not executor._precision_follow_active


def test_unacknowledged_stop_prevents_terminal_hand_ramp(monkeypatch):
    monkeypatch.setattr(_Stock, "_follow", _stock_follow_stub)
    goal = np.full(7, .2)
    arm = _Arm(goal, response={"type": "error"})
    executor = _wrapper(arm)
    with pytest.raises(RuntimeError, match="stop was not acknowledged"):
        executor._follow(np.vstack([np.zeros(7), goal]),
                         np.full((2, 6), 500.))
    assert executor.ramps == []
    assert arm.moves == []


def test_force_stop_is_acknowledged_before_caller_can_reverse(monkeypatch):
    def force_stop(self, _path, _hand, **kwargs):
        assert kwargs["stop_wrench_z"] == 12.
        self._last_stop_on_contact = True

    monkeypatch.setattr(_Stock, "_follow", force_stop)
    arm = _Arm(np.zeros(7))
    executor = _wrapper(arm)
    executor._follow(np.vstack((np.zeros(7), np.full(7, .2))),
                     stop_wrench_z=12., abort_on_contact=True)
    assert arm.stops == 1
    assert executor._last_stop_on_contact is True
    assert arm.moves == []


def test_stock_lineage_requires_exact_demo_wrapper(monkeypatch):
    stock_type = type("FrankaExecutor", (_Stock,), {
        "__module__": "src.execution.franka_executor"})
    stock = object.__new__(stock_type)
    with pytest.raises(RuntimeError, match="stalled stream"):
        _reject_known_stock_follower(stock)
    checked_type = type("Checked", (CompletionCheckedFrankaMixin, stock_type), {})
    with pytest.raises(RuntimeError, match="stalled stream"):
        _reject_known_stock_follower(object.__new__(checked_type))
    monkeypatch.setattr(franka_completion_checked, "_COMPOSED_TYPE", checked_type)
    _reject_known_stock_follower(object.__new__(checked_type))


def test_stop_acknowledgement_needs_daemon_response_and_stationary_feedback():
    moving = _wrapper(_Arm(np.zeros(7), np.ones(7) * .2))
    moving.stop_wait_s = 0.
    assert moving.stop_and_acknowledge()["acknowledged"] is False
    refused = _wrapper(_Arm(np.zeros(7), response={"type": "error"}))
    assert refused.stop_and_acknowledge()["acknowledged"] is False
    held = _wrapper(_Arm(np.zeros(7), np.ones(7) * .01))
    assert held.stop_and_acknowledge()["acknowledged"] is True


def test_held_motion_requests_reflex_abort_in_stock_follower(monkeypatch):
    seen = []

    def follow(self, _path, _hand, **kwargs):
        seen.append(kwargs["abort_on_contact"])

    monkeypatch.setattr(_Stock, "_follow", follow)
    executor = _wrapper(_Arm(np.zeros(7)))
    executor._holding_object = True
    executor._follow(np.zeros((2, 7)))
    assert seen == [True]
