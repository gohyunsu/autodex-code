"""The demo FR3 wrapper lands only on an empty-hand clear-view return."""

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
from autodex.utils.robot_config import FR3_CLEAR_VIEW  # noqa: E402


class _Arm:
    def __init__(self, q, velocity=None, *, response=None):
        self.q = np.asarray(q, dtype=float)
        self.velocity = np.zeros(7) if velocity is None else np.asarray(
            velocity, dtype=float)
        self.moves = []
        self.events = []
        self.stops = 0
        self.reads = 0
        self.response = {"type": "success"} if response is None else response

    def get_data(self):
        self.reads += 1
        return {"qpos": self.q.copy(), "qvel": self.velocity.copy()}

    def is_error(self):
        return False

    def move(self, target, **kwargs):
        self.moves.append(np.asarray(target).copy())
        self.events.append("land")
        self.q = np.asarray(target, dtype=float).copy()

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
    def ramp(q):
        executor.ramps.append(np.asarray(q).copy())
        arm.events.append("hand_ramp")
    executor._ramp_hand = ramp
    return executor


def _stock_follow_stub(self, path, hand_traj, **kwargs):
    # Model the derived follower's checked stop before terminal commands.
    response = self.arm.stop_streaming()
    if response.get("type") != "success":
        raise RuntimeError("FR3 stream stop failed")
    if hand_traj is not None:
        self._ramp_hand(hand_traj[-1])
    assert self.follow_tol == .04
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
    assert arm.stops == 1
    assert arm.reads == 1


def test_stock_sized_stationary_residual_completes_without_landing(monkeypatch):
    monkeypatch.setattr(_Stock, "_follow", _stock_follow_stub)
    goal = np.zeros(7)
    arm = _Arm(np.array([.039, 0, 0, 0, 0, 0, 0]))
    executor = _wrapper(arm)
    executor._follow(np.vstack([goal, goal]))
    assert arm.moves == []


def test_empty_hand_clear_view_uses_stock_landing_then_measured_completion(
        monkeypatch):
    monkeypatch.setattr(_Stock, "_follow", _stock_follow_stub)
    goal = np.asarray(FR3_CLEAR_VIEW, dtype=float)
    arm = _Arm(goal + np.array([.073, 0, 0, 0, 0, 0, 0]))
    executor = _wrapper(arm)
    executor._holding_object = False
    hand = np.full((2, 6), 500.)
    executor._follow(np.vstack([goal + .1, goal]), hand)
    assert len(arm.moves) == 1
    np.testing.assert_allclose(arm.moves[0], goal)
    np.testing.assert_allclose(arm.q, goal)
    np.testing.assert_array_equal(executor.ramps[0], hand[-1])
    assert arm.events == ["land", "hand_ramp"]
    assert executor.land_tol == .02
    assert arm.stops == 1


def test_held_clear_view_path_does_not_make_stock_landing(monkeypatch):
    monkeypatch.setattr(_Stock, "_follow", _stock_follow_stub)
    goal = np.asarray(FR3_CLEAR_VIEW, dtype=float)
    arm = _Arm(goal + np.array([.073, 0, 0, 0, 0, 0, 0]))
    executor = _wrapper(arm)
    executor._holding_object = True
    with pytest.raises(RuntimeError, match="stationary endpoint"):
        executor._follow(np.vstack([goal + .1, goal]))
    assert arm.moves == []


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
    with pytest.raises(RuntimeError, match="stream stop failed"):
        executor._follow(np.vstack([np.zeros(7), goal]),
                         np.full((2, 6), 500.))
    assert executor.ramps == []
    assert arm.moves == []
    assert arm.stops == 1


def test_force_stop_is_acknowledged_before_caller_can_reverse(monkeypatch):
    def force_stop(self, _path, _hand, **kwargs):
        assert kwargs["stop_wrench_z"] == 12.
        self._last_stop_on_contact = True
        assert self.arm.stop_streaming()["type"] == "success"

    monkeypatch.setattr(_Stock, "_follow", force_stop)
    arm = _Arm(np.zeros(7))
    executor = _wrapper(arm)
    executor._follow(np.vstack((np.zeros(7), np.full(7, .2))),
                     stop_wrench_z=12., abort_on_contact=True)
    assert arm.stops == 1
    assert arm.reads == 1
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


def test_transfer_final_reference_stall_reaches_visual_checkpoint(monkeypatch):
    def final_stall(self, path, _hand, **_kwargs):
        assert self.arm.stop_streaming()["type"] == "success"
        raise RuntimeError(
            f"FR3 stream incomplete (stalled reference) at waypoint "
            f"{len(path) - 1}/{len(path)}")

    monkeypatch.setattr(_Stock, "_follow", final_stall)
    goal = np.full(7, .2)
    arm = _Arm(goal - .02)
    executor = _wrapper(arm)
    executor._holding_object = True
    executor._precision_held_stage = "transfer"
    executor._follow(np.vstack([np.zeros(7), goal]),
                     np.full((2, 6), 500.))
    assert executor._precision_transfer_tail["terminal_goal_reached"] is False
    assert executor._precision_transfer_tail["terminal_joint_error_l2_rad"] > .04
    assert arm.moves == []
    assert executor.ramps == []

    executor._precision_held_stage = "lift"
    with pytest.raises(RuntimeError, match="stalled reference"):
        executor._follow(np.vstack([np.zeros(7), goal]),
                         np.full((2, 6), 500.))


def test_blocking_move_reports_daemon_reflex_instead_of_target_error():
    try:
        from src.execution.franka_executor import FrankaExecutor
    except RuntimeError as exc:
        if "No CUDA GPUs are available" not in str(exc):
            raise
        pytest.skip("stock FrankaExecutor import requires CUDA")

    class ReflexArm:
        def move(self, _target, **_kwargs):
            return {"type": "error", "message": "cartesian_reflex"}

        def is_error(self):
            raise AssertionError("do not recover and continue after reflex")

    executor = object.__new__(FrankaExecutor)
    executor.arm = ReflexArm()
    executor.arm_speed_scale = 1.0
    executor._last_release_wrist_reference = None
    with pytest.raises(RuntimeError, match=r"home: FR3 move command failed \(cartesian_reflex\)"):
        executor._move_to(np.zeros(7), what="home")
