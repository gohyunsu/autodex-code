"""The shared FR3 executor keeps stock-gripper and Inspire commands distinct."""

from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.execution.franka_executor import FrankaExecutor  # noqa: E402


class Arm:
    def __init__(self):
        self.open_calls = []
        self.grasp_calls = []

    def open_gripper(self, **kwargs):
        self.open_calls.append(kwargs)
        return {"type": "success"}

    def grasp(self, **kwargs):
        self.grasp_calls.append(kwargs)
        return {"type": "success", "final_width": 0.03}

    @staticmethod
    def get_data():
        return {"gripper_width": 0.03, "gripper_grasping": True}


def _executor(*, gripper: bool):
    executor = object.__new__(FrankaExecutor)
    executor._is_stock_gripper = gripper
    executor._is_robotiq = False
    executor._is_parallel_gripper = gripper
    executor.arm = Arm()
    executor.hand = None
    executor.dt = 0.
    executor.gripper_force_n = 60.
    executor.gripper_speed_m_s = .05
    executor._last_hand_qpos = np.zeros(2 if gripper else 6)
    executor._last_hand_action = np.zeros(2 if gripper else 6)
    return executor


def test_stock_gripper_open_and_grasp_use_daemon_metric_width(monkeypatch):
    monkeypatch.setattr("src.execution.franka_executor.time.sleep", lambda _: None)
    executor = _executor(gripper=True)

    executor._move_hand(np.array([.04, .04]))
    measured = executor._close_parallel_gripper()

    assert executor.arm.open_calls == [{"width": .08, "speed": .05}]
    assert executor.arm.grasp_calls == [{
        "force": 60., "speed": .05, "eps_inner": .005,
        "eps_outer": .005,
    }]
    np.testing.assert_allclose(measured, [.015, .015])
    np.testing.assert_allclose(executor._last_hand_qpos, measured)
    np.testing.assert_allclose(executor._last_hand_action, measured)


def test_inspire_move_does_not_call_stock_gripper_protocol(monkeypatch):
    monkeypatch.setattr("src.execution.franka_executor.time.sleep", lambda _: None)
    executor = _executor(gripper=False)
    moved = []
    executor.hand = type("Hand", (), {"move": lambda _self, q: moved.append(q)})()
    action = np.array([100., 200., 300., 400., 500., 600.])

    executor._move_hand(action)

    assert executor.arm.open_calls == []
    assert executor.arm.grasp_calls == []
    assert len(moved) == 1
    np.testing.assert_allclose(moved[0], action)
    np.testing.assert_allclose(executor._last_hand_action, action)


def test_stock_gripper_daemon_error_is_not_silently_accepted(monkeypatch):
    monkeypatch.setattr("src.execution.franka_executor.time.sleep", lambda _: None)
    executor = _executor(gripper=True)
    executor.arm.open_gripper = lambda **_kwargs: {
        "type": "error", "message": "gripper unavailable"}

    with pytest.raises(RuntimeError, match="gripper unavailable"):
        executor._move_hand(np.array([.04, .04]))
