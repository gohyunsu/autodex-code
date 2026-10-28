"""Real Inspire feedback is converted; commanded-only poses are rejected."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.executor.real import _convert_inspire  # noqa: E402
from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.live_robot_state import (  # noqa: E402
    FeedbackTimeAlignmentUnavailable, read_live_franka_inspire_state,
)


def _devices():
    hand_planner = np.array([0.3, 0.2, 0.6, 0.7, 0.8, 0.9])
    raw = _convert_inspire(hand_planner)
    arm_data = {"qpos": np.arange(7) * 0.01,
                "qvel": np.zeros(7),
                "wrench": np.arange(6) * 0.1, "time": 10.0}
    hand_data = {"qpos": raw.copy(), "action": raw.copy(), "time": 100.01}
    arm = SimpleNamespace(get_data=lambda: arm_data)
    hand = SimpleNamespace(get_data=lambda: hand_data)
    return arm, hand, arm_data, hand_data, hand_planner


def _read(arm, hand, monkeypatch):
    monkeypatch.setattr("precision_insertion.live_robot_state.time.time",
                        lambda: 100.0)
    monkeypatch.setattr("precision_insertion.live_robot_state.time.monotonic",
                        lambda: 10.0)
    return read_live_franka_inspire_state(
        arm=arm, hand=hand, max_arm_hand_skew_s=0.03,
        max_sample_age_s=0.05, max_hand_command_error_raw=30,
        max_arm_velocity_rad_s=0.05,
        now_s=100.02)


def test_measured_raw_hand_becomes_planner_joint_order(monkeypatch):
    arm, hand, _, _, expected_hand = _devices()
    result = _read(arm, hand, monkeypatch)
    np.testing.assert_allclose(result.full_q[:7], np.arange(7) * 0.01)
    np.testing.assert_allclose(result.full_q[7:], expected_hand, atol=1e-10)
    assert result.sample_timestamp_s == pytest.approx(100.005)
    assert result.arm_robot_uptime_s == pytest.approx(10.0)
    assert result.source == "robot_joint_feedback"
    assert result.to_record()["hand_raw_measured"] != expected_hand.tolist()


def test_missing_stale_or_disagreeing_feedback_fails_closed(monkeypatch):
    arm, hand, arm_data, hand_data, _ = _devices()
    hand_data["qpos"] = None
    with pytest.raises(ValueError, match="measured raw qpos"):
        _read(arm, hand, monkeypatch)
    arm, hand, arm_data, hand_data, _ = _devices()
    hand_data["time"] = 99.8
    with pytest.raises(FeedbackTimeAlignmentUnavailable,
                       match="stale or not time-aligned"):
        _read(arm, hand, monkeypatch)
    arm, hand, arm_data, hand_data, _ = _devices()
    hand_data["qpos"][0] -= 100
    held = _read(arm, hand, monkeypatch)
    assert held.max_hand_command_error_raw == 100
    np.testing.assert_allclose(
        held.full_q[7:], convert_inspire_raw(hand_data["qpos"][None, :])[0])
    arm, hand, arm_data, hand_data, _ = _devices()
    hand_data.pop("qpos")
    hand_data["joint_value"] = hand_data["action"]
    with pytest.raises(ValueError, match="measured raw qpos"):
        _read(arm, hand, monkeypatch)


def test_invalid_raw_range_and_wrench_are_rejected(monkeypatch):
    arm, hand, arm_data, hand_data, _ = _devices()
    hand_data["qpos"][0] = -1
    with pytest.raises(ValueError, match="outside 0..1000"):
        _read(arm, hand, monkeypatch)
    arm, hand, arm_data, hand_data, _ = _devices()
    arm_data["wrench"] = [0, 0, 0]
    with pytest.raises(ValueError, match="FR3 wrench"):
        _read(arm, hand, monkeypatch)
    arm, hand, arm_data, hand_data, _ = _devices()
    arm_data["qvel"][2] = 0.2
    with pytest.raises(ValueError, match="stationary post-lift hold"):
        _read(arm, hand, monkeypatch)


def test_recent_daemon_state_needs_no_second_packet_but_stale_state_rejects(
        monkeypatch):
    arm, hand, arm_data, _, _ = _devices()
    monkeypatch.setattr("precision_insertion.live_robot_state.time.time",
                        lambda: 100.0)
    monkeypatch.setattr("precision_insertion.live_robot_state.time.monotonic",
                        lambda: 10.0)
    result = read_live_franka_inspire_state(
        arm=arm, hand=hand, max_arm_hand_skew_s=0.03,
        max_sample_age_s=0.05, max_hand_command_error_raw=30,
        max_arm_velocity_rad_s=0.05, now_s=100.02)
    assert result.arm_robot_uptime_s == arm_data["time"]
    monkeypatch.setattr("precision_insertion.live_robot_state.time.monotonic",
                        lambda: 10.1)
    with pytest.raises(ValueError, match="stale or not on the host clock"):
        read_live_franka_inspire_state(
            arm=arm, hand=hand, max_arm_hand_skew_s=0.03,
            max_sample_age_s=0.05, max_hand_command_error_raw=30,
            max_arm_velocity_rad_s=0.05, now_s=100.02)


def test_live_read_reacquires_arm_packet_after_blocking_motion():
    _, _, arm_data, hand_data, _ = _devices()

    class Arm:
        calls = 0

        def get_data(self):
            self.calls += 1
            return dict(arm_data, time=time.monotonic() -
                        (0.3 if self.calls == 1 else 0.005))

    arm = Arm()
    hand = SimpleNamespace(get_data=lambda: dict(hand_data, time=time.time()))
    result = read_live_franka_inspire_state(
        arm=arm, hand=hand, max_arm_hand_skew_s=0.1,
        max_sample_age_s=0.5, max_hand_command_error_raw=30,
        max_arm_velocity_rad_s=0.05)
    assert arm.calls >= 2
    assert abs(result.arm_timestamp_s - result.hand_timestamp_s) < 0.1
