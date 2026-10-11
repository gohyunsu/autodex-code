"""Lift adapter records only measured controller samples, never a fake pass."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import franka_measured_lift as measured  # noqa: E402


class _Arm:
    def __init__(self, force=1.):
        self.force = force
        self.uptime = 1.

    def get_data(self):
        self.uptime += .001
        return {"qpos": np.zeros(7), "qvel": np.zeros(7),
                "wrench": np.array([self.force, 0., 0., 0., 0., 0.]),
                "time": self.uptime}

    def is_error(self):
        return False


class _Hand:
    def __init__(self):
        self.action = np.full(6, 500.)
        self.measured = self.action.copy()

    def get_data(self):
        return {"action": self.action.copy(),
                "joint_value": self.measured.copy(), "time": time.time()}


class _Executor:
    def __init__(self, force=1., *, sample_count=3):
        self.arm = _Arm(force)
        self.hand = _Hand()
        self._holding_object = True
        self.sample_count = sample_count
        self.stop_calls = 0

    def follow_joint_trajectory(self, path, raw_hold):
        assert path.shape == (3, 13)
        assert raw_hold.shape == (3, 6)
        for _ in range(self.sample_count):
            self.arm.get_data()

    def stop_and_acknowledge(self):
        self.stop_calls += 1
        return {"acknowledged": True}


def _adapter(tmp_path, monkeypatch, executor):
    monkeypatch.setattr(measured, "is_demo_completion_checked_executor",
                        lambda item: item is executor)
    binary = tmp_path / "franka_daemon"
    binary.write_bytes(b"test binary, never robot code")
    return measured.FrankaMeasuredLiftAdapter(
        executor=executor, daemon_binary_path=binary,
        trace_root=tmp_path / "traces",
        limits=measured.MeasuredLiftMonitorLimits(5., 2., 20., .1))


def _follow(adapter):
    return adapter.follow_lift(
        np.zeros((3, 13)), max_duration_s=1.,
        expected_hand_raw=np.full(6, 500.),
        trajectory_sha256="a" * 64)


def test_lift_adapter_records_actual_arm_and_hand_samples(tmp_path, monkeypatch):
    executor = _Executor()
    adapter = _adapter(tmp_path, monkeypatch, executor)
    result = _follow(adapter)
    assert result["trajectory_complete"] is True
    assert result["force_abort"] is False
    assert result["terminal_hold_acknowledged"] is True
    assert executor.arm.__class__ is _Arm  # proxy restored
    trace = json.loads(Path(result["controller_trace_path"]).read_text())
    assert trace["status"] == "completed"
    assert len(trace["samples"]) == 3
    assert trace["samples"][0]["force_norm_n"] == 1.
    assert trace["samples"][0]["hand_raw_measured"] == [500.] * 6


def test_lift_force_abort_is_not_mislabeled_complete(tmp_path, monkeypatch):
    executor = _Executor(force=10.)
    adapter = _adapter(tmp_path, monkeypatch, executor)
    with pytest.raises(RuntimeError, match="measured FR3 lift aborted"):
        _follow(adapter)
    assert executor.stop_calls == 1
    trace_path = next((tmp_path / "traces").glob("lift_*/controller_trace.json"))
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "aborted"
    assert trace["samples"][0]["force_norm_n"] == 10.


def test_lift_without_advancing_feedback_fails(tmp_path, monkeypatch):
    executor = _Executor(sample_count=0)
    adapter = _adapter(tmp_path, monkeypatch, executor)
    with pytest.raises(RuntimeError, match="measured FR3 lift aborted"):
        _follow(adapter)
    trace_path = next((tmp_path / "traces").glob("lift_*/controller_trace.json"))
    trace = json.loads(trace_path.read_text())
    assert trace["status"] == "aborted"
    assert trace["samples"] == []
    assert executor.stop_calls == 1


def test_lift_requires_prior_squeeze_and_checked_executor(tmp_path, monkeypatch):
    executor = _Executor()
    adapter = _adapter(tmp_path, monkeypatch, executor)
    executor._holding_object = False
    with pytest.raises(ValueError, match="active grasp"):
        _follow(adapter)
    assert not (tmp_path / "traces").exists()
    with pytest.raises(TypeError, match="checked FR3 executor"):
        measured.FrankaMeasuredLiftAdapter(
            executor=object(), daemon_binary_path=tmp_path / "daemon",
            trace_root=tmp_path / "other", limits=adapter.limits)
