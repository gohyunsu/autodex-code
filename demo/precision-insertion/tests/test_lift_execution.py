"""Fake-adapter tests; never command a real Franka or certify a watchdog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import lift_execution, measured_lift_preflight  # noqa: E402
from precision_insertion.lift_execution import (  # noqa: E402
    LiftExecutionLimits, execute_bound_measured_lift,
)
from test_measured_lift_preflight import _fixture, _state  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Adapter:
    def __init__(self, tmp_path):
        self.daemon_binary_path = tmp_path / "reviewed_daemon"
        self.daemon_binary_path.write_bytes(b"fake-test-binary")
        self.trace = tmp_path / "controller_trace.json"
        self.trace.write_text(json.dumps({"test_only": True}))
        self.calls = 0
        self.stops = 0
        self.fail = False

    def follow_lift(self, path, *, max_duration_s, expected_hand_raw,
                    trajectory_sha256):
        self.calls += 1
        assert path.shape == (2, 13)
        assert np.allclose(path[:, 7:], path[0, 7:])
        assert np.all(expected_hand_raw == 600.)
        assert max_duration_s == .5
        if self.fail:
            raise RuntimeError("simulated controller abort")
        return {
            "schema": "precision_insertion_external_lift_result_v1",
            "trajectory_complete": True, "force_abort": False,
            "terminal_hold_acknowledged": True,
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": 102.23, "completed_at_s": 102.5,
            "controller_trace_path": str(self.trace),
            "controller_trace_sha256": _sha(self.trace),
        }

    def stop_and_acknowledge(self):
        self.stops += 1
        return {"acknowledged": True}


def _setup(tmp_path, monkeypatch):
    args, _calls, _log, _marker = _fixture(tmp_path, monkeypatch)
    chain = measured_lift_preflight.plan_measured_lift_chain(**args)
    runner = args["runner"]
    measured_lift_preflight.write_measured_lift_chain(
        chain, tmp_path / "measured_chain")
    report = tmp_path / "measured_chain" / "report.json"
    runner._measured_lift_preflight = chain
    runner._measured_lift_report_path = report
    runner._measured_lift_report_sha256 = _sha(report)
    adapter = _Adapter(tmp_path)
    commissioning = tmp_path / "watchdog_review.json"
    commissioning.write_text(json.dumps({
        "schema": "precision_insertion_robot_watchdog_commissioning_v1",
        "reviewed_by": "fake-test-reviewer", "tested_at_s": 102.,
        "daemon_binary_path": str(adapter.daemon_binary_path),
        "daemon_binary_sha256": _sha(adapter.daemon_binary_path),
        "command_expiry_s": .05,
        "stale_command_timeout_test_passed": True,
        "disconnect_stop_test_passed": True,
        "stop_ack_test_passed": True,
        "e_stop_test_passed": True,
    }))
    limits = LiftExecutionLimits(
        max_start_age_s=.5, max_end_state_age_s=.5,
        max_execution_feedback_gap_s=.1,
        max_start_joint_error_rad=.01, max_end_joint_error_rad=.01,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    clock = iter([102.2, 102.21, 102.22, 102.7])
    monkeypatch.setattr(lift_execution, "_wall_time", lambda: next(clock))
    kwargs = dict(
        runner=runner, adapter=adapter, pre_state=args["joint_sample"],
        read_post_state=lambda: _state(
            102.55, np.array([.13, .03, .03, .03, .03, .03, .03])),
        limits=limits, commissioning_record_path=commissioning,
        motion_interlock=lambda: True)
    return kwargs


def test_measured_lift_requires_explicit_motion_opt_in(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="interlock"):
        execute_bound_measured_lift(**kwargs)
    assert kwargs["adapter"].calls == 0
    assert not (kwargs["runner"]._attempt_dir / "lift_started.json").exists()


def test_bound_lift_writes_observation_eligible_log_not_grasp_label(
        tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    result = execute_bound_measured_lift(
        **kwargs, enable_robot_motion=True)
    assert result["trajectory_complete"] is True
    assert result["force_abort"] is False
    assert result["post_state"]["full_q"][0] == pytest.approx(.13)
    assert "grasp_success" not in result
    assert kwargs["adapter"].calls == 1
    assert (kwargs["runner"]._attempt_dir / "lift_execution.json").is_file()
    with pytest.raises(FileExistsError, match="already started"):
        execute_bound_measured_lift(**kwargs, enable_robot_motion=True)


def test_lift_adapter_error_latches_failure_and_supervised_recovery(
        tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    kwargs["adapter"].fail = True
    with pytest.raises(RuntimeError, match="simulated controller abort"):
        execute_bound_measured_lift(**kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].stops == 1
    failure = json.loads((kwargs["runner"]._attempt_dir /
                          "lift_failure.json").read_text())
    assert failure["robot_state_after_failure"] == (
        "unknown_requires_supervised_recovery")
    assert not (kwargs["runner"]._attempt_dir / "lift_execution.json").exists()


def test_lift_refuses_unreviewed_or_changed_daemon_binary(
        tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    kwargs["adapter"].daemon_binary_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="no matching reviewed watchdog"):
        execute_bound_measured_lift(**kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].calls == 0


def test_lift_needs_prompt_measured_endpoint_after_controller_completion(
        tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    kwargs["read_post_state"] = lambda: _state(
        102.65, np.array([.13, .03, .03, .03, .03, .03, .03]))
    with pytest.raises(RuntimeError, match="endpoint differs"):
        execute_bound_measured_lift(**kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].stops == 1
