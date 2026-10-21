"""An observed XY vote reaches the same non-contact controller as transfer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import observed_retry_transfer_execution as module  # noqa: E402
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.retry_preflight import (  # noqa: E402
    XYRetryPreflight, write_xy_retry_preflight,
)
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion.transfer_execution import (  # noqa: E402
    TransferExecutionLimits,
)
from test_measured_lift_preflight import _state  # noqa: E402
from test_transfer_execution import _Adapter  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setup(tmp_path, monkeypatch):
    runner = object.__new__(SessionRunner)
    runner._attempt_dir = tmp_path / "attempt"
    runner._attempt_dir.mkdir()
    runner.session_sha256 = "s" * 64
    pre = _state(102.2, np.full(7, .03))
    final = _state(102.55, np.array([.13, .03, .03, .03, .03, .03, .03]))
    transfer = np.stack((pre.full_q, final.full_q))
    planning = InsertionPreflight(
        "sampled_planning_pass", None, transfer, transfer.copy(), {}, 0,
        pre.full_q[7:].copy(), "measured", ())
    target = SimpleNamespace(
        xy_offset_socket_m=(.001, 0.),
        to_record=lambda: {"xy_offset_socket_m": [.001, 0.]})
    plan = XYRetryPreflight(
        "sampled_retry_preflight_pass", ("table", "0", "1"),
        "x_plus_1mm", target, planning, {"endpoint_pass": True},
        pre.full_q.copy(), np.eye(4), 102., 102.)
    report_dir = write_xy_retry_preflight(plan, tmp_path / "retry_preflight")
    report = report_dir / "report.json"
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/1",
        labels={"insertion_success": False}, failure_code=None,
        _pending_retry=True, xy_offset_socket_m=(.001, 0.),
        events=[{"stage": "xy_retry", "value": "x_plus_1mm",
                 "evidence_refs": {"live_preflight": str(report)}}])
    monkeypatch.setattr(SessionRunner, "current_decision", lambda self:
                        SimpleNamespace(action="await_retry_execution_and_observation"))
    monkeypatch.setattr(SessionRunner, "verify_current_preflight_evidence",
                        lambda self: {})
    adapter = _Adapter(tmp_path)
    review = tmp_path / "watchdog_review.json"
    review.write_text(json.dumps({
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
    limits = TransferExecutionLimits(
        max_start_age_s=.5, max_end_state_age_s=.5,
        max_start_joint_error_rad=.01, max_end_joint_error_rad=.01,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    clock = iter([102.3, 102.31, 102.7])
    monkeypatch.setattr(module.time, "time", lambda: next(clock))
    return dict(
        runner=runner, preflight=plan, preflight_report_path=report,
        adapter=adapter, pre_state=pre,
        read_post_state=lambda: final, limits=limits,
        commissioning_record_path=review,
        motion_interlock=lambda: True)


def test_observed_retry_transfer_uses_bound_plan_and_measured_arrival(
        tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    result = module.execute_bound_observed_retry_transfer(
        **kwargs, enable_robot_motion=True)
    assert result["schema"] == (
        "precision_insertion_observed_xy_transfer_execution_v1")
    assert result["measurement"] == {
        "trajectory_complete": True, "safety_abort": False,
        "grasp_held": True}
    assert kwargs["adapter"].calls == 1
    assert (kwargs["runner"]._attempt_dir /
            "observed_retry_transfer_execution.json").is_file()


def test_observed_retry_transfer_refuses_changed_path(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    archive = kwargs["preflight_report_path"].parent / "planned_trajectories.npz"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        module.execute_bound_observed_retry_transfer(
            **kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].calls == 0
