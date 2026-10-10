"""The retry transfer is separately bound and never counts as arrival."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import postshift_path_handoff  # noqa: E402
from precision_insertion import postshift_transfer_execution as execution  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from test_postshift_path_handoff import _case  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Adapter:
    def __init__(self, root):
        self.daemon_binary_path = root / "reviewed_retry_transfer_daemon"
        self.daemon_binary_path.write_bytes(b"fake-test-only-binary")
        self.sources = {}
        for name in ("trajectory_feedback", "safety", "grasp_state"):
            path = root / f"retry_{name}.json"
            path.write_text(json.dumps({"test_only": name}))
            self.sources[name] = {"path": str(path), "sha256": _sha(path)}
        self.calls = 0
        self.stops = 0
        self.fail = False

    def follow_transfer(self, path, *, max_duration_s, expected_hand_raw,
                        trajectory_sha256):
        self.calls += 1
        assert path.shape == (2, 13)
        assert np.array_equal(path[:, 7:], np.repeat(path[:1, 7:], 2, axis=0))
        assert max_duration_s == .5
        if self.fail:
            raise RuntimeError("simulated retry transfer abort")
        return {
            "schema": "precision_insertion_external_transfer_result_v1",
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True, "terminal_hold_acknowledged": True,
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": 100.53, "completed_at_s": 100.6,
            "source_records": self.sources,
        }

    def stop_and_acknowledge(self):
        self.stops += 1
        return {"acknowledged": True}


def _setup(tmp_path, monkeypatch):
    upstream = _case(tmp_path, monkeypatch)
    runner = object.__new__(SessionRunner)
    runner._attempt_dir = tmp_path / "attempt"
    runner._attempt_dir.mkdir()
    report_path = postshift_path_handoff.prepare_postshift_path_handoff(
        output_dir=runner._attempt_dir / "postshift_path_handoffs" / "000",
        **upstream)
    handoff = json.loads(report_path.read_text())
    runner.mode = upstream["mode"]
    runner.shared_root = tmp_path
    runner.calibration = upstream["calibration"]
    runner.session_sha256 = handoff["session_calibration_sha256"]
    runner._postshift_handoff_used_reports = {
        upstream["preflight_report_path"].resolve()}
    runner._attempt = SimpleNamespace(
        attempt_id=handoff["attempt_id"], candidate_id=handoff["candidate_id"],
        labels={"grasp_success": True, "preinsert_reached": True,
                "insertion_success": False}, failure_code=None)
    monkeypatch.setattr(SessionRunner, "current_decision", lambda self:
                        SimpleNamespace(
                            action="guarded_withdrawal_then_xy_assessment"))
    monkeypatch.setattr(SessionRunner, "verify_current_preflight_evidence",
                        lambda self: {})
    before = replace(
        upstream["measured_start"], sample_timestamp_s=100.48,
        arm_timestamp_s=100.48, hand_timestamp_s=100.48)
    terminal = replace(
        before, full_q=np.asarray(handoff["transfer_end_q"]),
        sample_timestamp_s=100.65, arm_timestamp_s=100.65,
        hand_timestamp_s=100.65)
    adapter = _Adapter(tmp_path)
    commissioning = tmp_path / "retry_watchdog_review.json"
    commissioning.write_text(json.dumps({
        "schema": "precision_insertion_robot_watchdog_commissioning_v1",
        "reviewed_by": "fake-test-reviewer", "tested_at_s": 100.,
        "daemon_binary_path": str(adapter.daemon_binary_path),
        "daemon_binary_sha256": _sha(adapter.daemon_binary_path),
        "command_expiry_s": .05,
        "stale_command_timeout_test_passed": True,
        "disconnect_stop_test_passed": True,
        "stop_ack_test_passed": True, "e_stop_test_passed": True,
    }))
    limits = execution.PostShiftTransferLimits(
        max_start_age_s=.2, max_end_state_age_s=.3,
        max_execution_feedback_gap_s=.1,
        max_start_joint_error_rad=.01, max_end_joint_error_rad=.01,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    clock = iter([100.5, 100.52, 100.7, 100.8])
    monkeypatch.setattr(execution, "_wall_time", lambda: next(clock))
    return dict(
        runner=runner, expected=upstream["expected"],
        checkpoint=upstream["checkpoint"],
        shift_plan=upstream["shift_plan"],
        handoff_report_path=report_path, adapter=adapter,
        pre_state=before, read_post_state=lambda: terminal,
        limits=limits, max_handoff_age_s=.2,
        commissioning_record_path=commissioning,
        motion_interlock=lambda: True)


def test_retry_transfer_defaults_to_no_motion(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="interlock"):
        execution.execute_bound_postshift_transfer(**args)
    assert args["adapter"].calls == 0
    assert not (args["runner"]._attempt_dir /
                "postshift_transfer_executions/000/started.json").exists()


def test_retry_transfer_logs_only_noncontact_completion(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    result = execution.execute_bound_postshift_transfer(
        **args, enable_robot_motion=True)
    assert result["measurement"] == {
        "trajectory_complete": True, "safety_abort": False,
        "grasp_held": True}
    assert "preinsert_reached" not in result
    assert args["runner"]._attempt.labels["insertion_success"] is False
    assert args["adapter"].calls == 1
    log = (args["runner"]._attempt_dir /
           "postshift_transfer_executions/000/execution.json")
    assert execution.verify_postshift_transfer_execution(
        log, handoff_report_path=args["handoff_report_path"],
        expected=args["expected"], checkpoint=args["checkpoint"],
        shift_plan=args["shift_plan"], mode=args["runner"].mode,
        shared_root=args["runner"].shared_root,
        calibration=args["runner"].calibration) == result
    with pytest.raises(FileExistsError, match="already started"):
        execution.execute_bound_postshift_transfer(
            **args, enable_robot_motion=True)


def test_retry_transfer_rejects_changed_path_before_command(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    handoff = json.loads(args["handoff_report_path"].read_text())
    Path(handoff["trajectory_archive_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        execution.execute_bound_postshift_transfer(
            **args, enable_robot_motion=True)
    assert args["adapter"].calls == 0


def test_retry_transfer_rejects_expired_handoff_before_command(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="fresh measured hold"):
        execution.execute_bound_postshift_transfer(
            **{**args, "max_handoff_age_s": .01},
            enable_robot_motion=True)
    assert args["adapter"].calls == 0


def test_retry_transfer_latches_controller_failure(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    args["adapter"].fail = True
    with pytest.raises(RuntimeError, match="simulated retry transfer abort"):
        execution.execute_bound_postshift_transfer(
            **args, enable_robot_motion=True)
    assert args["adapter"].stops == 1
    failure = json.loads((args["runner"]._attempt_dir /
                          "postshift_transfer_executions/000/failure.json").read_text())
    assert failure["robot_state_after_failure"] == (
        "unknown_requires_supervised_recovery")


def test_retry_transfer_log_rejects_changed_source(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    execution.execute_bound_postshift_transfer(
        **args, enable_robot_motion=True)
    log = (args["runner"]._attempt_dir /
           "postshift_transfer_executions/000/execution.json")
    source = Path(args["adapter"].sources["grasp_state"]["path"])
    source.write_text('{"changed":true}')
    with pytest.raises((RuntimeError, ValueError), match="producer record"):
        execution.verify_postshift_transfer_execution(
            log, handoff_report_path=args["handoff_report_path"],
            expected=args["expected"], checkpoint=args["checkpoint"],
            shift_plan=args["shift_plan"], mode=args["runner"].mode,
            shared_root=args["runner"].shared_root,
            calibration=args["runner"].calibration)
