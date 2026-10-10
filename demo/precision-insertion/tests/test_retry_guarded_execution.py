"""A fake contact adapter exercises the retry boundary; no robot is used."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.guarded_axial_handoff import _state_from_record  # noqa: E402
from precision_insertion.guarded_contact import GuardedContactLimits  # noqa: E402
from precision_insertion import retry_guarded_execution as execution  # noqa: E402
from precision_insertion.retry_guarded_execution import (  # noqa: E402
    RetryGuardedExecutionLimits, execute_bound_retry_guarded_insertion,
)
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from test_retry_guarded_metric import _case_metric, _sha  # noqa: E402


class _Adapter:
    def __init__(self, root: Path, metric: Path):
        self.daemon_binary_path = root / "fake_reviewed_retry_contact_daemon"
        self.daemon_binary_path.write_bytes(b"test-only-controller")
        self.metric = metric
        self.calls = 0
        self.stops = 0
        self.bad_metric_hash = False

    def follow_guarded_insertion(
        self, axial, *, handoff_sha256, trajectory_archive_sha256,
        contact_limits, expected_hand_raw, max_duration_s,
    ):
        self.calls += 1
        assert axial.ndim == 2 and axial.shape[1] == 13
        assert np.allclose(axial[:, 7:], axial[0, 7:])
        assert contact_limits["target_depth_m"] == .020
        assert max_duration_s == .5
        assert np.all(expected_hand_raw == 500.)
        return {
            "schema": "precision_insertion_external_guarded_result_v1",
            "terminal_hold_acknowledged": True,
            "safety_abort": False,
            "handoff_sha256": handoff_sha256,
            "trajectory_archive_sha256": trajectory_archive_sha256,
            "started_at_s": 101.3, "completed_at_s": 101.5,
            "metric_record_path": str(self.metric),
            "metric_record_sha256": (
                "0" * 64 if self.bad_metric_hash else _sha(self.metric)),
        }

    def stop_and_acknowledge(self):
        self.stops += 1
        return {"acknowledged": True}


def _setup(tmp_path, monkeypatch):
    metric_path, source = _case_metric(tmp_path, monkeypatch)
    packet = source["handoff_report_path"]
    handoff = json.loads(packet.read_text())
    metric = json.loads(metric_path.read_text())
    attempt_record = json.loads(Path(handoff["pending_state_path"]).read_text())
    runner = object.__new__(SessionRunner)
    runner.mode = source["mode"]
    runner.shared_root = source["shared_root"]
    runner.calibration = source["calibration"]
    runner.session_sha256 = handoff["session_calibration_sha256"]
    runner._attempt_dir = tmp_path
    runner._attempt_index = 4
    runner._retry_axial_handoff_used_reports = {
        Path(handoff["replan_report_path"])}
    runner._attempt = SimpleNamespace(
        attempt_id=attempt_record["attempt_id"],
        candidate_id=attempt_record["candidate_id"],
        labels={name: attempt_record[name] for name in (
            "grasp_success", "preinsert_reached", "insertion_success")},
        failure_code=attempt_record["failure_code"],
        events=attempt_record["events"])
    runner.current_decision = lambda: SimpleNamespace(
        action="await_retry_execution_and_observation")
    runner.verify_current_preflight_evidence = lambda: {}
    state = _state_from_record(handoff["measured_start"])
    before = replace(
        state, sample_timestamp_s=101.2,
        arm_timestamp_s=101.2, hand_timestamp_s=101.2,
        arm_robot_uptime_s=11.2)
    after = replace(
        before, full_q=np.asarray(handoff["axial_end_q"]),
        sample_timestamp_s=101.53,
        arm_timestamp_s=101.53, hand_timestamp_s=101.53,
        arm_robot_uptime_s=11.53)
    adapter = _Adapter(tmp_path, metric_path)
    limits = RetryGuardedExecutionLimits(
        max_start_age_s=.2, max_end_state_age_s=.2,
        max_execution_feedback_gap_s=.1,
        max_start_joint_error_rad=.001, max_end_joint_error_rad=.001,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    contact_limits = GuardedContactLimits(**metric["contact_limits"])
    watchdog = tmp_path / "retry_contact_watchdog_review.json"
    watchdog.write_text(json.dumps({
        "schema": "precision_insertion_robot_watchdog_commissioning_v1",
        "reviewed_by": "synthetic-test-reviewer", "tested_at_s": 101.,
        "daemon_binary_path": str(adapter.daemon_binary_path),
        "daemon_binary_sha256": _sha(adapter.daemon_binary_path),
        "command_expiry_s": .05,
        "stale_command_timeout_test_passed": True,
        "disconnect_stop_test_passed": True,
        "stop_ack_test_passed": True, "e_stop_test_passed": True,
    }))
    contact = tmp_path / "retry_contact_review.json"
    contact.write_text(json.dumps({
        "schema": "precision_insertion_contact_controller_commissioning_v1",
        "reviewed_by": "synthetic-test-reviewer", "tested_at_s": 101.,
        "daemon_binary_path": str(adapter.daemon_binary_path),
        "daemon_binary_sha256": _sha(adapter.daemon_binary_path),
        "contact_limits": asdict(contact_limits),
        "force_limit_test_passed": True,
        "contact_abort_test_passed": True,
        "sample_gap_abort_test_passed": True,
        "terminal_hold_test_passed": True,
    }))
    clock = iter([101.25, 101.26, 101.51, 101.6])
    monkeypatch.setattr(execution, "_wall_time", lambda: next(clock))
    return dict(
        runner=runner, expected=source["expected"],
        previous=source["previous"], arrival=source["arrival"],
        checkpoint=source["checkpoint"], shift_plan=source["shift_plan"],
        handoff_report_path=packet, adapter=adapter,
        pre_state=before, read_post_state=lambda: after,
        limits=limits, contact_limits=contact_limits,
        max_handoff_age_s=.2,
        watchdog_commissioning_path=watchdog,
        contact_commissioning_path=contact,
        motion_interlock=lambda: True)


def test_retry_contact_defaults_to_no_motion(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="interlock"):
        execute_bound_retry_guarded_insertion(**args)
    assert args["adapter"].calls == 0
    assert not (tmp_path / "retry_guarded_executions/000/started.json").exists()


def test_retry_contact_records_external_stroke_without_success_label(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    record = execute_bound_retry_guarded_insertion(
        **args, enable_robot_motion=True)
    assert record["safety_abort"] is False
    assert record["robot_ready"] is False
    assert args["runner"]._attempt.labels["insertion_success"] is False
    assert args["adapter"].calls == 1
    saved = tmp_path / "retry_guarded_executions/000/execution.json"
    assert saved.is_file()
    replayed = execution.verify_retry_guarded_execution(
        saved, handoff_report_path=args["handoff_report_path"],
        expected=args["expected"], previous=args["previous"],
        arrival=args["arrival"], checkpoint=args["checkpoint"],
        shift_plan=args["shift_plan"], mode=args["runner"].mode,
        shared_root=args["runner"].shared_root,
        calibration=args["runner"].calibration)
    assert replayed == record
    with pytest.raises(FileExistsError, match="already started"):
        execute_bound_retry_guarded_insertion(
            **args, enable_robot_motion=True)


def test_retry_contact_rejects_stale_state_before_command(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    stale = replace(args["pre_state"], sample_timestamp_s=100.9,
                    arm_timestamp_s=100.9, hand_timestamp_s=100.9)
    with pytest.raises(ValueError, match="fresh held state"):
        execute_bound_retry_guarded_insertion(
            **{**args, "pre_state": stale}, enable_robot_motion=True)
    assert args["adapter"].calls == 0


def test_retry_contact_latches_metric_failure_and_requests_stop(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    args["adapter"].bad_metric_hash = True
    with pytest.raises(RuntimeError, match="metric record changed"):
        execute_bound_retry_guarded_insertion(
            **args, enable_robot_motion=True)
    assert args["adapter"].stops == 1
    failure = json.loads((tmp_path / "retry_guarded_executions/000/failure.json").read_text())
    assert failure["robot_state_after_failure"] == (
        "unknown_requires_supervised_recovery")


def test_retry_contact_rejects_unreviewed_limits_before_command(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    review = args["contact_commissioning_path"]
    record = json.loads(review.read_text())
    record["contact_limits"]["max_axial_force_n"] = 999.
    review.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="matching contact review"):
        execute_bound_retry_guarded_insertion(
            **args, enable_robot_motion=True)
    assert args["adapter"].calls == 0
    assert not (tmp_path / "retry_guarded_executions/000/started.json").exists()


def test_retry_contact_rejects_foreign_event_before_command(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    args["runner"]._attempt.events[-1]["value"] = "x_plus_1mm"
    with pytest.raises(ValueError, match="same failed held attempt"):
        execute_bound_retry_guarded_insertion(
            **args, enable_robot_motion=True)
    assert args["adapter"].calls == 0


def test_saved_retry_contact_rejects_changed_metric_after_execution(
        tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    execute_bound_retry_guarded_insertion(
        **args, enable_robot_motion=True)
    saved = tmp_path / "retry_guarded_executions/000/execution.json"
    metric = Path(json.loads(saved.read_text())["metric_record"]["path"])
    metric.write_text(metric.read_text() + " ")
    with pytest.raises(ValueError, match="metric source changed"):
        execution.verify_retry_guarded_execution(
            saved, handoff_report_path=args["handoff_report_path"],
            expected=args["expected"], previous=args["previous"],
            arrival=args["arrival"], checkpoint=args["checkpoint"],
            shift_plan=args["shift_plan"], mode=args["runner"].mode,
            shared_root=args["runner"].shared_root,
            calibration=args["runner"].calibration)
