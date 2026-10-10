"""Fake-adapter checks for the contact boundary; no test drives a robot."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.guarded_contact import GuardedContactLimits  # noqa: E402
from precision_insertion import guarded_insertion_execution as execution  # noqa: E402
from precision_insertion.guarded_insertion_execution import (  # noqa: E402
    GuardedInsertionExecutionLimits, execute_bound_guarded_insertion,
)
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from test_measured_lift_preflight import _state  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Adapter:
    def __init__(self, root: Path, metric: Path):
        self.daemon_binary_path = root / "reviewed_contact_daemon"
        self.daemon_binary_path.write_bytes(b"fake-test-only-binary")
        self.metric = metric
        self.calls = 0
        self.stops = 0
        self.bad_metric_hash = False

    def follow_guarded_insertion(
        self, axial, *, handoff_sha256, trajectory_archive_sha256,
        contact_limits, expected_hand_raw, max_duration_s,
    ):
        self.calls += 1
        assert axial.shape == (2, 13)
        assert max_duration_s == .5
        assert np.all(expected_hand_raw == 600.)
        assert contact_limits["target_depth_m"] == .020
        return {
            "schema": "precision_insertion_external_guarded_result_v1",
            "terminal_hold_acknowledged": True,
            "safety_abort": False,
            "handoff_sha256": handoff_sha256,
            "trajectory_archive_sha256": trajectory_archive_sha256,
            "started_at_s": 102.33,
            "completed_at_s": 102.5,
            "metric_record_path": str(self.metric),
            "metric_record_sha256": (
                "0" * 64 if self.bad_metric_hash else _sha(self.metric)),
        }

    def stop_and_acknowledge(self):
        self.stops += 1
        return {"acknowledged": True}


def _setup(tmp_path: Path, monkeypatch):
    runner = object.__new__(SessionRunner)
    runner.mode = select_mode("cylinder", 1.)
    runner.session_sha256 = "a" * 64
    runner._attempt_dir = tmp_path / "attempt"
    runner._attempt_dir.mkdir()
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/194",
        labels={"grasp_success": True, "preinsert_reached": True,
                "insertion_success": None},
        failure_code=None)
    monkeypatch.setattr(SessionRunner, "current_decision", lambda self:
                        SimpleNamespace(
                            action="await_guarded_insertion_and_observation"))
    runner._postlift_report_path = tmp_path / "postlift.json"
    runner._preinsert_report_path = tmp_path / "preinsert.json"
    runner._postlift_report_path.write_text("{}", encoding="utf-8")
    runner._preinsert_report_path.write_text("{}", encoding="utf-8")
    handoff_dir = runner._attempt_dir / "guarded_axial_handoffs" / "000"
    handoff_dir.mkdir(parents=True)
    handoff_report = handoff_dir / "report.json"
    handoff_report.write_text("{}", encoding="utf-8")
    measured = _state(102.0, np.full(7, .03))
    pre = _state(102.2, np.full(7, .03))
    end_arm = np.array([.04, .03, .03, .03, .03, .03, .03])
    post = _state(102.55, end_arm)
    axial = np.stack((pre.full_q, post.full_q))
    archive = tmp_path / "axial.npz"
    np.savez_compressed(archive, axial=axial)
    handoff = {
        "attempt_id": runner._attempt.attempt_id,
        "candidate_id": runner._attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "postlift_report_path": str(runner._postlift_report_path),
        "preinsert_report_path": str(runner._preinsert_report_path),
        "mode": {"family": "cylinder"},
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": _sha(archive),
        "axial_sample_count": 2,
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "decision_timestamp_s": 102.1,
        "measured_start": measured.to_record(),
    }
    monkeypatch.setattr(execution, "verify_guarded_axial_handoff",
                        lambda _path: handoff)
    metric_file = tmp_path / "external_metric.json"
    metric_file.write_text("{}", encoding="utf-8")
    metric = {
        "schema": "precision_insertion_guarded_execution_v2",
        "axial_handoff": {"path": str(handoff_report),
                            "sha256": _sha(handoff_report)},
        "measurement": {"safety_abort": False},
    }
    monkeypatch.setattr(execution, "_metric_record", lambda *_a, **_k:
                        (metric, 102.33, 102.5))
    adapter = _Adapter(tmp_path, metric_file)
    limits = GuardedInsertionExecutionLimits(
        max_start_age_s=.5, max_end_state_age_s=.5,
        max_execution_feedback_gap_s=.1,
        max_start_joint_error_rad=.01, max_end_joint_error_rad=.01,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    contact_limits = GuardedContactLimits(
        target_depth_m=.020, max_axial_force_n=10.,
        max_lateral_force_n=5., max_torque_nm=2.,
        max_lateral_error_m=.002, max_axis_tilt_deg=5.,
        max_yaw_error_deg=None, max_sample_age_s=.02,
        max_sample_gap_s=.03, max_duration_s=.5,
        max_depth_step_m=.005, max_depth_regression_m=.001,
        max_depth_overshoot_m=.001)
    watchdog = tmp_path / "watchdog_review.json"
    watchdog.write_text(json.dumps({
        "schema": "precision_insertion_robot_watchdog_commissioning_v1",
        "reviewed_by": "fake-test-reviewer", "tested_at_s": 102.,
        "daemon_binary_path": str(adapter.daemon_binary_path),
        "daemon_binary_sha256": _sha(adapter.daemon_binary_path),
        "command_expiry_s": .05,
        "stale_command_timeout_test_passed": True,
        "disconnect_stop_test_passed": True,
        "stop_ack_test_passed": True, "e_stop_test_passed": True,
    }), encoding="utf-8")
    contact_review = tmp_path / "contact_review.json"
    contact_review.write_text(json.dumps({
        "schema": "precision_insertion_contact_controller_commissioning_v1",
        "reviewed_by": "fake-test-reviewer", "tested_at_s": 102.,
        "daemon_binary_path": str(adapter.daemon_binary_path),
        "daemon_binary_sha256": _sha(adapter.daemon_binary_path),
        "contact_limits": asdict(contact_limits),
        "force_limit_test_passed": True,
        "contact_abort_test_passed": True,
        "sample_gap_abort_test_passed": True,
        "terminal_hold_test_passed": True,
    }), encoding="utf-8")
    clock = iter([102.3, 102.31, 102.32, 102.7, 102.8])
    monkeypatch.setattr(execution, "_wall_time", lambda: next(clock))
    return dict(
        runner=runner, adapter=adapter, handoff_report_path=handoff_report,
        pre_state=pre, read_post_state=lambda: post,
        limits=limits, contact_limits=contact_limits,
        max_handoff_age_s=.5,
        watchdog_commissioning_path=watchdog,
        contact_commissioning_path=contact_review,
        motion_interlock=lambda: True)


def test_guarded_insertion_defaults_to_no_motion(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="interlock"):
        execute_bound_guarded_insertion(**args)
    assert args["adapter"].calls == 0
    assert not (args["runner"]._attempt_dir /
                "guarded_insertion_started.json").exists()


def test_guarded_insertion_records_external_stroke_without_task_label(
    tmp_path, monkeypatch,
):
    args = _setup(tmp_path, monkeypatch)
    record = execute_bound_guarded_insertion(
        **args, enable_robot_motion=True)
    assert record["safety_abort"] is False
    assert record["robot_ready"] is False
    assert args["runner"]._attempt.labels["insertion_success"] is None
    assert args["adapter"].calls == 1
    assert (args["runner"]._attempt_dir /
            "guarded_insertion_execution.json").is_file()


def test_guarded_insertion_stops_on_changed_metric(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    args["adapter"].bad_metric_hash = True
    with pytest.raises(RuntimeError, match="metric record changed"):
        execute_bound_guarded_insertion(
            **args, enable_robot_motion=True)
    assert args["adapter"].stops == 1
    failure = json.loads((args["runner"]._attempt_dir /
                          "guarded_insertion_failure.json").read_text())
    assert failure["robot_state_after_failure"] == (
        "unknown_requires_supervised_recovery")


def test_guarded_insertion_rejects_unreviewed_contact_limits(
    tmp_path, monkeypatch,
):
    args = _setup(tmp_path, monkeypatch)
    review = args["contact_commissioning_path"]
    content = json.loads(review.read_text(encoding="utf-8"))
    content["contact_limits"]["max_axial_force_n"] = 999.
    review.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="matching contact review"):
        execute_bound_guarded_insertion(
            **args, enable_robot_motion=True)
    assert args["adapter"].calls == 0
    assert not (args["runner"]._attempt_dir /
                "guarded_insertion_started.json").exists()


def test_guarded_insertion_rejects_foreign_handoff_without_command(
    tmp_path, monkeypatch,
):
    args = _setup(tmp_path, monkeypatch)
    foreign = tmp_path / "other_attempt" / "report.json"
    foreign.parent.mkdir()
    foreign.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this attempt"):
        execute_bound_guarded_insertion(
            **{**args, "handoff_report_path": foreign},
            enable_robot_motion=True)
    assert args["adapter"].calls == 0
