"""Fake-controller checks for the held 1 mm VLM correction boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import lateral_execution  # noqa: E402
from precision_insertion.grounded_lateral import GroundedLateralPreflight  # noqa: E402
from precision_insertion.lateral_execution import (  # noqa: E402
    LateralExecutionLimits, execute_bound_lateral_shift,
)
from precision_insertion.lateral_preflight import LateralHoldPreflight  # noqa: E402
from precision_insertion.postshift_checkpoint import _execution_log  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from test_measured_lift_preflight import _state  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Adapter:
    def __init__(self, root: Path):
        self.daemon_binary_path = root / "reviewed_lateral_daemon"
        self.daemon_binary_path.write_bytes(b"fake-test-only-daemon")
        self.sources = {}
        for name in ("trajectory_feedback", "safety", "grasp_state"):
            source = root / f"{name}.json"
            source.write_text(json.dumps({"test_only": name}), encoding="utf-8")
            self.sources[name] = {"path": str(source), "sha256": _sha(source)}
        self.calls = 0
        self.stops = 0
        self.fail = False

    def follow_lateral(self, path, *, max_duration_s, expected_hand_raw,
                       trajectory_sha256):
        self.calls += 1
        assert path.shape == (2, 13)
        assert np.array_equal(path[:, 7:],
                              np.repeat(path[:1, 7:], 2, axis=0))
        assert np.all(expected_hand_raw == 600.)
        assert max_duration_s == .5
        if self.fail:
            raise RuntimeError("simulated lateral controller failure")
        return {
            "schema": "precision_insertion_external_lateral_result_v1",
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True, "terminal_hold_acknowledged": True,
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": 102.33, "completed_at_s": 102.5,
            "source_records": self.sources,
        }

    def stop_and_acknowledge(self):
        self.stops += 1
        return {"acknowledged": True}


def _setup(tmp_path: Path, monkeypatch):
    runner = object.__new__(SessionRunner)
    runner._attempt_dir = tmp_path / "attempt"
    runner._attempt_dir.mkdir()
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/194",
        labels={"grasp_success": True, "preinsert_reached": True,
                "insertion_success": False})
    monkeypatch.setattr(SessionRunner, "current_decision", lambda self:
                        SimpleNamespace(
                            action="guarded_withdrawal_then_xy_assessment"))
    first = _state(102.0, np.full(7, .03))
    pre = _state(102.2, np.full(7, .03))
    final_arm = np.array([.031, .03, .03, .03, .03, .03, .03])
    final = _state(102.55, final_arm)
    trajectory = np.stack((pre.full_q, final.full_q))
    lateral = LateralHoldPreflight(
        "sampled_lateral_hold_shift_pass", (.001, 0.),
        pre.full_q.copy(), np.eye(4), np.eye(4), np.eye(4),
        trajectory, {"success": True}, {"sampled_clear": True})
    dummy = tmp_path / "source.json"
    dummy.write_text("{}", encoding="utf-8")
    plan = GroundedLateralPreflight(
        "attempt_1", "table/0/194", dummy, _sha(dummy),
        dummy, _sha(dummy), dummy, _sha(dummy), first,
        102.1, .0002, 1., lateral)
    report_dir = (runner._attempt_dir / "lateral_hold_preflights" / "000")
    nested = report_dir / "lateral"
    nested.mkdir(parents=True)
    trajectory_path = nested / "lateral_trajectory.npy"
    np.save(trajectory_path, trajectory)
    (nested / "report.json").write_text(json.dumps({
        "lateral_trajectory": trajectory_path.name,
        "lateral_trajectory_sha256": _sha(trajectory_path),
    }), encoding="utf-8")
    report = report_dir / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(lateral_execution, "verify_grounded_lateral_preflight",
                        lambda _plan, _path: {})
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
    }), encoding="utf-8")
    limits = LateralExecutionLimits(
        max_start_age_s=.5, max_end_state_age_s=.5,
        max_execution_feedback_gap_s=.1,
        max_start_joint_error_rad=.01, max_end_joint_error_rad=.01,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    clock = iter([102.3, 102.31, 102.32, 102.7, 102.8])
    monkeypatch.setattr(lateral_execution, "_wall_time", lambda: next(clock))
    return dict(
        runner=runner, plan=plan, plan_report_path=report,
        adapter=adapter, pre_state=pre,
        read_post_state=lambda: final, limits=limits,
        max_plan_age_s=.5,
        commissioning_record_path=commissioning,
        motion_interlock=lambda: True)


def test_lateral_shift_defaults_to_no_motion(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="interlock"):
        execute_bound_lateral_shift(**args)
    assert args["adapter"].calls == 0
    assert not (args["runner"]._attempt_dir / "lateral_executions" /
                "000" / "started.json").exists()


def test_lateral_shift_log_is_accepted_by_postshift_checkpoint(
    tmp_path, monkeypatch,
):
    args = _setup(tmp_path, monkeypatch)
    record = execute_bound_lateral_shift(
        **args, enable_robot_motion=True)
    log_path = (args["runner"]._attempt_dir / "lateral_executions" /
                "000" / "execution.json")
    assert _execution_log(log_path, plan=args["plan"],
                          plan_report_path=args["plan_report_path"])[0] == record
    assert record["robot_ready"] is False
    assert args["runner"]._attempt.labels["insertion_success"] is False
    assert args["adapter"].calls == 1


def test_lateral_shift_stops_and_marks_failure(tmp_path, monkeypatch):
    args = _setup(tmp_path, monkeypatch)
    args["adapter"].fail = True
    with pytest.raises(RuntimeError, match="simulated lateral"):
        execute_bound_lateral_shift(**args, enable_robot_motion=True)
    assert args["adapter"].stops == 1
    failure = json.loads((args["runner"]._attempt_dir / "lateral_executions" /
                          "000" / "failure.json").read_text())
    assert failure["robot_state_after_failure"] == (
        "unknown_requires_supervised_recovery")


def test_lateral_shift_rejects_changed_saved_path_before_motion(
    tmp_path, monkeypatch,
):
    args = _setup(tmp_path, monkeypatch)
    path = (args["plan_report_path"].parent / "lateral" /
            "lateral_trajectory.npy")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory changed"):
        execute_bound_lateral_shift(**args, enable_robot_motion=True)
    assert args["adapter"].calls == 0
