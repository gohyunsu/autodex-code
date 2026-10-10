"""Fake controller tests for the non-contact held-transfer boundary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import transfer_execution  # noqa: E402
from precision_insertion.held_relation import HeldRelation  # noqa: E402
from precision_insertion.postlift_preflight import (  # noqa: E402
    PostLiftPreflight, write_postlift_preflight,
)
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.preinsert_checkpoint import _transfer_log  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion.transfer_execution import (  # noqa: E402
    TransferExecutionLimits, execute_bound_held_transfer,
)
from test_measured_lift_preflight import _state  # noqa: E402


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Adapter:
    def __init__(self, root):
        self.daemon_binary_path = root / "reviewed_daemon"
        self.daemon_binary_path.write_bytes(b"fake-test-binary")
        self.sources = {}
        for name in ("trajectory_feedback", "safety", "grasp_state"):
            source = root / f"{name}.json"
            source.write_text(json.dumps({"test_only": name}))
            self.sources[name] = {"path": str(source), "sha256": _sha(source)}
        self.calls = 0
        self.stops = 0
        self.fail = False

    def follow_transfer(self, path, *, max_duration_s, expected_hand_raw,
                        trajectory_sha256):
        self.calls += 1
        assert path.shape == (2, 13)
        assert np.array_equal(path[:, 7:], np.repeat(path[:1, 7:], 2, axis=0))
        assert np.all(expected_hand_raw == 600.)
        assert max_duration_s == .5
        if self.fail:
            raise RuntimeError("simulated transfer abort")
        return {
            "schema": "precision_insertion_external_transfer_result_v1",
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True, "terminal_hold_acknowledged": True,
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": 102.33, "completed_at_s": 102.5,
            "source_records": self.sources,
        }

    def stop_and_acknowledge(self):
        self.stops += 1
        return {"acknowledged": True}


def _setup(tmp_path, monkeypatch):
    runner = object.__new__(SessionRunner)
    runner._attempt_dir = tmp_path / "attempt"
    runner._attempt_dir.mkdir()
    runner.session_sha256 = "s" * 64
    runner.catalog_sha256 = "c" * 64
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/1",
        labels={"grasp_success": True, "preinsert_reached": None},
        failure_code=None)
    monkeypatch.setattr(SessionRunner, "current_decision", lambda self:
                        SimpleNamespace(action="transfer_execution_gate_required"))
    start = _state(102.0, np.full(7, .03))
    pre = _state(102.2, np.full(7, .03))
    final = _state(102.55, np.array([.13, .03, .03, .03, .03, .03, .03]))
    trajectory = np.stack((pre.full_q, final.full_q))
    planning = InsertionPreflight(
        "sampled_planning_pass", None, trajectory, None, {}, 0,
        pre.full_q[7:].copy(), "measured", ())
    identity = np.eye(4)
    relation = HeldRelation(identity, identity, np.zeros(3), 0., 0., "identity")
    targets = SimpleNamespace(to_record=lambda: {"T_key_hand": identity.tolist()})
    plan = PostLiftPreflight(
        "sampled_postlift_preflight_pass", "attempt_1", ("table", "0", "1"),
        runner.session_sha256, runner.catalog_sha256, "key_before", "key_after",
        101.9, 102.0, start.full_q.copy(), start, identity, identity,
        relation, {"endpoint_pass": True}, targets, planning)
    runner._postlift_preflight = plan
    output_dir = tmp_path / "postlift"
    write_postlift_preflight(plan, output_dir)
    runner._postlift_report_path = output_dir / "report.json"
    runner._postlift_report_sha256 = _sha(runner._postlift_report_path)
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
    limits = TransferExecutionLimits(
        max_start_age_s=.5, max_end_state_age_s=.5,
        max_execution_feedback_gap_s=.1,
        max_start_joint_error_rad=.01, max_end_joint_error_rad=.01,
        max_hand_drift_raw=5., max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_execution_duration_s=.5, max_watchdog_command_age_s=.1,
        max_commissioning_record_age_s=1.)
    clock = iter([102.3, 102.31, 102.32, 102.7])
    monkeypatch.setattr(transfer_execution, "_wall_time", lambda: next(clock))
    return dict(
        runner=runner, adapter=adapter, pre_state=pre,
        read_post_state=lambda: final, limits=limits,
        commissioning_record_path=commissioning,
        motion_interlock=lambda: True)


def test_transfer_defaults_to_no_motion(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    with pytest.raises(PermissionError, match="interlock"):
        execute_bound_held_transfer(**kwargs)
    assert kwargs["adapter"].calls == 0
    assert not (kwargs["runner"]._attempt_dir / "transfer_started.json").exists()


def test_transfer_writes_bound_log_for_later_preinsert_observation(
        tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    record = execute_bound_held_transfer(**kwargs, enable_robot_motion=True)
    assert record["measurement"] == {
        "trajectory_complete": True, "safety_abort": False,
        "grasp_held": True}
    assert "preinsert_reached" not in record
    path = kwargs["runner"]._attempt_dir / "transfer_execution.json"
    assert _transfer_log(path, kwargs["runner"]._attempt) == record
    assert kwargs["adapter"].calls == 1
    with pytest.raises(FileExistsError, match="already started"):
        execute_bound_held_transfer(**kwargs, enable_robot_motion=True)


def test_transfer_refuses_changed_saved_trajectory(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    archive = kwargs["runner"]._postlift_report_path.parent / "planned_trajectories.npz"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        execute_bound_held_transfer(**kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].calls == 0


def test_transfer_abort_latches_unknown_robot_state(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    kwargs["adapter"].fail = True
    with pytest.raises(RuntimeError, match="simulated transfer abort"):
        execute_bound_held_transfer(**kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].stops == 1
    assert not (kwargs["runner"]._attempt_dir / "transfer_execution.json").exists()
    failure = json.loads((kwargs["runner"]._attempt_dir /
                          "transfer_failure.json").read_text())
    assert failure["robot_state_after_failure"] == (
        "unknown_requires_supervised_recovery")


def test_transfer_rejects_late_measured_endpoint(tmp_path, monkeypatch):
    kwargs = _setup(tmp_path, monkeypatch)
    kwargs["read_post_state"] = lambda: _state(
        102.65, np.array([.13, .03, .03, .03, .03, .03, .03]))
    with pytest.raises(RuntimeError, match="hold differs"):
        execute_bound_held_transfer(**kwargs, enable_robot_motion=True)
    assert kwargs["adapter"].stops == 1
