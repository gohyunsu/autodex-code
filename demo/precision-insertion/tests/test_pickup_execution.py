"""The stock pickup is reused only after the demo's live start-state gate."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion import pickup_execution  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.pickup_execution import (  # noqa: E402
    PickupExecutionLimits, execute_bound_pickup,
)
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion.trial_preflight import TrialPreflight  # noqa: E402


def _state(stamp, arm, raw):
    measured = np.full(6, raw, dtype=float)
    return LiveRobotState(
        full_q=np.concatenate([arm, convert_inspire_raw(measured[None])[0]]),
        arm_qvel=np.zeros(7), sample_timestamp_s=stamp,
        arm_timestamp_s=stamp, arm_robot_uptime_s=stamp - 90,
        hand_timestamp_s=stamp, hand_raw_measured=measured,
        hand_raw_commanded=measured.copy(), max_hand_command_error_raw=0.,
        wrench=np.zeros(6))


class _Executor:
    def __init__(self, *, fail=False):
        self.calls = []
        self.fail = fail

    def execute(self, plan, **kwargs):
        self.calls.append((plan, kwargs))
        if self.fail:
            raise RuntimeError("simulated contact abort")
        return np.full(6, 600.0)


def _case(tmp_path, monkeypatch):
    monkeypatch.setattr(pickup_execution, "_wall_time", lambda: 101.1)
    pre = _state(101., np.zeros(7), 500.)
    post = _state(102., np.full(7, .03), 600.)
    planned = np.stack([pre.full_q.copy(), pre.full_q.copy()])
    planned[-1, :7] = post.full_q[:7]
    pickup = SimpleNamespace(
        traj=planned, success=True, scene_info=("table", "0", "1"),
        pregrasp_pose=np.zeros(6), grasp_pose=np.ones(6) * .2,
        wrist_se3=np.eye(4))
    trial = TrialPreflight(
        status="sampled_planning_pass", pose_class={"stem": "table_000"},
        attempted_candidates=(), selected_candidate_key=("table", "0", "1"),
        repose_target_stems=(), insertion_plan=SimpleNamespace(
            sampled_planning_pass=True), pickup_plan=pickup, trial_scene={
                "mesh": {}, "cuboid": {}},
        key_observation_id="key_1", key_capture_timestamp_s=99.,
        start_q_acquisition_timestamp_s=99., max_key_state_skew_s=.1,
        key_pose_world=np.eye(4), live_start_q=pre.full_q.copy(),
        attempted_before_trial=(), covered_scenes=(), limits=None,
        max_pose_error_deg=5., axial_waypoint_step_m=.002,
        max_candidate_attempts=None, session_calibration_sha256="s" * 64,
        catalog_sha256="c" * 64)
    report_dir = tmp_path / "preflight"
    report_dir.mkdir()
    archive = report_dir / "planned_trajectories.npz"
    np.savez_compressed(
        archive, pickup_approach=planned,
        pickup_pregrasp=pickup.pregrasp_pose,
        pickup_grasp=pickup.grasp_pose,
        pickup_wrist=pickup.wrist_se3)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    report = report_dir / "report.json"
    report.write_text(json.dumps({"artifacts": {
        "planned_trajectories": archive.name,
        "planned_trajectories_sha256": digest}}), encoding="utf-8")
    attempt_dir = tmp_path / "attempt_1"
    attempt_dir.mkdir()
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    (attempt_dir / "preflight_binding.json").write_text(json.dumps({
        "candidate_id": "table/0/1", "preflight_report": str(report),
        "preflight_report_sha256": report_hash,
        "session_calibration_sha256": "s" * 64,
        "measured_start_state_sha256": "m" * 64,
    }), encoding="utf-8")
    runner = object.__new__(SessionRunner)
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/1",
        started_at_s=100., events=[], failure_code=None)
    runner._preflight = trial
    runner._preflight_report_path = report
    runner._attempt_dir = attempt_dir
    runner.session_sha256 = "s" * 64
    monkeypatch.setattr(SessionRunner, "current_decision",
                        lambda self: SimpleNamespace(action="await_lift_observation"))
    monkeypatch.setattr(SessionRunner, "verify_current_preflight_evidence",
                        lambda self: {
                            "report_sha256": report_hash,
                            "measured_start_state_sha256": "m" * 64,
                        })
    limits = PickupExecutionLimits(
        max_start_age_s=.5, max_start_joint_error_rad=.01,
        max_arrival_arm_error_rad=.01, max_arm_hand_skew_s=.01,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01)
    executor = _Executor()
    kwargs = dict(runner=runner, executor=executor, planner=object(),
                  pre_state=pre, read_post_state=lambda: post, limits=limits,
                  motion_interlock=lambda: True, enable_robot_motion=True,
                  )
    return kwargs, archive, attempt_dir


def test_binds_pickup_without_replaying_lift(tmp_path, monkeypatch):
    kwargs, _archive, attempt_dir = _case(tmp_path, monkeypatch)
    report = execute_bound_pickup(**kwargs)
    assert report["status"] == "squeeze_command_and_feedback_complete"
    assert report["robot_ready"] is False
    assert len(kwargs["executor"].calls) == 1
    assert kwargs["executor"].calls[0][1]["skip_lift"] is True
    assert kwargs["executor"].calls[0][1]["start_from_current"] is True
    assert (attempt_dir / "pickup_execution.json").is_file()
    assert (attempt_dir / "pickup_started.json").is_file()
    with pytest.raises(FileExistsError):
        execute_bound_pickup(**kwargs)


@pytest.mark.parametrize("subclass", [False, True])
def test_known_stock_franka_follower_is_rejected_before_motion(
        tmp_path, monkeypatch, subclass):
    kwargs, _archive, attempt_dir = _case(tmp_path, monkeypatch)
    stock_type = type("FrankaExecutor", (_Executor,), {
        "__module__": "src.execution.franka_executor"})
    stock_like = (type("DerivedExecutor", (stock_type,), {})()
                  if subclass else stock_type())
    kwargs["executor"] = stock_like
    with pytest.raises(RuntimeError, match="stalled stream"):
        execute_bound_pickup(**kwargs)
    assert stock_like.calls == []
    assert not (attempt_dir / "pickup_started.json").exists()


@pytest.mark.parametrize("defect", ["disabled", "stale", "plan_changed",
                                     "hand_changed", "bad_report_hash",
                                     "interlock"])
def test_rejects_before_motor_command(tmp_path, monkeypatch, defect):
    kwargs, archive, attempt_dir = _case(tmp_path, monkeypatch)
    if defect == "disabled":
        kwargs["enable_robot_motion"] = False
    elif defect == "stale":
        monkeypatch.setattr(pickup_execution, "_wall_time", lambda: 103.)
    elif defect == "plan_changed":
        archive.write_bytes(b"tampered plan")
    elif defect == "hand_changed":
        kwargs["runner"]._preflight.pickup_plan.grasp_pose[0] += .1
    elif defect == "bad_report_hash":
        binding = attempt_dir / "preflight_binding.json"
        value = json.loads(binding.read_text())
        value["preflight_report_sha256"] = "0" * 64
        binding.write_text(json.dumps(value))
    else:
        kwargs["motion_interlock"] = lambda: False
    with pytest.raises((PermissionError, ValueError)):
        execute_bound_pickup(**kwargs)
    assert not kwargs["executor"].calls
    assert not (attempt_dir / "pickup_execution.json").exists()


def test_failed_motor_call_is_logged_without_grasp_label(tmp_path, monkeypatch):
    kwargs, _archive, attempt_dir = _case(tmp_path, monkeypatch)
    kwargs["executor"] = _Executor(fail=True)
    with pytest.raises(RuntimeError, match="simulated contact abort"):
        execute_bound_pickup(**kwargs)
    log = json.loads((attempt_dir / "pickup_execution.json").read_text())
    assert log["status"] == "execution_or_feedback_failed"
    assert "grasp_success" not in log
    assert kwargs["runner"].active_attempt.events == []


def test_post_command_feedback_mismatch_is_not_grasp_success(
        tmp_path, monkeypatch):
    kwargs, _archive, attempt_dir = _case(tmp_path, monkeypatch)
    kwargs["read_post_state"] = lambda: _state(
        102., np.full(7, .2), 600.)
    with pytest.raises(RuntimeError, match="post-squeeze"):
        execute_bound_pickup(**kwargs)
    assert len(kwargs["executor"].calls) == 1
    log = json.loads((attempt_dir / "pickup_execution.json").read_text())
    assert log["status"] == "execution_or_feedback_failed"
    assert log["post_state"]["full_q"][:7] == [.2] * 7
    assert kwargs["runner"].active_attempt.events == []


def test_orphaned_start_marker_blocks_repeated_motor_command(
        tmp_path, monkeypatch):
    kwargs, _archive, attempt_dir = _case(tmp_path, monkeypatch)
    (attempt_dir / "pickup_started.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already began"):
        execute_bound_pickup(**kwargs)
    assert not kwargs["executor"].calls


def test_slow_interlock_expires_measured_start_before_motion(
        tmp_path, monkeypatch):
    kwargs, _archive, attempt_dir = _case(tmp_path, monkeypatch)
    stamps = iter((101.1, 102.0))
    monkeypatch.setattr(pickup_execution, "_wall_time", lambda: next(stamps))
    with pytest.raises(ValueError, match="aged while waiting"):
        execute_bound_pickup(**kwargs)
    assert not kwargs["executor"].calls
    assert not (attempt_dir / "pickup_started.json").exists()
