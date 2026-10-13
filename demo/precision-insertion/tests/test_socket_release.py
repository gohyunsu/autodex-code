"""Offline socket release planning; fake planner never opens a robot."""

from pathlib import Path
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import socket_release  # noqa: E402
from precision_insertion import stock_socket_release  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT  # noqa: E402


def _setup(tmp_path, monkeypatch):
    attempt_dir = tmp_path / "attempt"
    handoff_path = attempt_dir / "guarded_axial_handoffs" / "000" / "report.json"
    handoff_path.parent.mkdir(parents=True)
    handoff_path.write_text("{}", encoding="utf-8")
    key = tmp_path / "key.obj"
    key.write_text("# fake key mesh\n", encoding="utf-8")
    postlift = tmp_path / "postlift.json"
    postlift.write_text(json.dumps({
        "targets": {"T_robot_key_verification": np.eye(4).tolist()}}),
        encoding="utf-8")
    axial = np.zeros((4, 13))
    axial[:, 0] = [0., .01, .02, .03]
    axial[:, 7:] = .2
    archive = tmp_path / "axial.npz"
    np.savez_compressed(archive, axial=axial)
    handoff = {
        "attempt_id": "attempt_1", "candidate_id": "table/0/3",
        "session_calibration_sha256": "a" * 64,
        "postlift_report_path": str(postlift),
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": socket_release._sha(archive),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
    }
    monkeypatch.setattr(socket_release, "verify_guarded_axial_handoff",
                        lambda _: handoff)
    monkeypatch.setattr(socket_release, "validated_frozen_socket_pose",
                        lambda **_: np.eye(4))
    scene = {"mesh": {"fixture_socket": {"file_path": "socket.obj"}},
             "cuboid": {"table": {"dims": [1., 1., .1]}}}
    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(attempt_id="attempt_1",
                                       candidate_id="table/0/3"),
        mode=SimpleNamespace(family="cylinder"),
        _attempt_dir=attempt_dir,
        _preflight=SimpleNamespace(trial_scene={
            "mesh": {"target": {"file_path": str(key)}}}),
        calibration=SimpleNamespace(collision_scene=scene),
        shared_root=tmp_path, session_sha256="a" * 64)
    limits = PathAuditLimits(.5, .05, 10., .01, 5., .01, 5., 0.)

    class Planner:
        def plan_js_to_init(self, world, start, *, start_hand_qpos,
                            goal_arm_qpos):
            assert world["mesh"]["target"]["file_path"] == str(key)
            np.testing.assert_allclose(start_hand_qpos, INSPIRE_INIT)
            return np.linspace(np.r_[start, start_hand_qpos],
                               np.r_[goal_arm_qpos, INSPIRE_INIT], 50)

    return runner, handoff_path, limits, Planner(), axial


def test_release_plan_opens_then_retraces_axis_and_stock_retract(
        tmp_path, monkeypatch):
    runner, handoff, limits, planner, axial = _setup(tmp_path, monkeypatch)
    worlds = []

    def check(_, world, path):
        worlds.append((world, path.copy()))
        return {"all_feasible": True, "first_invalid_index": None,
                "status": None, "backend": "fake_stock_rollout"}

    monkeypatch.setattr(socket_release, "_check_path", check)
    report_path = socket_release.plan_socket_release_exit(
        runner=runner, planner=planner, handoff_report_path=handoff,
        limits=limits, output_dir=runner._attempt_dir / "socket_release_plan")
    report, paths = socket_release.verify_socket_release_plan(report_path)
    assert report["robot_ready"] is False
    assert len(worlds) == 3
    assert "target" not in worlds[0][0]["mesh"]
    assert "target" in worlds[1][0]["mesh"]
    np.testing.assert_allclose(paths["opening"][0], axial[-1])
    np.testing.assert_allclose(paths["axial_exit"][:, :7], axial[::-1, :7])
    np.testing.assert_allclose(paths["retract"][-1, :7], FR3_CLEAR_VIEW)


def test_release_plan_rejects_axis_exit_collision_before_saving(
        tmp_path, monkeypatch):
    runner, handoff, limits, planner, _ = _setup(tmp_path, monkeypatch)
    calls = 0

    def check(_, world, path):
        nonlocal calls
        calls += 1
        return {"all_feasible": calls != 2,
                "first_invalid_index": 1 if calls == 2 else None,
                "status": "collision" if calls == 2 else None,
                "backend": "fake_stock_rollout"}

    monkeypatch.setattr(socket_release, "_check_path", check)
    with pytest.raises(ValueError, match="retrace"):
        socket_release.plan_socket_release_exit(
            runner=runner, planner=planner, handoff_report_path=handoff,
            limits=limits, output_dir=runner._attempt_dir / "socket_release_plan")
    assert not (runner._attempt_dir / "socket_release_plan").exists()


def test_stock_release_executes_preplanned_axis_exit_then_clear_view(
        tmp_path, monkeypatch):
    runner, handoff, limits, planner, axial = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(socket_release, "_check_path", lambda *_: {
        "all_feasible": True, "first_invalid_index": None,
        "status": None, "backend": "fake_stock_rollout"})
    monkeypatch.setattr(stock_socket_release,
                        "is_demo_completion_checked_executor", lambda _: True)
    now = time.time()
    metric = runner._attempt_dir / "stock_force_metric.json"
    metric.write_text(json.dumps({"completed_at_s": now - 1.}),
                      encoding="utf-8")
    metric_ref = {"path": str(metric),
                  "sha256": hashlib.sha256(metric.read_bytes()).hexdigest()}
    monkeypatch.setattr(stock_socket_release, "_metric_record",
                        lambda *_a, **_k: ({
                            "schema": "precision_insertion_stock_force_execution_v1",
                            "measurement": {"safety_abort": False}},
                            now - 2., now - 1.))
    q = axial[-1].copy()
    actions = []

    def state():
        stamp = time.time()
        return LiveRobotState(
            q.copy(), np.zeros(7), stamp, stamp, 10., stamp,
            np.zeros(6), np.zeros(6), 0., np.zeros(6))

    class Executor:
        def release(self, *_args, **_kwargs):
            actions.append("open")
            q[7:] = INSPIRE_INIT

        def stop_and_acknowledge(self):
            actions.append("stop")
            return {"acknowledged": True}

    def follow(path):
        actions.append("axis" if len(actions) == 2 else "retract")
        q[:] = path[-1]

    clear = SimpleNamespace(
        _validate_feedback=lambda *_a, **_k: None,
        _verify_release_waypoint=lambda *_a, **_k: actions.append("verify"),
        _follow_autodex=follow,
        verify_state=lambda measured, **_: measured,
        max_arm_error_rad=.05, max_hand_error_rad=.05)
    runner.current_decision = lambda: decision
    decision = SimpleNamespace(action="await_guarded_insertion_and_observation")
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner),
                           clear_view=clear)
    callback = stock_socket_release.StockSocketRelease(
        executor=Executor(), planner=planner,
        read_state=state, path_limits=limits)
    plan_path = callback.prepare(live, runner, decision, handoff)
    report = json.loads(plan_path.read_text(encoding="utf-8"))
    execution = {
        "safety_abort": False, "grasp_held": True,
        "axial_handoff": report["handoff"],
        "metric_record": metric_ref}
    release_path = callback(live, runner, decision, execution)
    saved = json.loads(release_path.read_text(encoding="utf-8"))
    assert saved["status"] == "released_and_clear_view"
    assert actions[0] == "open"
    assert actions.count("verify") == 2
    np.testing.assert_allclose(q[:7], FR3_CLEAR_VIEW)


def test_stock_release_never_opens_after_force_stop(tmp_path, monkeypatch):
    runner, handoff, limits, planner, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(socket_release, "_check_path", lambda *_: {
        "all_feasible": True, "first_invalid_index": None,
        "status": None, "backend": "fake_stock_rollout"})
    monkeypatch.setattr(stock_socket_release,
                        "is_demo_completion_checked_executor", lambda _: True)
    actions = []
    executor = SimpleNamespace(release=lambda *_a, **_k: actions.append("open"))
    callback = stock_socket_release.StockSocketRelease(
        executor=executor, planner=planner,
        read_state=lambda: None, path_limits=limits)
    runner.current_decision = lambda: decision
    decision = SimpleNamespace(action="await_guarded_insertion_and_observation")
    live = SimpleNamespace(opened=SimpleNamespace(runner=runner))
    callback.prepare(live, runner, decision, handoff)
    with pytest.raises(ValueError, match="completed held stroke"):
        callback(live, runner, decision, {"safety_abort": True,
                                          "grasp_held": True})
    assert actions == []
    next_runner, next_handoff, _, next_planner, next_axial = _setup(
        tmp_path / "next", monkeypatch)
    callback.planner = next_planner
    next_runner.active_attempt.attempt_id = "attempt_2"
    source = {
        "attempt_id": "attempt_2", "candidate_id": "table/0/3",
        "session_calibration_sha256": "a" * 64,
        "postlift_report_path": str(tmp_path / "next" / "postlift.json"),
        "trajectory_archive_path": str(tmp_path / "next" / "axial.npz"),
        "trajectory_archive_sha256": socket_release._sha(
            tmp_path / "next" / "axial.npz"),
        "axial_start_q": next_axial[0].tolist(),
        "axial_end_q": next_axial[-1].tolist(),
    }
    monkeypatch.setattr(socket_release, "verify_guarded_axial_handoff",
                        lambda _: source)
    next_runner.current_decision = lambda: decision
    next_live = SimpleNamespace(opened=SimpleNamespace(runner=next_runner))
    next_report = callback.prepare(next_live, next_runner, decision,
                                   next_handoff)
    assert next_report.is_file()
