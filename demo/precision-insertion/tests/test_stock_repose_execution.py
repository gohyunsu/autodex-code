"""Offline stock v8 reset execution with measured fake FR3 waypoints."""

import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT  # noqa: E402
from precision_insertion import stock_repose_execution as stock  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.repose_artifacts import (  # noqa: E402
    write_repose_preflight_artifacts,
)
from precision_insertion.repose_execution_evidence import (  # noqa: E402
    verify_repose_execution_evidence,
)
from precision_insertion.repose_preflight import ReposeHeldPreflight  # noqa: E402
from precision_insertion.repose_release import ReposeReleasePreflight  # noqa: E402
from precision_insertion.repose_transition import (  # noqa: E402
    ReposeTransitionPreflight,
)


def _setup(tmp_path, monkeypatch, *, restoration=False,
           squeeze_offset=0., squeeze_arm_offset=0.):
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    q0 = np.r_[FR3_CLEAR_VIEW, INSPIRE_INIT].astype(float)
    q1 = q0.copy()
    q1[7:] = .2
    q2 = q1.copy()
    q2[0] += .03
    q3 = q2.copy()
    q3[0] += .03
    q4 = q3.copy()
    q4[0] += .03
    q5 = q4.copy()
    q5[7:] = INSPIRE_INIT
    q6 = q5.copy()
    q6[0] += .03
    q7 = q0.copy()
    pickup = SimpleNamespace(success=True, traj=np.vstack((q0, q0)))
    held = ReposeHeldPreflight(
        "sampled_held_path_pass_release_unplanned",
        np.vstack((q1, q2)), np.vstack((q2, q3)),
        np.vstack((q3, q4)), {"sampled_clear": True}, (),
        np.eye(4), .12, q1[7:].copy(), "commanded_nominal")
    release = ReposeReleasePreflight(
        "nominal_release_exit_path_pass_drop_unobserved",
        np.vstack((q5, q6)), np.vstack((q6, q7)),
        {"sampled_clear": True}, (), INSPIRE_INIT.copy(),
        np.asarray(FR3_CLEAR_VIEW).copy())
    plan = ReposeTransitionPreflight(
        "nominal_reset_preflight_pass_drop_unobserved", "000", "001",
        12, np.eye(4), {"seed_id": "191"}, (), pickup, held, release,
        "observation_1", time.time() - 2., time.time() - 2., .1)
    source_files = {}
    for name in ("session", "catalog", "key_pose_world", "live_start_q",
                 "limits"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"source": name}), encoding="utf-8")
        source_files[name] = path
    output = (attempt_dir / "reset_restoration" / "000" / "preflight"
              if restoration else tmp_path / "preflight")
    folder = write_repose_preflight_artifacts(
        result=plan,
        trial_scene={"mesh": {"target": {}, "fixture_socket": {}},
                     "cuboid": {}},
        output_dir=output, source_files=source_files)
    report = folder / "report.json"
    actual = q0.copy()
    events = []

    def state():
        stamp = time.time()
        return LiveRobotState(
            actual.copy(), np.zeros(7), stamp, stamp, 100., stamp,
            np.zeros(6), np.zeros(6), 0., np.zeros(6))

    class Executor:
        def execute(self, _plan, **kwargs):
            assert kwargs["skip_lift"] is True
            assert kwargs["start_from_current"] is True
            events.append("pickup")
            actual[:] = q1
            actual[7] += squeeze_offset
            actual[0] += squeeze_arm_offset

        def stop_and_acknowledge(self):
            events.append("stop")
            return {"acknowledged": True}

    executor = Executor()

    class Clear:
        max_arm_error_rad = .05
        max_hand_error_rad = .05

        def __init__(self):
            self.executor = executor

        def _validate_feedback(self, measured, **_):
            assert isinstance(measured, LiveRobotState)

        def _verify_held_endpoint(self, expected, **_):
            if np.max(np.abs(actual[:7] - expected[:7])) > .05:
                raise ValueError("measured arm differs from v8 plan")
            return state()

        def _follow_autodex(self, path):
            events.append("follow")
            actual[:] = path[-1]

        def finish_planned_release_exit_with_autodex(
                self, *, release_plan, evidence_sink):
            assert release_plan is release
            for name, value in (("release_open", q5),
                                ("post_release_lift", q6),
                                ("post_release_retract", q7)):
                events.append(name)
                actual[:] = value
                evidence_sink.append({"name": name,
                                      "state": state().to_record()})
                time.sleep(.001)
            return state()

    monkeypatch.setattr(stock, "is_demo_completion_checked_executor",
                        lambda _: True)
    callback = stock.StockReposeExecutor(
        executor=executor, planner=object(), clear_view=Clear(),
        read_state=state, motion_interlock=lambda: True,
        enable_robot_motion=True)
    decision = SimpleNamespace(action=(
        "recover_released_key_then_reobserve" if restoration else
        "repose_execution_gate_required"))
    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(
            attempt_id="reset_1", started_at_s=time.time() - 10.),
        _attempt_dir=attempt_dir, _repose_report_path=report,
        session_sha256="a" * 64,
        current_decision=(
            (lambda: decision) if restoration else
            (lambda: SimpleNamespace(action="await_repose_observation"))))
    return callback, runner, decision, plan, report, source_files, events


@pytest.mark.parametrize("restoration", [False, True])
def test_stock_repose_runs_saved_v8_paths_and_replays_feedback(
        tmp_path, monkeypatch, restoration):
    callback, runner, decision, plan, report, _, events = _setup(
        tmp_path, monkeypatch, restoration=restoration)
    execution = (callback.execute_restoration(
        runner, decision, plan, report) if restoration else
        callback.execute_repose(runner, decision, plan))
    verified = verify_repose_execution_evidence(
        path=execution, attempt_id="reset_1",
        attempt_started_at_s=runner.active_attempt.started_at_s,
        preflight_report_path=report,
        preflight_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        session_calibration_sha256=runner.session_sha256,
        selected_seed=plan.selected_seed)
    assert verified["scope"] == "stock_measured_waypoints_not_key_landing_pose"
    assert verified["source_records"] == {
        "stock_feedback": verified["stock_feedback"]}
    assert events == ["pickup", "follow", "follow", "follow",
                      "release_open", "post_release_lift",
                      "post_release_retract", "stop"]


def test_stock_repose_rejects_changed_source_before_robot_command(
        tmp_path, monkeypatch):
    callback, runner, decision, plan, _, sources, events = _setup(
        tmp_path, monkeypatch)
    sources["key_pose_world"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        callback.execute_repose(runner, decision, plan)
    assert events == []


def test_stock_repose_accepts_measured_finger_offset_under_grasp(
        tmp_path, monkeypatch):
    callback, runner, decision, plan, _, _, events = _setup(
        tmp_path, monkeypatch, squeeze_offset=.2)
    callback.execute_repose(runner, decision, plan)
    assert events[:2] == ["pickup", "follow"]


def test_stock_repose_stops_before_lift_when_arm_misses_squeeze_waypoint(
        tmp_path, monkeypatch):
    callback, runner, decision, plan, _, _, events = _setup(
        tmp_path, monkeypatch, squeeze_arm_offset=.2)
    with pytest.raises(ValueError, match="measured arm differs"):
        callback.execute_repose(runner, decision, plan)
    assert events == ["pickup", "stop"]
    assert (runner._attempt_dir / "stock_repose_failure.json").is_file()
