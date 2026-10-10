"""Post-squeeze replan uses measured q, not the nominal pickup lift start."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion import measured_lift_preflight as lift  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion.uncertainty_margin import (  # noqa: E402
    SurfaceDeviationBounds,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(stamp, arm, raw=600.):
    hand = np.full(6, raw)
    return LiveRobotState(
        full_q=np.concatenate([arm, convert_inspire_raw(hand[None])[0]]),
        arm_qvel=np.zeros(7), sample_timestamp_s=stamp,
        arm_timestamp_s=stamp, arm_robot_uptime_s=stamp - 90,
        hand_timestamp_s=stamp, hand_raw_measured=hand,
        hand_raw_commanded=hand.copy(), max_hand_command_error_raw=0.,
        wrench=np.zeros(6))


class _Planner:
    def __init__(self):
        self.starts = []

    def fk_wrist(self, q):
        pose = np.eye(4)
        pose[2, 3] = float(q[0])
        return pose

    def plan_lift_preflight(self, start, _scene, *, lift_h,
                            timing_phase):
        assert timing_phase == "precision_insertion_measured_lift"
        assert lift_h == .10
        self.starts.append(np.asarray(start).copy())
        end = np.asarray(start, dtype=float).copy()
        end[0] += .10
        return SimpleNamespace(
            traj=np.stack([start, end]), start_full_qpos=start)


def _fixture(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    state = _state(102.1, np.full(7, .03))
    post = _state(102.0, np.full(7, .03))
    attempt_dir = tmp_path / "attempt_1"
    attempt_dir.mkdir()
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    np.save(candidate / "wrist_se3.npy", np.eye(4))
    plan_dir = tmp_path / "trial"
    plan_dir.mkdir()
    trial_report = plan_dir / "report.json"
    plan_sha = "p" * 64
    trial_report.write_text(json.dumps({"artifacts": {
        "planned_trajectories_sha256": plan_sha}}), encoding="utf-8")
    start_marker = attempt_dir / "pickup_started.json"
    start_marker.write_text(json.dumps({
        "status": "command_requested", "attempt_id": "attempt_1",
        "candidate_id": "table/0/1"}), encoding="utf-8")
    log_file = attempt_dir / "pickup_execution.json"
    log_file.write_text(json.dumps({
        "schema": "precision_insertion_pickup_execution_v1",
        "status": "squeeze_command_and_feedback_complete",
        "attempt_id": "attempt_1", "candidate_id": "table/0/1",
        "preflight_report_sha256": "r" * 64,
        "planned_trajectories_sha256": plan_sha,
        "pickup_started_sha256": _sha(start_marker),
        "command": "stock_franka_execute_skip_lift_start_from_current",
        "completed_at_s": 102.0, "post_state": post.to_record(),
        "squeeze_action_raw": [600.] * 6,
    }), encoding="utf-8")
    runner = object.__new__(SessionRunner)
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/1",
        tabletop_pose_stem="000", started_at_s=100.,
        events=[], failure_code=None)
    runner._preflight = SimpleNamespace(
        status="sampled_planning_pass",
        selected_candidate_key=("table", "0", "1"),
        trial_scene={"mesh": {}, "cuboid": {}})
    runner._attempt_dir = attempt_dir
    runner._preflight_report_path = trial_report
    runner._measured_lift_preflight = None
    runner._measured_lift_report_path = None
    runner._measured_lift_report_sha256 = None
    runner._measured_lift_index = 0
    runner.session_sha256 = "s" * 64
    runner.catalog = {"minimum_hand_clearance_m": .001}
    runner.mode = mode
    runner.shared_root = tmp_path
    runner.calibration = object()
    monkeypatch.setattr(SessionRunner, "current_decision",
                        lambda self: SimpleNamespace(action="await_lift_observation"))
    monkeypatch.setattr(SessionRunner, "verify_current_preflight_evidence",
                        lambda self: {"report_sha256": "r" * 64})
    monkeypatch.setattr(lift, "_wall_time", lambda: 102.2)
    monkeypatch.setattr(lift, "select_pose_candidates", lambda *args, **kwargs: {
        "status": "candidates_available", "candidates": [{
            "key": ["table", "0", "1"], "candidate_dir": str(candidate)}]})
    def screen(**kwargs):
        return {
            "endpoint_pass": True,
            "T_key_hand": kwargs["T_key_hand_override"].tolist(),
            "xy_offset_socket_m": [0., 0.],
            "hold_pose_screens": {"measured_squeeze": {
                "hand_q": kwargs["hand_poses_override"][
                    "measured_squeeze"].tolist()}},
        }
    monkeypatch.setattr(lift, "screen_grasp_endpoint", screen)
    targets = SimpleNamespace(to_record=lambda: {"test": "target"})
    monkeypatch.setattr(lift, "build_rigid_insertion_targets",
                        lambda **kwargs: targets)
    planned_calls = []

    def full_plan(**kwargs):
        planned_calls.append(kwargs)
        return InsertionPreflight(
            "sampled_planning_pass", kwargs["lift_trajectory"],
            np.stack([kwargs["start_q"], kwargs["start_q"]]),
            np.stack([kwargs["start_q"], kwargs["start_q"]]),
            {"sampled_clear": True}, 1, kwargs["held_hand_q"],
            "measured", ())

    monkeypatch.setattr(lift, "plan_held_transfer_and_axial", full_plan)
    monkeypatch.setattr(lift, "audit_sampled_uncertainty_margins",
                        lambda **kwargs: {"sampled_margin_pass": True})
    limits = PathAuditLimits(.02, .005, 1., .001, 1., .001, 1., .001)
    bounds = SurfaceDeviationBounds(
        .001, .001, "commissioned_future_trial_surface_bound")
    planner = _Planner()
    args = dict(runner=runner, planner=planner,
                pickup_execution_log=log_file, joint_sample=state,
                bounds=bounds, limits=limits, max_state_age_s=.5,
                max_post_squeeze_arm_drift_rad=.01,
                max_post_squeeze_hand_drift_raw=10.,
                max_arm_hand_skew_s=.01,
                max_hand_command_error_raw=10.,
                max_arm_velocity_rad_s=.01,
                axial_waypoint_step_m=.002)
    return args, planned_calls, log_file, start_marker


def test_measured_q_replans_lift_and_full_chain(tmp_path, monkeypatch):
    args, planned_calls, _log, _started = _fixture(tmp_path, monkeypatch)
    result = lift.plan_measured_lift_chain(**args)
    assert result.status == "sampled_measured_chain_pass"
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["cartesian_planner_mode"] == "default"
    assert result.relation_source.startswith("nominal_BODex")
    np.testing.assert_allclose(args["planner"].starts[0],
                               args["joint_sample"].full_q, atol=1e-7)
    assert len(planned_calls) == 1
    assert planned_calls[0]["held_hand_source"] == "measured"
    output = lift.write_measured_lift_chain(result, tmp_path / "new_chain")
    assert (output / "planned_trajectories.npz").is_file()
    assert lift.verify_measured_lift_chain(
        output / "report.json", expected=result)["status"] == result.status
    with pytest.raises(FileExistsError):
        lift.write_measured_lift_chain(result, output)


@pytest.mark.parametrize("defect", ["missing_marker", "changed_log_plan",
                                     "stale_state", "hand_drift"])
def test_bad_squeeze_evidence_blocks_planner(tmp_path, monkeypatch, defect):
    args, planned_calls, log_file, marker = _fixture(tmp_path, monkeypatch)
    if defect == "missing_marker":
        marker.rename(marker.with_suffix(".moved"))
    elif defect == "changed_log_plan":
        data = json.loads(log_file.read_text())
        data["planned_trajectories_sha256"] = "x" * 64
        log_file.write_text(json.dumps(data))
    elif defect == "stale_state":
        monkeypatch.setattr(lift, "_wall_time", lambda: 104.)
    else:
        args["joint_sample"] = _state(102.1, np.full(7, .09))
    with pytest.raises(ValueError):
        lift.plan_measured_lift_chain(**args)
    assert not args["planner"].starts
    assert not planned_calls


def test_post_squeeze_replan_requires_the_bound_cartesian_mode(
        tmp_path, monkeypatch):
    args, planned_calls, _log, _marker = _fixture(tmp_path, monkeypatch)
    args["planner"]._native_pose_constraints_enabled = True
    with pytest.raises(ValueError, match="planner mode differs"):
        lift.plan_measured_lift_chain(**args)
    assert not args["planner"].starts
    assert not planned_calls

    args["runner"]._preflight.cartesian_planner_mode = (
        "native-locked-experimental")
    trial_file = args["runner"]._preflight_report_path
    saved = json.loads(trial_file.read_text())
    saved["cartesian_planner_mode"] = "native-locked-experimental"
    trial_file.write_text(json.dumps(saved), encoding="utf-8")
    result = lift.plan_measured_lift_chain(**args)
    assert result.status == "sampled_measured_chain_pass"
    assert result.to_record()["cartesian_planner_mode"] == (
        "native-locked-experimental")


def test_endpoint_and_uncertainty_margins_fail_closed(tmp_path, monkeypatch):
    args, planned_calls, _log, _marker = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(lift, "screen_grasp_endpoint",
                        lambda **kwargs: {
                            "endpoint_pass": False,
                            "T_key_hand": kwargs["T_key_hand_override"].tolist(),
                            "xy_offset_socket_m": [0., 0.],
                            "hold_pose_screens": {"measured_squeeze": {
                                "hand_q": kwargs["hand_poses_override"][
                                    "measured_squeeze"].tolist()}},
                        })
    result = lift.plan_measured_lift_chain(**args)
    assert result.status == "measured_squeeze_endpoint_rejected"
    assert not args["planner"].starts
    assert not planned_calls
    monkeypatch.setattr(lift, "screen_grasp_endpoint", lambda **kwargs: {
        "endpoint_pass": True,
        "T_key_hand": kwargs["T_key_hand_override"].tolist(),
        "xy_offset_socket_m": [0., 0.],
        "hold_pose_screens": {"measured_squeeze": {
            "hand_q": kwargs["hand_poses_override"][
                "measured_squeeze"].tolist()}},
    })
    monkeypatch.setattr(lift, "audit_sampled_uncertainty_margins",
                        lambda **kwargs: {"sampled_margin_pass": False})
    result = lift.plan_measured_lift_chain(**args)
    assert result.status == "sampled_uncertainty_margin_rejected"
    assert len(planned_calls) == 1


def test_saved_chain_rejects_changed_trajectory_or_pickup_log(
        tmp_path, monkeypatch):
    args, _planned_calls, log_file, _marker = _fixture(tmp_path, monkeypatch)
    result = lift.plan_measured_lift_chain(**args)
    output = lift.write_measured_lift_chain(result, tmp_path / "new_chain")
    archive = output / "planned_trajectories.npz"
    original = archive.read_bytes()
    archive.write_bytes(b"different trajectory")
    with pytest.raises(ValueError, match="path bytes changed"):
        lift.verify_measured_lift_chain(output / "report.json")
    archive.write_bytes(original)
    log_file.write_text(log_file.read_text() + " ")
    with pytest.raises(ValueError, match="pickup evidence changed"):
        lift.verify_measured_lift_chain(output / "report.json")


def test_uncommissioned_error_bound_is_not_an_execution_substitute(
        tmp_path, monkeypatch):
    args, _planned_calls, _log, _marker = _fixture(tmp_path, monkeypatch)
    args["bounds"] = SurfaceDeviationBounds(.001, .001, "one_trial_scatter")
    with pytest.raises(ValueError, match="future-trial surface bound"):
        lift.plan_measured_lift_chain(**args)
    assert not args["planner"].starts


def test_runner_saves_single_passing_measured_lift_chain(
        tmp_path, monkeypatch):
    args, _planned_calls, _log, _marker = _fixture(tmp_path, monkeypatch)
    runner = args.pop("runner")
    result = runner.prepare_measured_lift_chain(**args)
    assert result.status == "sampled_measured_chain_pass"
    report = runner._measured_lift_report_path
    assert report == runner._attempt_dir / "measured_lift_preflights/000/report.json"
    assert len(runner._measured_lift_report_sha256) == 64
    assert lift.verify_measured_lift_chain(report, expected=result)
    with pytest.raises(ValueError, match="already preflighted"):
        runner.prepare_measured_lift_chain(**args)


def test_endpoint_cannot_silently_use_other_xy_or_hand(
        tmp_path, monkeypatch):
    args, planned_calls, _log, _marker = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(lift, "screen_grasp_endpoint", lambda **kwargs: {
        "endpoint_pass": True,
        "T_key_hand": kwargs["T_key_hand_override"].tolist(),
        "xy_offset_socket_m": [.001, 0.],
        "hold_pose_screens": {"measured_squeeze": {
            "hand_q": kwargs["hand_poses_override"][
                "measured_squeeze"].tolist()}},
    })
    with pytest.raises(ValueError, match="different held relation or hand"):
        lift.plan_measured_lift_chain(**args)
    assert not args["planner"].starts
    assert not planned_calls
