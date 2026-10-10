"""A fresh retry arrival must replace, not replay, the old axial path."""

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

from precision_insertion import postshift_arrival_checkpoint as checkpoint  # noqa: E402
from precision_insertion import postshift_arrival_replan as replan  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from precision_insertion import session_runner as runner_module  # noqa: E402
from precision_insertion.session_policy import decide_after_attempt  # noqa: E402
from precision_insertion.uncertainty_margin import SurfaceDeviationBounds  # noqa: E402
from test_postshift_arrival_checkpoint import _case  # noqa: E402


def _setup(tmp_path, monkeypatch):
    args, visual = _case(tmp_path, monkeypatch)
    visual.update({
        "schema": "precision_insertion_grounded_alignment_v2",
        "insertion_axis_socket": [0., 0., -1.],
        "robot_ready": False,
    })
    observed = checkpoint.assess_postshift_arrival(**args)
    assert observed.status == "visual_alignment_within_budget"
    arrival_path = checkpoint.write_postshift_arrival_checkpoint(
        observed, tmp_path / "fresh_arrival")
    previous = args["preflight"]
    handoff = json.loads(args["handoff_report_path"].read_text())
    previous_path = Path(handoff["postshift_preflight_report_path"])
    # Original source verifiers have their own integration tests. The parent
    # test fixture carries a deliberately abbreviated simulated CAD screen.
    monkeypatch.setattr(replan, "verify_postshift_insertion_preflight",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(replan, "verify_postshift_arrival_checkpoint",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(replan, "validated_frozen_socket_pose",
                        lambda **_kwargs: np.eye(4))
    monkeypatch.setattr(replan, "_load_mesh", lambda _path: SimpleNamespace(
        vertices=np.array([[0., 0., 0.], [0., 0., .08]])))
    monkeypatch.setattr(replan, "audit_sampled_uncertainty_margins",
                        lambda **_kwargs: {"sampled_margin_pass": True})
    wrist = args["planner"].fk_wrist(observed.joint_sample.full_q)
    key_goal = np.eye(4)
    key_goal[2, 3] = .115
    fake_targets = SimpleNamespace(
        T_key_hand=None, T_robot_key_verification=key_goal,
        T_robot_hand_preinsert=wrist,
        to_record=lambda: {"schema": "test_arrival_target"})

    def build(**kwargs):
        fake_targets.T_key_hand = kwargs["T_key_hand"]
        return fake_targets

    monkeypatch.setattr(replan, "build_rigid_insertion_targets", build)
    seen = {}

    def screen(**kwargs):
        seen["screen"] = kwargs
        return {"endpoint_pass": True,
                "T_socket_key_tested": key_goal.tolist()}

    def axial(**kwargs):
        seen["axial"] = kwargs
        start = kwargs["start_q"]
        end = start.copy()
        end[0] += .0001
        return InsertionPreflight(
            "sampled_planning_pass", None,
            np.stack([start, start]), np.stack([start, end]),
            {"sampled_clear": True}, 1, start[7:], "measured", ())

    monkeypatch.setattr(replan, "plan_held_transfer_and_axial", axial)
    trial = SimpleNamespace(
        trial_scene={},
        limits=SimpleNamespace(minimum_hand_clearance_m=.0002),
        axial_waypoint_step_m=.005)
    call = dict(
        planner=args["planner"], mode=args["mode"], shared_root=tmp_path,
        calibration=args["calibration"], trial=trial, previous=previous,
        previous_report_path=previous_path, arrival=observed,
        arrival_report_path=arrival_path, checkpoint=args["checkpoint"],
        shift_plan=args["shift_plan"],
        bounds=SurfaceDeviationBounds(
            .02, .01, "commissioned_future_trial_surface_bound"),
        max_visual_tip_error_m=.0002,
        max_visual_axis_error_deg=1.,
        max_axis_prior_residual_deg=3., screen=screen)
    return call, seen


def test_replans_fresh_arrival_endpoint_and_axial_only(tmp_path, monkeypatch):
    call, seen = _setup(tmp_path, monkeypatch)
    result = replan.plan_postshift_arrival_axial(**call)
    assert result.status == "sampled_arrival_20mm_axial_preflight_pass"
    assert result.to_record()["old_axial_path_reusable"] is False
    assert result.to_record()["axial_contact_authorized"] is False
    assert seen["axial"]["start_at_preinsert"] is True
    np.testing.assert_array_equal(
        seen["axial"]["start_q"], call["arrival"].joint_sample.full_q)
    np.testing.assert_array_equal(
        seen["screen"]["hand_poses_override"]["measured_arrival"],
        call["arrival"].joint_sample.full_q[7:])
    assert seen["screen"]["cylinder_yaw_gauge_socket_rad"] == (
        result.yaw_gauge_socket_rad)

    report = replan.write_postshift_arrival_replan(
        result, tmp_path / "fresh_axial")
    saved = json.loads(report.read_text())
    assert saved["planned_axial"] == "planned_axial.npz"
    assert "transfer" not in np.load(report.parent / "planned_axial.npz").files
    assert saved["old_axial_path_reusable"] is False


def test_arrival_abstain_or_underbounded_error_cannot_replan(
        tmp_path, monkeypatch):
    call, seen = _setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="newly aligned"):
        replan.plan_postshift_arrival_axial(**{
            **call, "arrival": replace(call["arrival"],
                                       status="visual_abstain")})
    with pytest.raises(ValueError, match="omit prior or fresh"):
        replan.plan_postshift_arrival_axial(**{
            **call, "bounds": SurfaceDeviationBounds(
                .001, .01, "commissioned_future_trial_surface_bound")})
    assert not seen


def test_fresh_key_relation_that_moves_target_requires_new_hold(
        tmp_path, monkeypatch):
    call, _ = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(replan, "plan_held_transfer_and_axial",
                        lambda **kwargs: InsertionPreflight(
                            "arrival_hold_goal_residual", None, None, None,
                            None, 0, kwargs["held_hand_q"], "measured", ()))
    result = replan.plan_postshift_arrival_axial(**call)
    assert result.status == "arrival_hold_goal_residual"
    assert result.planning.axial_trajectory is None
    assert result.to_record()["axial_contact_authorized"] is False


def test_saved_arrival_axial_rejects_mutated_path_and_cad(
        tmp_path, monkeypatch):
    call, _ = _setup(tmp_path, monkeypatch)
    result = replan.plan_postshift_arrival_axial(**call)
    paths = AssetPaths(tmp_path, call["mode"])
    candidate = paths.candidate_dir / "table" / "0" / "0"
    files = {
        "key_mesh": paths.raw_mesh(call["mode"].key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
        "wrist_se3": candidate / "wrist_se3.npy",
        "pregrasp_pose": candidate / "pregrasp_pose.npy",
        "grasp_pose": candidate / "grasp_pose.npy",
    }
    for name, file in files.items():
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.is_file():
            file.write_bytes(name.encode("ascii"))
    endpoint = {
        **result.endpoint,
        "T_key_hand": result.hypothesis.T_key_hand.tolist(),
        "cylinder_yaw_gauge_socket_rad": result.yaw_gauge_socket_rad,
        "input_sha256": {name: hashlib.sha256(file.read_bytes()).hexdigest()
                         for name, file in files.items()},
    }
    result = replace(result, candidate_dir=candidate, endpoint=endpoint)
    report = replan.write_postshift_arrival_replan(
        result, tmp_path / "saved_arrival_axial")

    def verify():
        return replan.verify_postshift_arrival_replan(
            report, expected=result, previous=call["previous"],
            arrival=call["arrival"], checkpoint=call["checkpoint"],
            shift_plan=call["shift_plan"], mode=call["mode"],
            shared_root=tmp_path, calibration=call["calibration"])

    assert verify()["status"] == result.status
    archive = report.parent / "planned_axial.npz"
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        verify()
    other = replan.write_postshift_arrival_replan(
        result, tmp_path / "saved_arrival_axial_other")
    files["key_mesh"].write_bytes(b"changed CAD")
    with pytest.raises(ValueError, match="endpoint CAD"):
        replan.verify_postshift_arrival_replan(
            other, expected=result, previous=call["previous"],
            arrival=call["arrival"], checkpoint=call["checkpoint"],
            shift_plan=call["shift_plan"], mode=call["mode"],
            shared_root=tmp_path, calibration=call["calibration"])


def test_session_records_verified_continuous_shift_as_pending_retry(
        tmp_path, monkeypatch):
    call, _ = _setup(tmp_path, monkeypatch)
    result = replan.plan_postshift_arrival_axial(**call)
    attempt_dir = tmp_path / "attempt_record"
    attempt_dir.mkdir()
    shift_file = attempt_dir / "lateral_hold_preflights" / "000" / "report.json"
    shift_file.parent.mkdir(parents=True)
    shift_file.write_text("{}")
    arrival_file = (attempt_dir / "postshift_arrival_checkpoints" / "000" /
                    "report.json")
    arrival_file.parent.mkdir(parents=True)
    arrival_file.write_text("{}")
    replan_file = (attempt_dir / "postshift_arrival_axial_replans" / "000" /
                   "report.json")
    replan_file.parent.mkdir(parents=True)
    replan_file.write_text("{}")
    result = replace(result, arrival_report_path=arrival_file)
    checkpoint_record = replace(
        call["checkpoint"], lateral_preflight_report_path=shift_file)
    mode = call["mode"]
    session = {"schema": "precision_insertion_session_calibration_v1",
               "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                        "key_object": mode.key_object,
                        "socket_object": mode.socket_object}}
    attempt = begin_attempt(
        attempt_id=result.attempt_id, mode=mode, session_record=session,
        candidate_id=result.candidate_id, tabletop_pose_stem="000",
        xy_offset_socket_m=(0., 0.), started_at_s=1.)
    attempt.record_stage("grasp_success", True, timestamp_s=2.,
                         evidence_refs={"vlm_observation": "vlm/lift.json",
                                        "key_wrist_check": "grip.json"})
    attempt.record_stage("preinsert_reached", True, timestamp_s=3.,
                         evidence_refs={
                             "trajectory": "transfer.json",
                             "key_socket_pose": "hold.json",
                             "grasp_state": "grip.json",
                             "postlift_preflight": "postlift.json"})
    attempt.record_insertion_evidence(
        InsertionEvidence("partial", (.001, .002), "key_pose_multiview",
                          True, False, True), timestamp_s=4.,
        evidence_refs={"vlm_observation": "vlm/final.json",
                       "key_depth": "depth.json", "alignment": "axis.json",
                       "force_trace": "force.json"})
    runner = object.__new__(SessionRunner)
    runner.mode = mode
    runner.shared_root = tmp_path
    runner.calibration = call["calibration"]
    runner.max_xy_retries = 2
    runner._attempt = attempt
    runner._attempt_dir = attempt_dir
    runner._attempt_index = 3
    runner._attempted = set()
    runner._postlift_preflight = None
    runner._lift_checkpoint = None
    runner._postshift_arrival_replan_saved_reports = {replan_file}
    runner._postshift_arrival_replan_committed_reports = set()
    runner._postshift_arrival_replan_used_reports = {arrival_file}
    runner._postshift_arrival_saved_reports = {arrival_file}
    monkeypatch.setattr(
        SessionRunner, "current_decision",
        lambda self: decide_after_attempt(
            self._attempt, max_xy_retries=self.max_xy_retries))
    monkeypatch.setattr(SessionRunner, "verify_current_preflight_evidence",
                        lambda _self: {})
    monkeypatch.setattr(runner_module, "verify_grounded_lateral_preflight",
                        lambda *_args, **_kwargs: {})
    monkeypatch.setattr(replan, "verify_postshift_arrival_replan",
                        lambda *_args, **_kwargs: {})
    timestamp = call["arrival"].decision_timestamp_s + .1
    kwargs = dict(
        replan=result, replan_report_path=replan_file,
        previous=call["previous"], arrival=call["arrival"],
        checkpoint=checkpoint_record, shift_plan=call["shift_plan"],
        shift_plan_report_path=shift_file, timestamp_s=timestamp)
    updated = runner.record_grounded_xy_retry(**kwargs)
    assert updated.events[-1]["value"] == "grounded_continuous_xy"
    assert updated.xy_offset_socket_m == pytest.approx(
        call["shift_plan"].lateral.increment_socket_xy_m)
    assert runner.current_decision().action == (
        "await_retry_execution_and_observation")
    assert (attempt_dir / "state_004.json").is_file()
    with pytest.raises(ValueError, match="failed held attempt"):
        runner.record_grounded_xy_retry(**kwargs)
