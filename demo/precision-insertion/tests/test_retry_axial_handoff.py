"""The continuous XY retry needs its own fresh-contact source packet."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import retry_axial_handoff as handoff  # noqa: E402
from precision_insertion import postshift_arrival_replan as replan  # noqa: E402
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from test_postshift_arrival_replan import _setup  # noqa: E402


def _case(tmp_path, monkeypatch):
    call, _ = _setup(tmp_path, monkeypatch)
    # The upstream arrival fixture deliberately reuses a square calibration
    # record while mocking geometry. This packet test needs a cylinder-mode
    # attempt record, so bind its synthetic session metadata to that mode.
    mode = call["mode"]
    session_record = dict(call["calibration"].record)
    session_record["mode"] = {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
    }
    call["calibration"].record = session_record
    result = replan.plan_postshift_arrival_axial(**call)
    start = call["arrival"].joint_sample.full_q.copy()
    end = start.copy()
    end[0] += .0001
    planning = replace(
        result.planning,
        transfer_trajectory=np.stack([start, start]),
        axial_trajectory=np.stack([start, end]),
        planner_query_records=(
            {"stage": "arrival_hold", "success": True,
             "planner_api": "measured_fk_no_transfer",
             "executable_transfer": False},
            {"stage": "axial_waypoint", "index": 1, "success": True}),
        axial_waypoint_count=1,
        sampled_held_path_audit={"sampled_clear": True})
    result = replace(result, planning=planning)
    replan_report = replan.write_postshift_arrival_replan(
        result, tmp_path / "postshift_arrival_axial_replans" / "000")
    # The full replan source/CAD verifier is covered separately; here the
    # packet's state-event, path and freshness binding is under test.
    monkeypatch.setattr(handoff, "verify_postshift_arrival_replan",
                        lambda path, **_kwargs: json.loads(Path(path).read_text()))

    attempt = begin_attempt(
        attempt_id=result.attempt_id, mode=mode,
        session_record=call["calibration"].record,
        candidate_id=result.candidate_id, tabletop_pose_stem="000",
        xy_offset_socket_m=(0., 0.), started_at_s=99.)
    attempt.record_stage(
        "grasp_success", True, timestamp_s=99.2,
        evidence_refs={"vlm_observation": "vlm/lift.json",
                       "key_wrist_check": "grip.json"})
    attempt.record_stage(
        "preinsert_reached", True, timestamp_s=99.4,
        evidence_refs={"trajectory": "transfer.json",
                       "key_socket_pose": "hold.json",
                       "grasp_state": "grip.json",
                       "postlift_preflight": "postlift.json"})
    attempt.record_insertion_evidence(
        InsertionEvidence("partial", (.001, .002), "key_pose_multiview",
                          True, False, True), timestamp_s=99.6,
        evidence_refs={"vlm_observation": "vlm/final.json",
                       "key_depth": "depth.json", "alignment": "axis.json",
                       "force_trace": "force.json"})
    shift = call["shift_plan"]
    checkpoint = call["checkpoint"]
    diagnostic = json.loads(shift.diagnostic_report_path.read_text())
    attempt.record_grounded_retry(
        increment_socket_xy_m=shift.lateral.increment_socket_xy_m,
        supporting_cameras=tuple(diagnostic["alignment"]["inlier_cameras"]),
        timestamp_s=101.,
        evidence_refs={
            "axial_withdrawal": str(shift.withdrawal_evidence_path),
            "grounded_xy": str(shift.diagnostic_report_path),
            "lateral_preflight": str(
                checkpoint.lateral_preflight_report_path),
            "lateral_execution": str(checkpoint.lateral_execution_path),
            "postshift_arrival": str(result.arrival_report_path),
            "arrival_axial_preflight": str(replan_report),
        })
    pending_state = attempt.write_new(tmp_path / "state_004.json")
    arrived = call["arrival"].joint_sample
    measured = replace(
        arrived, sample_timestamp_s=101.1,
        arm_timestamp_s=101.1, hand_timestamp_s=101.1,
        arm_robot_uptime_s=11.1)
    args = dict(
        replan_report_path=replan_report, expected=result,
        previous=call["previous"], arrival=call["arrival"],
        checkpoint=checkpoint, shift_plan=shift,
        pending_state_path=pending_state, mode=mode,
        shared_root=tmp_path, calibration=call["calibration"],
        measured_start=measured, decision_timestamp_s=101.12,
        max_state_age_s=.2, max_arrival_age_s=.5,
        max_start_joint_error_rad=.001, max_hand_drift_raw=5.,
        max_arm_hand_skew_s=.01, max_hand_command_error_raw=10.,
        max_arm_velocity_rad_s=.01)
    return args


def test_retry_packet_binds_new_axial_path_and_pending_state(
        tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    packet = handoff.prepare_retry_axial_handoff(
        output_dir=tmp_path / "retry_handoff", **args)
    saved = handoff.verify_retry_axial_handoff(
        packet, expected=args["expected"], previous=args["previous"],
        arrival=args["arrival"], checkpoint=args["checkpoint"],
        shift_plan=args["shift_plan"], mode=args["mode"],
        shared_root=tmp_path, calibration=args["calibration"])
    assert saved["robot_ready"] is False
    assert saved["axial_start_q"] == args["arrival"].joint_sample.full_q.tolist()
    assert saved["pending_state_path"] == str(args["pending_state_path"])
    assert saved["trajectory_archive_path"] == str(
        args["replan_report_path"].parent / "planned_axial.npz")
    with pytest.raises(FileExistsError):
        handoff.prepare_retry_axial_handoff(
            output_dir=tmp_path / "retry_handoff", **args)


def test_retry_packet_rejects_stale_arrival_or_wrong_pending_event(
        tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="stale"):
        handoff.prepare_retry_axial_handoff(
            output_dir=tmp_path / "stale",
            **{**args, "decision_timestamp_s": 101.8})
    pending = json.loads(args["pending_state_path"].read_text())
    pending["events"][-1]["evidence_refs"]["arrival_axial_preflight"] = (
        "another_trial/report.json")
    args["pending_state_path"].write_text(json.dumps(pending))
    with pytest.raises(ValueError, match="pending event differs"):
        handoff.prepare_retry_axial_handoff(
            output_dir=tmp_path / "changed_event", **args)


def test_retry_packet_rejects_changed_axial_archive(tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    packet = handoff.prepare_retry_axial_handoff(
        output_dir=tmp_path / "retry_handoff", **args)
    archive = args["replan_report_path"].parent / "planned_axial.npz"
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="archive bytes changed"):
        handoff.verify_retry_axial_handoff(
            packet, expected=args["expected"], previous=args["previous"],
            arrival=args["arrival"], checkpoint=args["checkpoint"],
            shift_plan=args["shift_plan"], mode=args["mode"],
            shared_root=tmp_path, calibration=args["calibration"])


def test_retry_packet_rejects_joint_drift_and_nonaxial_archive(
        tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    moved = args["measured_start"].full_q.copy()
    moved[0] += .01
    with pytest.raises(ValueError, match="measured arrival hold"):
        handoff.prepare_retry_axial_handoff(
            output_dir=tmp_path / "moved",
            **{**args, "measured_start": replace(
                args["measured_start"], full_q=moved)})
    archive = args["replan_report_path"].parent / "planned_axial.npz"
    axial = np.load(archive, allow_pickle=False)["axial"]
    np.savez(archive, axial=axial, transfer=axial)
    report = json.loads(args["replan_report_path"].read_text())
    report["planned_axial_sha256"] = handoff._sha(archive)
    args["replan_report_path"].write_text(json.dumps(report))
    with pytest.raises(ValueError, match="non-axial motion"):
        handoff.prepare_retry_axial_handoff(
            output_dir=tmp_path / "nonaxial", **args)


def test_session_runner_accepts_only_this_pending_retry_once(
        tmp_path, monkeypatch):
    args = _case(tmp_path, monkeypatch)
    runner = object.__new__(SessionRunner)
    runner.mode = args["mode"]
    runner.shared_root = tmp_path
    runner.calibration = args["calibration"]
    runner._attempt_dir = tmp_path
    runner._attempt_index = 4
    runner._attempt = SimpleNamespace(
        attempt_id=args["expected"].attempt_id,
        candidate_id=args["expected"].candidate_id,
        events=[{"evidence_refs": {
            "arrival_axial_preflight": str(args["replan_report_path"])}}])
    runner._postshift_arrival_replan_committed_reports = {
        args["replan_report_path"].resolve()}
    runner._retry_axial_handoff_used_reports = set()
    runner._retry_axial_handoff_index = 0
    runner.current_decision = lambda: SimpleNamespace(
        action="await_retry_execution_and_observation")
    runner.verify_current_preflight_evidence = lambda: {}
    kwargs = dict(
        replan=args["expected"],
        replan_report_path=args["replan_report_path"],
        previous=args["previous"], arrival=args["arrival"],
        checkpoint=args["checkpoint"], shift_plan=args["shift_plan"],
        measured_start=args["measured_start"],
        decision_timestamp_s=args["decision_timestamp_s"],
        **{name: value for name, value in args.items()
           if name.startswith("max_")})
    packet = runner.prepare_retry_axial_handoff(**kwargs)
    assert packet == (
        tmp_path / "retry_guarded_axial_handoffs/000/report.json")
    assert packet.is_file()
    with pytest.raises(ValueError, match="not the pending XY event"):
        runner.prepare_retry_axial_handoff(**kwargs)
    runner.current_decision = lambda: SimpleNamespace(action="select_grasp")
    with pytest.raises(ValueError, match="pending grounded attempt"):
        runner.prepare_retry_axial_handoff(**kwargs)
