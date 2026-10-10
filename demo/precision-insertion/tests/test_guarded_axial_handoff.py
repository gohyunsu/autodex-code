"""The saved 20 mm path must follow this attempt's observed socket hold."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.guarded_axial_handoff import (  # noqa: E402
    prepare_guarded_axial_handoff, verify_guarded_axial_handoff,
)
from precision_insertion.preinsert_checkpoint import (  # noqa: E402
    assess_preinsert_checkpoint, write_preinsert_checkpoint,
)
from precision_insertion.session_runner import SessionRunner  # noqa: E402
from test_preinsert_checkpoint import _setup  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, bad_target=False, bad_waypoint=False,
             changes_grasp=False):
    args = _setup(tmp_path)
    # Synthetic planner evidence only: the handoff tests binding/continuity,
    # not FR3 IK, real force control, or a true key-depth measurement.
    start = args["joint_sample"].full_q.copy()
    before = start.copy()
    before[0] -= .02
    end = start.copy()
    end[0] += .03
    if changes_grasp:
        end[7] += .01
    transfer = np.stack((before, start))
    axial = np.stack((start, end))
    archive = tmp_path / "planned_trajectories.npz"
    np.savez_compressed(archive, transfer=transfer, axial=axial)
    path = args["postlift_report_path"]
    postlift = json.loads(path.read_text(encoding="utf-8"))
    postlift["schema"] = "precision_insertion_postlift_preflight_v1"
    postlift["robot_ready"] = False
    postlift["planned_trajectories"] = archive.name
    postlift["planned_trajectories_sha256"] = _sha(archive)
    mode = args["mode"]
    pre = np.asarray(postlift["targets"]["T_robot_hand_preinsert"])
    final = pre.copy()
    final[2, 3] -= .04 if bad_target else .05
    postlift["targets"].update({
        "schema": "precision_insertion_rigid_targets_v1",
        "mode": {
            "family": mode.family, "gap_mm": mode.gap_mm,
            "key_object": mode.key_object,
            "socket_object": mode.socket_object,
            "target_depth_m": mode.target_depth_m,
        },
        "T_robot_hand_verification": final.tolist(),
        "insertion_axis_robot": [0., 0., -1.],
        "preinsert_clearance_m": .03,
    })
    postlift["planning"] = {
        "status": "sampled_planning_pass",
        "sampled_planning_pass": True,
        "sampled_held_path_audit": {"sampled_clear": True},
        "axial_waypoint_count": 10,
        "planner_query_records": [
            {"stage": "transfer", "success": True},
            *({"stage": "axial_waypoint", "index": i,
               "success": not (bad_waypoint and i == 4)}
              for i in range(1, 11)),
        ],
        "sample_counts": {"transfer": 2, "axial": 2},
        "held_hand_q": start[7:].tolist(),
    }
    path.write_text(json.dumps(postlift), encoding="utf-8")
    observed = assess_preinsert_checkpoint(**args)
    assert observed.preinsert_reached is True
    preinsert_path = write_preinsert_checkpoint(
        observed, tmp_path / "arrival")
    state = replace(
        args["joint_sample"], sample_timestamp_s=11.02,
        arm_timestamp_s=11.02, hand_timestamp_s=11.02)
    handoff = dict(
        postlift_report_path=path, preinsert_report_path=preinsert_path,
        mode=mode, attempt_id=args["attempt"].attempt_id,
        candidate_id=args["attempt"].candidate_id,
        session_calibration_sha256=(
            args["attempt"].session_calibration_sha256),
        measured_start=state, decision_timestamp_s=11.03,
        max_state_age_s=.02, max_start_joint_error_rad=.01,
        max_arm_hand_skew_s=.02, max_hand_command_error_raw=30.,
        max_arm_velocity_rad_s=.05,
        output_dir=tmp_path / "axial_handoff",
    )
    return handoff, archive, args["attempt"]


def test_axial_handoff_binds_observed_hold_and_exact_saved_path(tmp_path):
    args, archive, _attempt = _fixture(tmp_path)
    report = prepare_guarded_axial_handoff(**args)
    replay = verify_guarded_axial_handoff(report)
    assert replay["target_depth_m"] == .020
    assert replay["axial_sample_count"] == 2
    assert replay["robot_ready"] is False
    assert replay["scope"] == (
        "read_only_axial_handoff_not_motion_or_key_depth_success")
    cli = subprocess.run([
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "run_pipeline.py"),
        "verify-guarded-axial-handoff", "--report", str(report),
    ], capture_output=True, text=True, check=False)
    assert cli.returncode == 0, cli.stderr
    assert json.loads(cli.stdout)["robot_ready"] is False
    with pytest.raises(FileExistsError):
        prepare_guarded_axial_handoff(**args)
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        verify_guarded_axial_handoff(report)


def test_axial_handoff_rejects_wrong_attempt_or_stale_start(tmp_path):
    args, _archive, _attempt = _fixture(tmp_path)
    with pytest.raises(ValueError, match="not bound"):
        prepare_guarded_axial_handoff(
            **{**args, "candidate_id": "table/0/other"})
    stale = replace(
        args["measured_start"], sample_timestamp_s=11.001,
        arm_timestamp_s=11.001, hand_timestamp_s=11.001)
    with pytest.raises(ValueError, match="stale or differs"):
        prepare_guarded_axial_handoff(
            **{**args, "measured_start": stale})
    shifted = args["measured_start"].full_q.copy()
    shifted[0] += .02
    with pytest.raises(ValueError, match="stale or differs"):
        prepare_guarded_axial_handoff(**{
            **args, "measured_start": replace(
                args["measured_start"], full_q=shifted)})


def test_axial_handoff_rejects_wrong_20mm_target(tmp_path):
    args, _archive, _attempt = _fixture(tmp_path, bad_target=True)
    with pytest.raises(ValueError, match="20 mm socket stroke"):
        prepare_guarded_axial_handoff(**args)


@pytest.mark.parametrize("condition,message", [
    ({"bad_waypoint": True}, "waypoints did not all pass"),
    ({"changes_grasp": True}, "changes the grasp"),
])
def test_axial_handoff_rejects_failed_waypoint_or_opening_hand(
        tmp_path, condition, message):
    args, _archive, _attempt = _fixture(tmp_path, **condition)
    with pytest.raises(ValueError, match=message):
        prepare_guarded_axial_handoff(**args)


def test_session_handoff_requires_recorded_positive_arrival(tmp_path, monkeypatch):
    args, _archive, attempt = _fixture(tmp_path)
    runner = object.__new__(SessionRunner)
    runner.mode = args["mode"]
    runner.session_sha256 = args["session_calibration_sha256"]
    runner._attempt = attempt
    runner._attempt_dir = tmp_path / "attempt"
    runner._attempt_dir.mkdir()
    runner._postlift_report_path = args["postlift_report_path"]
    runner._postlift_report_sha256 = _sha(args["postlift_report_path"])
    runner._preinsert_report_path = args["preinsert_report_path"]
    runner._preinsert_report_sha256 = _sha(args["preinsert_report_path"])
    runner._guarded_axial_index = 0
    monkeypatch.setattr(SessionRunner, "current_decision", lambda self:
                        SimpleNamespace(action=(
                            "await_guarded_insertion_and_observation")))
    params = {key: value for key, value in args.items() if key in {
        "measured_start", "decision_timestamp_s", "max_state_age_s",
        "max_start_joint_error_rad", "max_arm_hand_skew_s",
        "max_hand_command_error_raw", "max_arm_velocity_rad_s",
    }}
    with pytest.raises(ValueError, match="observed hold"):
        runner.prepare_guarded_axial_handoff(**params)
    attempt.record_stage(
        "preinsert_reached", True, timestamp_s=11.01,
        evidence_refs={
            "trajectory": "saved/transfer.json",
            "key_socket_pose": str(args["preinsert_report_path"]),
            "grasp_state": "saved/transfer.json",
            "postlift_preflight": str(args["postlift_report_path"]),
            "preinsert_checkpoint": str(args["preinsert_report_path"]),
        })
    report = runner.prepare_guarded_axial_handoff(**params)
    assert report.parent.name == "000"
    assert verify_guarded_axial_handoff(report)["candidate_id"] == (
        attempt.candidate_id)
