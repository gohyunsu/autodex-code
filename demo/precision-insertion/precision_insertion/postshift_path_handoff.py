"""Bind a fresh post-shift hold to both saved transfer and 20 mm axial paths.

The post-shift cuRobo result starts at the *observed shifted hold*, not
necessarily at the axial start. This read-only packet makes that distinction
explicit. It neither executes the transfer nor licenses contact motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode
from .geometry import validate_se3
from .grounded_lateral import GroundedLateralPreflight
from .guarded_axial_handoff import _state_from_record
from .live_robot_state import LiveRobotState
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import (
    PostShiftInsertionPreflight, verify_postshift_insertion_preflight,
)
from .retry_session import _digest


_SCHEMA = "precision_insertion_postshift_path_handoff_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive commissioned limit")
    return float(value)


def _build_record(
    *, preflight_report_path: Path, expected: PostShiftInsertionPreflight,
    checkpoint: PostShiftCheckpoint, shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
    measured_start: LiveRobotState, decision_timestamp_s: float,
    max_state_age_s: float, max_start_joint_error_rad: float,
    max_hand_drift_raw: float, max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
) -> dict:
    if (not isinstance(expected, PostShiftInsertionPreflight) or
            not isinstance(checkpoint, PostShiftCheckpoint) or
            not isinstance(shift_plan, GroundedLateralPreflight) or
            not isinstance(mode, TaskMode) or mode.family != "cylinder" or
            not math.isclose(mode.target_depth_m, .020, abs_tol=1e-9) or
            not isinstance(measured_start, LiveRobotState)):
        raise ValueError("post-shift handoff needs a held 20 mm cylinder plan")
    limits = {
        "max_state_age_s": _positive(max_state_age_s, "state age"),
        "max_start_joint_error_rad": _positive(
            max_start_joint_error_rad, "start joint error"),
        "max_hand_drift_raw": _positive(max_hand_drift_raw, "hand drift"),
        "max_arm_hand_skew_s": _positive(max_arm_hand_skew_s, "arm/hand skew"),
        "max_hand_command_error_raw": _positive(
            max_hand_command_error_raw, "hand tracking error"),
        "max_arm_velocity_rad_s": _positive(
            max_arm_velocity_rad_s, "hold velocity"),
    }
    report_path = Path(preflight_report_path).expanduser().resolve()
    saved = verify_postshift_insertion_preflight(
        report_path, expected=expected, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=shared_root,
        calibration=calibration)
    planning = saved.get("planning")
    audit = planning.get("sampled_held_path_audit") if isinstance(
        planning, dict) else None
    if (saved.get("status") != "sampled_postshift_20mm_preflight_pass" or
            saved.get("attempt_id") != shift_plan.attempt_id or
            saved.get("candidate_id") != shift_plan.candidate_id or
            checkpoint.attempt_id != expected.attempt_id or
            checkpoint.candidate_id != expected.candidate_id):
        raise ValueError("post-shift handoff lacks a passing source-bound plan")
    if (saved.get("postshift_report_path") != str(
                expected.postshift_report_path.resolve()) or
            saved.get("postshift_report_sha256") !=
                _sha(expected.postshift_report_path) or
            saved.get("postlift_report_path") != str(
                shift_plan.postlift_report_path.resolve()) or
            not isinstance(planning, dict) or
            planning.get("status") != "sampled_planning_pass" or
            planning.get("sampled_planning_pass") is not True or
            planning.get("held_hand_source") != "measured" or
            not isinstance(audit, dict) or audit.get("sampled_clear") is not True or
            saved.get("uncertainty_margin", {}).get(
                "sampled_margin_pass") is not True):
        raise ValueError("post-shift handoff lacks measured, audited held paths")
    queries = planning.get("planner_query_records")
    waypoints = planning.get("axial_waypoint_count")
    if (not isinstance(queries, list) or
            type(waypoints) is not int or waypoints < 1 or
            len([row for row in queries if isinstance(row, dict) and
                 row.get("stage") == "axial_waypoint"]) != waypoints or
            not any(isinstance(row, dict) and row.get("stage") == "transfer"
                    and row.get("success") is True for row in queries) or
            any(not isinstance(row, dict) or row.get("success") is not True
                for row in queries)):
        raise ValueError("post-shift planner queries did not all pass")
    targets = saved.get("targets")
    if (not isinstance(targets, dict) or
            targets.get("schema") != "precision_insertion_rigid_targets_v1" or
            targets.get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm,
                "key_object": mode.key_object,
                "socket_object": mode.socket_object,
                "target_depth_m": mode.target_depth_m,
            }):
        raise ValueError("post-shift targets differ from the selected mode")
    pre = validate_se3(targets.get("T_robot_hand_preinsert"),
                       name="post-shift preinsert hand target")
    final = validate_se3(targets.get("T_robot_hand_verification"),
                         name="post-shift 20 mm hand target")
    axis = np.asarray(targets.get("insertion_axis_robot"), dtype=float)
    clearance = targets.get("preinsert_clearance_m")
    if (axis.shape != (3,) or not np.all(np.isfinite(axis)) or
            not np.isclose(np.linalg.norm(axis), 1., atol=1e-8) or
            type(clearance) not in (int, float) or
            not math.isfinite(clearance) or clearance <= 0 or
            not np.allclose(pre[:3, :3], final[:3, :3], atol=1e-8, rtol=0) or
            not np.allclose(final[:3, 3] - pre[:3, 3],
                            (clearance + mode.target_depth_m) * axis,
                            atol=1e-8, rtol=0)):
        raise ValueError("post-shift axial targets do not describe 20 mm")
    archive = (report_path.parent / saved["planned_trajectories"]).resolve()
    with np.load(archive, allow_pickle=False) as data:
        transfer = np.asarray(data["transfer"], dtype=np.float64)
        axial = np.asarray(data["axial"], dtype=np.float64)
    held = np.asarray(planning.get("held_hand_q"), dtype=float)
    if (transfer.ndim != 2 or axial.ndim != 2 or
            transfer.shape[1:] != (13,) or axial.shape[1:] != (13,) or
            len(transfer) < 2 or len(axial) < 2 or
            held.shape != (6,) or not np.all(np.isfinite(held)) or
            not np.all(np.isfinite(transfer)) or
            not np.all(np.isfinite(axial)) or
            not np.allclose(transfer[-1], axial[0], atol=1e-4, rtol=0) or
            not np.allclose(transfer[:, 7:], held, atol=1e-8, rtol=0) or
            not np.allclose(axial[:, 7:], held, atol=1e-8, rtol=0)):
        raise ValueError("post-shift transfer/axial path is discontinuous")
    measured_start.validate(
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    decision = float(decision_timestamp_s)
    sample = float(measured_start.sample_timestamp_s)
    if (not math.isfinite(decision) or
            not checkpoint.decision_timestamp_s < sample <= decision or
            decision - sample > limits["max_state_age_s"] or
            np.max(np.abs(measured_start.full_q - transfer[0])) >
                limits["max_start_joint_error_rad"] or
            np.max(np.abs(measured_start.hand_raw_measured -
                          checkpoint.joint_sample.hand_raw_measured)) >
                limits["max_hand_drift_raw"]):
        raise ValueError("fresh measured hold differs from post-shift transfer start")
    return {
        "schema": _SCHEMA,
        "attempt_id": expected.attempt_id,
        "candidate_id": expected.candidate_id,
        "session_calibration_sha256": _digest(calibration.record),
        "mode": targets["mode"],
        "postshift_preflight_report_path": str(report_path),
        "postshift_preflight_report_sha256": _sha(report_path),
        "postshift_checkpoint_path": str(expected.postshift_report_path),
        "postshift_checkpoint_sha256": expected.postshift_report_sha256,
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": _sha(archive),
        "transfer_start_q": transfer[0].tolist(),
        "transfer_end_q": transfer[-1].tolist(),
        "transfer_sample_count": len(transfer),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "axial_sample_count": len(axial),
        "transfer_required": bool(np.max(np.abs(
            transfer[-1] - transfer[0])) > 1e-8),
        "target_depth_m": mode.target_depth_m,
        "measured_start": measured_start.to_record(),
        "decision_timestamp_s": decision,
        "limits": limits,
        "scope": "read_only_transfer_then_axial_handoff_not_motion_or_success",
        "robot_ready": False,
    }


def prepare_postshift_path_handoff(
    *, output_dir: Path, **kwargs,
) -> Path:
    """Persist one exclusive read-only path packet."""
    record = _build_record(**kwargs)
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def verify_postshift_path_handoff(
    report_path: Path, *, expected: PostShiftInsertionPreflight,
    checkpoint: PostShiftCheckpoint, shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> dict:
    """Replay source and measured-start checks, without motion approval."""
    record = json.loads(Path(report_path).expanduser().resolve().read_text(
        encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != _SCHEMA or
            record.get("robot_ready") is not False or
            record.get("scope") !=
            "read_only_transfer_then_axial_handoff_not_motion_or_success"):
        raise ValueError("invalid post-shift path handoff")
    limits = record["limits"]
    rebuilt = _build_record(
        preflight_report_path=Path(record["postshift_preflight_report_path"]),
        expected=expected, checkpoint=checkpoint, shift_plan=shift_plan,
        mode=mode, shared_root=shared_root, calibration=calibration,
        measured_start=_state_from_record(record["measured_start"]),
        decision_timestamp_s=record["decision_timestamp_s"],
        max_state_age_s=limits["max_state_age_s"],
        max_start_joint_error_rad=limits["max_start_joint_error_rad"],
        max_hand_drift_raw=limits["max_hand_drift_raw"],
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    if record != rebuilt:
        raise ValueError("post-shift path handoff differs from replay")
    return record
