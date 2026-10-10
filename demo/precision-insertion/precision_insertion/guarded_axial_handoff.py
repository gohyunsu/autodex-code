"""Bind an observed socket hold to one saved 20 mm axial joint path.

This is an evidence packet for a future independently safeguarded contact
controller. It does not command Franka, certify continuous collision safety,
or turn a commanded wrist stroke into observed key penetration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode, select_mode
from .geometry import validate_se3
from .live_robot_state import LiveRobotState
from .preinsert_checkpoint import verify_preinsert_checkpoint


_SCHEMA = "precision_insertion_guarded_axial_handoff_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value: float, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a commissioned positive limit")
    return float(value)


def _state_from_record(record: dict) -> LiveRobotState:
    if (not isinstance(record, dict) or
            record.get("schema") != "precision_insertion_live_robot_state_v1"):
        raise ValueError("axial handoff has no measured robot state")
    try:
        return LiveRobotState(
            full_q=np.asarray(record["full_q"], dtype=float),
            arm_qvel=np.asarray(record["arm_qvel"], dtype=float),
            sample_timestamp_s=record["sample_timestamp_s"],
            arm_timestamp_s=record["arm_timestamp_s"],
            arm_robot_uptime_s=record["arm_robot_uptime_s"],
            hand_timestamp_s=record["hand_timestamp_s"],
            hand_raw_measured=np.asarray(record["hand_raw_measured"], dtype=float),
            hand_raw_commanded=np.asarray(record["hand_raw_commanded"], dtype=float),
            max_hand_command_error_raw=record["max_hand_command_error_raw"],
            wrench=np.asarray(record["wrench"], dtype=float),
            source=record["source"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("axial handoff robot state is malformed") from exc


def _build_record(
    *, postlift_report_path: Path, preinsert_report_path: Path,
    mode: TaskMode, attempt_id: str, candidate_id: str,
    session_calibration_sha256: str, measured_start: LiveRobotState,
    decision_timestamp_s: float, max_state_age_s: float,
    max_start_joint_error_rad: float, max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
) -> dict:
    """Recompute every handoff field from immutable upstream evidence."""
    if (not isinstance(mode, TaskMode) or
            not isinstance(attempt_id, str) or not attempt_id.strip() or
            not isinstance(candidate_id, str) or not candidate_id.strip() or
            not isinstance(session_calibration_sha256, str) or
            len(session_calibration_sha256) != 64 or
            any(char not in "0123456789abcdef"
                for char in session_calibration_sha256) or
            not isinstance(measured_start, LiveRobotState)):
        raise ValueError("axial handoff needs mode, attempt and measured state")
    limits = {
        "max_state_age_s": _positive(max_state_age_s, "state age"),
        "max_start_joint_error_rad": _positive(
            max_start_joint_error_rad, "start joint error"),
        "max_arm_hand_skew_s": _positive(max_arm_hand_skew_s, "arm/hand skew"),
        "max_hand_command_error_raw": _positive(
            max_hand_command_error_raw, "hand tracking error"),
        "max_arm_velocity_rad_s": _positive(
            max_arm_velocity_rad_s, "hold velocity"),
    }
    measured_start.validate(
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    postlift_path = Path(postlift_report_path).expanduser().resolve()
    preinsert_path = Path(preinsert_report_path).expanduser().resolve()
    if not postlift_path.is_file() or not preinsert_path.is_file():
        raise ValueError("axial handoff source report is missing")
    postlift = json.loads(postlift_path.read_text(encoding="utf-8"))
    preinsert = verify_preinsert_checkpoint(preinsert_path)
    if (postlift.get("schema") not in {
                "precision_insertion_postlift_preflight_v1",
                "precision_insertion_bounded_postlift_preflight_v1"} or
            postlift.get("status") != "sampled_postlift_preflight_pass" or
            postlift.get("robot_ready") is not False or
            postlift.get("attempt_id") != attempt_id or
            postlift.get("candidate_key") != candidate_id.split("/") or
            postlift.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            preinsert.get("preinsert_reached") is not True or
            preinsert.get("attempt_id") != attempt_id or
            preinsert.get("candidate_id") != candidate_id or
            preinsert.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            preinsert.get("postlift_report_path") != str(postlift_path) or
            preinsert.get("postlift_report_sha256") != _sha(postlift_path)):
        raise ValueError("axial path is not bound to this observed socket hold")
    planning = postlift.get("planning")
    targets = postlift.get("targets")
    if (not isinstance(planning, dict) or
            planning.get("sampled_planning_pass") is not True or
            planning.get("status") != "sampled_planning_pass" or
            not isinstance(planning.get("sampled_held_path_audit"), dict) or
            planning["sampled_held_path_audit"].get("sampled_clear") is not True or
            type(planning.get("axial_waypoint_count")) is not int or
            planning["axial_waypoint_count"] < 1 or
            not isinstance(targets, dict) or
            targets.get("schema") != "precision_insertion_rigid_targets_v1" or
            targets.get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm,
                "key_object": mode.key_object,
                "socket_object": mode.socket_object,
                "target_depth_m": mode.target_depth_m,
            } or
            not math.isclose(mode.target_depth_m, .020,
                             abs_tol=1e-9, rel_tol=0)):
        raise ValueError("axial handoff lacks a passing 20 mm path audit")
    queries = planning.get("planner_query_records")
    if not isinstance(queries, list):
        raise ValueError("axial handoff lacks individual planner results")
    axial_queries = [row for row in queries if isinstance(row, dict) and
                     row.get("stage") == "axial_waypoint"]
    if (len(axial_queries) != planning["axial_waypoint_count"] or
            [row.get("index") for row in axial_queries] !=
            list(range(1, len(axial_queries) + 1)) or
            any(row.get("success") is not True for row in axial_queries) or
            not any(isinstance(row, dict) and row.get("stage") == "transfer"
                    and row.get("success") is True for row in queries)):
        raise ValueError("axial handoff planner waypoints did not all pass")
    pre_pose = validate_se3(targets.get("T_robot_hand_preinsert"),
                            name="handoff preinsert target")
    final_pose = validate_se3(targets.get("T_robot_hand_verification"),
                              name="handoff 20 mm target")
    axis = np.asarray(targets.get("insertion_axis_robot"), dtype=float)
    clearance = targets.get("preinsert_clearance_m")
    if (axis.shape != (3,) or not np.all(np.isfinite(axis)) or
            not np.isclose(np.linalg.norm(axis), 1., atol=1e-8) or
            type(clearance) not in (int, float) or
            not math.isfinite(clearance) or clearance <= 0 or
            not np.allclose(final_pose[:3, :3], pre_pose[:3, :3],
                            atol=1e-8, rtol=0) or
            not np.allclose(final_pose[:3, 3] - pre_pose[:3, 3],
                            (clearance + .020) * axis, atol=1e-8, rtol=0)):
        raise ValueError("axial targets do not describe the 20 mm socket stroke")
    if (postlift.get("planned_trajectories") !=
            "planned_trajectories.npz"):
        raise ValueError("saved axial trajectory archive is absent")
    archive_path = postlift_path.parent / "planned_trajectories.npz"
    if (_sha(archive_path) !=
            postlift.get("planned_trajectories_sha256")):
        raise ValueError("saved axial trajectory bytes changed")
    with np.load(archive_path, allow_pickle=False) as archive:
        axial = np.asarray(archive["axial"], dtype=np.float64)
        transfer = np.asarray(archive["transfer"], dtype=np.float64)
    held = np.asarray(planning.get("held_hand_q"), dtype=float)
    if (axial.ndim != 2 or axial.shape[1] != 13 or len(axial) < 2 or
            transfer.ndim != 2 or transfer.shape[1] != 13 or
            len(transfer) < 2 or held.shape != (6,) or
            not np.all(np.isfinite(axial)) or
            not np.all(np.isfinite(transfer)) or
            not np.all(np.isfinite(held)) or
            not isinstance(planning.get("sample_counts"), dict) or
            planning["sample_counts"].get("axial") != len(axial) or
            planning["sample_counts"].get("transfer") !=
            len(transfer) or
            not np.array_equal(axial[0], transfer[-1]) or
            not np.allclose(axial[:, 7:], held, atol=1e-8, rtol=0) or
            not np.allclose(transfer[:, 7:], held, atol=1e-8, rtol=0)):
        raise ValueError("axial path is discontinuous or changes the grasp")
    observed_at = float(preinsert["observation_completed_at_s"])
    decision_at = float(decision_timestamp_s)
    sample_at = float(measured_start.sample_timestamp_s)
    if (not all(math.isfinite(value) for value in
                (observed_at, decision_at, sample_at)) or
            not observed_at < sample_at <= decision_at or
            decision_at - sample_at > limits["max_state_age_s"] or
            np.max(np.abs(measured_start.full_q - axial[0])) >
            limits["max_start_joint_error_rad"]):
        raise ValueError("measured hold is stale or differs from the axial start")
    return {
        "schema": _SCHEMA,
        "attempt_id": attempt_id,
        "candidate_id": candidate_id,
        "session_calibration_sha256": session_calibration_sha256,
        "mode": targets["mode"],
        "postlift_report_path": str(postlift_path),
        "postlift_report_sha256": _sha(postlift_path),
        "preinsert_report_path": str(preinsert_path),
        "preinsert_report_sha256": _sha(preinsert_path),
        "trajectory_archive_path": str(archive_path.resolve()),
        "trajectory_archive_sha256": _sha(archive_path),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "axial_sample_count": len(axial),
        "target_depth_m": mode.target_depth_m,
        "measured_start": measured_start.to_record(),
        "decision_timestamp_s": decision_at,
        "limits": limits,
        "scope": "read_only_axial_handoff_not_motion_or_key_depth_success",
        "robot_ready": False,
    }


def prepare_guarded_axial_handoff(
    *, postlift_report_path: Path, preinsert_report_path: Path,
    mode: TaskMode, attempt_id: str, candidate_id: str,
    session_calibration_sha256: str, measured_start: LiveRobotState,
    decision_timestamp_s: float, max_state_age_s: float,
    max_start_joint_error_rad: float, max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
    output_dir: Path,
) -> Path:
    """Save an immutable-by-convention packet; never request robot motion."""
    record = _build_record(
        postlift_report_path=postlift_report_path,
        preinsert_report_path=preinsert_report_path, mode=mode,
        attempt_id=attempt_id, candidate_id=candidate_id,
        session_calibration_sha256=session_calibration_sha256,
        measured_start=measured_start,
        decision_timestamp_s=decision_timestamp_s,
        max_state_age_s=max_state_age_s,
        max_start_joint_error_rad=max_start_joint_error_rad,
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    report = target / "report.json"
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def verify_guarded_axial_handoff(report_path: Path) -> dict:
    """Replay a packet against the same original plans and camera checkpoint."""
    record = json.loads(Path(report_path).expanduser().resolve().read_text(
        encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema") != _SCHEMA or
            record.get("robot_ready") is not False or
            record.get("scope") !=
            "read_only_axial_handoff_not_motion_or_key_depth_success"):
        raise ValueError("invalid guarded axial handoff report")
    mode_record = record["mode"]
    mode = select_mode(mode_record["family"], mode_record["gap_mm"])
    limits = record["limits"]
    expected = _build_record(
        postlift_report_path=Path(record["postlift_report_path"]),
        preinsert_report_path=Path(record["preinsert_report_path"]),
        mode=mode, attempt_id=record["attempt_id"],
        candidate_id=record["candidate_id"],
        session_calibration_sha256=record["session_calibration_sha256"],
        measured_start=_state_from_record(record["measured_start"]),
        decision_timestamp_s=record["decision_timestamp_s"],
        max_state_age_s=limits["max_state_age_s"],
        max_start_joint_error_rad=limits["max_start_joint_error_rad"],
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    if record != expected:
        raise ValueError("guarded axial handoff differs from replay")
    return record
