"""Read-only guarded-contact packet for a verified continuous XY retry.

Unlike the first-insertion handoff, this binds the *post-transfer* multi-view
arrival, fresh axial-only replan and append-only grounded retry event. The
packet contains no executable transfer and does not command or authorize
contact motion. A separately commissioned retry controller must still verify
this packet immediately before any guarded stroke.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode
from .grounded_lateral import GroundedLateralPreflight
from .guarded_axial_handoff import _positive, _state_from_record
from .live_robot_state import LiveRobotState
from .postshift_arrival_checkpoint import PostShiftArrivalCheckpoint
from .postshift_arrival_replan import (
    PostShiftArrivalReplan, verify_postshift_arrival_replan,
)
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import PostShiftInsertionPreflight
from .retry_session import _digest


_SCHEMA = "precision_insertion_retry_axial_handoff_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_record(
    *, replan_report_path: Path, expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint, checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    pending_state_path: Path, mode: TaskMode, shared_root: Path,
    calibration, measured_start: LiveRobotState,
    decision_timestamp_s: float, max_state_age_s: float,
    max_arrival_age_s: float, max_start_joint_error_rad: float,
    max_hand_drift_raw: float, max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
) -> dict:
    if (not isinstance(expected, PostShiftArrivalReplan) or
            not isinstance(arrival, PostShiftArrivalCheckpoint) or
            not isinstance(measured_start, LiveRobotState) or
            not isinstance(mode, TaskMode) or mode.family != "cylinder" or
            not math.isclose(mode.target_depth_m, .020, abs_tol=1e-9)):
        raise ValueError("retry axial handoff needs a measured 20 mm cylinder")
    limits = {
        "max_state_age_s": _positive(max_state_age_s, "state age"),
        "max_arrival_age_s": _positive(max_arrival_age_s, "arrival age"),
        "max_start_joint_error_rad": _positive(
            max_start_joint_error_rad, "start joint error"),
        "max_hand_drift_raw": _positive(max_hand_drift_raw, "hand drift"),
        "max_arm_hand_skew_s": _positive(
            max_arm_hand_skew_s, "arm/hand skew"),
        "max_hand_command_error_raw": _positive(
            max_hand_command_error_raw, "hand tracking error"),
        "max_arm_velocity_rad_s": _positive(
            max_arm_velocity_rad_s, "hold velocity"),
    }
    source = Path(replan_report_path).expanduser().resolve()
    saved = verify_postshift_arrival_replan(
        source, expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=mode,
        shared_root=shared_root, calibration=calibration)
    if (saved.get("status") !=
            "sampled_arrival_20mm_axial_preflight_pass" or
            saved.get("endpoint", {}).get("endpoint_pass") is not True or
            saved.get("uncertainty_margin", {}).get(
                "sampled_margin_pass") is not True or
            saved.get("old_axial_path_reusable") is not False or
            saved.get("axial_contact_authorized") is not False):
        raise ValueError("retry handoff lacks a fresh passing axial replan")
    planning = saved.get("planning")
    audit = (planning.get("sampled_held_path_audit")
             if isinstance(planning, dict) else None)
    if (not isinstance(planning, dict) or
            planning.get("sampled_planning_pass") is not True or
            planning.get("held_hand_source") != "measured" or
            not isinstance(audit, dict) or
            audit.get("sampled_clear") is not True):
        raise ValueError("retry axial path lacks measured held-key audit")
    queries = planning.get("planner_query_records")
    count = planning.get("axial_waypoint_count")
    if (not isinstance(queries, list) or type(count) is not int or
            count < 1 or not queries or queries[0] != {
                "stage": "arrival_hold", "success": True,
                "planner_api": "measured_fk_no_transfer",
                "executable_transfer": False,
            } or
            len(queries) != count + 1 or
            any(not isinstance(row, dict) or
                row.get("stage") != "axial_waypoint" or
                row.get("index") != index or
                row.get("success") is not True
                for index, row in enumerate(queries[1:], 1))):
        raise ValueError("retry axial queries include a transfer or failed step")
    archive_name = saved.get("planned_axial")
    if archive_name != "planned_axial.npz":
        raise ValueError("retry axial archive is missing")
    archive = (source.parent / archive_name).resolve()
    if (not archive.is_relative_to(source.parent) or not archive.is_file() or
            _sha(archive) != saved.get("planned_axial_sha256")):
        raise ValueError("retry axial archive bytes changed")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"axial"}:
            raise ValueError("retry archive contains non-axial motion")
        axial = np.asarray(data["axial"], dtype=np.float64)
    held = np.asarray(planning.get("held_hand_q"), dtype=float)
    if (axial.ndim != 2 or axial.shape[1] != 13 or len(axial) < 2 or
            not np.all(np.isfinite(axial)) or held.shape != (6,) or
            not np.all(np.isfinite(held)) or
            not np.array_equal(axial, expected.planning.axial_trajectory) or
            not np.allclose(axial[:, 7:], held, atol=1e-8, rtol=0) or
            not np.array_equal(axial[0], arrival.joint_sample.full_q) or
            planning.get("sample_counts", {}).get("axial") != len(axial)):
        raise ValueError("retry axial samples differ from measured arrival")

    pending_path = Path(pending_state_path).expanduser().resolve()
    pending = json.loads(pending_path.read_text(encoding="utf-8"))
    if not isinstance(pending, dict):
        raise ValueError("retry pending state must be an attempt record")
    calibration_hash = _digest(calibration.record)
    events = pending.get("events")
    event = events[-1] if isinstance(events, list) and events else None
    if not isinstance(event, dict) or not isinstance(
            event.get("evidence_refs"), dict):
        raise ValueError("retry pending state has no grounded event")
    if (pending.get("schema") != "precision_insertion_attempt_v1" or
            pending.get("attempt_id") != expected.attempt_id or
            pending.get("candidate_id") != expected.candidate_id or
            pending.get("session_calibration_sha256") != calibration_hash or
            pending.get("mode") != mode.family or
            pending.get("gap_mm") != mode.gap_mm or
            pending.get("insertion_success") is not False or
            pending.get("grasp_success") is not True or
            pending.get("preinsert_reached") is not True or
            pending.get("failure_code") is not None or
            event.get("stage") != "xy_retry" or
            event.get("value") != "grounded_continuous_xy" or
            event.get("scope") !=
                "executed_observed_metric_shift_not_contact_authorization" or
            event.get("evidence_refs", {}).get(
                "arrival_axial_preflight") != str(source) or
            event.get("evidence_refs", {}).get(
                "postshift_arrival") != str(expected.arrival_report_path) or
            event.get("evidence_refs", {}).get(
                "grounded_xy") != str(shift_plan.diagnostic_report_path) or
            event.get("evidence_refs", {}).get(
                "axial_withdrawal") != str(shift_plan.withdrawal_evidence_path) or
            event.get("evidence_refs", {}).get(
                "lateral_execution") != str(checkpoint.lateral_execution_path) or
            event.get("evidence_refs", {}).get(
                "lateral_preflight") != str(
                    checkpoint.lateral_preflight_report_path) or
            event.get("increment_socket_xy_m") != list(
                shift_plan.lateral.increment_socket_xy_m) or
            event.get("offset_socket_m") != pending.get("xy_offset_socket_m") or
            event.get("supporting_cameras") != json.loads(
                shift_plan.diagnostic_report_path.read_text(
                    encoding="utf-8"))["alignment"]["inlier_cameras"]):
        raise ValueError("retry pending event differs from verified source chain")
    event_time = float(event["timestamp_s"])
    decision = float(decision_timestamp_s)
    sample = float(measured_start.sample_timestamp_s)
    if (not all(math.isfinite(value) for value in
                (event_time, decision, sample)) or
            not arrival.decision_timestamp_s < event_time < sample <= decision or
            decision - sample > limits["max_state_age_s"] or
            decision - arrival.decision_timestamp_s >
                limits["max_arrival_age_s"]):
        raise ValueError("retry handoff or post-transfer observation is stale")
    measured_start.validate(
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    if (np.max(np.abs(measured_start.full_q - axial[0])) >
            limits["max_start_joint_error_rad"] or
            np.max(np.abs(measured_start.hand_raw_measured -
                          arrival.joint_sample.hand_raw_measured)) >
            limits["max_hand_drift_raw"]):
        raise ValueError("retry handoff differs from measured arrival hold")
    return {
        "schema": _SCHEMA,
        "attempt_id": expected.attempt_id,
        "candidate_id": expected.candidate_id,
        "session_calibration_sha256": calibration_hash,
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object,
                 "target_depth_m": mode.target_depth_m},
        "replan_report_path": str(source),
        "replan_report_sha256": _sha(source),
        "arrival_report_path": str(expected.arrival_report_path),
        "arrival_report_sha256": expected.arrival_report_sha256,
        "pending_state_path": str(pending_path),
        "pending_state_sha256": _sha(pending_path),
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": _sha(archive),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "axial_sample_count": len(axial),
        "measured_start": measured_start.to_record(),
        "decision_timestamp_s": decision,
        "limits": limits,
        "scope": "read_only_grounded_retry_axial_packet_not_contact_permission",
        "robot_ready": False,
    }


def prepare_retry_axial_handoff(*, output_dir: Path, **kwargs) -> Path:
    """Write one exclusive source-bound packet; no robot calls."""
    record = _build_record(**kwargs)
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    path = target / "report.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path


def verify_retry_axial_handoff(
    report_path: Path, *, expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint, checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> dict:
    """Recompute packet and source checks using its saved measured state."""
    path = Path(report_path).expanduser().resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(saved, dict) or saved.get("schema") != _SCHEMA or
            saved.get("robot_ready") is not False or
            saved.get("scope") !=
                "read_only_grounded_retry_axial_packet_not_contact_permission"):
        raise ValueError("invalid retry axial handoff")
    limits = saved["limits"]
    rebuilt = _build_record(
        replan_report_path=Path(saved["replan_report_path"]),
        expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan,
        pending_state_path=Path(saved["pending_state_path"]),
        mode=mode, shared_root=shared_root, calibration=calibration,
        measured_start=_state_from_record(saved["measured_start"]),
        decision_timestamp_s=saved["decision_timestamp_s"],
        max_state_age_s=limits["max_state_age_s"],
        max_arrival_age_s=limits["max_arrival_age_s"],
        max_start_joint_error_rad=limits["max_start_joint_error_rad"],
        max_hand_drift_raw=limits["max_hand_drift_raw"],
        max_arm_hand_skew_s=limits["max_arm_hand_skew_s"],
        max_hand_command_error_raw=limits["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=limits["max_arm_velocity_rad_s"])
    if saved != rebuilt:
        raise ValueError("retry axial handoff differs from replay")
    return saved
