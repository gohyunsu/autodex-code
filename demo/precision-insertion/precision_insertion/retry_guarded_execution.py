"""Opt-in guarded contact boundary for one *observed* continuous XY retry.

The same external controller contract as the first insertion is reused, but
the source chain is deliberately different. This boundary never sets task
success, authenticates a physical key-depth sensor or replaces the robot-side
watchdog, force servo, stop acknowledgement and human motion interlock.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable
from types import SimpleNamespace

import numpy as np

from .config import TaskMode
from .grounded_lateral import GroundedLateralPreflight
from .guarded_axial_handoff import _state_from_record
from .guarded_contact import GuardedContactLimits
from .guarded_insertion_execution import _verify_contact_commissioning
from .lift_execution import (
    LiftExecutionLimits as RetryGuardedExecutionLimits,
    _verify_controller_commissioning, _write_new,
)
from .live_robot_state import LiveRobotState
from .pickup_execution import _reject_known_stock_follower
from .postshift_arrival_checkpoint import PostShiftArrivalCheckpoint
from .postshift_arrival_replan import PostShiftArrivalReplan
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import PostShiftInsertionPreflight
from .retry_axial_handoff import verify_retry_axial_handoff
from .retry_guarded_metric import verify_retry_guarded_metric
from .session_runner import SessionRunner


_SCHEMA = "precision_insertion_retry_guarded_motion_boundary_v1"
_SCOPE = "external_retry_contact_not_physical_key_depth_or_task_success"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def execute_bound_retry_guarded_insertion(
    *, runner: SessionRunner, expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint,
    checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    handoff_report_path: Path, adapter, pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: RetryGuardedExecutionLimits,
    contact_limits: GuardedContactLimits,
    max_handoff_age_s: float,
    watchdog_commissioning_path: Path,
    contact_commissioning_path: Path,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Request one bound 20 mm stroke through an external safe controller.

    The Python process cannot protect motion if it dies. The adapter must
    enforce an independent robot-side watchdog, force/contact abort,
    grip-loss stop and terminal hold before being commissioned.
    """
    limits.validate()
    if (type(max_handoff_age_s) not in (int, float) or
            not math.isfinite(max_handoff_age_s) or
            max_handoff_age_s <= 0 or
            not isinstance(runner, SessionRunner) or
            runner.current_decision().action !=
                "await_retry_execution_and_observation" or
            not isinstance(expected, PostShiftArrivalReplan) or
            not isinstance(previous, PostShiftInsertionPreflight) or
            not isinstance(arrival, PostShiftArrivalCheckpoint) or
            not isinstance(checkpoint, PostShiftCheckpoint) or
            not isinstance(shift_plan, GroundedLateralPreflight) or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or
            not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_guarded_insertion", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("retry contact needs a pending XY event and safe adapter")
    contact_limits.validate(family=runner.mode.family)
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    attempt_dir = runner._attempt_dir
    if (attempt is None or attempt_dir is None or
            attempt.attempt_id != expected.attempt_id or
            attempt.candidate_id != expected.candidate_id or
            attempt.labels.get("grasp_success") is not True or
            attempt.labels.get("preinsert_reached") is not True or
            attempt.labels.get("insertion_success") is not False or
            attempt.failure_code is not None or
            not attempt.events or
            attempt.events[-1].get("stage") != "xy_retry" or
            attempt.events[-1].get("value") != "grounded_continuous_xy"):
        raise ValueError("retry contact needs the same failed held attempt")
    report_path = Path(handoff_report_path).expanduser().resolve()
    root = (attempt_dir / "retry_guarded_axial_handoffs").resolve()
    if (report_path.name != "report.json" or
            report_path.parent.parent != root or
            not report_path.parent.name.isdigit()):
        raise ValueError("retry contact handoff is outside this attempt")
    runner.verify_current_preflight_evidence()
    handoff = verify_retry_axial_handoff(
        report_path, expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=runner.mode,
        shared_root=runner.shared_root, calibration=runner.calibration)
    current_state = (attempt_dir /
                     f"state_{runner._attempt_index:03d}.json").resolve()
    source = Path(handoff["replan_report_path"])
    if (source not in runner._retry_axial_handoff_used_reports or
            handoff["pending_state_path"] != str(current_state) or
            handoff["pending_state_sha256"] != _sha(current_state) or
            handoff["attempt_id"] != attempt.attempt_id or
            handoff["candidate_id"] != attempt.candidate_id or
            handoff["session_calibration_sha256"] != runner.session_sha256 or
            handoff["mode"] != {
                "family": runner.mode.family, "gap_mm": runner.mode.gap_mm,
                "key_object": runner.mode.key_object,
                "socket_object": runner.mode.socket_object,
                "target_depth_m": runner.mode.target_depth_m,
            }):
        raise ValueError("retry contact handoff is not the current XY event")
    archive = Path(handoff["trajectory_archive_path"])
    if _sha(archive) != handoff["trajectory_archive_sha256"]:
        raise ValueError("retry axial archive changed after handoff")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"axial"}:
            raise ValueError("retry contact archive contains non-axial motion")
        axial = np.asarray(data["axial"], dtype=np.float64)
    if (axial.shape != (handoff["axial_sample_count"], 13) or
            not np.all(np.isfinite(axial)) or
            not np.array_equal(axial[0], handoff["axial_start_q"]) or
            not np.array_equal(axial[-1], handoff["axial_end_q"])):
        raise ValueError("retry contact path differs from handoff")
    output_dir = (attempt_dir / "retry_guarded_executions" /
                  report_path.parent.name)
    output = output_dir / "execution.json"
    started_path = output_dir / "started.json"
    failure_path = output_dir / "failure.json"
    if any(path.exists() for path in (output, started_path, failure_path)):
        raise FileExistsError("this retry contact stroke already started")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = _wall_time()
    handoff_state = handoff["measured_start"]
    if (pre_state.sample_timestamp_s <= handoff["decision_timestamp_s"] or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            now - handoff["decision_timestamp_s"] > max_handoff_age_s or
            np.max(np.abs(pre_state.full_q - axial[0])) >
                limits.max_start_joint_error_rad or
            np.max(np.abs(pre_state.hand_raw_measured -
                          handoff_state["hand_raw_measured"])) >
                limits.max_hand_drift_raw):
        raise ValueError("fresh held state differs from retry axial start")
    watchdog = _verify_controller_commissioning(
        path=watchdog_commissioning_path, adapter=adapter,
        limits=limits, now_s=now)
    contact = _verify_contact_commissioning(
        path=contact_commissioning_path, adapter=adapter,
        limits=limits, contact_limits=contact_limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("retry contact needs an explicit live interlock")
    requested = _wall_time()
    if (requested - pre_state.sample_timestamp_s > limits.max_start_age_s or
            requested - handoff["decision_timestamp_s"] > max_handoff_age_s):
        raise ValueError("retry contact start aged while waiting for interlock")
    base = {
        "schema": _SCHEMA, "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "retry_axial_handoff": {"path": str(report_path),
                                "sha256": _sha(report_path)},
        "trajectory_archive_sha256": handoff["trajectory_archive_sha256"],
        "execution_limits": asdict(limits),
        "max_handoff_age_s": float(max_handoff_age_s),
        "watchdog_commissioning": watchdog,
        "contact_commissioning": contact,
        "pre_state": pre_state.to_record(),
        "scope": _SCOPE, "robot_ready": False,
    }
    _write_new(started_path, {**base, "status": "command_requested",
                              "requested_at_s": requested})
    base["started_marker_sha256"] = _sha(started_path)
    try:
        result = adapter.follow_guarded_insertion(
            axial.copy(),
            handoff_sha256=base["retry_axial_handoff"]["sha256"],
            trajectory_archive_sha256=handoff["trajectory_archive_sha256"],
            contact_limits=asdict(contact_limits),
            expected_hand_raw=pre_state.hand_raw_measured.copy(),
            max_duration_s=limits.max_execution_duration_s)
        if (not isinstance(result, dict) or
                result.get("schema") !=
                    "precision_insertion_external_guarded_result_v1" or
                result.get("terminal_hold_acknowledged") is not True or
                type(result.get("safety_abort")) is not bool or
                result.get("handoff_sha256") !=
                    base["retry_axial_handoff"]["sha256"] or
                result.get("trajectory_archive_sha256") !=
                    handoff["trajectory_archive_sha256"]):
            raise RuntimeError("retry controller lacks a bound terminal hold")
        metric_raw = result.get("metric_record_path")
        if not isinstance(metric_raw, str):
            raise RuntimeError("retry controller lacks a metric record")
        metric_path = Path(metric_raw).expanduser().resolve()
        if (not Path(metric_raw).is_absolute() or
                not metric_path.is_file() or
                result.get("metric_record_sha256") != _sha(metric_path)):
            raise RuntimeError("retry controller metric record changed")
        metric, begun, completed = verify_retry_guarded_metric(
            metric_path, handoff_report_path=report_path,
            expected=expected, previous=previous, arrival=arrival,
            checkpoint=checkpoint, shift_plan=shift_plan,
            mode=runner.mode, shared_root=runner.shared_root,
            calibration=runner.calibration)
        if (metric["contact_limits"] != asdict(contact_limits) or
                metric["retry_axial_handoff"] !=
                    base["retry_axial_handoff"] or
                result.get("started_at_s") != begun or
                result.get("completed_at_s") != completed or
                result["safety_abort"] is not
                    metric["measurement"]["safety_abort"] or
                not requested <= begun < completed <= _wall_time() or
                completed - begun > limits.max_execution_duration_s):
            raise RuntimeError("retry result differs from replayed contact trace")
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("retry contact has no measured terminal state")
        post.validate(
            max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
            max_hand_command_error_raw=limits.max_hand_command_error_raw,
            max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
        now = _wall_time()
        if (post.sample_timestamp_s < completed or
                post.sample_timestamp_s - completed >
                    limits.max_execution_feedback_gap_s or
                post.sample_timestamp_s > now or
                now - post.sample_timestamp_s >
                    limits.max_end_state_age_s or
                np.max(np.abs(post.hand_raw_measured -
                              pre_state.hand_raw_measured)) >
                    limits.max_hand_drift_raw or
                (not result["safety_abort"] and
                 np.max(np.abs(post.full_q - axial[-1])) >
                    limits.max_end_joint_error_rad)):
            raise RuntimeError("retry terminal hold differs from contact result")
        record = {
            **base, "started_at_s": begun, "completed_at_s": completed,
            "safety_abort": result["safety_abort"],
            "metric_record": {"path": str(metric_path),
                              "sha256": _sha(metric_path)},
            "post_state": post.to_record(), "controller_result": result,
        }
        _write_new(output, record)
    except Exception as exc:
        try:
            stop_ack = adapter.stop_and_acknowledge()
        except Exception as stop_exc:
            stop_ack = {"acknowledged": False, "error": str(stop_exc)}
        _write_new(failure_path, {
            **base, "status": "execution_or_feedback_failed",
            "error_type": type(exc).__name__, "error": str(exc),
            "stop_acknowledgement": stop_ack,
            "robot_state_after_failure":
                "unknown_requires_supervised_recovery",
        })
        raise
    return record


def verify_retry_guarded_execution(
    log_path: Path, *, handoff_report_path: Path,
    expected: PostShiftArrivalReplan,
    previous: PostShiftInsertionPreflight,
    arrival: PostShiftArrivalCheckpoint,
    checkpoint: PostShiftCheckpoint,
    shift_plan: GroundedLateralPreflight,
    mode: TaskMode, shared_root: Path, calibration,
) -> dict:
    """Replay a saved controller call and sources without commanding motion."""
    path = Path(log_path).expanduser().resolve()
    handoff_path = Path(handoff_report_path).expanduser().resolve()
    attempt_dir = handoff_path.parents[2]
    if (handoff_path.name != "report.json" or
            handoff_path.parent.parent !=
                attempt_dir / "retry_guarded_axial_handoffs" or
            not handoff_path.parent.name.isdigit() or
            path != (attempt_dir / "retry_guarded_executions" /
                     handoff_path.parent.name / "execution.json")):
        raise ValueError("retry contact log is outside its attempt")
    handoff = verify_retry_axial_handoff(
        handoff_path, expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=mode,
        shared_root=shared_root, calibration=calibration)
    record = json.loads(path.read_text(encoding="utf-8"))
    reference = {"path": str(handoff_path), "sha256": _sha(handoff_path)}
    if (not isinstance(record, dict) or record.get("schema") != _SCHEMA or
            record.get("attempt_id") != handoff["attempt_id"] or
            record.get("candidate_id") != handoff["candidate_id"] or
            record.get("session_calibration_sha256") !=
                handoff["session_calibration_sha256"] or
            record.get("retry_axial_handoff") != reference or
            record.get("trajectory_archive_sha256") !=
                handoff["trajectory_archive_sha256"] or
            record.get("scope") != _SCOPE or
            record.get("robot_ready") is not False):
        raise ValueError("retry contact log differs from current handoff")
    try:
        limits = RetryGuardedExecutionLimits(**record["execution_limits"])
        limits.validate()
        max_age = float(record["max_handoff_age_s"])
        pre = _state_from_record(record["pre_state"])
        post = _state_from_record(record["post_state"])
        begun = float(record["started_at_s"])
        completed = float(record["completed_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("retry contact log has malformed feedback") from exc
    if not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("retry contact log has invalid handoff age")
    pre.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    post.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    started_path = path.parent / "started.json"
    started = json.loads(started_path.read_text(encoding="utf-8"))
    requested = started.get("requested_at_s")
    if (record.get("started_marker_sha256") != _sha(started_path) or
            started.get("status") != "command_requested" or
            any(started.get(key) != value for key, value in record.items()
                if key not in {
                    "started_marker_sha256", "started_at_s",
                    "completed_at_s", "safety_abort", "metric_record",
                    "post_state", "controller_result"}) or
            type(requested) not in (int, float) or
            not all(math.isfinite(value) for value in
                    (requested, begun, completed)) or
            not handoff["decision_timestamp_s"] <
                pre.sample_timestamp_s <= requested <= begun < completed <=
                post.sample_timestamp_s or
            requested - handoff["decision_timestamp_s"] > max_age or
            requested - pre.sample_timestamp_s > limits.max_start_age_s or
            completed - begun > limits.max_execution_duration_s or
            post.sample_timestamp_s - completed >
                limits.max_execution_feedback_gap_s):
        raise ValueError("retry contact timeline differs from started marker")
    archive = Path(handoff["trajectory_archive_path"])
    if _sha(archive) != handoff["trajectory_archive_sha256"]:
        raise ValueError("retry contact axial archive changed")
    with np.load(archive, allow_pickle=False) as data:
        if set(data.files) != {"axial"}:
            raise ValueError("retry contact archive contains non-axial motion")
        axial = np.asarray(data["axial"], dtype=np.float64)
    if (axial.shape != (handoff["axial_sample_count"], 13) or
            not np.array_equal(axial[0], handoff["axial_start_q"]) or
            not np.array_equal(axial[-1], handoff["axial_end_q"]) or
            np.max(np.abs(pre.full_q - axial[0])) >
                limits.max_start_joint_error_rad or
            np.max(np.abs(pre.hand_raw_measured -
                          handoff["measured_start"]["hand_raw_measured"])) >
                limits.max_hand_drift_raw or
            np.max(np.abs(post.hand_raw_measured -
                          pre.hand_raw_measured)) > limits.max_hand_drift_raw or
            (record.get("safety_abort") is False and
             np.max(np.abs(post.full_q - axial[-1])) >
                limits.max_end_joint_error_rad)):
        raise ValueError("retry contact feedback differs from axial path")
    metric_reference = record.get("metric_record")
    if (not isinstance(metric_reference, dict) or
            set(metric_reference) != {"path", "sha256"} or
            not isinstance(metric_reference["path"], str) or
            not Path(metric_reference["path"]).is_absolute() or
            _sha(Path(metric_reference["path"])) !=
                metric_reference["sha256"]):
        raise ValueError("retry contact metric source changed")
    metric, metric_begun, metric_completed = verify_retry_guarded_metric(
        Path(metric_reference["path"]), handoff_report_path=handoff_path,
        expected=expected, previous=previous, arrival=arrival,
        checkpoint=checkpoint, shift_plan=shift_plan, mode=mode,
        shared_root=shared_root, calibration=calibration)
    if (metric_begun != begun or metric_completed != completed or
            metric["measurement"]["safety_abort"] is not
                record.get("safety_abort")):
        raise ValueError("retry contact metric differs from execution log")
    controller = record.get("controller_result")
    if (not isinstance(controller, dict) or
            controller.get("schema") !=
                "precision_insertion_external_guarded_result_v1" or
            controller.get("handoff_sha256") != reference["sha256"] or
            controller.get("trajectory_archive_sha256") !=
                handoff["trajectory_archive_sha256"] or
            controller.get("started_at_s") != begun or
            controller.get("completed_at_s") != completed or
            controller.get("terminal_hold_acknowledged") is not True or
            controller.get("safety_abort") is not record["safety_abort"] or
            controller.get("metric_record_path") !=
                metric_reference["path"] or
            controller.get("metric_record_sha256") !=
                metric_reference["sha256"]):
        raise ValueError("retry controller result differs from saved evidence")
    adapter_record = {}
    for name in ("watchdog_commissioning", "contact_commissioning"):
        ref = record.get(name)
        if (not isinstance(ref, dict) or
                not isinstance(ref.get("path"), str) or
                not Path(ref["path"]).is_absolute() or
                _sha(Path(ref["path"])) != ref.get("sha256")):
            raise ValueError(f"retry {name} review changed")
        review = json.loads(Path(ref["path"]).read_text(encoding="utf-8"))
        binary = Path(review["daemon_binary_path"])
        if (not binary.is_absolute() or
                _sha(binary) != ref.get("daemon_binary_sha256")):
            raise ValueError(f"retry {name} controller binary changed")
        adapter_record[name] = (ref, SimpleNamespace(
            daemon_binary_path=str(binary)))
    verified_watchdog = _verify_controller_commissioning(
        path=Path(record["watchdog_commissioning"]["path"]),
        adapter=adapter_record["watchdog_commissioning"][1],
        limits=limits, now_s=requested)
    contact_limits = GuardedContactLimits(**metric["contact_limits"])
    contact_limits.validate(family=mode.family)
    verified_contact = _verify_contact_commissioning(
        path=Path(record["contact_commissioning"]["path"]),
        adapter=adapter_record["contact_commissioning"][1],
        limits=limits, contact_limits=contact_limits, now_s=requested)
    if (verified_watchdog != record["watchdog_commissioning"] or
            verified_contact != record["contact_commissioning"]):
        raise ValueError("retry contact review differs from replay")
    return record
