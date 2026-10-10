"""Opt-in non-contact transfer after a VLM-guided held XY correction.

This is the same externally commissioned `follow_transfer` controller
contract used for the first transfer, but a different source chain: the
post-shift checkpoint and replan. A completed trajectory is not a new
pre-insertion visual observation or permission to insert.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .grounded_lateral import GroundedLateralPreflight
from .lift_execution import (
    LiftExecutionLimits as PostShiftTransferLimits,
    _verify_controller_commissioning, _write_new,
)
from .live_robot_state import LiveRobotState
from .pickup_execution import _reject_known_stock_follower
from .postshift_checkpoint import PostShiftCheckpoint
from .postshift_insertion import PostShiftInsertionPreflight
from .postshift_path_handoff import verify_postshift_path_handoff
from .guarded_axial_handoff import _state_from_record
from .session_runner import SessionRunner
from .transfer_execution import _source_records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def execute_bound_postshift_transfer(
    *, runner: SessionRunner, expected: PostShiftInsertionPreflight,
    checkpoint: PostShiftCheckpoint, shift_plan: GroundedLateralPreflight,
    handoff_report_path: Path, adapter, pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: PostShiftTransferLimits, max_handoff_age_s: float,
    commissioning_record_path: Path, motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Follow only the saved transfer; require new observation before contact.

    The injected controller must independently enforce a robot-side dead-man,
    timeout, force/contact stop, grasp-loss stop and terminal hold. This Python
    boundary cannot protect motion if the Python process dies.
    """
    limits.validate()
    if (type(max_handoff_age_s) not in (int, float) or
            not math.isfinite(max_handoff_age_s) or max_handoff_age_s <= 0 or
            not isinstance(runner, SessionRunner) or
            runner.current_decision().action !=
                "guarded_withdrawal_then_xy_assessment" or
            not isinstance(expected, PostShiftInsertionPreflight) or
            not isinstance(checkpoint, PostShiftCheckpoint) or
            not isinstance(shift_plan, GroundedLateralPreflight) or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_transfer", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("post-shift transfer needs a held retry and safe adapter")
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    attempt_dir = runner._attempt_dir
    if (attempt is None or attempt_dir is None or
            attempt.attempt_id != expected.attempt_id or
            attempt.candidate_id != expected.candidate_id or
            attempt.labels.get("grasp_success") is not True or
            attempt.labels.get("preinsert_reached") is not True or
            attempt.labels.get("insertion_success") is not False or
            attempt.failure_code is not None):
        raise ValueError("post-shift transfer needs the same failed held attempt")
    report_path = Path(handoff_report_path).expanduser().resolve()
    root = (attempt_dir / "postshift_path_handoffs").resolve()
    if (report_path.parent.parent != root or report_path.name != "report.json"
            or not report_path.parent.name.isdigit()):
        raise ValueError("post-shift transfer handoff is outside this attempt")
    runner.verify_current_preflight_evidence()
    handoff = verify_postshift_path_handoff(
        report_path, expected=expected, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=runner.mode,
        shared_root=runner.shared_root, calibration=runner.calibration)
    preflight_path = Path(handoff["postshift_preflight_report_path"])
    if (handoff["attempt_id"] != attempt.attempt_id or
            handoff["candidate_id"] != attempt.candidate_id or
            handoff["session_calibration_sha256"] != runner.session_sha256 or
            handoff["mode"]["family"] != runner.mode.family or
            preflight_path not in runner._postshift_handoff_used_reports or
            handoff["transfer_required"] is not True):
        raise ValueError("post-shift handoff is not a needed transfer here")
    archive_path = Path(handoff["trajectory_archive_path"])
    if _sha(archive_path) != handoff["trajectory_archive_sha256"]:
        raise ValueError("post-shift planned path changed after handoff")
    with np.load(archive_path, allow_pickle=False) as data:
        transfer = np.asarray(data["transfer"], dtype=np.float64)
        axial = np.asarray(data["axial"], dtype=np.float64)
    if (transfer.shape != (handoff["transfer_sample_count"], 13) or
            axial.shape != (handoff["axial_sample_count"], 13) or
            not np.all(np.isfinite(transfer)) or
            not np.all(np.isfinite(axial)) or
            not np.array_equal(transfer[0], handoff["transfer_start_q"]) or
            not np.array_equal(transfer[-1], handoff["transfer_end_q"]) or
            not np.array_equal(axial[0], handoff["axial_start_q"]) or
            not np.array_equal(axial[-1], handoff["axial_end_q"]) or
            not np.allclose(transfer[-1], axial[0], atol=1e-4, rtol=0) or
            not np.allclose(transfer[:, 7:], transfer[:1, 7:],
                            atol=1e-8, rtol=0)):
        raise ValueError("post-shift transfer differs from handoff")
    output_dir = (attempt_dir / "postshift_transfer_executions" /
                  report_path.parent.name)
    output = output_dir / "execution.json"
    started_path = output_dir / "started.json"
    failure_path = output_dir / "failure.json"
    if any(path.exists() for path in (output, started_path, failure_path)):
        raise FileExistsError("this post-shift transfer already started")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = _wall_time()
    handoff_start = handoff["measured_start"]
    if (pre_state.sample_timestamp_s <= handoff["decision_timestamp_s"] or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            now - handoff["decision_timestamp_s"] > max_handoff_age_s or
            np.max(np.abs(pre_state.full_q - transfer[0])) >
                limits.max_start_joint_error_rad or
            np.max(np.abs(pre_state.hand_raw_measured -
                          handoff_start["hand_raw_measured"])) >
                limits.max_hand_drift_raw):
        raise ValueError("fresh measured hold differs from retry transfer start")
    commissioning = _verify_controller_commissioning(
        path=commissioning_record_path, adapter=adapter,
        limits=limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("post-shift transfer needs a live motion interlock")
    requested_at = _wall_time()
    if (requested_at - pre_state.sample_timestamp_s >
            limits.max_start_age_s or
            requested_at - handoff["decision_timestamp_s"] >
            max_handoff_age_s):
        raise ValueError("post-shift transfer start became stale at interlock")
    base = {
        "schema": "precision_insertion_postshift_transfer_execution_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "postshift_handoff": {"path": str(report_path),
                              "sha256": _sha(report_path)},
        "trajectory_archive_sha256": handoff["trajectory_archive_sha256"],
        "execution_limits": asdict(limits),
        "max_handoff_age_s": float(max_handoff_age_s),
        "commissioning": commissioning,
        "pre_state": pre_state.to_record(),
        "scope": "external_noncontact_transfer_not_arrival_or_insertion",
        "robot_ready": False,
    }
    _write_new(started_path, {**base, "status": "command_requested",
                              "requested_at_s": requested_at})
    base["started_marker_sha256"] = _sha(started_path)
    try:
        result = adapter.follow_transfer(
            transfer.copy(), max_duration_s=limits.max_execution_duration_s,
            expected_hand_raw=pre_state.hand_raw_measured.copy(),
            trajectory_sha256=handoff["trajectory_archive_sha256"])
        if (not isinstance(result, dict) or
                result.get("schema") !=
                    "precision_insertion_external_transfer_result_v1" or
                result.get("trajectory_complete") is not True or
                result.get("safety_abort") is not False or
                result.get("grasp_held") is not True or
                result.get("terminal_hold_acknowledged") is not True or
                result.get("trajectory_sha256") !=
                    handoff["trajectory_archive_sha256"]):
            raise RuntimeError("retry transfer controller did not complete safely")
        begun = float(result["started_at_s"])
        completed = float(result["completed_at_s"])
        requested = json.loads(started_path.read_text(
            encoding="utf-8"))["requested_at_s"]
        if (not all(math.isfinite(value) for value in (begun, completed)) or
                not requested <= begun < completed <= _wall_time() or
                completed - begun > limits.max_execution_duration_s):
            raise RuntimeError("retry transfer completion times are invalid")
        sources = _source_records(result)
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("retry transfer has no measured terminal state")
        post.validate(
            max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
            max_hand_command_error_raw=limits.max_hand_command_error_raw,
            max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
        now = _wall_time()
        if (post.sample_timestamp_s < completed or
                post.sample_timestamp_s - completed >
                    limits.max_execution_feedback_gap_s or
                post.sample_timestamp_s > now or
                now - post.sample_timestamp_s > limits.max_end_state_age_s or
                np.max(np.abs(post.full_q - transfer[-1])) >
                    limits.max_end_joint_error_rad or
                np.max(np.abs(post.hand_raw_measured -
                              pre_state.hand_raw_measured)) >
                    limits.max_hand_drift_raw):
            raise RuntimeError("retry transfer terminal hold differs from path")
        record = {
            **base, "started_at_s": begun, "completed_at_s": completed,
            "measurement": {"trajectory_complete": True,
                            "safety_abort": False, "grasp_held": True},
            "source_records": sources, "controller_result": result,
            "post_state": post.to_record(),
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
            "robot_state_after_failure": "unknown_requires_supervised_recovery",
        })
        raise
    return record


def verify_postshift_transfer_execution(
    execution_report_path: Path, *, handoff_report_path: Path,
    expected: PostShiftInsertionPreflight,
    checkpoint: PostShiftCheckpoint, shift_plan: GroundedLateralPreflight,
    mode, shared_root: Path, calibration,
) -> dict:
    """Replay file/feedback consistency for a later *fresh* camera checkpoint.

    Producer files and hashes are not authentication of physical behavior.
    The caller must still capture new images after this transfer.
    """
    path = Path(execution_report_path).expanduser().resolve()
    handoff_path = Path(handoff_report_path).expanduser().resolve()
    attempt_dir = handoff_path.parents[2]
    if (handoff_path.name != "report.json" or
            handoff_path.parent.parent.name != "postshift_path_handoffs" or
            path.name != "execution.json" or
            path.parent.name != handoff_path.parent.name or
            path.parent.parent != attempt_dir / "postshift_transfer_executions"):
        raise ValueError("retry transfer log is outside its handoff attempt")
    handoff = verify_postshift_path_handoff(
        handoff_path, expected=expected, checkpoint=checkpoint,
        shift_plan=shift_plan, mode=mode, shared_root=shared_root,
        calibration=calibration)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or
            record.get("schema") !=
                "precision_insertion_postshift_transfer_execution_v1" or
            record.get("attempt_id") != handoff["attempt_id"] or
            record.get("candidate_id") != handoff["candidate_id"] or
            record.get("session_calibration_sha256") !=
                handoff["session_calibration_sha256"] or
            record.get("postshift_handoff") != {
                "path": str(handoff_path), "sha256": _sha(handoff_path)} or
            record.get("trajectory_archive_sha256") !=
                handoff["trajectory_archive_sha256"] or
            record.get("measurement") != {
                "trajectory_complete": True, "safety_abort": False,
                "grasp_held": True} or
            record.get("scope") !=
                "external_noncontact_transfer_not_arrival_or_insertion" or
            record.get("robot_ready") is not False):
        raise ValueError("retry transfer log differs from the handoff")
    limits_raw = record.get("execution_limits")
    handoff_age = record.get("max_handoff_age_s")
    if (not isinstance(limits_raw, dict) or
            type(handoff_age) not in (int, float) or
            not math.isfinite(handoff_age) or handoff_age <= 0):
        raise ValueError("retry transfer lacks saved execution limits")
    try:
        limits = PostShiftTransferLimits(**limits_raw)
        limits.validate()
        pre = _state_from_record(record["pre_state"])
        post = _state_from_record(record["post_state"])
        begun = float(record["started_at_s"])
        completed = float(record["completed_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("retry transfer feedback is malformed") from exc
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
    request_time = started.get("requested_at_s")
    if (record.get("started_marker_sha256") != _sha(started_path) or
            started.get("status") != "command_requested" or
            any(started.get(key) != value for key, value in record.items()
                if key not in {
                    "started_marker_sha256", "started_at_s", "completed_at_s",
                    "measurement", "source_records", "controller_result",
                    "post_state"}) or
            type(request_time) not in (int, float) or
            not all(math.isfinite(value) for value in
                    (request_time, begun, completed)) or
            not handoff["decision_timestamp_s"] < pre.sample_timestamp_s <=
                request_time <= begun < completed <= post.sample_timestamp_s or
            request_time - handoff["decision_timestamp_s"] > handoff_age or
            request_time - pre.sample_timestamp_s > limits.max_start_age_s or
            completed - begun > limits.max_execution_duration_s or
            post.sample_timestamp_s - completed >
                limits.max_execution_feedback_gap_s or
            np.max(np.abs(pre.full_q - handoff["transfer_start_q"])) >
                limits.max_start_joint_error_rad or
            np.max(np.abs(post.full_q - handoff["transfer_end_q"])) >
                limits.max_end_joint_error_rad or
            np.max(np.abs(post.hand_raw_measured -
                          pre.hand_raw_measured)) > limits.max_hand_drift_raw):
        raise ValueError("retry transfer feedback differs from measured path")
    controller = record.get("controller_result")
    if (not isinstance(controller, dict) or
            controller.get("schema") !=
                "precision_insertion_external_transfer_result_v1" or
            controller.get("trajectory_sha256") !=
                handoff["trajectory_archive_sha256"] or
            controller.get("started_at_s") != begun or
            controller.get("completed_at_s") != completed or
            any(controller.get(name) is not True for name in
                ("trajectory_complete", "grasp_held",
                 "terminal_hold_acknowledged")) or
            controller.get("safety_abort") is not False or
            _source_records(controller) != record.get("source_records")):
        raise ValueError("retry transfer producer records differ")
    commission = record.get("commissioning")
    if (not isinstance(commission, dict) or
            not isinstance(commission.get("path"), str) or
            _sha(Path(commission["path"])) != commission.get("sha256")):
        raise ValueError("retry transfer watchdog review changed")
    return record
