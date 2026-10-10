"""Opt-in boundary to a separately commissioned 20 mm contact controller.

This module binds one observed pre-insertion hold to its saved axial path,
checks explicit live interlocks, then accepts independently recorded force,
depth, alignment and grasp evidence. It cannot implement the robot-side
dead-man, watchdog, force servo or physical key-depth measurement. No task
success label is set here, even when the commanded stroke completes.
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

from .guarded_axial_handoff import verify_guarded_axial_handoff
from .guarded_contact import GuardedContactLimits
from .insertion_checkpoint import _metric_record
from .lift_execution import (
    LiftExecutionLimits as GuardedInsertionExecutionLimits,
    _verify_controller_commissioning, _write_new,
)
from .live_robot_state import LiveRobotState
from .pickup_execution import _reject_known_stock_follower
from .session_runner import SessionRunner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def _verify_contact_commissioning(
    *, path: Path, adapter, limits: GuardedInsertionExecutionLimits,
    contact_limits: GuardedContactLimits, now_s: float,
) -> dict:
    """Require a reviewed contact/force test of the exact controller binary."""
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    binary = Path(adapter.daemon_binary_path).expanduser().resolve()
    tests = (
        "force_limit_test_passed", "contact_abort_test_passed",
        "sample_gap_abort_test_passed", "terminal_hold_test_passed",
    )
    if (not isinstance(record, dict) or
            record.get("schema") !=
            "precision_insertion_contact_controller_commissioning_v1" or
            not isinstance(record.get("reviewed_by"), str) or
            not record["reviewed_by"].strip() or
            any(record.get(name) is not True for name in tests) or
            not binary.is_file() or
            record.get("daemon_binary_path") != str(binary) or
            record.get("daemon_binary_sha256") != _sha(binary) or
            record.get("contact_limits") != asdict(contact_limits)):
        raise ValueError("guarded controller has no matching contact review")
    tested = record.get("tested_at_s")
    if (type(tested) not in (float, int) or not math.isfinite(tested) or
            tested > now_s or
            now_s - tested > limits.max_commissioning_record_age_s):
        raise ValueError("guarded contact review is stale")
    return {"path": str(source), "sha256": _sha(source),
            "daemon_binary_sha256": _sha(binary)}


def execute_bound_guarded_insertion(
    *, runner: SessionRunner, adapter, handoff_report_path: Path,
    pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: GuardedInsertionExecutionLimits,
    contact_limits: GuardedContactLimits,
    max_handoff_age_s: float,
    watchdog_commissioning_path: Path,
    contact_commissioning_path: Path,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Run one *bound* external contact stroke; never infer key insertion.

    The external adapter must provide `follow_guarded_insertion` with its own
    robot-side watchdog, force/contact abort, grip-loss stop and terminal
    hold. This Python boundary is not a substitute for any of those. An
    exception after command onset leaves an exclusive failure marker and
    demands supervised recovery.
    """
    limits.validate()
    if (type(max_handoff_age_s) not in (int, float) or
            not math.isfinite(max_handoff_age_s) or
            max_handoff_age_s <= 0 or
            not isinstance(runner, SessionRunner) or
            runner.current_decision().action !=
            "await_guarded_insertion_and_observation" or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or
            not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_guarded_insertion", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("guarded insertion needs an observed hold and safe adapter")
    contact_limits.validate(family=runner.mode.family)
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    attempt_dir = runner._attempt_dir
    if (attempt is None or attempt_dir is None or
            attempt.labels.get("grasp_success") is not True or
            attempt.labels.get("preinsert_reached") is not True or
            attempt.labels.get("insertion_success") is not None or
            attempt.failure_code is not None):
        raise ValueError("guarded insertion needs a successful observed arrival")
    report_path = Path(handoff_report_path).expanduser().resolve()
    handoff_root = (attempt_dir / "guarded_axial_handoffs").resolve()
    if (not report_path.is_relative_to(handoff_root) or
            report_path.name != "report.json"):
        raise ValueError("guarded handoff is outside this attempt")
    handoff = verify_guarded_axial_handoff(report_path)
    if (handoff["attempt_id"] != attempt.attempt_id or
            handoff["candidate_id"] != attempt.candidate_id or
            handoff["session_calibration_sha256"] != runner.session_sha256 or
            handoff["postlift_report_path"] !=
            str(runner._postlift_report_path) or
            handoff["preinsert_report_path"] !=
            str(runner._preinsert_report_path) or
            handoff["mode"]["family"] != runner.mode.family):
        raise ValueError("guarded path is not bound to this observed attempt")
    archive_path = Path(handoff["trajectory_archive_path"])
    if _sha(archive_path) != handoff["trajectory_archive_sha256"]:
        raise ValueError("guarded axial archive changed after handoff verification")
    with np.load(archive_path, allow_pickle=False) as archive:
        axial = np.asarray(archive["axial"], dtype=np.float64)
    if (axial.shape != (handoff["axial_sample_count"], 13) or
            not np.all(np.isfinite(axial)) or
            not np.array_equal(axial[0], handoff["axial_start_q"]) or
            not np.array_equal(axial[-1], handoff["axial_end_q"])):
        raise ValueError("guarded axial archive differs from handoff")
    output = attempt_dir / "guarded_insertion_execution.json"
    started_path = attempt_dir / "guarded_insertion_started.json"
    failure_path = attempt_dir / "guarded_insertion_failure.json"
    if any(path.exists() for path in (output, started_path, failure_path)):
        raise FileExistsError("this attempt already started guarded insertion")
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
            float(np.max(np.abs(pre_state.full_q - axial[0]))) >
            limits.max_start_joint_error_rad or
            float(np.max(np.abs(pre_state.hand_raw_measured -
                                handoff_state["hand_raw_measured"]))) >
            limits.max_hand_drift_raw):
        raise ValueError("fresh held state differs from the axial handoff")
    watchdog = _verify_controller_commissioning(
        path=watchdog_commissioning_path, adapter=adapter,
        limits=limits, now_s=now)
    contact = _verify_contact_commissioning(
        path=contact_commissioning_path, adapter=adapter,
        limits=limits, contact_limits=contact_limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("contact stroke needs an explicit live interlock")
    if _wall_time() - pre_state.sample_timestamp_s > limits.max_start_age_s:
        raise ValueError("held start aged while waiting for contact interlock")
    base = {
        "schema": "precision_insertion_guarded_motion_boundary_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "axial_handoff": {"path": str(report_path), "sha256": _sha(report_path)},
        "trajectory_archive_sha256": handoff["trajectory_archive_sha256"],
        "watchdog_commissioning": watchdog,
        "contact_commissioning": contact,
        "pre_state": pre_state.to_record(),
        "scope": "external_guarded_contact_not_physical_key_depth_or_task_success",
        "robot_ready": False,
    }
    _write_new(started_path, {**base, "status": "command_requested",
                              "requested_at_s": _wall_time()})
    base["started_marker_sha256"] = _sha(started_path)
    try:
        result = adapter.follow_guarded_insertion(
            axial.copy(), handoff_sha256=base["axial_handoff"]["sha256"],
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
                base["axial_handoff"]["sha256"] or
                result.get("trajectory_archive_sha256") !=
                handoff["trajectory_archive_sha256"]):
            raise RuntimeError("guarded controller lacks a bound terminal hold")
        metric_raw = result.get("metric_record_path")
        if not isinstance(metric_raw, str):
            raise RuntimeError("guarded controller lacks a metric record")
        metric_path = Path(metric_raw).expanduser().resolve()
        if (not Path(metric_raw).is_absolute() or
                not metric_path.is_file() or
                result.get("metric_record_sha256") != _sha(metric_path)):
            raise RuntimeError("guarded controller metric record changed")
        metric, begun, completed = _metric_record(
            metric_path, attempt_id=attempt.attempt_id,
            candidate_id=attempt.candidate_id,
            session_calibration_sha256=runner.session_sha256)
        requested = json.loads(started_path.read_text(
            encoding="utf-8"))["requested_at_s"]
        if (metric["schema"] != "precision_insertion_guarded_execution_v2" or
                metric["axial_handoff"] != base["axial_handoff"] or
                result.get("started_at_s") != begun or
                result.get("completed_at_s") != completed or
                result["safety_abort"] is not
                metric["measurement"]["safety_abort"] or
                not requested <= begun < completed <= _wall_time() or
                completed - begun > limits.max_execution_duration_s):
            raise RuntimeError("guarded result differs from replayed contact record")
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("guarded stroke has no measured terminal state")
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
                float(np.max(np.abs(post.hand_raw_measured -
                                    pre_state.hand_raw_measured))) >
                limits.max_hand_drift_raw or
                (not result["safety_abort"] and
                 float(np.max(np.abs(post.full_q - axial[-1]))) >
                 limits.max_end_joint_error_rad)):
            raise RuntimeError("measured terminal hold differs from contact result")
        record = {
            **base, "started_at_s": begun, "completed_at_s": completed,
            "safety_abort": result["safety_abort"],
            "metric_record": {"path": str(metric_path),
                              "sha256": _sha(metric_path)},
            "post_state": post.to_record(),
            "controller_result": result,
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
