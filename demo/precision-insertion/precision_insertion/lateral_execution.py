"""Opt-in, source-bound execution boundary for one VLM-guided held XY shift.

The shift is at most 1 mm in socket XY and follows a saved, sampled-audited
Franka/Inspire path. A separately commissioned adapter owns all actual motion
and force/grip stops. This boundary logs its result for the *later* fresh
post-shift camera checkpoint; it never records an insertion retry or success.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .grounded_lateral import (
    GroundedLateralPreflight, verify_grounded_lateral_preflight,
)
from .lift_execution import (
    LiftExecutionLimits as LateralExecutionLimits,
    _verify_controller_commissioning, _write_new,
)
from .live_robot_state import LiveRobotState
from .pickup_execution import _reject_known_stock_follower
from .session_runner import SessionRunner
from .transfer_execution import _source_records


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def execute_bound_lateral_shift(
    *, runner: SessionRunner, plan: GroundedLateralPreflight,
    plan_report_path: Path, adapter, pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: LateralExecutionLimits, max_plan_age_s: float,
    commissioning_record_path: Path,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Execute only the verified held shift; require new images afterward.

    No adapter is included with this demo. Its robot-side dead-man/watchdog,
    contact abort and grip-loss hold must be independently commissioned.
    Returning from this method is not a post-shift alignment or insertion
    verdict; `assess_postshift_lateral_alignment` still requires new frames.
    """
    limits.validate()
    if (type(max_plan_age_s) not in (float, int) or
            not math.isfinite(max_plan_age_s) or max_plan_age_s <= 0 or
            not isinstance(runner, SessionRunner) or
            runner.current_decision().action !=
            "guarded_withdrawal_then_xy_assessment" or
            not isinstance(plan, GroundedLateralPreflight) or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or
            not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_lateral", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("lateral shift needs failed insertion and safe adapter")
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    attempt_dir = runner._attempt_dir
    if (attempt is None or attempt_dir is None or
            attempt.attempt_id != plan.attempt_id or
            attempt.candidate_id != plan.candidate_id or
            attempt.labels.get("grasp_success") is not True or
            attempt.labels.get("preinsert_reached") is not True or
            attempt.labels.get("insertion_success") is not False or
            plan.lateral.status != "sampled_lateral_hold_shift_pass" or
            plan.lateral.trajectory is None or
            plan.lateral.sampled_audit is None or
            plan.lateral.sampled_audit.get("sampled_clear") is not True):
        raise ValueError("lateral shift lacks a passing held-failure preflight")
    report_path = Path(plan_report_path).expanduser().resolve()
    expected_root = (attempt_dir / "lateral_hold_preflights").resolve()
    if (report_path.parent.parent != expected_root or
            report_path.name != "report.json" or
            not report_path.parent.name.isdigit()):
        raise ValueError("lateral plan is outside this attempt")
    verify_grounded_lateral_preflight(plan, report_path)
    trajectory_path = (report_path.parent / "lateral" /
                       "lateral_trajectory.npy").resolve()
    nested = json.loads((report_path.parent / "lateral" /
                         "report.json").read_text(encoding="utf-8"))
    if (nested.get("lateral_trajectory") != trajectory_path.name or
            nested.get("lateral_trajectory_sha256") != _sha(trajectory_path)):
        raise ValueError("lateral trajectory changed after source verification")
    trajectory = np.asarray(
        np.load(trajectory_path, allow_pickle=False), dtype=np.float64)
    if (trajectory.ndim != 2 or trajectory.shape[1] != 13 or
            len(trajectory) < 2 or
            not np.array_equal(trajectory, plan.lateral.trajectory) or
            not np.allclose(trajectory[:, 7:], trajectory[0, 7:],
                            atol=1e-8, rtol=0)):
        raise ValueError("lateral command differs from the audited held path")
    output_dir = (attempt_dir / "lateral_executions" /
                  report_path.parent.name)
    output = output_dir / "execution.json"
    started_path = output_dir / "started.json"
    failure_path = output_dir / "failure.json"
    if any(path.exists() for path in (output, started_path, failure_path)):
        raise FileExistsError("this lateral preflight already started execution")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = _wall_time()
    if (pre_state.sample_timestamp_s <= plan.decision_timestamp_s or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            now - plan.decision_timestamp_s > max_plan_age_s or
            float(np.max(np.abs(pre_state.full_q - trajectory[0]))) >
            limits.max_start_joint_error_rad or
            float(np.max(np.abs(pre_state.hand_raw_measured -
                                plan.joint_sample.hand_raw_measured))) >
            limits.max_hand_drift_raw):
        raise ValueError("fresh withdrawn hold differs from lateral start")
    commissioning = _verify_controller_commissioning(
        path=commissioning_record_path, adapter=adapter,
        limits=limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("lateral shift needs explicit live motion interlock")
    if _wall_time() - pre_state.sample_timestamp_s > limits.max_start_age_s:
        raise ValueError("lateral start aged while waiting for interlock")
    base = {
        "schema": "precision_insertion_lateral_hold_execution_v1",
        "source": "commissioned_lateral_controller",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "preflight_report_path": str(report_path),
        "preflight_report_sha256": _sha(report_path),
        "trajectory_sha256": _sha(trajectory_path),
        "commissioning": commissioning,
        "pre_state": pre_state.to_record(),
        "scope": "external_held_shift_not_postshift_alignment_or_insertion",
        "robot_ready": False,
    }
    _write_new(started_path, {**base, "status": "command_requested",
                              "requested_at_s": _wall_time()})
    base["started_marker_sha256"] = _sha(started_path)
    try:
        result = adapter.follow_lateral(
            trajectory.copy(), max_duration_s=limits.max_execution_duration_s,
            expected_hand_raw=pre_state.hand_raw_measured.copy(),
            trajectory_sha256=base["trajectory_sha256"])
        if (not isinstance(result, dict) or
                result.get("schema") !=
                "precision_insertion_external_lateral_result_v1" or
                result.get("trajectory_complete") is not True or
                result.get("safety_abort") is not False or
                result.get("grasp_held") is not True or
                result.get("terminal_hold_acknowledged") is not True or
                result.get("trajectory_sha256") != base["trajectory_sha256"]):
            raise RuntimeError("lateral controller did not complete safely")
        begun = float(result["started_at_s"])
        completed = float(result["completed_at_s"])
        requested = json.loads(started_path.read_text(
            encoding="utf-8"))["requested_at_s"]
        if (not all(math.isfinite(value) for value in (begun, completed)) or
                not requested <= begun < completed <= _wall_time() or
                completed - begun > limits.max_execution_duration_s):
            raise RuntimeError("lateral controller completion times are invalid")
        sources = _source_records(result)
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("lateral shift has no measured terminal hold")
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
                float(np.max(np.abs(post.full_q - trajectory[-1]))) >
                limits.max_end_joint_error_rad or
                float(np.max(np.abs(post.hand_raw_measured -
                                    pre_state.hand_raw_measured))) >
                limits.max_hand_drift_raw):
            raise RuntimeError("measured lateral hold differs from saved plan")
        record = {
            **base, "started_at_s": begun, "completed_at_s": completed,
            "measurement": {"trajectory_complete": True,
                            "safety_abort": False, "grasp_held": True},
            "source_records": sources,
            "controller_result": result,
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
