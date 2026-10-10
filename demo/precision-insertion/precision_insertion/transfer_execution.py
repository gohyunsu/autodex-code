"""Opt-in held transfer to the pre-insertion hold; never axial contact.

Only a separately safeguarded controller may follow this plan. This Python
boundary binds its returned records to the measured post-lift replan; it
cannot implement a robot-side watchdog or verify physical contact itself.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .assets import AssetPaths
from .bounded_postlift import (
    BoundedPostLiftPreflight, verify_bounded_postlift_preflight,
)
from .lift_execution import (
    LiftExecutionLimits as TransferExecutionLimits,
    _verify_controller_commissioning, _write_new,
)
from .live_robot_state import LiveRobotState
from .pickup_execution import _reject_known_stock_follower
from .postlift_preflight import PostLiftPreflight
from .session_runner import SessionRunner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def _source_records(result: dict) -> dict:
    """Retain exact independent controller files, not one invented summary."""
    sources = result.get("source_records")
    if not isinstance(sources, dict) or set(sources) != {
            "trajectory_feedback", "safety", "grasp_state"}:
        raise RuntimeError("transfer controller lacks three producer records")
    checked = {}
    resolved_paths = set()
    for name, source in sources.items():
        if (not isinstance(source, dict) or set(source) != {"path", "sha256"}
                or not isinstance(source["path"], str)):
            raise RuntimeError(f"invalid transfer producer reference: {name}")
        raw_path = Path(source["path"]).expanduser()
        path = raw_path.resolve()
        if (not raw_path.is_absolute() or not path.is_file() or
                path in resolved_paths or source["sha256"] != _sha(path)):
            raise RuntimeError(f"transfer producer record changed: {name}")
        resolved_paths.add(path)
        checked[name] = {"path": str(path), "sha256": source["sha256"]}
    return checked


def execute_bound_held_transfer(
    *, runner: SessionRunner, adapter, pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: TransferExecutionLimits, commissioning_record_path: Path,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Follow one saved non-contact transfer from a fresh measured held state.

    ``adapter.follow_transfer`` must have independently commissioned
    robot-side timeout, force and grip-loss stops. A successful return only
    allows the *later* same-trial camera/robot pre-insertion checkpoint; it
    does not set the ``preinsert_reached`` task label.
    """
    limits.validate()
    if (not isinstance(runner, SessionRunner) or
            runner.current_decision().action !=
            "transfer_execution_gate_required" or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or
            not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_transfer", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("transfer needs a passing held replan and guarded adapter")
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    plan = runner._postlift_preflight
    report_path = runner._postlift_report_path
    attempt_dir = runner._attempt_dir
    if (attempt is None or
            not isinstance(plan, (PostLiftPreflight,
                                  BoundedPostLiftPreflight)) or
            plan.status != "sampled_postlift_preflight_pass" or
            plan.planning is None or not plan.planning.sampled_planning_pass or
            plan.planning.transfer_trajectory is None or
            attempt.labels["grasp_success"] is not True or
            attempt.labels["preinsert_reached"] is not None or
            attempt.failure_code is not None or
            plan.attempt_id != attempt.attempt_id or
            "/".join(plan.candidate_key) != attempt.candidate_id or
            plan.session_calibration_sha256 != runner.session_sha256 or
            plan.catalog_sha256 != runner.catalog_sha256 or
            report_path is None or attempt_dir is None or
            runner._postlift_report_sha256 != _sha(report_path)):
        raise ValueError("transfer is not bound to this passing post-lift plan")
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    if any(saved.get(key) != value for key, value in plan.to_record().items()):
        raise ValueError("saved post-lift plan differs from selected replan")
    if isinstance(plan, BoundedPostLiftPreflight):
        candidate_dir = (AssetPaths(runner.shared_root, runner.mode).candidate_dir /
                         Path(*plan.candidate_key))
        verify_bounded_postlift_preflight(
            report_path, mode=runner.mode, shared_root=runner.shared_root,
            candidate_dir=candidate_dir)
    source = saved.get("planned_trajectories")
    if source != "planned_trajectories.npz":
        raise ValueError("saved post-lift transfer trajectory is absent")
    trajectory_file = (report_path.parent / source).resolve()
    if (not trajectory_file.is_relative_to(report_path.parent.resolve()) or
            saved.get("planned_trajectories_sha256") != _sha(trajectory_file)):
        raise ValueError("saved post-lift trajectory bytes changed")
    with np.load(trajectory_file, allow_pickle=False) as archive:
        transfer = np.asarray(archive["transfer"], dtype=np.float64)
    if (transfer.ndim != 2 or transfer.shape[1] != 13 or len(transfer) < 2 or
            not np.all(np.isfinite(transfer)) or
            not np.array_equal(transfer, plan.planning.transfer_trajectory) or
            not np.allclose(transfer[:, 7:], plan.planning.held_hand_q,
                            atol=1e-8, rtol=0) or
            not np.allclose(transfer[:, 7:], transfer[0, 7:],
                            atol=1e-8, rtol=0)):
        raise ValueError("saved transfer does not hold the planned Inspire grasp")
    output = attempt_dir / "transfer_execution.json"
    started_path = attempt_dir / "transfer_started.json"
    failure_path = attempt_dir / "transfer_failure.json"
    if any(path.exists() for path in (output, started_path, failure_path)):
        raise FileExistsError("this attempt already started a held transfer")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = _wall_time()
    if (pre_state.sample_timestamp_s <= max(plan.joint_timestamp_s,
                                            plan.key_capture_timestamp_s) or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            float(np.max(np.abs(pre_state.full_q - transfer[0]))) >
            limits.max_start_joint_error_rad or
            float(np.max(np.abs(pre_state.hand_raw_measured -
                                plan.joint_feedback.hand_raw_measured))) >
            limits.max_hand_drift_raw):
        raise ValueError("live held start differs from the post-lift transfer")
    commissioning = _verify_controller_commissioning(
        path=commissioning_record_path, adapter=adapter,
        limits=limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("held transfer needs explicit live motion interlock")
    if _wall_time() - pre_state.sample_timestamp_s > limits.max_start_age_s:
        raise ValueError("held transfer start became stale at interlock")
    base = {
        "schema": "precision_insertion_transfer_execution_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "postlift_report_path": str(report_path),
        "postlift_report_sha256": _sha(report_path),
        "trajectory_sha256": _sha(trajectory_file),
        "commissioning": commissioning,
        "pre_state": pre_state.to_record(),
        "scope": "external_guarded_transfer_not_arrival_or_insertion_success",
        "robot_ready": False,
    }
    _write_new(started_path, {**base, "status": "command_requested",
                              "requested_at_s": _wall_time()})
    base["transfer_started_sha256"] = _sha(started_path)
    try:
        result = adapter.follow_transfer(
            transfer.copy(), max_duration_s=limits.max_execution_duration_s,
            expected_hand_raw=pre_state.hand_raw_measured.copy(),
            trajectory_sha256=base["trajectory_sha256"])
        if (not isinstance(result, dict) or
                result.get("schema") !=
                "precision_insertion_external_transfer_result_v1" or
                result.get("trajectory_complete") is not True or
                result.get("safety_abort") is not False or
                result.get("grasp_held") is not True or
                result.get("terminal_hold_acknowledged") is not True or
                result.get("trajectory_sha256") != base["trajectory_sha256"]):
            raise RuntimeError("guarded transfer controller did not complete safely")
        begun = float(result["started_at_s"])
        completed = float(result["completed_at_s"])
        requested = json.loads(started_path.read_text(encoding="utf-8"))["requested_at_s"]
        if (not all(math.isfinite(value) for value in (begun, completed)) or
                not requested <= begun < completed or
                completed - begun > limits.max_execution_duration_s):
            raise RuntimeError("guarded transfer completion times are invalid")
        sources = _source_records(result)
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("held transfer has no measured final robot feedback")
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
                float(np.max(np.abs(post.full_q - transfer[-1]))) >
                limits.max_end_joint_error_rad or
                float(np.max(np.abs(post.hand_raw_measured -
                                    pre_state.hand_raw_measured))) >
                limits.max_hand_drift_raw):
            raise RuntimeError("measured transfer hold differs from plan")
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
