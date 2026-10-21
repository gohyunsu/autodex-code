"""Execute a saved observed-key XY retry's non-contact transfer.

The square-key route already has same-frame FoundPose, two-view VLM voting,
exact 20 mm endpoint screening and a fresh cuRobo plan. This boundary reuses
the *same* external held-transfer controller contract as the first transfer;
it does not command axial contact or infer that the key arrived correctly.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .lift_execution import (
    LiftExecutionLimits as TransferExecutionLimits,
    _verify_controller_commissioning, _write_new,
)
from .live_robot_state import LiveRobotState
from .pickup_execution import _reject_known_stock_follower
from .retry_preflight import XYRetryPreflight
from .session_runner import SessionRunner
from .transfer_execution import verify_transfer_controller_result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_bound_observed_retry_transfer(
    *, runner: SessionRunner, preflight: XYRetryPreflight,
    preflight_report_path: Path, adapter, pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: TransferExecutionLimits, commissioning_record_path: Path,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Follow only the source-bound XY transfer, then retain measured arrival.

    The caller must next capture and assess the corrected pre-insertion hold
    before any guarded axial stroke. No retry success label is written here.
    """
    limits.validate()
    if (not isinstance(runner, SessionRunner) or
            runner.current_decision().action !=
            "await_retry_execution_and_observation" or
            not isinstance(preflight, XYRetryPreflight) or
            preflight.status != "sampled_retry_preflight_pass" or
            preflight.planning.sampled_planning_pass is not True or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or
            not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_transfer", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("observed XY transfer needs a pending planned retry")
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    attempt_dir = runner._attempt_dir
    report = Path(preflight_report_path).expanduser().resolve()
    if (attempt is None or attempt_dir is None or
            attempt.candidate_id != "/".join(preflight.candidate_key) or
            attempt.labels.get("insertion_success") is not False or
            attempt.failure_code is not None or
            not attempt._pending_retry or not attempt.events or
            attempt.events[-1].get("stage") != "xy_retry" or
            attempt.events[-1].get("value") != preflight.choice_id or
            attempt.events[-1].get("evidence_refs", {}).get(
                "live_preflight") != str(report) or
            tuple(attempt.xy_offset_socket_m) !=
            tuple(preflight.targets.xy_offset_socket_m) or
            not report.is_file()):
        raise ValueError("observed XY transfer differs from pending retry event")
    runner.verify_current_preflight_evidence()
    saved = json.loads(report.read_text(encoding="utf-8"))
    if (not isinstance(saved, dict) or
            any(saved.get(key) != value for key, value in
                preflight.to_record().items()) or
            saved.get("planned_trajectories") != "planned_trajectories.npz"):
        raise ValueError("saved observed XY preflight differs from selected plan")
    archive = (report.parent / saved["planned_trajectories"]).resolve()
    if (not archive.is_relative_to(report.parent) or
            _sha(archive) != saved.get("planned_trajectories_sha256")):
        raise ValueError("observed XY transfer trajectory bytes changed")
    with np.load(archive, allow_pickle=False) as data:
        transfer = np.asarray(data["transfer"], dtype=np.float64)
    planned = preflight.planning.transfer_trajectory
    if (planned is None or transfer.ndim != 2 or
            transfer.shape[1] != 13 or len(transfer) < 2 or
            not np.all(np.isfinite(transfer)) or
            not np.array_equal(transfer, planned) or
            not np.allclose(transfer[:, 7:], transfer[0, 7:], atol=1e-8,
                            rtol=0)):
        raise ValueError("observed XY transfer does not hold the planned grasp")
    output = attempt_dir / "observed_retry_transfer_execution.json"
    started = attempt_dir / "observed_retry_transfer_started.json"
    failure = attempt_dir / "observed_retry_transfer_failure.json"
    if any(path.exists() for path in (output, started, failure)):
        raise FileExistsError("this observed XY transfer already started")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = time.time()
    if (pre_state.sample_timestamp_s <= preflight.start_q_timestamp_s or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            np.max(np.abs(pre_state.full_q - transfer[0])) >
            limits.max_start_joint_error_rad):
        raise ValueError("live withdrawn hold differs from XY transfer start")
    commissioning = _verify_controller_commissioning(
        path=commissioning_record_path, adapter=adapter,
        limits=limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("observed XY transfer needs a live motion interlock")
    requested = time.time()
    if requested - pre_state.sample_timestamp_s > limits.max_start_age_s:
        raise ValueError("observed XY start aged while waiting for interlock")
    base = {
        "schema": "precision_insertion_observed_xy_transfer_execution_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "preflight_report_path": str(report),
        "preflight_report_sha256": _sha(report),
        "trajectory_sha256": _sha(archive),
        "commissioning": commissioning,
        "pre_state": pre_state.to_record(),
        "scope": "external_retry_transfer_not_arrival_or_insertion_success",
        "robot_ready": False,
    }
    _write_new(started, {**base, "status": "command_requested",
                         "requested_at_s": requested})
    base["started_marker_sha256"] = _sha(started)
    try:
        result = adapter.follow_transfer(
            transfer.copy(), max_duration_s=limits.max_execution_duration_s,
            expected_hand_raw=pre_state.hand_raw_measured.copy(),
            trajectory_sha256=base["trajectory_sha256"])
        begun, completed, sources = verify_transfer_controller_result(
            result, trajectory_sha256=base["trajectory_sha256"],
            requested_at_s=requested,
            max_execution_duration_s=limits.max_execution_duration_s)
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("observed XY transfer lacks measured arrival")
        post.validate(
            max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
            max_hand_command_error_raw=limits.max_hand_command_error_raw,
            max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
        now = time.time()
        if (post.sample_timestamp_s < begun or
                abs(post.sample_timestamp_s - completed) >
                limits.max_execution_feedback_gap_s or
                post.sample_timestamp_s > now or
                now - post.sample_timestamp_s > limits.max_end_state_age_s or
                np.max(np.abs(post.full_q - transfer[-1])) >
                limits.max_end_joint_error_rad or
                np.max(np.abs(post.hand_raw_measured -
                              pre_state.hand_raw_measured)) >
                limits.max_hand_drift_raw):
            raise RuntimeError("measured XY arrival differs from transfer plan")
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
        _write_new(failure, {
            **base, "status": "execution_or_feedback_failed",
            "error_type": type(exc).__name__, "error": str(exc),
            "stop_acknowledgement": stop_ack,
            "robot_state_after_failure": "unknown_requires_supervised_recovery",
        })
        raise
    return record
