"""Demo-local, opt-in boundary for an independently safeguarded held lift.

This consumes the *measured post-squeeze* v8 lift replan, never the original
nominal candidate lift. It cannot implement or certify the external Franka
controller, robot-side stale-command watchdog or E-stop. Transfer and
contact insertion are intentionally separate stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable

import numpy as np

from .live_robot_state import LiveRobotState
from .measured_lift_preflight import (
    MeasuredLiftChain, verify_measured_lift_chain,
)
from .pickup_execution import _reject_known_stock_follower
from .session_runner import SessionRunner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def _write_new(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class LiftExecutionLimits:
    max_start_age_s: float
    max_end_state_age_s: float
    max_execution_feedback_gap_s: float
    max_start_joint_error_rad: float
    max_end_joint_error_rad: float
    max_hand_drift_raw: float
    max_arm_hand_skew_s: float
    max_hand_command_error_raw: float
    max_arm_velocity_rad_s: float
    max_execution_duration_s: float
    max_watchdog_command_age_s: float
    max_commissioning_record_age_s: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            if (type(value) not in (int, float) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} needs a finite commissioned limit")


def _verify_controller_commissioning(
    *, path: Path, adapter, limits: LiftExecutionLimits, now_s: float,
) -> dict:
    """Check a reviewed record against the exact binary used by the adapter.

    This catches stale/wrong binaries and absent stated tests. It cannot
    establish that an operator honestly performed the tests.
    """
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    binary = Path(adapter.daemon_binary_path).expanduser().resolve()
    required_true = (
        "stale_command_timeout_test_passed", "disconnect_stop_test_passed",
        "stop_ack_test_passed", "e_stop_test_passed",
    )
    if (not isinstance(record, dict) or
            record.get("schema") !=
            "precision_insertion_robot_watchdog_commissioning_v1" or
            not isinstance(record.get("reviewed_by"), str) or
            not record["reviewed_by"].strip() or
            any(record.get(name) is not True for name in required_true) or
            not binary.is_file() or
            record.get("daemon_binary_path") != str(binary) or
            record.get("daemon_binary_sha256") != _sha(binary)):
        raise ValueError("lift controller has no matching reviewed watchdog record")
    tested = record.get("tested_at_s")
    timeout = record.get("command_expiry_s")
    if (type(tested) not in (int, float) or not math.isfinite(tested) or
            tested > now_s or
            now_s - tested > limits.max_commissioning_record_age_s or
            type(timeout) not in (int, float) or
            not math.isfinite(timeout) or timeout <= 0 or
            timeout > limits.max_watchdog_command_age_s):
        raise ValueError("lift watchdog review is stale or its expiry is too slow")
    return {"path": str(source), "sha256": _sha(source),
            "daemon_binary_sha256": _sha(binary),
            "command_expiry_s": float(timeout)}


def execute_bound_measured_lift(
    *, runner: SessionRunner, adapter, pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: LiftExecutionLimits, commissioning_record_path: Path,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Command only a saved measured-start lift after explicit live gates.

    ``adapter.follow_lift`` must itself enforce a robot-side watchdog and
    force/velocity/stall stops throughout the call. This Python code can
    check returned evidence but cannot stop motion if its process dies.
    """
    limits.validate()
    if (not isinstance(runner, SessionRunner) or
            runner.current_decision().action != "await_lift_observation" or
            not isinstance(pre_state, LiveRobotState) or
            not callable(read_post_state) or
            not callable(motion_interlock) or
            not callable(getattr(adapter, "follow_lift", None)) or
            not callable(getattr(adapter, "stop_and_acknowledge", None))):
        raise ValueError("lift needs a selected attempt and guarded adapter")
    _reject_known_stock_follower(adapter)
    attempt = runner.active_attempt
    chain = runner._measured_lift_preflight
    report_path = runner._measured_lift_report_path
    attempt_dir = runner._attempt_dir
    if (attempt is None or not isinstance(chain, MeasuredLiftChain) or
            chain.status != "sampled_measured_chain_pass" or
            report_path is None or attempt_dir is None or
            chain.attempt_id != attempt.attempt_id or
            chain.candidate_id != attempt.candidate_id or
            chain.session_calibration_sha256 != runner.session_sha256 or
            chain.planning is None or chain.planning.lift_trajectory is None or
            runner._measured_lift_report_sha256 != _sha(report_path)):
        raise ValueError("lift is not bound to this passing measured chain")
    saved = verify_measured_lift_chain(report_path, expected=chain)
    if (saved["status"] != "sampled_measured_chain_pass" or
            saved["session_calibration_sha256"] != runner.session_sha256):
        raise ValueError("saved lift chain changed")
    output = attempt_dir / "lift_execution.json"
    started_path = attempt_dir / "lift_started.json"
    failure_path = attempt_dir / "lift_failure.json"
    if any(path.exists() for path in (output, started_path, failure_path)):
        raise FileExistsError("this attempt already started a held lift")
    trajectory_file = (report_path.parent /
                       saved["planned_trajectories"]).resolve()
    if (not trajectory_file.is_relative_to(report_path.parent) or
            _sha(trajectory_file) != saved["planned_trajectories_sha256"]):
        raise ValueError("held lift trajectory bytes changed")
    with np.load(trajectory_file, allow_pickle=False) as archive:
        lift = np.asarray(archive["lift"], dtype=np.float64)
    if (lift.ndim != 2 or lift.shape[1] != 13 or len(lift) < 2 or
            not np.all(np.isfinite(lift)) or
            not np.array_equal(lift, chain.planning.lift_trajectory) or
            not np.allclose(lift[:, 7:], lift[0, 7:], atol=1e-8, rtol=0)):
        raise ValueError("saved measured lift does not hold the Inspire grasp")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = _wall_time()
    if (pre_state.sample_timestamp_s < chain.measured_start.sample_timestamp_s or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            float(np.max(np.abs(pre_state.full_q - lift[0]))) >
            limits.max_start_joint_error_rad or
            float(np.max(np.abs(pre_state.hand_raw_measured -
                                chain.measured_start.hand_raw_measured))) >
            limits.max_hand_drift_raw):
        raise ValueError("live held start differs from the measured lift plan")
    commissioning = _verify_controller_commissioning(
        path=commissioning_record_path, adapter=adapter,
        limits=limits, now_s=now)
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("held lift needs explicit live motion interlock")
    if _wall_time() - pre_state.sample_timestamp_s > limits.max_start_age_s:
        raise ValueError("held start became stale while waiting for interlock")
    base = {
        "schema": "precision_insertion_lift_execution_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "session_calibration_sha256": runner.session_sha256,
        "measured_lift_report_path": str(report_path),
        "measured_lift_report_sha256": _sha(report_path),
        "trajectory_sha256": _sha(trajectory_file),
        "commissioning": commissioning,
        "pre_state": pre_state.to_record(),
        "scope": "external_guarded_held_lift_not_grasp_success_or_insertion",
        "robot_ready": False,
    }
    _write_new(started_path, {
        **base, "status": "command_requested",
        "requested_at_s": _wall_time(),
    })
    base["lift_started_sha256"] = _sha(started_path)
    try:
        result = adapter.follow_lift(
            lift.copy(), max_duration_s=limits.max_execution_duration_s,
            expected_hand_raw=pre_state.hand_raw_measured.copy(),
            trajectory_sha256=base["trajectory_sha256"])
        if (not isinstance(result, dict) or
                result.get("schema") !=
                "precision_insertion_external_lift_result_v1" or
                result.get("trajectory_complete") is not True or
                result.get("force_abort") is not False or
                result.get("terminal_hold_acknowledged") is not True or
                result.get("trajectory_sha256") != base[
                    "trajectory_sha256"]):
            raise RuntimeError("guarded lift controller did not complete safely")
        begun = float(result["started_at_s"])
        completed = float(result["completed_at_s"])
        requested = json.loads(started_path.read_text(encoding="utf-8"))[
            "requested_at_s"]
        if (not all(math.isfinite(value) for value in (begun, completed)) or
                not requested <= begun < completed or
                completed - begun > limits.max_execution_duration_s):
            raise RuntimeError("guarded lift completion times are invalid")
        source_trace = Path(result["controller_trace_path"]).expanduser()
        trace_path = source_trace.resolve()
        if (not source_trace.is_absolute() or not trace_path.is_file() or
                result.get("controller_trace_sha256") != _sha(trace_path)):
            raise RuntimeError("guarded lift controller trace is absent or changed")
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise RuntimeError("held lift has no measured final robot feedback")
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
                float(np.max(np.abs(post.full_q - lift[-1]))) >
                limits.max_end_joint_error_rad or
                float(np.max(np.abs(post.hand_raw_measured -
                                    pre_state.hand_raw_measured))) >
                limits.max_hand_drift_raw):
            raise RuntimeError("measured held lift endpoint differs from plan")
        record = {
            **base, "trajectory_complete": True, "force_abort": False,
            "started_at_s": begun, "completed_at_s": completed,
            "controller_result": result, "post_state": post.to_record(),
        }
        _write_new(output, record)
    except Exception as exc:
        try:
            stop_ack = adapter.stop_and_acknowledge()
        except Exception as stop_exc:
            stop_ack = {"error": str(stop_exc), "acknowledged": False}
        _write_new(failure_path, {
            **base, "status": "execution_or_feedback_failed",
            "error_type": type(exc).__name__, "error": str(exc),
            "stop_acknowledgement": stop_ack,
            "robot_state_after_failure": "unknown_requires_supervised_recovery",
        })
        raise
    return record
