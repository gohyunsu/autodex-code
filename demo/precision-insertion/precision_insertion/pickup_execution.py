"""Demo-only pickup boundary for a separately commissioned motion adapter.

This module may command *approach, grasp and squeeze* after an explicit live
interlock. The unchanged stock FrankaExecutor is refused: its follower can
fall through from a stall to a blocking endpoint move. This module never
replays held lift, transfer or contact insertion. A completed squeeze is not
an observed grasp-success label.
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
from .session_runner import SessionRunner
from .trial_preflight import TrialPreflight


@dataclass(frozen=True)
class PickupExecutionLimits:
    max_start_age_s: float
    max_start_joint_error_rad: float
    max_arrival_arm_error_rad: float
    max_arm_hand_skew_s: float
    max_hand_command_error_raw: float
    max_arm_velocity_rad_s: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            if (type(value) not in (int, float) or
                    not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


def _write_new(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _reject_known_stock_follower(executor) -> None:
    """Do not accidentally promote the known stock follower to demo motion.

    This class-lineage check is only a *negative* gate; passing it does not
    certify a replacement executor or its robot daemon. Those still require
    the external hardware interlock and separate safety commissioning.
    """
    if any(kind.__name__ == "FrankaExecutor" and
           kind.__module__.endswith("franka_executor")
           for kind in type(executor).__mro__):
        raise RuntimeError(
            "unchanged FrankaExecutor is not a fail-closed pickup follower: "
            "a stalled stream may make a blocking final landing move")


def execute_bound_pickup(
    *, runner: SessionRunner, executor, planner,
    pre_state: LiveRobotState,
    read_post_state: Callable[[], LiveRobotState],
    limits: PickupExecutionLimits,
    motion_interlock: Callable[[], bool],
    enable_robot_motion: bool = False,
) -> dict:
    """Execute only v8 pickup through squeeze, then check measured feedback.

    The selected approach is required to be byte-bound to the saved trial
    preflight and its *current* measured 13-DOF starting state. The caller
    must provide a separately commissioned hardware/intervention interlock;
    this function cannot commission a camera clock, force controller or E-stop.
    An exception after command onset leaves an exclusive failure record and
    the arm/hand for supervised recovery. No attempt label is set here.
    """
    limits.validate()
    if (not isinstance(runner, SessionRunner) or
            runner.current_decision().action != "await_lift_observation" or
            not callable(read_post_state) or not callable(motion_interlock)):
        raise ValueError("pickup needs an active attempt and live interlock")
    _reject_known_stock_follower(executor)
    attempt = runner.active_attempt
    trial = runner._preflight  # same demo's immutable, hash-bound selection
    if (attempt is None or not isinstance(trial, TrialPreflight) or
            trial.status != "sampled_planning_pass" or
            trial.selected_candidate_key is None or
            attempt.candidate_id != "/".join(trial.selected_candidate_key) or
            attempt.events or attempt.failure_code is not None or
            trial.pickup_plan is None or
            trial.insertion_plan is None or
            not trial.insertion_plan.sampled_planning_pass):
        raise ValueError("pickup is not bound to a fresh passing v8 trial")
    verified = runner.verify_current_preflight_evidence()
    report_path = runner._preflight_report_path
    attempt_dir = runner._attempt_dir
    if report_path is None or attempt_dir is None:
        raise ValueError("pickup is missing saved attempt/preflight evidence")
    output = attempt_dir / "pickup_execution.json"
    started = attempt_dir / "pickup_started.json"
    if output.exists() or started.exists():
        raise FileExistsError(
            f"pickup already began or has an execution record: {attempt_dir}")
    binding_path = attempt_dir / "preflight_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if (binding.get("candidate_id") != attempt.candidate_id or
            binding.get("preflight_report") != str(report_path) or
            binding.get("preflight_report_sha256") !=
            verified["report_sha256"] or
            binding.get("session_calibration_sha256") !=
            runner.session_sha256 or
            binding.get("measured_start_state_sha256") !=
            verified["measured_start_state_sha256"]):
        raise ValueError("attempt binding differs from selected preflight")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = report.get("artifacts", {})
    relative = artifacts.get("planned_trajectories")
    if not isinstance(relative, str):
        raise ValueError("selected pickup has no saved dense trajectory")
    path = (report_path.parent / relative).resolve()
    if (not path.is_relative_to(report_path.parent) or
            _hash(path) != artifacts.get("planned_trajectories_sha256")):
        raise ValueError("saved pickup trajectory changed")
    with np.load(path, allow_pickle=False) as saved:
        approach = np.asarray(saved["pickup_approach"], dtype=np.float64)
        pregrasp = np.asarray(saved["pickup_pregrasp"], dtype=np.float64)
        grasp = np.asarray(saved["pickup_grasp"], dtype=np.float64)
        wrist = np.asarray(saved["pickup_wrist"], dtype=np.float64)
    planned = np.asarray(trial.pickup_plan.traj, dtype=np.float64)
    if (trial.pickup_plan.success is not True or
            tuple(trial.pickup_plan.scene_info) !=
            trial.selected_candidate_key or
            planned.ndim != 2 or planned.shape[1] != 13 or len(planned) < 2 or
            not np.all(np.isfinite(planned)) or
            not np.array_equal(approach, planned) or
            not np.array_equal(pregrasp, np.asarray(
                trial.pickup_plan.pregrasp_pose, dtype=np.float64)) or
            not np.array_equal(grasp, np.asarray(
                trial.pickup_plan.grasp_pose, dtype=np.float64)) or
            not np.array_equal(wrist, np.asarray(
                trial.pickup_plan.wrist_se3, dtype=np.float64)) or
            not np.allclose(planned[0], trial.live_start_q, atol=1e-4, rtol=0)):
        raise ValueError("pickup command differs from its saved v8 plan/start")
    if not isinstance(pre_state, LiveRobotState):
        raise TypeError("pickup requires measured FR3/Inspire feedback")
    pre_state.validate(
        max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
        max_hand_command_error_raw=limits.max_hand_command_error_raw,
        max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
    now = _wall_time()
    if (not math.isfinite(now) or
            pre_state.sample_timestamp_s <= attempt.started_at_s or
            pre_state.sample_timestamp_s > now or
            now - pre_state.sample_timestamp_s > limits.max_start_age_s or
            float(np.max(np.abs(pre_state.full_q - planned[0]))) >
            limits.max_start_joint_error_rad):
        raise ValueError("fresh measured start does not match v8 approach")
    # Validate all read-only evidence before touching the interlock/motors.
    if enable_robot_motion is not True or motion_interlock() is not True:
        raise PermissionError("pickup motion needs explicit live interlock")
    if (_wall_time() - pre_state.sample_timestamp_s >
            limits.max_start_age_s):
        raise ValueError("measured start aged while waiting for motion interlock")

    base = {
        "schema": "precision_insertion_pickup_execution_v1",
        "attempt_id": attempt.attempt_id,
        "candidate_id": attempt.candidate_id,
        "preflight_report": str(report_path),
        "preflight_report_sha256": verified["report_sha256"],
        "planned_trajectories_sha256": _hash(path),
        "pre_state": pre_state.to_record(),
        "command": "stock_franka_execute_skip_lift_start_from_current",
        "scope": "approach_grasp_squeeze_only_not_lift_or_grasp_label",
    }
    _write_new(started, {**base, "status": "command_requested",
                         "requested_at_s": _wall_time(),
                         "robot_ready": False})
    base["pickup_started_sha256"] = _hash(started)
    raw = None
    post = None
    try:
        squeeze_action = executor.execute(
            trial.pickup_plan, planner=planner,
            scene_cfg=trial.trial_scene,
            skip_lift=True, start_from_current=True)
        raw = np.asarray(squeeze_action, dtype=np.float64)
        if (raw.shape != (6,) or not np.all(np.isfinite(raw)) or
                np.any(raw < 0) or np.any(raw > 1000)):
            raise RuntimeError("stock executor returned no valid squeeze action")
        post = read_post_state()
        if not isinstance(post, LiveRobotState):
            raise TypeError("post-squeeze FR3/Inspire feedback is missing")
        post.validate(
            max_arm_hand_skew_s=limits.max_arm_hand_skew_s,
            max_hand_command_error_raw=limits.max_hand_command_error_raw,
            max_arm_velocity_rad_s=limits.max_arm_velocity_rad_s)
        if (post.sample_timestamp_s <= pre_state.sample_timestamp_s or
                float(np.max(np.abs(post.full_q[:7] - planned[-1, :7]))) >
                limits.max_arrival_arm_error_rad or
                not np.allclose(post.hand_raw_commanded, raw,
                                atol=limits.max_hand_command_error_raw,
                                rtol=0)):
            raise RuntimeError("post-squeeze arm/hand feedback differs from command")
        record = {**base, "status": "squeeze_command_and_feedback_complete",
                  "squeeze_action_raw": raw.tolist(),
                  "post_state": post.to_record(),
                  "completed_at_s": post.sample_timestamp_s,
                  "robot_ready": False}
    except Exception as exc:
        failure = {**base, "status": "execution_or_feedback_failed",
                   "error_type": type(exc).__name__,
                   "error": str(exc), "robot_ready": False}
        if raw is not None and raw.shape == (6,) and np.all(np.isfinite(raw)):
            failure["squeeze_action_raw"] = raw.tolist()
        if isinstance(post, LiveRobotState):
            try:
                candidate_state = post.to_record()
                json.dumps(candidate_state, allow_nan=False)
            except (TypeError, ValueError):
                pass
            else:
                failure["post_state"] = candidate_state
        _write_new(output, failure)
        raise
    _write_new(output, record)
    return record
