"""Plan a bounded socket-plane hold shift after a grounded XY diagnostic.

This module deliberately stops before another insertion. A positive result
only means that cuRobo planned the arm path and sampled full key/Inspire CAD
cleared the frozen world with supplied future-trial surface bounds. A new
camera observation and insertion-specific gates are still required.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode
from .geometry import validate_se3
from .path_audit import PathAuditLimits, audit_held_lateral_path
from .preflight import _goal_met, _path
from .uncertainty_margin import SurfaceDeviationBounds
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


@dataclass(frozen=True)
class LateralHoldPreflight:
    status: str
    increment_socket_xy_m: tuple[float, float]
    start_q: np.ndarray
    T_robot_hand_start: np.ndarray
    T_robot_hand_goal: np.ndarray
    T_key_hand: np.ndarray
    trajectory: np.ndarray | None
    planner_query: dict
    sampled_audit: dict | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_lateral_hold_preflight_v1",
            "status": self.status,
            "increment_socket_xy_m": list(self.increment_socket_xy_m),
            "start_q": self.start_q.tolist(),
            "T_robot_hand_start": self.T_robot_hand_start.tolist(),
            "T_robot_hand_goal": self.T_robot_hand_goal.tolist(),
            "T_key_hand": self.T_key_hand.tolist(),
            "trajectory_sample_count": (
                None if self.trajectory is None else len(self.trajectory)),
            "planner_query": self.planner_query,
            "sampled_audit": self.sampled_audit,
            "scope": "read_only_lateral_hold_plan_requires_fresh_postshift_observation",
            "insertion_replan_allowed": False,
            "robot_ready": False,
        }


def plan_lateral_hold_shift(
    *, planner, mode: TaskMode, shared_root: Path, calibration,
    trial_scene: dict, start_q: np.ndarray, expected_hold_pose: np.ndarray,
    T_key_hand: np.ndarray, increment_socket_xy_m: tuple[float, float],
    bounds: SurfaceDeviationBounds, limits: PathAuditLimits,
    max_path_deviation_m: float, max_hold_height_deviation_m: float,
    max_hold_rotation_deg: float,
) -> LateralHoldPreflight:
    """Start at the measured withdrawn wrist and plan <=1 mm in socket XY.

    ``expected_hold_pose`` must come from the saved, source-verified prior
    preinsert plan. This lower-level function does not admit a VLM diagnostic
    or physical calibration by itself; the session caller must bind those
    sources and measured joint timestamps before using its result.
    """
    limits.validate()
    bounds.validate()
    increment = np.asarray(increment_socket_xy_m, dtype=np.float64)
    if (increment.shape != (2,) or not np.all(np.isfinite(increment)) or
            not 0 < np.linalg.norm(increment) <= 0.001 + 1e-12):
        raise ValueError("hold shift requires nonzero socket XY <= 1 mm")
    if not all(math.isfinite(float(v)) and v > 0 for v in (
            max_path_deviation_m, max_hold_height_deviation_m,
            max_hold_rotation_deg)):
        raise ValueError("commissioned lateral path tolerances must be positive")
    if (getattr(planner, "_n_arm", None) != 7 or
            getattr(planner, "_hand", None) != "fr3_inspire" or
            getattr(planner, "_robot_cfg", {}).get(
                "kinematics", {}).get("ee_link") != "base_link"):
        raise ValueError("lateral preflight requires the stock FR3/Inspire planner")
    root = Path(shared_root).expanduser().resolve()
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    held_scene = build_held_scene_from_trial(
        trial_scene=trial_scene, calibration=calibration)
    start = np.asarray(start_q, dtype=np.float64)
    if start.shape != (13,) or not np.all(np.isfinite(start)):
        raise ValueError("lateral preflight needs measured 13-DOF start joints")
    held = start[7:].copy()
    relation = validate_se3(T_key_hand, name="held T_key_hand")
    wrist = validate_se3(planner.fk_wrist(start), name="measured start wrist FK")
    expected = validate_se3(expected_hold_pose, name="saved withdrawn hold pose")
    if not _goal_met(wrist, expected, limits):
        raise ValueError("measured wrist is not at the saved withdrawn hold")
    goal = wrist.copy()
    goal[:3, 3] += socket[:3, :2] @ increment
    query = planner.plan_cartesian_pose(
        start, goal, scene_cfg=held_scene, include_obj_obstacle=False,
        return_result=True, lock_hand=True,
        timing_phase="precision_insertion_lateral_hold_preflight")
    query_record = {
        "stage": "lateral_hold_shift", "planner_api": "plan_cartesian_pose",
        "success": bool(query.success),
        "constraint_mode": getattr(query, "constraint_mode", None),
        "failure_stage": getattr(query, "failure_stage", None),
    }

    def result(status: str, trajectory=None, audit=None):
        return LateralHoldPreflight(
            status, (float(increment[0]), float(increment[1])),
            start.copy(), wrist.copy(), goal.copy(), relation.copy(),
            trajectory, query_record, audit)

    if not query.success or query.trajectory is None:
        return result("lateral_curobo_unreachable")
    trajectory = _path(query.trajectory, "lateral hold shift", start, held)
    if not _goal_met(validate_se3(planner.fk_wrist(trajectory[-1]),
                                   name="lateral goal FK"), goal, limits):
        return result("lateral_goal_residual", trajectory)
    audit = audit_held_lateral_path(
        shared_root=root, mode=mode, calibration=calibration,
        planner=planner, trajectory=trajectory, held_hand_q=held,
        T_key_hand=relation,
        increment_socket_xy_m=(float(increment[0]), float(increment[1])),
        limits=limits, max_path_deviation_m=max_path_deviation_m,
        max_hold_height_deviation_m=max_hold_height_deviation_m,
        max_hold_rotation_deg=max_hold_rotation_deg,
        key_surface_bound_m=bounds.key_surface_m,
        hand_surface_bound_m=bounds.hand_surface_m)
    return result("sampled_lateral_hold_shift_pass" if audit[
        "sampled_clear"] else "sampled_lateral_hold_shift_rejected",
        trajectory, audit)


def write_lateral_hold_preflight(
    result: LateralHoldPreflight, output_dir: Path,
) -> Path:
    """Save exact cuRobo joint bytes and audit without creating retry state."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if result.trajectory is not None:
        trajectory_file = target / "lateral_trajectory.npy"
        np.save(trajectory_file, result.trajectory, allow_pickle=False)
        record["lateral_trajectory"] = trajectory_file.name
        record["lateral_trajectory_sha256"] = hashlib.sha256(
            trajectory_file.read_bytes()).hexdigest()
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target
