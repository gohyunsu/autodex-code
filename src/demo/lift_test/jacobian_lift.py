"""Compatibility API for the production Jacobian vertical-stroke planner.

The lift experiment originally owned this solver.  The implementation now
lives in :mod:`autodex.planner.jacobian_stroke` so offline grids and the real
AutoDex pipeline exercise the same numerical continuation and validation
contract.  This module preserves the experiment's historical return shape.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from autodex.planner.jacobian_stroke import (
    FRANKA_VERTICAL_PAYLOAD_SPEED_SCALE,
    JacobianStrokeOptions,
    XARM_VERTICAL_PAYLOAD_SPEED_SCALE,
    numerical_wrist_jacobian,
    plan_jacobian_vertical_stroke,
)


@dataclass(frozen=True)
class LiftOptions:
    height_m: float = 0.10
    step_m: float = 0.005
    finite_difference_rad: float = 1.0e-4
    damping: float = 2.0e-2
    max_iterations: int = 24
    max_joint_step_rad: float = 0.10
    position_tolerance_m: float = 0.0015
    orientation_tolerance_rad: float = np.deg2rad(2.0)
    rotation_weight_m_per_rad: float = 0.10
    max_waypoint_delta_rad: float = 0.45
    max_segment_joint_delta_rad: float = 0.02
    table_clearance_tolerance_m: float = 0.001


def _stroke_options(options: LiftOptions, n_arm: int) -> JacobianStrokeOptions:
    return JacobianStrokeOptions(
        step_m=options.step_m,
        finite_difference_rad=options.finite_difference_rad,
        damping=options.damping,
        max_iterations=options.max_iterations,
        max_joint_step_rad=options.max_joint_step_rad,
        position_tolerance_m=options.position_tolerance_m,
        orientation_tolerance_rad=options.orientation_tolerance_rad,
        rotation_weight_m_per_rad=options.rotation_weight_m_per_rad,
        max_waypoint_delta_rad=options.max_waypoint_delta_rad,
        max_segment_joint_delta_rad=options.max_segment_joint_delta_rad,
        support_clearance_tolerance_m=options.table_clearance_tolerance_m,
        held_object_speed_scale=(
            XARM_VERTICAL_PAYLOAD_SPEED_SCALE if int(n_arm) == 6
            else FRANKA_VERTICAL_PAYLOAD_SPEED_SCALE),
    )


def continue_vertical_lift(
    planner,
    start_full_q: np.ndarray,
    *,
    hand_q: np.ndarray,
    mesh_vertices: np.ndarray,
    object_pose_at_grasp: np.ndarray,
    table_surface_z_m: float,
    options: LiftOptions,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[np.ndarray | None, list[dict[str, Any]], dict[str, Any]]:
    """Preserve the old +Z experiment API using the production solver."""
    start = np.asarray(start_full_q, dtype=np.float32).reshape(-1).copy()
    hand = np.asarray(hand_q, dtype=np.float32).reshape(-1)
    n_arm = int(planner._n_arm)
    if len(start) != n_arm + len(hand):
        raise ValueError("start q and fixed hand q dimensions do not match planner")
    start[n_arm:] = hand
    if options.height_m <= 0.0:
        raise ValueError("lift height must be positive")
    target = planner.fk_wrist(start)
    target[2, 3] += float(options.height_m)
    result = plan_jacobian_vertical_stroke(
        planner, start, target,
        options=_stroke_options(options, n_arm),
        attached_object_vertices=np.asarray(mesh_vertices, dtype=np.float64),
        attached_object_pose_at_start=np.asarray(
            object_pose_at_grasp, dtype=np.float64),
        support_surface_z_m=float(table_surface_z_m),
        expected_travel_m=float(options.height_m),
        progress_callback=progress_callback,
    )
    completed_steps = sum(bool(record.get("success"))
                          for record in result.step_records)
    requested_steps = int(np.ceil(options.height_m / options.step_m))
    info: dict[str, Any] = {
        "success": result.success,
        "reason": result.failure_detail,
        "failure_code": result.failure_code,
        "completed_steps": completed_steps,
        "requested_steps": requested_steps,
        "start_wrist_pose": planner.fk_wrist(start).tolist(),
        "production_stroke_timing": result.timing,
        "production_stroke_validation": result.validation,
    }
    if result.success:
        info["execution_qpos"] = np.asarray(
            result.collision_checked_qpos, dtype=np.float32)
        info["execution_sample_contract"] = {
            "source": "collision_checked_linear_joint_chords",
            "max_joint_delta_rad": options.max_segment_joint_delta_rad,
            "sample_count": len(result.collision_checked_qpos),
            "production_c2_also_validated": True,
        }
    geometry = (None if not result.success else
                np.asarray(result.geometric_qpos, dtype=np.float32))
    return geometry, result.step_records, info


def options_as_dict(options: LiftOptions) -> dict[str, Any]:
    return asdict(options)


def save_step_csv(path: Path, records: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else [
        "step", "target_z_m", "direction", "success", "failure_code",
        "failure_detail", "iterations", "position_error_m",
        "orientation_error_rad", "delta_q_norm", "min_singular_value",
        "condition_number", "joint_limit_hit", "segment_samples",
        "segment_failure_alpha", "total_s",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


__all__ = [
    "LiftOptions", "continue_vertical_lift", "numerical_wrist_jacobian",
    "options_as_dict", "save_step_csv",
]
