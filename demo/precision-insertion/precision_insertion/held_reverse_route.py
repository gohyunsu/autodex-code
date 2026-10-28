"""Plan a held-key return by reversing this attempt's executed joint paths.

This is a read-only preflight. It neither opens the hand nor commands the arm.
The observed landing must still be re-perceived and restored by the existing
reset pipeline after an eventual execution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from autodex.utils.conversion import cart2se3, se32cart
from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT

from .axis_jacobian import _fixed_world_cfg
from .geometry import pose_angle_deg, validate_se3
from .path_audit import PathAuditLimits, _audit_held_geometry_samples


@dataclass(frozen=True)
class HeldReverseRoute:
    held_trajectory: np.ndarray
    release_key_pose_robot: np.ndarray
    release_scene: dict
    clear_view_trajectory: np.ndarray
    sampled_geometry: dict


def _joint_path(path: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(path, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 13 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError(f"{name} needs a finite 13-joint trajectory")
    return q.copy()


def _join(
    start: np.ndarray, end: np.ndarray, *, planner, limits: PathAuditLimits,
    arm_velocity_rad_s: float, traj_dt_s: float,
) -> np.ndarray:
    """Sample a short join using the stock follower's joint velocity scale."""
    step = min(limits.max_joint_step_rad,
               arm_velocity_rad_s * traj_dt_s * .75)
    count = max(2, math.ceil(float(np.max(np.abs(end[:7] - start[:7]))) /
                             step) + 1)
    while True:
        path = np.linspace(start, end, count, dtype=np.float64)
        wrist = [validate_se3(planner.fk_wrist(q), name="reverse join FK")
                 for q in path]
        if all(
            np.max(np.abs(path[i, :7] - path[i - 1, :7])) <=
            limits.max_joint_step_rad and
            np.linalg.norm(wrist[i][:3, 3] - wrist[i - 1][:3, 3]) <=
            limits.max_wrist_step_m and
            pose_angle_deg(wrist[i], wrist[i - 1]) <=
            limits.max_wrist_rotation_deg
            for i in range(1, len(path))
        ):
            return path
        count *= 2


def assemble_reverse_held_trajectory(
    *, planner, current_q: np.ndarray,
    executed_xy_trajectories: Sequence[np.ndarray],
    executed_transfer_trajectory: np.ndarray,
    executed_lift_trajectory: np.ndarray,
    limits: PathAuditLimits, arm_velocity_rad_s: float, traj_dt_s: float,
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Join the measured state to reversed XY, transfer, then lift paths."""
    limits.validate()
    if (not math.isfinite(arm_velocity_rad_s) or
            not math.isfinite(traj_dt_s) or arm_velocity_rad_s <= 0 or
            traj_dt_s <= 0):
        raise ValueError("reverse route needs follower timing")
    current = np.asarray(current_q, dtype=np.float64)
    if current.shape != (13,) or not np.isfinite(current).all():
        raise ValueError("reverse route needs measured FR3/Inspire joints")
    held_hand = current[7:].copy()
    xy_source = [
        (f"xy_{i}", _joint_path(path, f"executed XY {i}"))
        for i, path in enumerate(executed_xy_trajectories)
    ]
    source = list(reversed(xy_source))
    source.extend((
        ("transfer", _joint_path(executed_transfer_trajectory,
                                 "executed transfer")),
        ("lift", _joint_path(executed_lift_trajectory, "executed lift")),
    ))
    reversed_parts = []
    for name, path in source:
        segment = path[::-1].copy()
        segment[:, 7:] = held_hand
        reversed_parts.append((f"reverse_{name}", segment))
    parts = []
    previous = current
    for name, segment in reversed_parts:
        bridge = _join(
            previous, segment[0], planner=planner, limits=limits,
            arm_velocity_rad_s=arm_velocity_rad_s, traj_dt_s=traj_dt_s)
        parts.append((f"join_to_{name}", bridge))
        parts.append((name, segment))
        previous = segment[-1]
    held_path = np.concatenate([part if i == 0 else part[1:]
                                for i, (_, part) in enumerate(parts)])
    return held_path, parts


def plan_held_reverse_route(
    *, planner, trial_scene: dict, shared_root: Path, mode, calibration,
    current_q: np.ndarray, current_key_pose_robot: np.ndarray,
    original_key_pose_robot: np.ndarray,
    executed_xy_trajectories: Sequence[np.ndarray],
    executed_transfer_trajectory: np.ndarray,
    executed_lift_trajectory: np.ndarray,
    limits: PathAuditLimits, arm_velocity_rad_s: float, traj_dt_s: float,
) -> HeldReverseRoute:
    """Screen reversed motion and the stock open-hand clear-view retract.

    The current key pose is the same multiview-XY/nominal-orientation
    hypothesis used by the corrected insertion handoff. No new grip sensor
    claim is made. The original source trajectories remain unchanged.
    """
    if (not isinstance(trial_scene, dict) or
            not isinstance(trial_scene.get("mesh"), dict) or
            not isinstance(trial_scene["mesh"].get("target"), dict)):
        raise ValueError("reverse route needs this attempt's trial scene")
    key = validate_se3(current_key_pose_robot, name="current held key pose")
    original = validate_se3(original_key_pose_robot,
                            name="original tabletop key pose")
    planner._set_motion_world(_fixed_world_cfg(trial_scene))
    held_path, parts = assemble_reverse_held_trajectory(
        planner=planner, current_q=current_q,
        executed_xy_trajectories=executed_xy_trajectories,
        executed_transfer_trajectory=executed_transfer_trajectory,
        executed_lift_trajectory=executed_lift_trajectory, limits=limits,
        arm_velocity_rad_s=arm_velocity_rad_s, traj_dt_s=traj_dt_s)
    current = np.asarray(current_q, dtype=np.float64)
    held_hand = current[7:].copy()
    held_relation = np.linalg.inv(key) @ validate_se3(
        planner.fk_wrist(current), name="current held wrist FK")
    wrist = {name: [validate_se3(planner.fk_wrist(q), name=f"{name} FK")
                    for q in path] for name, path in parts}
    failures, distances, hashes = _audit_held_geometry_samples(
        shared_root=Path(shared_root), mode=mode, calibration=calibration,
        wrist=wrist, T_key_hand=held_relation, held_hand_q=held_hand,
        minimum_hand_clearance_m=limits.minimum_hand_clearance_m,
        stop_on_first_failure=True, collision_only_if_clear=True)
    if failures:
        raise ValueError(f"reverse held geometry failed: {failures[0]}")

    import torch
    q_tensor = torch.as_tensor(
        held_path, dtype=torch.float32,
        device=planner._tensor_args.device).unsqueeze(1)
    feasible = (planner._motion_gen.rollout_fn
                .rollout_constraint(q_tensor, use_batch_env=False)
                .feasible.detach().cpu().numpy().reshape(-1))
    if not bool(np.all(feasible)):
        raise ValueError(
            f"reverse arm world/self collision at sample "
            f"{int(np.flatnonzero(~feasible)[0])}")

    release_key = validate_se3(
        planner.fk_wrist(held_path[-1]) @ np.linalg.inv(held_relation),
        name="reverse release key pose")
    scene = deepcopy(trial_scene)
    target = scene["mesh"]["target"]
    mesh_offset = np.linalg.inv(original) @ cart2se3(
        np.asarray(target["pose"], dtype=np.float64))
    target["pose"] = se32cart(release_key @ mesh_offset).tolist()
    retract = planner.plan_js_to_init(
        scene, held_path[-1, :7],
        start_hand_qpos=np.asarray(INSPIRE_INIT),
        goal_arm_qpos=np.asarray(FR3_CLEAR_VIEW))
    if retract is None:
        raise ValueError("reverse release has no open-hand clear-view path")
    retract = _joint_path(retract, "open-hand clear-view retract")
    audit = {
        "held_sample_count": len(held_path),
        "segments": {name: len(path) for name, path in parts},
        "arm_world_self_feasible_count": int(feasible.sum()),
        "sampled_key_hand_world_distances_m": distances,
        "geometry_hashes": hashes,
        "clear_view_sample_count": len(retract),
        "scope": "sampled_offline_plan_not_executed_drop_or_contact",
    }
    return HeldReverseRoute(held_path, release_key, scene, retract, audit)
