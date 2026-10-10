"""Compose unchanged AutoDex planning primitives into an insertion preflight.

The caller has already selected one v8 grasp and obtained AutoDex's pickup
``PlanResult`` for the fresh key scene. This module does *no* robot I/O. It
replans the lift at the specified held Inspire pose, plans transfer and short
socket-axis waypoints with the existing cuRobo planner, and audits the held
key/hand along those dense trajectories. There is no guarded contact control,
between-sample swept proof, release or physical success claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from autodex.utils.conversion import cart2se3

from .geometry import pose_angle_deg, validate_se3
from .path_audit import PathAuditLimits, audit_held_joint_paths
from .targets import InsertionTargets


@dataclass(frozen=True)
class InsertionPreflight:
    status: str
    lift_trajectory: np.ndarray | None
    transfer_trajectory: np.ndarray | None
    axial_trajectory: np.ndarray | None
    sampled_held_path_audit: dict[str, Any] | None
    axial_waypoint_count: int
    held_hand_q: np.ndarray
    held_hand_source: str
    planner_query_records: tuple[dict[str, Any], ...]

    @property
    def sampled_planning_pass(self) -> bool:
        return self.status == "sampled_planning_pass"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "precision_insertion_planning_preflight_v1",
            "status": self.status,
            "sampled_planning_pass": self.sampled_planning_pass,
            "axial_waypoint_count": self.axial_waypoint_count,
            "held_hand_q": self.held_hand_q.tolist(),
            "held_hand_source": self.held_hand_source,
            "planner_query_records": list(self.planner_query_records),
            "sample_counts": {
                name: None if path is None else int(len(path))
                for name, path in (
                    ("lift", self.lift_trajectory),
                    ("transfer", self.transfer_trajectory),
                    ("axial", self.axial_trajectory),
                )
            },
            "sampled_held_path_audit": self.sampled_held_path_audit,
            "not_validated": [
                "measured Inspire joint state or grasp rigidity after pickup",
                "initial key/table support contact during squeeze",
                "executor replay contract for this separately replanned held lift",
                "continuous swept geometry between FK samples",
                "guarded force/contact insertion and physical task success",
            ],
            "robot_ready": False,
        }


class HeldHandPoseDrift(ValueError):
    """A nominal planner path changed the hand that must hold the key."""

    def __init__(self, name: str, max_abs_delta_rad: float):
        self.max_abs_delta_rad = max_abs_delta_rad
        super().__init__(
            f"{name} does not keep the held Inspire pose "
            f"(max_abs_delta_rad={max_abs_delta_rad:.6g})")


def _path(value: Any, name: str, start_q: np.ndarray,
          held_q: np.ndarray) -> np.ndarray:
    path = np.asarray(value, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 13 or len(path) < 2:
        raise ValueError(f"{name} must be a nonempty dense 13-DOF path")
    if not np.all(np.isfinite(path)):
        raise ValueError(f"{name} contains non-finite joints")
    if not np.allclose(path[0], start_q, atol=1e-4, rtol=0):
        delta = float(np.max(np.abs(path[0] - start_q)))
        raise ValueError(
            f"{name} does not start at the previous joint state "
            f"(max_abs_delta_rad={delta:.6g})")
    if not np.allclose(path[:, 7:], held_q, atol=1e-4, rtol=0):
        delta = float(np.max(np.abs(path[:, 7:] - held_q)))
        raise HeldHandPoseDrift(name, delta)
    return path


def _goal_met(actual: np.ndarray, expected: np.ndarray,
              limits: PathAuditLimits) -> bool:
    return (np.linalg.norm(actual[:3, 3] - expected[:3, 3]) <=
            limits.goal_position_tolerance_m and
            pose_angle_deg(actual, expected) <=
            limits.goal_rotation_tolerance_deg)


def plan_insertion_after_pickup(
    *, planner, pickup_plan, trial_scene: dict,
    shared_root: Path, calibration, targets: InsertionTargets,
    held_hand_q: np.ndarray, held_hand_source: str,
    limits: PathAuditLimits, axial_waypoint_step_m: float,
) -> InsertionPreflight:
    """Attempt one grasp's nominal lift, transfer and 20 mm axial preflight.

    ``pickup_plan`` must be the unchanged AutoDex planner's successful
    approach/lift result for this exact fresh key scene and a *single* already
    endpoint-screened v8 candidate. ``held_hand_q`` is the hand configuration
    modeled throughout lift and transfer; label its provenance truthfully.
    Supplying ``commanded_nominal`` never proves the physical grip.

    A failed cuRobo query returns a stage-specific status. CUDA/solver faults,
    stale assets or contradictory frames raise: they must not be converted
    into ordinary infeasibility or a fallback robot motion.
    """
    limits.validate()
    step = float(axial_waypoint_step_m)
    if not math.isfinite(step) or not 0 < step <= 0.005:
        raise ValueError("axial waypoint step must be finite and <= 5 mm")
    held = np.asarray(held_hand_q, dtype=np.float64)
    if held.shape != (6,) or not np.all(np.isfinite(held)):
        raise ValueError("held_hand_q must be six finite Inspire joints")
    if held_hand_source not in {"measured", "commanded_nominal"}:
        raise ValueError("held_hand_source must be measured or commanded_nominal")
    if (getattr(planner, "_n_arm", None) != 7 or
            getattr(planner, "_hand", None) != "fr3_inspire" or
            getattr(planner, "_robot_cfg", {}).get(
                "kinematics", {}).get("ee_link") != "base_link"):
        raise ValueError("preflight requires the original FR3/Inspire base_link planner")
    if not isinstance(trial_scene, dict) or not isinstance(
            trial_scene.get("mesh"), dict):
        raise ValueError("trial scene is missing the key/socket collision meshes")
    target_mesh = trial_scene["mesh"].get("target")
    if not isinstance(target_mesh, dict):
        raise ValueError("fresh trial key target is missing")
    frozen = calibration.collision_scene
    if (trial_scene.get("cuboid") != frozen.get("cuboid") or
            {name: value for name, value in trial_scene["mesh"].items()
             if name != "target"} != frozen.get("mesh")):
        raise ValueError("trial scene differs from frozen socket/table world")

    key_pose = validate_se3(cart2se3(np.asarray(target_mesh["pose"], dtype=float)),
                            name="fresh T_robot_key")
    if not pickup_plan.success or pickup_plan.lift_preflight is None:
        raise ValueError("AutoDex pickup did not pass approach and lift preflight")
    expected_wrist = key_pose @ targets.T_key_hand
    requested_wrist = validate_se3(pickup_plan.wrist_se3,
                                   name="selected pickup wrist")
    if not _goal_met(requested_wrist, expected_wrist, limits):
        raise ValueError("pickup wrist does not match fixed key/hand grasp")
    pickup = np.asarray(pickup_plan.traj, dtype=np.float64)
    if (pickup.ndim != 2 or pickup.shape[1] != 13 or len(pickup) < 2 or
            not np.all(np.isfinite(pickup))):
        raise ValueError("AutoDex pickup has no valid 13-DOF trajectory")

    # Original plan() models the grasp-pose hand for its lift, but the
    # executor may continue squeezing after grasp. Replan from the same arm
    # endpoint with the explicitly selected held hand state; never silently
    # reuse a lift checked with different finger geometry.
    lift_start = pickup[-1].copy()
    lift_start[7:] = held
    if not _goal_met(validate_se3(planner.fk_wrist(lift_start),
                                  name="post-grasp FK"), requested_wrist, limits):
        raise ValueError("post-grasp FK disagrees with the selected wrist")

    query_records: list[dict[str, Any]] = []

    def result(status: str, *, lift=None, transfer=None, axial=None,
               audit=None, waypoints=0) -> InsertionPreflight:
        return InsertionPreflight(status, lift, transfer, axial, audit,
                                  waypoints, held.copy(), held_hand_source,
                                  tuple(query_records))

    lift_plan = planner.plan_lift_preflight(lift_start, trial_scene, lift_h=0.10)
    query_records.append({"stage": "held_lift", "success": lift_plan is not None,
                          "planner_api": "plan_lift_preflight"})
    if lift_plan is None:
        return result("held_lift_unreachable")
    try:
        lift = _path(lift_plan.traj, "held lift", lift_start, held)
    except HeldHandPoseDrift as exc:
        query_records[-1]["held_hand_lock_verified"] = False
        query_records[-1]["max_abs_hand_delta_rad"] = exc.max_abs_delta_rad
        return result("held_lift_hand_drift")
    lift_goal = validate_se3(planner.fk_wrist(lift[-1]), name="lift endpoint FK")
    expected_lift = requested_wrist.copy()
    expected_lift[2, 3] += 0.10
    if not _goal_met(lift_goal, expected_lift, limits):
        raise ValueError("held lift does not reach AutoDex's 10 cm lift target")

    return plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial_scene, shared_root=shared_root,
        calibration=calibration, targets=targets, start_q=lift[-1],
        held_hand_q=held, held_hand_source=held_hand_source, limits=limits,
        axial_waypoint_step_m=step, lift_trajectory=lift,
        prior_query_records=tuple(query_records))


def plan_held_transfer_and_axial(
    *, planner, trial_scene: dict, shared_root: Path, calibration,
    targets: InsertionTargets, start_q: np.ndarray,
    held_hand_q: np.ndarray, held_hand_source: str,
    limits: PathAuditLimits, axial_waypoint_step_m: float,
    lift_trajectory: np.ndarray | None = None,
    prior_query_records: tuple[dict[str, Any], ...] = (),
) -> InsertionPreflight:
    """Plan from a measured held state to preinsert and nominal 20 mm pose.

    Initial trials call this after the held lift; bounded XY retries call it
    after confirmed axial withdrawal. Both reuse the same cuRobo queries and
    full-key/whole-hand sampled audit. This never executes contact motion.
    """
    limits.validate()
    step = float(axial_waypoint_step_m)
    if not math.isfinite(step) or not 0 < step <= 0.005:
        raise ValueError("axial waypoint step must be finite and <= 5 mm")
    held = np.asarray(held_hand_q, dtype=np.float64)
    if held.shape != (6,) or not np.all(np.isfinite(held)):
        raise ValueError("held_hand_q must be six finite Inspire joints")
    if held_hand_source not in {"measured", "commanded_nominal"}:
        raise ValueError("held_hand_source must be measured or commanded_nominal")
    if (getattr(planner, "_n_arm", None) != 7 or
            getattr(planner, "_hand", None) != "fr3_inspire" or
            getattr(planner, "_robot_cfg", {}).get(
                "kinematics", {}).get("ee_link") != "base_link"):
        raise ValueError("preflight requires the original FR3/Inspire base_link planner")
    frozen = calibration.collision_scene
    if (not isinstance(trial_scene, dict) or
            trial_scene.get("cuboid") != frozen.get("cuboid") or
            not isinstance(trial_scene.get("mesh"), dict) or
            {name: value for name, value in trial_scene["mesh"].items()
             if name != "target"} != frozen.get("mesh")):
        raise ValueError("trial scene differs from frozen socket/table world")
    start = np.asarray(start_q, dtype=np.float64)
    if (start.shape != (13,) or not np.all(np.isfinite(start)) or
            not np.allclose(start[7:], held, atol=1e-4, rtol=0)):
        raise ValueError("held transfer start must be finite 13-DOF with fixed hand")
    if lift_trajectory is not None:
        lift = np.asarray(lift_trajectory, dtype=np.float64)
        if not np.allclose(lift[-1], start, atol=1e-4, rtol=0):
            raise ValueError("held lift does not end at transfer start")
    else:
        lift = None
    query_records: list[dict[str, Any]] = list(prior_query_records)

    def result(status: str, *, transfer=None, axial=None,
               audit=None, waypoints=0) -> InsertionPreflight:
        return InsertionPreflight(
            status, lift, transfer, axial, audit, waypoints, held.copy(),
            held_hand_source, tuple(query_records))

    transfer_result = planner.plan_cartesian_pose(
        start, targets.T_robot_hand_preinsert,
        scene_cfg=trial_scene, include_obj_obstacle=False,
        return_result=True, lock_hand=True,
        timing_phase="precision_insertion_preflight")
    query_records.append({
        "stage": "transfer", "success": bool(transfer_result.success),
        "planner_api": "plan_cartesian_pose",
        "constraint_mode": getattr(transfer_result, "constraint_mode", None),
        "failure_stage": getattr(transfer_result, "failure_stage", None),
    })
    if not transfer_result.success or transfer_result.trajectory is None:
        return result("transfer_unreachable")
    try:
        transfer = _path(transfer_result.trajectory, "held transfer", start, held)
    except HeldHandPoseDrift as exc:
        query_records[-1]["held_hand_lock_verified"] = False
        query_records[-1]["max_abs_hand_delta_rad"] = exc.max_abs_delta_rad
        return result("transfer_hand_drift")
    if not _goal_met(validate_se3(planner.fk_wrist(transfer[-1]),
                                  name="transfer endpoint FK"),
                     targets.T_robot_hand_preinsert, limits):
        return result("transfer_goal_residual", transfer=transfer)

    travel = targets.preinsert_clearance_m + targets.mode.target_depth_m
    count = int(math.ceil(travel / step))
    if count < 1:
        raise ValueError("insertion travel is invalid")
    segments: list[np.ndarray] = []
    current = transfer[-1].copy()
    for index in range(1, count + 1):
        fraction = index / count
        goal = targets.T_robot_hand_preinsert.copy()
        goal[:3, 3] += fraction * travel * targets.insertion_axis_robot
        query = planner.plan_cartesian_pose(
            current, goal, scene_cfg=trial_scene,
            include_obj_obstacle=False, return_result=True, lock_hand=True,
            timing_phase="precision_insertion_axial_preflight")
        query_records.append({
            "stage": "axial_waypoint", "index": index,
            "success": bool(query.success),
            "planner_api": "plan_cartesian_pose",
            "constraint_mode": getattr(query, "constraint_mode", None),
            "failure_stage": getattr(query, "failure_stage", None),
            "T_robot_hand_goal": goal.tolist(),
        })
        if not query.success or query.trajectory is None:
            return result("axial_waypoint_unreachable", transfer=transfer,
                          waypoints=index - 1)
        try:
            segment = _path(query.trajectory, f"axial waypoint {index}",
                            current, held)
        except HeldHandPoseDrift as exc:
            query_records[-1]["held_hand_lock_verified"] = False
            query_records[-1]["max_abs_hand_delta_rad"] = exc.max_abs_delta_rad
            return result("axial_waypoint_hand_drift", transfer=transfer,
                          waypoints=index - 1)
        if not _goal_met(validate_se3(planner.fk_wrist(segment[-1]),
                                      name=f"axial FK[{index}]"), goal, limits):
            return result("axial_waypoint_goal_residual", transfer=transfer,
                          waypoints=index - 1)
        segments.append(segment[1:])
        current = segment[-1].copy()
    axial = np.concatenate([transfer[-1:], *segments], axis=0)
    sampled = audit_held_joint_paths(
        shared_root=shared_root, calibration=calibration, targets=targets,
        planner=planner, transfer_trajectory=transfer,
        descent_trajectory=axial, held_hand_q=held, limits=limits,
        lift_trajectory=lift)
    if not sampled["sampled_clear"]:
        return result("sampled_held_path_rejected",
                      transfer=transfer, axial=axial, audit=sampled,
                      waypoints=count)
    return result("sampled_planning_pass", transfer=transfer,
                  axial=axial, audit=sampled, waypoints=count)
