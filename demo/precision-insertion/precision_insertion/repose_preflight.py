"""Compose unchanged Franka planning primitives for a held-key reset path.

This module plans only through the *still-grasped* release waypoint. It does
not open the hand, predict the drop, retreat, command motors, or label a reset
successful. The original AutoDex reset planner's empty-mesh lift scene is
not reused because that removes the fixed precision-insertion socket.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from autodex.utils.conversion import cart2se3
from autodex.utils.tabletop_geometry import table_surface_z

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import _coal_mesh, _load_mesh
from .geometry import validate_se3
from .path_audit import (
    PathAuditLimits, _fixed_world_models, _held_pair_report,
)
from .preflight import _goal_met, _path
from .repose_path_audit import audit_repose_held_paths
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


@dataclass(frozen=True)
class ReposeHeldPreflight:
    status: str
    lift_trajectory: np.ndarray | None
    transfer_trajectory: np.ndarray | None
    descent_trajectory: np.ndarray | None
    sampled_held_path_audit: dict | None
    planner_query_records: tuple[dict[str, Any], ...]
    T_robot_key_rest: np.ndarray
    release_height_m: float
    held_hand_q: np.ndarray
    held_hand_source: str

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_repose_held_preflight_v1",
            "status": self.status,
            "sampled_held_path_pass": (
                self.status == "sampled_held_path_pass_release_unplanned"),
            "planner_query_records": list(self.planner_query_records),
            "sample_counts": {
                name: None if path is None else len(path)
                for name, path in (
                    ("lift", self.lift_trajectory),
                    ("transfer", self.transfer_trajectory),
                    ("descent", self.descent_trajectory),
                )
            },
            "T_robot_key_rest": self.T_robot_key_rest.tolist(),
            "release_height_m": self.release_height_m,
            "held_hand_q": self.held_hand_q.tolist(),
            "held_hand_source": self.held_hand_source,
            "sampled_held_path_audit": self.sampled_held_path_audit,
            "not_validated": [
                "observed post-squeeze key-in-hand pose and controller tracking",
                "open-hand release, retreat and falling-key landing pose",
                "tabletop workspace/board-boundary coverage at release",
                "continuous swept geometry between sampled held-key poses",
                "physical reset or insertion success",
            ],
            "robot_ready": False,
        }


def build_v8_repose_rest_pose(
    *, shared_root: Path, mode: TaskMode, calibration,
    target_pose_stem: str, release_xy_robot_m: tuple[float, float],
    asset_support_tolerance_m: float,
) -> np.ndarray:
    """Place one genuine v8 tabletop pose at an explicitly chosen XY site.

    v8 tabletop transforms are expressed relative to a z=0 support plane.
    Preserve their exact orientation and local vertical offset, then add
    only the session's measured table height. The caller must independently
    select a location inside the usable board area and clear of the socket.
    """
    stem = str(target_pose_stem)
    if not stem.isdigit() or stem != f"{int(stem):03d}":
        raise ValueError("target pose stem must be a three-digit v8 ID")
    xy = np.asarray(release_xy_robot_m, dtype=np.float64)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("release XY must be two finite robot-frame meters")
    tolerance = float(asset_support_tolerance_m)
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("asset support tolerance must be positive")
    assets = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    pose_path = assets.key_tabletop_dir / f"{stem}.npy"
    if not pose_path.is_file():
        raise FileNotFoundError(f"v8 tabletop pose missing: {pose_path}")
    rest = validate_se3(np.load(pose_path, allow_pickle=False),
                        name=f"v8 tabletop pose {stem}")
    mesh = _load_mesh(assets.raw_mesh(mode.key_object))
    local_bottom = float((mesh.vertices @ rest[:3, :3].T +
                          rest[:3, 3])[:, 2].min())
    if abs(local_bottom) > tolerance:
        raise ValueError("v8 tabletop asset does not rest on its z=0 plane")
    rest[:2, 3] = xy
    rest[2, 3] += float(table_surface_z(calibration.board))
    return rest


def validate_repose_rest_target(
    *, shared_root: Path, mode: TaskMode, calibration,
    T_robot_key_rest: np.ndarray, support_tolerance_m: float,
    minimum_rest_socket_clearance_m: float,
) -> dict:
    """Fail closed on a floating/buried target or a key resting on socket.

    The caller still owns board-boundary/fixture-footprint selection and
    post-release observation; table support alone is not a safe drop policy.
    """
    tolerance = float(support_tolerance_m)
    clearance = float(minimum_rest_socket_clearance_m)
    if (not math.isfinite(tolerance) or tolerance <= 0 or
            not math.isfinite(clearance) or clearance <= 0):
        raise ValueError("support tolerance and socket clearance must be positive")
    rest = validate_se3(T_robot_key_rest, name="reset target tabletop key")
    assets = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    key_mesh = _load_mesh(assets.raw_mesh(mode.key_object))
    if not key_mesh.is_watertight:
        raise ValueError("reset key CAD is not watertight")
    vertices = key_mesh.vertices @ rest[:3, :3].T + rest[:3, 3]
    min_z = float(vertices[:, 2].min())
    table_z = float(table_surface_z(calibration.board))
    if abs(min_z - table_z) > tolerance:
        raise ValueError("reset target key bottom does not match measured table")
    fixed, _ = _fixed_world_models(calibration)
    socket_model, socket_pose, socket_mesh, occupancy = fixed["mesh/fixture_socket"]
    relative = np.linalg.inv(socket_pose) @ rest
    result = _held_pair_report(
        _coal_mesh(key_mesh), key_mesh, socket_model, socket_mesh,
        occupancy, relative)
    distance = float(result["minimum_surface_distance_m"])
    if result["colliding"] or distance < clearance:
        raise ValueError("reset target key collides with or approaches socket")
    return {"key_bottom_z_m": min_z, "table_surface_z_m": table_z,
            "key_socket_clearance_m": distance,
            "minimum_required_socket_clearance_m": clearance}


def plan_repose_held_chain(
    *, planner, pickup_plan, trial_scene: dict, shared_root: Path,
    calibration, mode: TaskMode, T_key_hand: np.ndarray,
    T_robot_key_rest: np.ndarray, release_height_m: float,
    minimum_rest_socket_clearance_m: float,
    held_hand_q: np.ndarray, held_hand_source: str,
    limits: PathAuditLimits, lift_height_m: float = 0.10,
    preplace_vertical_travel_m: float = 0.10,
) -> ReposeHeldPreflight:
    """Plan pickup-end → held lift → high transfer → straight-down release.

    ``pickup_plan`` is the unchanged AutoDex v8 planner result for one exact
    reset seed and the fresh key scene. The selected key-to-hand transform is
    held fixed in the nominal plan, then audited against full CAD and the
    frozen socket. A measured post-lift relation is required before any real
    transfer; this planner-only result never grants motor authorization.
    """
    limits.validate()
    if (getattr(planner, "_n_arm", None) != 7 or
            getattr(planner, "_hand", None) != "fr3_inspire" or
            getattr(planner, "_robot_cfg", {}).get(
                "kinematics", {}).get("ee_link") != "base_link"):
        raise ValueError("repose preflight needs the original FR3/Inspire planner")
    if held_hand_source not in {"measured", "commanded_nominal"}:
        raise ValueError("held hand source must be measured or commanded_nominal")
    held = np.asarray(held_hand_q, dtype=np.float64)
    if held.shape != (6,) or not np.all(np.isfinite(held)):
        raise ValueError("held_hand_q must be six finite Inspire joints")
    height = float(release_height_m)
    travel = float(preplace_vertical_travel_m)
    lift_h = float(lift_height_m)
    if (not math.isfinite(height) or height <= 0 or
            not math.isfinite(travel) or travel <= 0 or
            not math.isfinite(lift_h) or lift_h <= 0):
        raise ValueError("lift, preplace travel and release height must be positive")
    key_hand = validate_se3(T_key_hand, name="reset T_key_hand")
    rest = validate_se3(T_robot_key_rest, name="reset tabletop key target")
    root = Path(shared_root).expanduser().resolve()
    validated_frozen_socket_pose(mode=mode, shared_root=root,
                                 calibration=calibration)
    carried_scene = build_held_scene_from_trial(
        trial_scene=trial_scene, calibration=calibration)
    target = trial_scene["mesh"]["target"]
    initial = validate_se3(cart2se3(np.asarray(target["pose"], dtype=float)),
                           name="fresh reset T_robot_key")
    validate_repose_rest_target(
        shared_root=root, mode=mode, calibration=calibration,
        T_robot_key_rest=rest,
        support_tolerance_m=limits.goal_position_tolerance_m,
        minimum_rest_socket_clearance_m=minimum_rest_socket_clearance_m)
    if (not pickup_plan.success or pickup_plan.lift_preflight is None or
            pickup_plan.traj is None):
        raise ValueError("AutoDex reset pickup did not pass approach/lift preflight")
    pickup = np.asarray(pickup_plan.traj, dtype=np.float64)
    if (pickup.ndim != 2 or pickup.shape[1] != 13 or len(pickup) < 2 or
            not np.all(np.isfinite(pickup))):
        raise ValueError("AutoDex reset pickup has no valid 13-DOF path")
    requested_wrist = validate_se3(pickup_plan.wrist_se3,
                                   name="selected reset pickup wrist")
    if not _goal_met(requested_wrist, initial @ key_hand, limits):
        raise ValueError("reset pickup wrist differs from fresh key/seed")
    lift_start = pickup[-1].copy()
    lift_start[7:] = held
    if not _goal_met(validate_se3(planner.fk_wrist(lift_start),
                                  name="reset post-grasp FK"),
                     requested_wrist, limits):
        raise ValueError("reset post-grasp FK differs from selected wrist")

    queries: list[dict[str, Any]] = []

    def result(status: str, *, lift=None, transfer=None, descent=None,
               audit=None) -> ReposeHeldPreflight:
        return ReposeHeldPreflight(
            status, lift, transfer, descent, audit, tuple(queries), rest.copy(),
            height, held.copy(), held_hand_source)

    lift_plan = planner.plan_lift_preflight(
        lift_start, trial_scene, lift_h=lift_h)
    queries.append({"stage": "held_lift", "planner_api": "plan_lift_preflight",
                    "success": lift_plan is not None})
    if lift_plan is None:
        return result("held_lift_unreachable")
    lift = _path(lift_plan.traj, "repose held lift", lift_start, held)
    expected_lift = requested_wrist.copy()
    expected_lift[2, 3] += lift_h
    if not _goal_met(validate_se3(planner.fk_wrist(lift[-1]),
                                  name="reset lift FK"), expected_lift, limits):
        return result("held_lift_goal_residual", lift=lift)

    release_key = rest.copy()
    release_key[2, 3] += height
    preplace_key = release_key.copy()
    preplace_key[2, 3] += travel
    release_wrist = release_key @ key_hand
    preplace_wrist = preplace_key @ key_hand
    transfer_plan = planner.plan_cartesian_pose(
        lift[-1], preplace_wrist, scene_cfg=carried_scene,
        include_obj_obstacle=False, return_result=True, lock_hand=True,
        timing_phase="precision_insertion_repose_preflight")
    queries.append({
        "stage": "held_transfer", "planner_api": "plan_cartesian_pose",
        "success": bool(transfer_plan.success),
        "constraint_mode": getattr(transfer_plan, "constraint_mode", None),
        "failure_stage": getattr(transfer_plan, "failure_stage", None),
    })
    if not transfer_plan.success or transfer_plan.trajectory is None:
        return result("held_transfer_unreachable", lift=lift)
    transfer = _path(transfer_plan.trajectory, "repose held transfer",
                     lift[-1], held)
    if not _goal_met(validate_se3(planner.fk_wrist(transfer[-1]),
                                  name="reset preplace FK"),
                     preplace_wrist, limits):
        return result("held_transfer_goal_residual", lift=lift,
                      transfer=transfer)

    stroke = planner.plan_vertical_stroke(
        transfer[-1], preplace_wrist, release_wrist,
        expected_travel_m=travel, travel_tolerance_m=1e-5,
        scene_cfg=trial_scene, include_obj_obstacle=False,
        attached_object_pose_at_start=(
            validate_se3(planner.fk_wrist(transfer[-1]),
                         name="reset preplace actual FK") @
            np.linalg.inv(key_hand)),
        label="precision insertion repose held descent",
        timing_phase="precision_insertion_repose_preflight",
        return_result=True)
    queries.append({
        "stage": "held_descent", "planner_api": "plan_vertical_stroke",
        "success": bool(stroke.success),
        "failure_code": getattr(stroke, "failure_code", None),
    })
    if not stroke.success or stroke.trajectory is None:
        return result("held_descent_unreachable", lift=lift,
                      transfer=transfer)
    descent = _path(stroke.trajectory, "repose held descent",
                    transfer[-1], held)
    if not _goal_met(validate_se3(planner.fk_wrist(descent[-1]),
                                  name="reset release FK"),
                     release_wrist, limits):
        return result("held_descent_goal_residual", lift=lift,
                      transfer=transfer, descent=descent)

    audit = audit_repose_held_paths(
        shared_root=root, mode=mode, calibration=calibration, planner=planner,
        lift_trajectory=lift, transfer_trajectory=transfer,
        descent_trajectory=descent, held_hand_q=held, T_key_hand=key_hand,
        T_robot_key_initial=initial, T_robot_key_rest=rest,
        release_height_m=height, limits=limits)
    if not audit["sampled_clear"]:
        return result("sampled_held_path_rejected", lift=lift,
                      transfer=transfer, descent=descent, audit=audit)
    return result("sampled_held_path_pass_release_unplanned", lift=lift,
                  transfer=transfer, descent=descent, audit=audit)
