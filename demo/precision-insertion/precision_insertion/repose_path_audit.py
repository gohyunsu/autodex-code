"""Sampled held-key audit for a socket-aware repose/reorient plan.

This is a diagnostic on *planned* FR3/Inspire joint paths. It does not
execute a reset, model a falling key, or certify continuous swept collision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import _coal_mesh, _hand_link_meshes, _load_mesh
from .geometry import pose_angle_deg, validate_se3
from .path_audit import (
    PathAuditLimits, _array_sha256, _fixed_world_models, _held_pair_report,
    _joint_path,
    _poses_match, _sha256,
)


def audit_repose_held_paths(
    *, shared_root: Path, mode: TaskMode, calibration, planner,
    lift_trajectory: np.ndarray, transfer_trajectory: np.ndarray,
    descent_trajectory: np.ndarray, held_hand_q: np.ndarray,
    T_key_hand: np.ndarray, T_robot_key_initial: np.ndarray,
    T_robot_key_rest: np.ndarray, release_height_m: float,
    limits: PathAuditLimits,
) -> dict:
    """Check one fixed-grasp lift/transfer/descent against table and socket.

    The release goal is the caller's selected tabletop key pose translated
    vertically by ``release_height_m``. A pass is *not* a reset authorization:
    the post-lift relative key pose, open-hand release, retreat, and actual
    landing pose require separate checks.
    """
    limits.validate()
    height = float(release_height_m)
    if not np.isfinite(height) or height <= 0:
        raise ValueError("release height must be finite and positive")
    held = np.asarray(held_hand_q, dtype=np.float64)
    if held.shape != (6,) or not np.all(np.isfinite(held)):
        raise ValueError("held_hand_q must be six finite Inspire joints")
    key_hand = validate_se3(T_key_hand, name="reset T_key_hand")
    initial = validate_se3(T_robot_key_initial, name="initial T_robot_key")
    rest = validate_se3(T_robot_key_rest, name="target tabletop T_robot_key")
    release = rest.copy()
    release[2, 3] += height
    stages = {
        "lift": _joint_path(lift_trajectory, "repose lift", held),
        "transfer": _joint_path(transfer_trajectory, "repose transfer", held),
        "descent": _joint_path(descent_trajectory, "repose descent", held),
    }
    if (not np.allclose(stages["lift"][-1], stages["transfer"][0],
                        atol=1e-4, rtol=0) or
            not np.allclose(stages["transfer"][-1], stages["descent"][0],
                            atol=1e-4, rtol=0)):
        raise ValueError("repose held trajectories are discontinuous")
    wrist = {
        name: [validate_se3(planner.fk_wrist(q), name=f"{name} FK[{i}]")
               for i, q in enumerate(path)]
        for name, path in stages.items()
    }
    if not _poses_match(wrist["lift"][0], initial @ key_hand, limits):
        raise ValueError("repose grasp wrist differs from observed key/seed")

    failures = []
    final_key = wrist["descent"][-1] @ np.linalg.inv(key_hand)
    if not _poses_match(final_key, release, limits):
        failures.append({"stage": "descent", "sample": len(stages["descent"]) - 1,
                         "reason": "release_goal_residual",
                         "position_error_m": float(np.linalg.norm(
                             final_key[:3, 3] - release[:3, 3])),
                         "rotation_error_deg": pose_angle_deg(final_key, release)})
    for stage, path in stages.items():
        for i in range(1, len(path)):
            joint_step = float(np.max(np.abs(path[i] - path[i - 1])))
            wrist_step = float(np.linalg.norm(
                wrist[stage][i][:3, 3] - wrist[stage][i - 1][:3, 3]))
            angle = pose_angle_deg(wrist[stage][i], wrist[stage][i - 1])
            if (joint_step > limits.max_joint_step_rad or
                    wrist_step > limits.max_wrist_step_m or
                    angle > limits.max_wrist_rotation_deg):
                failures.append({"stage": stage, "sample": i,
                                 "reason": "path_sampling_too_sparse",
                                 "joint_step_rad": joint_step,
                                 "wrist_step_m": wrist_step,
                                 "wrist_rotation_deg": angle})
        if stage in ("lift", "descent"):
            direction = 1 if stage == "lift" else -1
            prior = float(wrist[stage][0][2, 3])
            for i, pose in enumerate(wrist[stage][1:], start=1):
                z = float(pose[2, 3])
                if direction * (z - prior) < -limits.goal_position_tolerance_m:
                    failures.append({"stage": stage, "sample": i,
                                     "reason": "nonmonotone_vertical_stroke",
                                     "previous_z_m": prior, "z_m": z})
                prior = z

    fixed, fixed_hashes = _fixed_world_models(
        calibration, mode=mode, shared_root=shared_root)
    assets = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    key_path = assets.raw_mesh(mode.key_object)
    urdf = assets.robot_urdf
    if not key_path.is_file() or not urdf.is_file():
        raise FileNotFoundError("full key CAD and Franka/Inspire URDF are required")
    key_mesh = _load_mesh(key_path)
    if not key_mesh.is_watertight:
        raise ValueError("full key CAD mesh is not watertight")
    moving = {"key": (_coal_mesh(key_mesh), key_mesh)}
    moving.update({f"hand/{name}": (_coal_mesh(mesh), mesh)
                   for name, mesh in _hand_link_meshes(urdf, held).items()})
    key_in_hand = np.linalg.inv(key_hand)
    minimum_distances = {}
    for stage, poses in wrist.items():
        for i, hand_pose in enumerate(poses):
            key_pose = hand_pose @ key_in_hand
            for moving_name, (model, moving_mesh) in moving.items():
                moving_pose = key_pose if moving_name == "key" else hand_pose
                for fixed_name, (obstacle, obstacle_pose, fixed_mesh,
                                 occupancy) in fixed.items():
                    if (stage == "lift" and i == 0 and moving_name == "key"
                            and fixed_name == "cuboid/table"):
                        continue  # Initial tabletop support contact only.
                    relative = np.linalg.inv(obstacle_pose) @ moving_pose
                    report = _held_pair_report(
                        model, moving_mesh, obstacle, fixed_mesh,
                        occupancy, relative)
                    pair = f"{moving_name}->{fixed_name}"
                    distance = report["minimum_surface_distance_m"]
                    minimum_distances[pair] = min(
                        distance, minimum_distances.get(pair, float("inf")))
                    required = (limits.minimum_hand_clearance_m
                                if moving_name != "key" else 0.0)
                    if report["colliding"] or distance < required:
                        failures.append({"stage": stage, "sample": i,
                                         "reason": "held_geometry_collision_or_clearance",
                                         "moving": moving_name,
                                         "obstacle": fixed_name,
                                         "colliding": report["colliding"],
                                         "distance_m": distance,
                                         "required_clearance_m": required})
    return {
        "schema": "precision_insertion_sampled_repose_held_path_audit_v1",
        "scope": "sampled_full_key_and_inspire_links_vs_frozen_socket_and_table",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "sample_counts": {name: len(path) for name, path in stages.items()},
        "release_height_m": height,
        "limits": vars(limits).copy(),
        "sampled_clear": not failures,
        "failures": failures,
        "minimum_surface_distances_m": minimum_distances,
        "input_sha256": {
            "key_mesh": _sha256(key_path), "robot_urdf": _sha256(urdf),
            "session_calibration_record": hashlib.sha256(json.dumps(
                calibration.record, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "fixed_collision_scene": hashlib.sha256(json.dumps(
                calibration.collision_scene, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "T_key_hand": _array_sha256(key_hand),
            "T_robot_key_initial": _array_sha256(initial),
            "T_robot_key_rest": _array_sha256(rest),
            "held_hand_q": _array_sha256(held),
            **{f"{name}_trajectory": _array_sha256(path)
               for name, path in stages.items()}, **fixed_hashes,
        },
        "not_validated": [
            "continuous swept geometry between FK samples",
            "actual post-squeeze key-in-hand pose and grasp stability",
            "open-hand release, retreat and falling-key landing pose",
            "independent cuRobo arm/self collision and controller tracking",
        ],
        "robot_ready": False,
    }
