"""Read-only sampled collision audit of a held key on planned FR3 paths.

This complements, but does not replace, the unchanged cuRobo arm/world checks.
Input trajectories must be the actual planned 13-DOF paths, not interpolated
goal poses. Sampling never proves swept-volume or force/contact safety between
samples, and a passing report is never a robot motion authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from autodex.utils.conversion import cart2se3
from autodex.utils.tabletop_geometry import table_cuboid

from .assets import AssetPaths
from .endpoint import (_coal_mesh, _coal_models_report, _hand_link_meshes,
                       _load_mesh)
from .geometry import pose_angle_deg, validate_se3
from .targets import InsertionTargets, build_rigid_insertion_targets


@dataclass(frozen=True)
class PathAuditLimits:
    max_joint_step_rad: float
    max_wrist_step_m: float
    max_wrist_rotation_deg: float
    goal_position_tolerance_m: float
    goal_rotation_tolerance_deg: float
    axial_lateral_tolerance_m: float
    axial_rotation_tolerance_deg: float
    minimum_hand_clearance_m: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _poses_match(actual: np.ndarray, expected: np.ndarray,
                 limits: PathAuditLimits) -> bool:
    return (np.linalg.norm(actual[:3, 3] - expected[:3, 3]) <=
            limits.goal_position_tolerance_m and
            pose_angle_deg(actual, expected) <=
            limits.goal_rotation_tolerance_deg)


def _joint_path(value: np.ndarray, name: str,
                held_hand_q: np.ndarray) -> np.ndarray:
    path = np.asarray(value, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 13 or len(path) < 2:
        raise ValueError(f"{name} must be a dense (N>=2, 13) FR3/Inspire path")
    if not np.all(np.isfinite(path)):
        raise ValueError(f"{name} contains non-finite joints")
    if not np.allclose(path[:, 7:], held_hand_q, atol=1e-4, rtol=0):
        raise ValueError(f"{name} changes Inspire joints while holding the key")
    return path


def _fixed_world_models(calibration) -> tuple[dict, dict]:
    """Build Coal models for the frozen robot-frame collision world."""
    import coal  # noqa: F401 -- must precede trimesh on the AutoDex host
    import trimesh

    scene = calibration.collision_scene
    if not isinstance(scene, dict) or not isinstance(scene.get("mesh"), dict):
        raise ValueError("session has no fixed mesh collision world")
    if not isinstance(scene.get("cuboid"), dict):
        raise ValueError("session has no measured table cuboid")
    if "target" in scene["mesh"]:
        raise ValueError("held key must not remain a fixed target obstacle")
    if scene["cuboid"].get("table") != table_cuboid(calibration.board):
        raise ValueError("table cuboid differs from frozen ChArUco measurement")
    models = {}
    source_hashes = {}
    for name, spec in sorted(scene["mesh"].items()):
        path = Path(spec["file_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"fixed scene mesh missing: {path}")
        pose = validate_se3(cart2se3(np.asarray(spec["pose"], dtype=float)),
                            name=f"fixed mesh {name} pose")
        mesh = _load_mesh(path)
        if name == "fixture_socket" and not mesh.is_watertight:
            raise ValueError("exact socket collision mesh is not watertight")
        models[f"mesh/{name}"] = (_coal_mesh(mesh), pose)
        source_hashes[f"mesh/{name}"] = _sha256(path)
    for name, spec in sorted(scene["cuboid"].items()):
        dims = np.asarray(spec["dims"], dtype=np.float64)
        if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
            raise ValueError(f"invalid fixed cuboid {name} dimensions")
        pose = validate_se3(cart2se3(np.asarray(spec["pose"], dtype=float)),
                            name=f"fixed cuboid {name} pose")
        models[f"cuboid/{name}"] = (
            _coal_mesh(trimesh.creation.box(extents=dims)), pose)
    if "mesh/fixture_socket" not in models:
        raise ValueError("frozen socket is absent from collision world")
    return models, source_hashes


def audit_held_joint_paths(
    *, shared_root: Path, calibration, targets: InsertionTargets,
    planner, transfer_trajectory: np.ndarray, descent_trajectory: np.ndarray,
    held_hand_q: np.ndarray, limits: PathAuditLimits,
) -> dict[str, Any]:
    """Audit FK samples of the actual planned transfer and axial approach.

    ``planner.fk_wrist(q)`` is the unchanged AutoDex planner FK; its cuRobo
    arm/world collision result must be checked independently by the caller.
    The hand-to-key transform and six Inspire joints are fixed at all samples.
    A tilted socket is supported *only for checking a supplied axis-aligned
    path*; this function does not create such a path or contact controller.
    """
    limits.validate()
    root = Path(shared_root).expanduser().resolve()
    mode = targets.mode
    paths = AssetPaths(root, mode)
    # Revalidate CAD hashes, frozen fixture and every target; a stale target
    # must not be silently checked against a newly modified socket/geometry.
    refreshed = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=targets.T_key_hand,
        xy_offset_socket_m=targets.xy_offset_socket_m)
    for name in ("T_robot_hand_preinsert", "T_robot_hand_entry",
                 "T_robot_hand_verification", "T_robot_key_verification"):
        if not np.allclose(getattr(refreshed, name), getattr(targets, name),
                           atol=1e-8):
            raise ValueError(f"stale insertion target: {name}")
    if refreshed.task_geometry_sha256 != targets.task_geometry_sha256:
        raise ValueError("task geometry changed since targets were constructed")
    if (refreshed.socket_collision_mesh_sha256 !=
            targets.socket_collision_mesh_sha256):
        raise ValueError("socket collision mesh changed since target construction")

    hand_q = np.asarray(held_hand_q, dtype=np.float64)
    if hand_q.shape != (6,) or not np.all(np.isfinite(hand_q)):
        raise ValueError("held_hand_q must be six finite Inspire joints")
    transfer = _joint_path(transfer_trajectory, "transfer_trajectory", hand_q)
    descent = _joint_path(descent_trajectory, "descent_trajectory", hand_q)
    if not np.allclose(transfer[-1], descent[0], atol=1e-4, rtol=0):
        raise ValueError("transfer and descent joint trajectories are discontinuous")

    stages = {"transfer": transfer, "descent": descent}
    wrist = {
        stage: [validate_se3(planner.fk_wrist(q), name=f"{stage} FK[{i}]")
                for i, q in enumerate(path)]
        for stage, path in stages.items()
    }
    if not _poses_match(wrist["transfer"][-1], targets.T_robot_hand_preinsert,
                        limits):
        raise ValueError("transfer does not reach pre-insertion hand target")
    if not _poses_match(wrist["descent"][-1], targets.T_robot_hand_verification,
                        limits):
        raise ValueError("descent does not reach 20 mm hand target")

    failures: list[dict] = []
    axis = targets.insertion_axis_robot
    origin = targets.T_robot_hand_preinsert[:3, 3]
    full_depth = targets.preinsert_clearance_m + mode.target_depth_m
    progress_previous = -float("inf")
    for stage, path in stages.items():
        for i in range(1, len(path)):
            q_step = float(np.max(np.abs(path[i] - path[i - 1])))
            d = float(np.linalg.norm(
                wrist[stage][i][:3, 3] - wrist[stage][i - 1][:3, 3]))
            angle = pose_angle_deg(wrist[stage][i], wrist[stage][i - 1])
            if (q_step > limits.max_joint_step_rad or
                    d > limits.max_wrist_step_m or
                    angle > limits.max_wrist_rotation_deg):
                failures.append({"stage": stage, "sample": i,
                                 "reason": "path_sampling_too_sparse",
                                 "joint_step_rad": q_step,
                                 "wrist_step_m": d,
                                 "wrist_rotation_deg": angle})
        if stage == "descent":
            for i, pose in enumerate(wrist[stage]):
                displacement = pose[:3, 3] - origin
                progress = float(np.dot(displacement, axis))
                lateral = float(np.linalg.norm(displacement - progress * axis))
                rotation = pose_angle_deg(
                    pose, targets.T_robot_hand_preinsert)
                if (progress < -limits.goal_position_tolerance_m or
                        progress > full_depth + limits.goal_position_tolerance_m or
                        progress + limits.goal_position_tolerance_m <
                        progress_previous or
                        lateral > limits.axial_lateral_tolerance_m or
                        rotation > limits.axial_rotation_tolerance_deg):
                    failures.append({"stage": stage, "sample": i,
                                     "reason": "not_monotone_socket_axis_stroke",
                                     "progress_m": progress,
                                     "lateral_error_m": lateral,
                                     "rotation_error_deg": rotation})
                progress_previous = progress

    fixed, world_hashes = _fixed_world_models(calibration)
    key_path = paths.raw_mesh(mode.key_object)
    robot_path = paths.robot_urdf
    if not key_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError("full key CAD and Franka/Inspire URDF are required")
    key_mesh = _load_mesh(key_path)
    if not key_mesh.is_watertight:
        raise ValueError("full key CAD mesh is not watertight")
    key_model = _coal_mesh(key_mesh)
    hand_models = {
        name: _coal_mesh(mesh) for name, mesh in
        _hand_link_meshes(robot_path, hand_q).items()
    }
    moving = {"key": key_model, **{f"hand/{n}": model
                                  for n, model in hand_models.items()}}
    key_in_hand = np.linalg.inv(targets.T_key_hand)
    minimum_distances: dict[str, float] = {}
    for stage, poses in wrist.items():
        for i, hand_pose in enumerate(poses):
            key_pose = hand_pose @ key_in_hand
            for moving_name, model in moving.items():
                moving_pose = key_pose if moving_name == "key" else hand_pose
                for fixed_name, (fixed_model, fixed_pose) in fixed.items():
                    relative = np.linalg.inv(fixed_pose) @ moving_pose
                    result = _coal_models_report(model, fixed_model, relative)
                    pair = f"{moving_name}->{fixed_name}"
                    distance = result["minimum_surface_distance_m"]
                    minimum_distances[pair] = min(
                        distance, minimum_distances.get(pair, float("inf")))
                    required = (limits.minimum_hand_clearance_m
                                if moving_name != "key" else 0.0)
                    if result["colliding"] or distance < required:
                        failures.append({"stage": stage, "sample": i,
                                         "reason": "held_geometry_collision_or_clearance",
                                         "moving": moving_name,
                                         "obstacle": fixed_name,
                                         "colliding": result["colliding"],
                                         "distance_m": distance,
                                         "required_clearance_m": required})

    return {
        "schema": "precision_insertion_sampled_held_path_audit_v1",
        "scope": "sampled_full_key_and_inspire_links_vs_frozen_world",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "sample_counts": {name: len(path) for name, path in stages.items()},
        "limits": vars(limits).copy(),
        "sampled_clear": not failures,
        "failures": failures,
        "minimum_surface_distances_m": minimum_distances,
        "input_sha256": {
            "key_mesh": _sha256(key_path),
            "robot_urdf": _sha256(robot_path),
            "task_geometry": targets.task_geometry_sha256,
            "session_calibration_record": hashlib.sha256(json.dumps(
                calibration.record, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "fixed_collision_scene": hashlib.sha256(json.dumps(
                calibration.collision_scene, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "transfer_trajectory": _array_sha256(transfer),
            "descent_trajectory": _array_sha256(descent),
            "held_hand_q": _array_sha256(hand_q),
            **world_hashes,
        },
        "not_validated": [
            "continuous swept geometry between FK samples",
            "independent cuRobo Franka arm/world collision and IK result",
            "grasp stability, contact force, jam response or physical success",
            "measured hand joints, grip slip or controller tracking",
        ],
        "robot_ready": False,
    }
