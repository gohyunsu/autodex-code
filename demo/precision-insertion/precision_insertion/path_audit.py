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
                       _load_mesh, validate_task_geometry)
from .geometry import pose_angle_deg, validate_se3
from .solid_occupancy import CylinderSocketOccupancy, SolidMeshOccupancy
from .targets import InsertionTargets, build_rigid_insertion_targets
from .world import validated_frozen_socket_pose


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


def _fixed_world_models(calibration, *, mode, shared_root: Path) -> tuple[dict, dict]:
    """Build Coal surfaces and solid occupancy for the frozen world."""
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
    cylinder_geometry = None
    if mode.family == "cylinder":
        geometry_path = AssetPaths(shared_root, mode).task_geometry
        geometry_bytes = geometry_path.read_bytes()
        cylinder_geometry = json.loads(geometry_bytes)
        validate_task_geometry(cylinder_geometry, mode)
        source_hashes["task_geometry"] = hashlib.sha256(
            geometry_bytes).hexdigest()
    for name, spec in sorted(scene["mesh"].items()):
        path = Path(spec["file_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"fixed scene mesh missing: {path}")
        pose = validate_se3(cart2se3(np.asarray(spec["pose"], dtype=float)),
                            name=f"fixed mesh {name} pose")
        mesh = _load_mesh(path)
        if name == "fixture_socket" and not mesh.is_watertight:
            raise ValueError("exact socket collision mesh is not watertight")
        occupancy = (
            CylinderSocketOccupancy(mesh, cylinder_geometry)
            if name == "fixture_socket" and cylinder_geometry is not None else
            SolidMeshOccupancy(mesh))
        models[f"mesh/{name}"] = (
            _coal_mesh(mesh), pose, mesh, occupancy)
        source_hashes[f"mesh/{name}"] = _sha256(path)
    for name, spec in sorted(scene["cuboid"].items()):
        dims = np.asarray(spec["dims"], dtype=np.float64)
        if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
            raise ValueError(f"invalid fixed cuboid {name} dimensions")
        pose = validate_se3(cart2se3(np.asarray(spec["pose"], dtype=float)),
                            name=f"fixed cuboid {name} pose")
        mesh = trimesh.creation.box(extents=dims)
        models[f"cuboid/{name}"] = (
            _coal_mesh(mesh), pose, mesh, SolidMeshOccupancy(mesh))
    if "mesh/fixture_socket" not in models:
        raise ValueError("frozen socket is absent from collision world")
    return models, source_hashes


def _held_pair_report(moving_model, moving_mesh, fixed_model,
                      fixed_mesh, fixed_occupancy,
                      T_fixed_moving: np.ndarray) -> dict:
    """Check surface crossing *and* a moving mesh enclosed by solid fixture.

    Coal's BVH checks triangle intersection, not solid containment. The
    inexpensive fixed AABB guard confines ray-parity occupancy calls to
    moving vertices that could lie inside a fixed obstacle. Checking only
    wholly enclosed links would miss a disconnected finger-link mesh whose
    one component is enclosed and another component remains outside.
    """
    report = _coal_models_report(moving_model, fixed_model, T_fixed_moving)
    if report["colliding"]:
        return report
    vertices = (moving_mesh.vertices @ T_fixed_moving[:3, :3].T +
                T_fixed_moving[:3, 3])
    within = np.all((vertices >= fixed_mesh.bounds[0] - 1e-9) &
                    (vertices <= fixed_mesh.bounds[1] + 1e-9), axis=1)
    if np.any(within):
        occupied = fixed_occupancy.classify(vertices[within])
        report["moving_vertices_inside_fixed"] = occupied.inside_vertices
        report["moving_vertices_occupancy_ambiguous"] = occupied.ambiguous_vertices
        if occupied.intersects_solid:
            report["colliding"] = True
            report["minimum_surface_distance_m"] = 0.0
    return report


def _audit_held_geometry_samples(
    *, shared_root: Path, mode, calibration, wrist: dict[str, list[np.ndarray]],
    T_key_hand: np.ndarray, held_hand_q: np.ndarray,
    minimum_hand_clearance_m: float,
    allow_initial_key_table_support: bool = False,
) -> tuple[list[dict], dict[str, float], dict[str, str]]:
    """One collision implementation for held transfer and lateral-shift paths.

    The result is a sampled surface/solid-containment check, not a swept
    volume or contact-force proof. In particular, a lateral hold shift never
    exempts its first key/table sample.
    """
    fixed, world_hashes = _fixed_world_models(
        calibration, mode=mode, shared_root=shared_root)
    paths = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    key_path = paths.raw_mesh(mode.key_object)
    robot_path = paths.robot_urdf
    if not key_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError("full key CAD and Franka/Inspire URDF are required")
    key_mesh = _load_mesh(key_path)
    if not key_mesh.is_watertight:
        raise ValueError("full key CAD mesh is not watertight")
    moving = {"key": (_coal_mesh(key_mesh), key_mesh)}
    moving.update({f"hand/{name}": (_coal_mesh(mesh), mesh)
                   for name, mesh in _hand_link_meshes(
                       robot_path, held_hand_q).items()})
    key_in_hand = np.linalg.inv(validate_se3(
        T_key_hand, name="sampled held T_key_hand"))
    failures: list[dict] = []
    minimum_distances: dict[str, float] = {}
    for stage, poses in wrist.items():
        for i, hand_pose in enumerate(poses):
            key_pose = hand_pose @ key_in_hand
            for moving_name, (model, moving_mesh) in moving.items():
                moving_pose = key_pose if moving_name == "key" else hand_pose
                for fixed_name, (fixed_model, fixed_pose, fixed_mesh,
                                 occupancy) in fixed.items():
                    if (allow_initial_key_table_support and stage == "lift" and
                            i == 0 and moving_name == "key" and
                            fixed_name == "cuboid/table"):
                        continue
                    relative = np.linalg.inv(fixed_pose) @ moving_pose
                    result = _held_pair_report(
                        model, moving_mesh, fixed_model, fixed_mesh,
                        occupancy, relative)
                    pair = f"{moving_name}->{fixed_name}"
                    distance = result["minimum_surface_distance_m"]
                    minimum_distances[pair] = min(
                        distance, minimum_distances.get(pair, float("inf")))
                    required = (minimum_hand_clearance_m
                                if moving_name != "key" else 0.0)
                    if result["colliding"] or distance < required:
                        failures.append({"stage": stage, "sample": i,
                                         "reason": "held_geometry_collision_or_clearance",
                                         "moving": moving_name,
                                         "obstacle": fixed_name,
                                         "colliding": result["colliding"],
                                         "distance_m": distance,
                                         "required_clearance_m": required})
    return failures, minimum_distances, {
        "key_mesh": _sha256(key_path), "robot_urdf": _sha256(robot_path),
        **world_hashes,
    }


def audit_held_joint_paths(
    *, shared_root: Path, calibration, targets: InsertionTargets,
    planner, transfer_trajectory: np.ndarray, descent_trajectory: np.ndarray,
    held_hand_q: np.ndarray, limits: PathAuditLimits,
    lift_trajectory: np.ndarray | None = None,
) -> dict[str, Any]:
    """Audit FK samples of lift, transfer and axial approach.

    ``planner.fk_wrist(q)`` is the unchanged AutoDex planner FK; its cuRobo
    arm/world collision result must be checked independently by the caller.
    The hand-to-key transform and six Inspire joints are fixed at all samples.
    A tilted socket is supported *only for checking a supplied axis-aligned
    path*; this function does not create such a path or contact controller.
    """
    limits.validate()
    root = Path(shared_root).expanduser().resolve()
    mode = targets.mode
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
    lift = None
    if lift_trajectory is not None:
        lift = _joint_path(lift_trajectory, "lift_trajectory", hand_q)
        if not np.allclose(lift[-1], transfer[0], atol=1e-4, rtol=0):
            raise ValueError("lift and transfer joint trajectories are discontinuous")

    stages = ({"lift": lift, "transfer": transfer, "descent": descent}
              if lift is not None else
              {"transfer": transfer, "descent": descent})
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

    # These are rigid key/hand model metrics, not a measurement of the key
    # after physical squeeze. Persist them separately from the Boolean audit
    # so a loose commissioned goal tolerance cannot masquerade as exactly
    # 20 mm of achieved insertion.
    final_hand = wrist["descent"][-1]
    final_delta = final_hand[:3, 3] - targets.T_robot_hand_preinsert[:3, 3]
    final_axial_progress = float(np.dot(
        final_delta, targets.insertion_axis_robot))
    final_lateral = float(np.linalg.norm(
        final_delta - final_axial_progress * targets.insertion_axis_robot))
    endpoint_metrics = {
        "axial_progress_from_preinsert_m": final_axial_progress,
        "rigid_model_depth_past_entry_m": (
            final_axial_progress - targets.preinsert_clearance_m),
        "nominal_target_depth_m": mode.target_depth_m,
        "lateral_from_socket_axis_m": final_lateral,
        "hand_target_translation_error_m": float(np.linalg.norm(
            final_hand[:3, 3] -
            targets.T_robot_hand_verification[:3, 3])),
        "hand_target_rotation_error_deg": pose_angle_deg(
            final_hand, targets.T_robot_hand_verification),
        "scope": "rigid_model_fk_not_measured_key_or_physical_insertion",
    }

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
        elif stage == "lift":
            first = wrist[stage][0]
            prior_height = float(first[2, 3])
            for i, pose in enumerate(wrist[stage]):
                lateral = float(np.linalg.norm(
                    pose[:2, 3] - first[:2, 3]))
                rotation = pose_angle_deg(pose, first)
                height = float(pose[2, 3])
                if (lateral > limits.axial_lateral_tolerance_m or
                        rotation > limits.axial_rotation_tolerance_deg or
                        height + limits.goal_position_tolerance_m < prior_height):
                    failures.append({"stage": stage, "sample": i,
                                     "reason": "not_monotone_world_z_lift",
                                     "lateral_error_m": lateral,
                                     "rotation_error_deg": rotation,
                                     "height_m": height})
                prior_height = height

    geometry_failures, minimum_distances, geometry_hashes = (
        _audit_held_geometry_samples(
            shared_root=root, mode=mode, calibration=calibration,
            wrist=wrist, T_key_hand=targets.T_key_hand,
            held_hand_q=hand_q,
            minimum_hand_clearance_m=limits.minimum_hand_clearance_m,
            allow_initial_key_table_support=lift is not None))
    failures.extend(geometry_failures)

    return {
        "schema": "precision_insertion_sampled_held_path_audit_v1",
        "scope": "sampled_full_key_and_inspire_links_vs_frozen_world",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "sample_counts": {name: len(path) for name, path in stages.items()},
        "nominal_endpoint_metrics": endpoint_metrics,
        "limits": vars(limits).copy(),
        "sampled_clear": not failures,
        "failures": failures,
        "minimum_surface_distances_m": minimum_distances,
        "input_sha256": {
            "key_mesh": geometry_hashes["key_mesh"],
            "robot_urdf": geometry_hashes["robot_urdf"],
            "task_geometry": targets.task_geometry_sha256,
            "session_calibration_record": hashlib.sha256(json.dumps(
                calibration.record, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "fixed_collision_scene": hashlib.sha256(json.dumps(
                calibration.collision_scene, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "transfer_trajectory": _array_sha256(transfer),
            "descent_trajectory": _array_sha256(descent),
            **({"lift_trajectory": _array_sha256(lift)}
               if lift is not None else {}),
            "held_hand_q": _array_sha256(hand_q),
            **{k: v for k, v in geometry_hashes.items()
               if k not in {"key_mesh", "robot_urdf"}},
        },
        "not_validated": [
            "continuous swept geometry between FK samples",
            "independent cuRobo Franka arm/world collision and IK result",
            "grasp stability, contact force, jam response or physical success",
            "measured hand joints, grip slip or controller tracking",
            *(["initial key/table support contact at lift sample zero"]
              if lift is not None else []),
        ],
        "robot_ready": False,
    }


def audit_held_lateral_path(
    *, shared_root: Path, mode, calibration, planner,
    trajectory: np.ndarray, held_hand_q: np.ndarray,
    T_key_hand: np.ndarray, increment_socket_xy_m: tuple[float, float],
    limits: PathAuditLimits, max_path_deviation_m: float,
    max_hold_height_deviation_m: float,
    max_hold_rotation_deg: float, key_surface_bound_m: float,
    hand_surface_bound_m: float,
) -> dict[str, Any]:
    """Audit a planned <=1 mm hold shift, never an insertion stroke.

    Every FK sample must stay near the socket-plane segment and its original
    height/orientation. Collision and future-trial surface bounds are checked
    against the same exact CAD and frozen world as the insertion path audit.
    """
    limits.validate()
    values = (max_path_deviation_m, max_hold_height_deviation_m,
              max_hold_rotation_deg, key_surface_bound_m,
              hand_surface_bound_m)
    if not all(math.isfinite(float(v)) and v > 0 for v in values):
        raise ValueError("lateral path and future surface limits must be positive")
    increment = np.asarray(increment_socket_xy_m, dtype=np.float64)
    if (increment.shape != (2,) or not np.all(np.isfinite(increment)) or
            not 0 < np.linalg.norm(increment) <= 0.001 + 1e-12):
        raise ValueError("lateral increment must be nonzero and at most 1 mm")
    held = np.asarray(held_hand_q, dtype=np.float64)
    if held.shape != (6,) or not np.all(np.isfinite(held)):
        raise ValueError("held hand must have six finite Inspire joints")
    path = _joint_path(trajectory, "lateral_trajectory", held)
    wrist = [validate_se3(planner.fk_wrist(q), name=f"lateral FK[{i}]")
             for i, q in enumerate(path)]
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=shared_root, calibration=calibration)
    initial = wrist[0]
    goal = initial.copy()
    goal[:3, 3] += socket[:3, :2] @ increment
    failures: list[dict] = []
    if not _poses_match(wrist[-1], goal, limits):
        failures.append({"stage": "lateral", "sample": len(path) - 1,
                         "reason": "lateral_goal_residual",
                         "position_error_m": float(np.linalg.norm(
                             wrist[-1][:3, 3] - goal[:3, 3])),
                         "rotation_error_deg": pose_angle_deg(wrist[-1], goal)})
    distance = float(np.linalg.norm(increment))
    unit = increment / distance
    previous_progress = -float("inf")
    for i, pose in enumerate(wrist):
        local = socket[:3, :3].T @ (pose[:3, 3] - initial[:3, 3])
        progress = float(np.dot(local[:2], unit))
        deviation = float(np.linalg.norm(local[:2] - progress * unit))
        rotation = pose_angle_deg(pose, initial)
        if (progress < -max_path_deviation_m or
                progress > distance + max_path_deviation_m or
                progress + max_path_deviation_m < previous_progress or
                deviation > max_path_deviation_m or
                abs(local[2]) > max_hold_height_deviation_m or
                rotation > max_hold_rotation_deg):
            failures.append({"stage": "lateral", "sample": i,
                             "reason": "not_socket_plane_lateral_segment",
                             "progress_m": progress,
                             "cross_track_m": deviation,
                             "height_error_m": float(local[2]),
                             "rotation_error_deg": rotation})
        previous_progress = progress
        if i:
            joint_step = float(np.max(np.abs(path[i] - path[i - 1])))
            wrist_step = float(np.linalg.norm(
                pose[:3, 3] - wrist[i - 1][:3, 3]))
            rotation_step = pose_angle_deg(pose, wrist[i - 1])
            if (joint_step > limits.max_joint_step_rad or
                    wrist_step > limits.max_wrist_step_m or
                    rotation_step > limits.max_wrist_rotation_deg):
                failures.append({"stage": "lateral", "sample": i,
                                 "reason": "path_sampling_too_sparse",
                                 "joint_step_rad": joint_step,
                                 "wrist_step_m": wrist_step,
                                 "wrist_rotation_deg": rotation_step})
    geometry_failures, distances, geometry_hashes = (
        _audit_held_geometry_samples(
            shared_root=shared_root, mode=mode, calibration=calibration,
            wrist={"lateral": wrist}, T_key_hand=T_key_hand,
            held_hand_q=held,
            minimum_hand_clearance_m=limits.minimum_hand_clearance_m))
    failures.extend(geometry_failures)
    if ("key->mesh/fixture_socket" not in distances or
            not any(pair.startswith("hand/") and pair.endswith(
                "->mesh/fixture_socket") for pair in distances) or
            not all(math.isfinite(float(value)) and value >= 0
                    for value in distances.values())):
        raise ValueError("lateral audit lacks finite key/hand fixture distances")
    margins = {}
    for pair, surface_distance in sorted(distances.items()):
        required = (key_surface_bound_m if pair.startswith("key->") else
                    hand_surface_bound_m + limits.minimum_hand_clearance_m)
        margin = float(surface_distance) - required
        margins[pair] = {"distance_m": float(surface_distance),
                         "required_m": required, "remaining_m": margin,
                         "clear": margin > 0}
        if margin <= 0:
            failures.append({"stage": "lateral", "reason":
                             "future_surface_bound_exceeds_sampled_clearance",
                             "pair": pair, "remaining_m": margin})
    root = Path(shared_root).expanduser().resolve()
    geometry_path = AssetPaths(root, mode).task_geometry
    return {
        "schema": "precision_insertion_sampled_lateral_hold_audit_v1",
        "scope": "sampled_held_key_and_hand_socket_plane_shift_not_execution",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "sample_count": len(path),
        "increment_socket_xy_m": increment.tolist(),
        "T_robot_hand_start": initial.tolist(),
        "T_robot_hand_goal": goal.tolist(),
        "sampled_clear": not failures,
        "failures": failures,
        "minimum_surface_distances_m": distances,
        "future_surface_margins": margins,
        "limits": {**vars(limits),
                   "max_path_deviation_m": max_path_deviation_m,
                   "max_hold_height_deviation_m": max_hold_height_deviation_m,
                   "max_hold_rotation_deg": max_hold_rotation_deg,
                   "key_surface_bound_m": key_surface_bound_m,
                   "hand_surface_bound_m": hand_surface_bound_m},
        "input_sha256": {
            **geometry_hashes,
            "task_geometry": _sha256(geometry_path),
            "session_calibration_record": hashlib.sha256(json.dumps(
                calibration.record, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "fixed_collision_scene": hashlib.sha256(json.dumps(
                calibration.collision_scene, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "T_key_hand": _array_sha256(np.asarray(T_key_hand)),
            "held_hand_q": _array_sha256(held),
            "lateral_trajectory": _array_sha256(path),
        },
        "not_validated": [
            "continuous swept geometry between FK samples",
            "physical authenticity or coverage of future surface bounds",
            "grasp slip, force/contact safety and controller tracking",
            "post-shift camera reobservation and insertion endpoint",
        ],
        "robot_ready": False,
    }
