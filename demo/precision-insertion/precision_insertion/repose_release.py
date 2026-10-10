"""Planning-only reset release and retreat under two key-pose hypotheses.

This follows AutoDex Franka's release-to-pregrasp, +10 cm clearance and
joint-space retract primitives without importing its hardware executor. The
key may remain at the release height or fall to the selected tabletop rest
pose; both static meshes are kept in the conservative planning world. Real
drop dynamics and post-release object perception are not simulated here.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from autodex.utils.conversion import se32cart

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import _coal_mesh, _hand_link_meshes, _load_mesh
from .geometry import pose_angle_deg, validate_se3
from .path_audit import (
    PathAuditLimits, _array_sha256, _fixed_world_models, _held_pair_report,
    _sha256,
)
from .preflight import _goal_met, _path
from .repose_preflight import ReposeHeldPreflight
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


@dataclass(frozen=True)
class ReposeReleasePreflight:
    status: str
    post_release_lift_trajectory: np.ndarray | None
    retract_trajectory: np.ndarray | None
    release_geometry_audit: dict | None
    planner_query_records: tuple[dict[str, Any], ...]
    release_hand_q: np.ndarray
    retreat_goal_arm_q: np.ndarray

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_repose_release_preflight_v1",
            "status": self.status,
            "nominal_path_pass": (
                self.status == "nominal_release_exit_path_pass_drop_unobserved"),
            "sample_counts": {
                "post_release_lift": (
                    None if self.post_release_lift_trajectory is None
                    else len(self.post_release_lift_trajectory)),
                "retract": (None if self.retract_trajectory is None
                            else len(self.retract_trajectory)),
            },
            "release_hand_q": self.release_hand_q.tolist(),
            "retreat_goal_arm_q": self.retreat_goal_arm_q.tolist(),
            "planner_query_records": list(self.planner_query_records),
            "release_geometry_audit": self.release_geometry_audit,
            "not_validated": [
                "actual hand opening and key detachment",
                "dynamic key drop path and actual landing pose",
                "measured robot/controller state before replay",
                "continuous swept visual geometry between FK samples",
                "physical reset or insertion success",
            ],
            "robot_ready": False,
        }


def build_repose_release_world(
    *, trial_scene: dict, calibration, T_robot_key_rest: np.ndarray,
    release_height_m: float,
) -> tuple[dict, np.ndarray]:
    """Keep socket/table plus both possible key locations after opening."""
    carried = build_held_scene_from_trial(
        trial_scene=trial_scene, calibration=calibration)
    rest = validate_se3(T_robot_key_rest, name="reset tabletop rest pose")
    height = float(release_height_m)
    if not math.isfinite(height) or height <= 0:
        raise ValueError("release height must be positive")
    floating = rest.copy()
    floating[2, 3] += height
    key_spec = trial_scene["mesh"]["target"]
    if (not isinstance(key_spec.get("file_path"), str) or
            not Path(key_spec["file_path"]).is_file()):
        raise ValueError("trial key planning mesh is missing")
    release_world = copy.deepcopy(carried)
    release_world["mesh"]["target"] = {
        "file_path": key_spec["file_path"],
        "pose": se32cart(floating).tolist(),
    }
    release_world["mesh"]["released_rest_key"] = {
        "file_path": key_spec["file_path"],
        "pose": se32cart(rest).tolist(),
    }
    return release_world, floating


def audit_repose_release_open_and_lift(
    *, shared_root: Path, mode: TaskMode, calibration, planner,
    held_plan: ReposeHeldPreflight, release_hand_q: np.ndarray,
    post_release_lift_trajectory: np.ndarray,
    minimum_release_key_clearance_m: float,
    limits: PathAuditLimits,
) -> dict:
    """Sample opening and vertical exit against socket, table and both keys.

    Hand/key contact during the opening ramp is expected and not adjudicated.
    At the *fully open* state and throughout the upward exit, every visible
    hand link must clear both the floating and tabletop-rest hypotheses.
    Franka-arm/self collision remains the original planner's responsibility.
    """
    limits.validate()
    release_q = np.asarray(release_hand_q, dtype=np.float64)
    if release_q.shape != (6,) or not np.all(np.isfinite(release_q)):
        raise ValueError("release hand must have six finite Inspire joints")
    clearance = float(minimum_release_key_clearance_m)
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("release key clearance must be positive")
    if (held_plan.status != "sampled_held_path_pass_release_unplanned" or
            held_plan.descent_trajectory is None):
        raise ValueError("held reset path has not passed sampled audit")
    descent_end = np.asarray(held_plan.descent_trajectory[-1], dtype=np.float64)
    post_start = descent_end.copy()
    post_start[7:] = release_q
    up = _path(post_release_lift_trajectory, "post-release lift",
               post_start, release_q)
    release_wrist = validate_se3(planner.fk_wrist(post_start),
                                 name="post-release wrist")
    rest = validate_se3(held_plan.T_robot_key_rest, name="reset rest key")
    floating = rest.copy()
    floating[2, 3] += held_plan.release_height_m
    fixed, hashes = _fixed_world_models(calibration)
    assets = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    key_path = assets.raw_mesh(mode.key_object)
    urdf = assets.robot_urdf
    if not key_path.is_file() or not urdf.is_file():
        raise FileNotFoundError("full key mesh and Franka/Inspire URDF required")
    key_mesh = _load_mesh(key_path)
    if not key_mesh.is_watertight:
        raise ValueError("full key CAD is not watertight")
    key_model = _coal_mesh(key_mesh)
    obstacle_models = dict(fixed)
    from .solid_occupancy import SolidMeshOccupancy

    solid_key = SolidMeshOccupancy(key_mesh)
    obstacle_models["key/floating_release"] = (
        key_model, floating, key_mesh, solid_key)
    obstacle_models["key/tabletop_rest"] = (
        key_model, rest, key_mesh, solid_key)
    failures = []
    minimum_distances = {}
    opening_steps = max(10, int(math.ceil(float(np.max(np.abs(
        release_q - held_plan.held_hand_q))) / limits.max_joint_step_rad)))
    if opening_steps > 1000:
        raise ValueError("release hand ramp exceeds 1000 audit steps")
    opening = np.repeat(descent_end[None], opening_steps + 1, axis=0)
    for i, alpha in enumerate(np.linspace(0.0, 1.0, len(opening))):
        opening[i, 7:] = (
            (1.0 - alpha) * held_plan.held_hand_q + alpha * release_q)
    stages = {"opening": opening, "post_release_lift": up}
    hand_mesh_cache = {}
    for stage, path in stages.items():
        poses = [validate_se3(planner.fk_wrist(q),
                              name=f"release {stage} FK[{i}]")
                 for i, q in enumerate(path)]
        previous_z = float(poses[0][2, 3])
        for i in range(1, len(path)):
            joint_step = float(np.max(np.abs(path[i] - path[i - 1])))
            wrist_step = float(np.linalg.norm(
                poses[i][:3, 3] - poses[i - 1][:3, 3]))
            if (joint_step > limits.max_joint_step_rad or
                    wrist_step > limits.max_wrist_step_m):
                failures.append({"stage": stage, "sample": i,
                                 "reason": "path_sampling_too_sparse",
                                 "joint_step_rad": joint_step,
                                 "wrist_step_m": wrist_step})
        for i, (q, wrist) in enumerate(zip(path, poses)):
            if stage == "opening" and not _goal_met(
                    wrist, release_wrist, limits):
                failures.append({"stage": stage, "sample": i,
                                 "reason": "wrist_moved_while_opening"})
            if stage == "post_release_lift":
                if (np.linalg.norm(wrist[:2, 3] - release_wrist[:2, 3]) >
                        limits.axial_lateral_tolerance_m or
                        wrist[2, 3] + limits.goal_position_tolerance_m <
                        previous_z or
                        pose_angle_deg(wrist, release_wrist) >
                        limits.axial_rotation_tolerance_deg):
                    failures.append({"stage": stage, "sample": i,
                                     "reason": "not_vertical_post_release_exit"})
                previous_z = float(wrist[2, 3])
            hand_key = tuple(float(value) for value in q[7:])
            if hand_key not in hand_mesh_cache:
                hand_mesh_cache[hand_key] = {
                    name: (mesh, _coal_mesh(mesh)) for name, mesh in
                    _hand_link_meshes(urdf, q[7:]).items()}
            for link, (mesh, model) in hand_mesh_cache[hand_key].items():
                for obstacle_name, (obstacle_model, obstacle_pose,
                                    obstacle_mesh, occupancy) in obstacle_models.items():
                    # Contact with the held key is expected until the final
                    # open state. Fixed socket/table collisions are *never*
                    # exempt, even while the hand is opening.
                    if obstacle_name.startswith("key/") and (stage == "opening"
                            and i < len(path) - 1):
                        continue
                    relative = np.linalg.inv(obstacle_pose) @ wrist
                    report = _held_pair_report(
                        model, mesh, obstacle_model, obstacle_mesh,
                        occupancy, relative)
                    pair = f"hand/{link}->{obstacle_name}"
                    distance = float(report["minimum_surface_distance_m"])
                    minimum_distances[pair] = min(
                        distance, minimum_distances.get(pair, float("inf")))
                    required = (clearance if obstacle_name.startswith("key/")
                                else limits.minimum_hand_clearance_m)
                    if report["colliding"] or distance < required:
                        failures.append({"stage": stage, "sample": i,
                                         "reason": "release_geometry_collision_or_clearance",
                                         "moving": f"hand/{link}",
                                         "obstacle": obstacle_name,
                                         "colliding": report["colliding"],
                                         "distance_m": distance,
                                         "required_clearance_m": required})
    expected_high = release_wrist.copy()
    expected_high[2, 3] += 0.10
    if not _goal_met(validate_se3(planner.fk_wrist(up[-1]),
                                  name="post-release lift endpoint FK"),
                     expected_high, limits):
        failures.append({"stage": "post_release_lift", "sample": len(up) - 1,
                         "reason": "post_release_lift_goal_residual"})
    return {
        "schema": "precision_insertion_sampled_repose_release_audit_v1",
        "sampled_clear": not failures,
        "sample_counts": {name: len(path) for name, path in stages.items()},
        "failures": failures,
        "minimum_surface_distances_m": minimum_distances,
        "input_sha256": {
            "key_mesh": _sha256(key_path), "robot_urdf": _sha256(urdf),
            "session_calibration_record": hashlib.sha256(json.dumps(
                calibration.record, sort_keys=True, allow_nan=False,
            ).encode("utf-8")).hexdigest(),
            "opening_trajectory": _array_sha256(opening),
            "post_release_lift_trajectory": _array_sha256(up),
            "T_robot_key_rest": _array_sha256(rest), **hashes,
        },
        "not_validated": [
            "grip contact physics while opening",
            "dynamic key drop between the two static pose hypotheses",
            "full visual-mesh audit of subsequent joint-space retract",
            "continuous swept geometry between FK samples",
        ],
        "robot_ready": False,
    }


def plan_repose_release_exit(
    *, planner, trial_scene: dict, shared_root: Path, calibration,
    mode: TaskMode, held_plan: ReposeHeldPreflight,
    release_hand_q: np.ndarray, retreat_goal_arm_q: np.ndarray,
    minimum_release_key_clearance_m: float,
    limits: PathAuditLimits,
) -> ReposeReleasePreflight:
    """Preflight hand opening, +10 cm exit and stock joint-space retract.

    This is still a nominal *drop* plan. It must be followed by measured
    release/lift observations and a fresh tabletop pose estimate before the
    next insertion trial or any reorientation success label.
    """
    limits.validate()
    release_q = np.asarray(release_hand_q, dtype=np.float64)
    retreat = np.asarray(retreat_goal_arm_q, dtype=np.float64)
    if (release_q.shape != (6,) or retreat.shape != (7,) or
            not np.all(np.isfinite(release_q)) or
            not np.all(np.isfinite(retreat))):
        raise ValueError("release hand and retreat arm must be 6/7 finite joints")
    if (held_plan.status != "sampled_held_path_pass_release_unplanned" or
            held_plan.descent_trajectory is None):
        raise ValueError("release preflight requires a passed held descent")
    root = Path(shared_root).expanduser().resolve()
    validated_frozen_socket_pose(mode=mode, shared_root=root,
                                 calibration=calibration)
    held_audit = held_plan.sampled_held_path_audit
    if (not isinstance(held_audit, dict) or
            held_audit.get("schema") !=
            "precision_insertion_sampled_repose_held_path_audit_v1" or
            held_audit.get("sampled_clear") is not True or
            held_audit.get("mode") !=
            {"family": mode.family, "gap_mm": mode.gap_mm}):
        raise ValueError("release preflight lacks matching held-path evidence")
    saved_hashes = held_audit.get("input_sha256", {})
    key_path = AssetPaths(root, mode).raw_mesh(mode.key_object)
    current_world_hash = hashlib.sha256(json.dumps(
        calibration.collision_scene, sort_keys=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if (saved_hashes.get("key_mesh") != _sha256(key_path) or
            saved_hashes.get("fixed_collision_scene") != current_world_hash or
            saved_hashes.get("descent_trajectory") !=
            _array_sha256(held_plan.descent_trajectory) or
            saved_hashes.get("T_robot_key_rest") !=
            _array_sha256(held_plan.T_robot_key_rest)):
        raise ValueError("held reset evidence changed before release preflight")
    scene, _ = build_repose_release_world(
        trial_scene=trial_scene, calibration=calibration,
        T_robot_key_rest=held_plan.T_robot_key_rest,
        release_height_m=held_plan.release_height_m)
    release_start = np.asarray(held_plan.descent_trajectory[-1],
                               dtype=np.float64).copy()
    release_start[7:] = release_q
    wrist_low = validate_se3(planner.fk_wrist(release_start),
                             name="repose release wrist")
    wrist_high = wrist_low.copy()
    wrist_high[2, 3] += 0.10
    queries = []

    def result(status: str, *, up=None, retract=None,
               audit=None) -> ReposeReleasePreflight:
        return ReposeReleasePreflight(
            status, up, retract, audit, tuple(queries),
            release_q.copy(), retreat.copy())

    up_result = planner.plan_vertical_stroke(
        release_start, wrist_low, wrist_high,
        expected_travel_m=0.10, travel_tolerance_m=1e-5,
        scene_cfg=scene, include_obj_obstacle=True,
        label="precision insertion repose post-release lift",
        timing_phase="precision_insertion_repose_preflight",
        return_result=True)
    queries.append({"stage": "post_release_lift",
                    "planner_api": "plan_vertical_stroke",
                    "success": bool(up_result.success),
                    "failure_code": getattr(up_result, "failure_code", None)})
    if not up_result.success or up_result.trajectory is None:
        return result("post_release_lift_unreachable")
    up = _path(up_result.trajectory, "repose post-release lift",
               release_start, release_q)
    audit = audit_repose_release_open_and_lift(
        shared_root=shared_root, mode=mode, calibration=calibration,
        planner=planner, held_plan=held_plan, release_hand_q=release_q,
        post_release_lift_trajectory=up,
        minimum_release_key_clearance_m=minimum_release_key_clearance_m,
        limits=limits)
    if not audit["sampled_clear"]:
        return result("sampled_release_exit_rejected", up=up, audit=audit)
    retract_raw = planner.plan_js_to_init(
        scene, up[-1, :7], start_hand_qpos=release_q,
        goal_arm_qpos=retreat)
    queries.append({"stage": "post_release_retract",
                    "planner_api": "plan_js_to_init",
                    "success": retract_raw is not None})
    if retract_raw is None:
        return result("post_release_retract_unreachable", up=up, audit=audit)
    retract = np.asarray(retract_raw, dtype=np.float64)
    if (retract.ndim != 2 or retract.shape[1] != 13 or len(retract) < 2 or
            not np.all(np.isfinite(retract)) or
            not np.allclose(retract[0], up[-1], atol=1e-4, rtol=0) or
            not np.allclose(retract[-1, :7], retreat, atol=1e-3, rtol=0)):
        raise ValueError("stock retract path is malformed or misses requested goal")
    return result("nominal_release_exit_path_pass_drop_unobserved",
                  up=up, retract=retract, audit=audit)
