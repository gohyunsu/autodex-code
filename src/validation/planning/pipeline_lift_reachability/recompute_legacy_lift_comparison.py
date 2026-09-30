#!/usr/bin/env python3
"""Reproduce legacy endpoint lift for saved Jacobian replays and rank wobble."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validation.planning.pipeline_lift_reachability.core import (
    build_scene, load_tabletop_transform, tabletop_files, write_json)


# Exact mask used by the pre-Jacobian production lift call.  The legacy
# implementation projected wrist orientation and XY onto start FK, but only at
# the endpoint; plan_single_js imposed no corresponding path constraint.
LEGACY_LIFT_HOLD_MASK = np.asarray([1, 1, 1, 1, 1, 0], dtype=np.float32)


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    result = np.repeat(np.eye(4)[None], len(position), axis=0)
    result[:, :3, 3] = position
    result[:, :3, :3] = Rotation.from_quat(
        quaternion_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    return result


def _fk(planner, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch
    with torch.inference_mode():
        state = planner._motion_gen.kinematics.get_state(torch.as_tensor(
            qpos, dtype=torch.float32, device=planner._tensor_args.device))
    position = state.ee_position.detach().cpu().numpy()
    quaternion = state.ee_quaternion.detach().cpu().numpy()
    return position, _pose_matrix(position, quaternion)


def _metrics(poses: np.ndarray) -> dict[str, float | int]:
    position = poses[:, :3, 3]
    lateral = np.linalg.norm(position[:, :2] - position[0, :2], axis=1)
    relative = np.einsum("ij,njk->nik", poses[0, :3, :3].T,
                         poses[:, :3, :3])
    orientation = Rotation.from_matrix(relative).magnitude()
    dz = np.diff(position[:, 2])
    return {
        "max_lateral_deviation_m": float(lateral.max()),
        "endpoint_lateral_deviation_m": float(lateral[-1]),
        "max_orientation_deviation_deg": float(np.degrees(orientation).max()),
        "endpoint_orientation_deviation_deg": float(np.degrees(orientation[-1])),
        "min_delta_z_m": float(dz.min()) if len(dz) else 0.0,
        "nonmonotonic_z_steps": int((dz < -1.0e-5).sum()),
        "final_vertical_travel_m": float(position[-1, 2] - position[0, 2]),
        "trajectory_samples": int(len(poses)),
    }


def _same_full_path_validation(planner, qpos: np.ndarray, poses: np.ndarray,
                               object_poses: np.ndarray,
                               object_vertices: np.ndarray,
                               table_z: float = 0.040) -> dict:
    """Apply the current final trajectory checks to either planner output."""
    from autodex.planner.jacobian_stroke import (
        JacobianStrokeOptions, _check_states_batch, _joint_bounds,
        _object_bottom_z)

    options = JacobianStrokeOptions()
    lower, upper = _joint_bounds(planner, qpos.shape[1])
    joint_limits = bool(np.all(
        (qpos >= lower[None] - 1.0e-6) & (qpos <= upper[None] + 1.0e-6)))
    valid, collision_status, collision_batch = _check_states_batch(planner, qpos)
    metrics = _metrics(poses)
    bottoms = _object_bottom_z(object_vertices, object_poses)
    object_clearance = bool(np.all(
        bottoms >= table_z - options.support_clearance_tolerance_m))
    vertical = bool(
        metrics["max_lateral_deviation_m"] <= options.position_tolerance_m
        and np.radians(metrics["max_orientation_deviation_deg"])
        <= options.orientation_tolerance_rad
        and metrics["min_delta_z_m"] >= -options.monotonic_tolerance_m
        and abs(metrics["final_vertical_travel_m"] - 0.10)
        <= options.position_tolerance_m)
    return {
        "passed": bool(joint_limits and valid.all() and object_clearance and vertical),
        "joint_limits_pass": joint_limits,
        "world_self_collision_pass": bool(valid.all()),
        "collision_status": collision_status,
        "collision_batch": collision_batch,
        "object_table_clearance_pass": object_clearance,
        "object_bottom_min_z_m": float(bottoms.min()),
        "vertical_path_pass": vertical,
        "thresholds": {
            "lateral_m": options.position_tolerance_m,
            "orientation_deg": float(np.degrees(options.orientation_tolerance_rad)),
            "monotonic_tolerance_m": options.monotonic_tolerance_m,
            "object_support_tolerance_m": options.support_clearance_tolerance_m,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    np.random.seed(args.seed)
    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir
                  else trajectory_dir / "legacy_lift_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_manifest = json.loads((trajectory_dir / "replay_manifest.json").read_text())
    source_run = Path(replay_manifest["source_run"])
    if not (source_run / "manifest.json").is_file():
        source_run = trajectory_dir.parent
    manifest = json.loads((source_run / "manifest.json").read_text())
    pose_file = next(path for path in tabletop_files(
        manifest["object"], manifest["version"])
        if path.stem == manifest["tabletop_pose_stem"])

    from autodex.planner import GraspPlanner
    from autodex.planner.planner import _to_curobo_world, _without_target_mesh
    planner = GraspPlanner(hand=manifest["planner_robot"],
                           use_cuda_graph=args.cuda_graph == "on")
    first_data = np.load(sorted(trajectory_dir.glob("[0-9][0-9]_*.npz"))[0])
    raw = load_tabletop_transform(
        pose_file, float(first_data["r_m"]), float(first_data["theta_deg"]),
        manifest["table_surface_z_m"])
    warm_scene, _ = build_scene(manifest["object"], manifest["version"], raw,
                                manifest["table_surface_z_m"])
    planner.warmup(warm_scene)
    import trimesh
    object_vertices = np.asarray(trimesh.load(
        warm_scene["mesh"]["target"]["file_path"], force="mesh",
        process=False).vertices)

    rows = []
    for path in sorted(trajectory_dir.glob("[0-9][0-9]_*.npz")):
        data = np.load(path)
        phases = np.asarray(data["phases"]).astype(str)
        current_qpos = np.asarray(data["qpos"])[phases == "lift"]
        start = current_qpos[0].astype(np.float32)
        raw = load_tabletop_transform(
            pose_file, float(data["r_m"]), float(data["theta_deg"]),
            manifest["table_surface_z_m"])
        scene, object_pose = build_scene(
            manifest["object"], manifest["version"], raw,
            manifest["table_surface_z_m"])
        # This is the implementation retained in planner.py specifically as
        # the previous production endpoint approximation: endpoint IK,
        # followed by unconstrained joint-space graph search + TrajOpt.
        planner._set_motion_world(_without_target_mesh(_to_curobo_world(scene)))
        start_wrist = planner.fk_wrist(start)
        target_wrist = start_wrist.copy()
        target_wrist[2, 3] += 0.10
        legacy = planner._plan_endpoint_approximation(
            start, target_wrist, LEGACY_LIFT_HOLD_MASK,
            scene_cfg=scene, include_obj_obstacle=False,
            debug_dump_dir=None, return_result=True,
            timing_parent_id=None, timing_phase="validation")
        row = {
            "source_npz": path.name,
            "candidate_key": [str(value) for value in data["candidate_key"]],
            "cell_id": str(data["cell_id"]),
            "r_m": float(data["r_m"]),
            "theta_deg": float(data["theta_deg"]),
            "legacy_success": bool(legacy.success),
            "legacy_failure_stage": legacy.failure_stage,
        }
        if not legacy.success or legacy.trajectory is None:
            rows.append(row)
            print(f"[legacy fail] {path.name}: {legacy.failure_stage}", flush=True)
            continue
        legacy_qpos = np.asarray(legacy.trajectory, dtype=np.float32)
        legacy_position, legacy_poses = _fk(planner, legacy_qpos)
        current_position, current_poses = _fk(planner, current_qpos)
        relative = np.linalg.inv(current_poses[0]) @ object_pose
        legacy_object = np.matmul(legacy_poses, relative)
        current_object = np.matmul(current_poses, relative)
        row["legacy_metrics"] = _metrics(legacy_poses)
        row["current_metrics"] = _metrics(current_poses)
        row["legacy_same_full_path_validation"] = _same_full_path_validation(
            planner, legacy_qpos, legacy_poses, legacy_object, object_vertices)
        row["current_same_full_path_validation"] = _same_full_path_validation(
            planner, current_qpos, current_poses, current_object, object_vertices)
        comparison_file = output_dir / f"{path.stem}_legacy_vs_jacobian.npz"
        np.savez_compressed(
            comparison_file,
            legacy_qpos=legacy_qpos, legacy_ee_position=legacy_position,
            legacy_object_poses=legacy_object,
            current_qpos=current_qpos, current_ee_position=current_position,
            current_object_poses=current_object,
            start_wrist_pose=current_poses[0], target_wrist_pose=target_wrist,
            object_pose_initial=object_pose,
            candidate_key=data["candidate_key"], cell_id=data["cell_id"],
            r_m=data["r_m"], theta_deg=data["theta_deg"])
        row["comparison_npz"] = comparison_file.name
        rows.append(row)
        print(f"[legacy ok] {path.name}: lateral="
              f"{row['legacy_metrics']['max_lateral_deviation_m'] * 1000:.1f} mm, "
              f"rotation={row['legacy_metrics']['max_orientation_deviation_deg']:.1f} deg",
              flush=True)

    eligible = [
        row for row in rows
        if row["legacy_success"]
        and row["legacy_same_full_path_validation"]["joint_limits_pass"]
        and row["legacy_same_full_path_validation"]["world_self_collision_pass"]
        and row["legacy_same_full_path_validation"]["object_table_clearance_pass"]]
    if not eligible:
        raise RuntimeError(
            "no legacy replay preserves joint/collision/object-table constraints")
    selected = max(eligible, key=lambda row:
                   row["legacy_metrics"]["max_lateral_deviation_m"])
    report = {
        "definition": {
            "legacy": (
                "historical hold-mask [rx,ry,rz,x,y]=1, z=0 endpoint IK + "
                "unconstrained joint-space graph search/TrajOpt"),
            "current": "5 mm-node damped-least-squares Jacobian world-Z lift",
            "historical_acceptance": (
                "endpoint IK success followed by cuRobo plan_single_js success; "
                "the target mesh was removed from the collision world and no "
                "wrist-path or attached-object clearance test was applied"),
            "selection": (
                "largest legacy FK lateral deviation after requiring legacy "
                "success, joint limits, world/self collision, and attached-object "
                "table clearance; the latter checks are stricter post-hoc audit "
                "conditions, not historical legacy acceptance conditions"),
        },
        "selected": selected,
        "all_cases": rows,
    }
    write_json(output_dir / "comparison_summary.json", report)
    print(f"[selected] {selected['comparison_npz']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
