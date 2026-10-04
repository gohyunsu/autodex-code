#!/usr/bin/env python3
"""Export the exact FR3+Inspire approach, hand-close, and lift plan.

The planner result is intentionally separated from rendering.  Run this
script in ``autodex_bodex`` so cuRobo validates the same candidate, full key
mesh, table, and 10 cm held-object lift used by the precision-insertion asset
check.  The resulting NPZ is hardware-free evidence, not an executable robot
program and not evidence that the grasp succeeds physically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID
from autodex.utils.conversion import se32cart
from autodex.utils.path import get_obj_root


TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2


def _place(rest_pose: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rotation_z = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pose = np.asarray(rest_pose, dtype=np.float64).copy()
    pose[:3, :3] = rotation_z @ pose[:3, :3]
    pose[:3, 3] = [x, y, float(rest_pose[2, 3]) + TABLE_SURFACE_Z]
    return pose


def _scene(object_root: Path, object_name: str, pose: np.ndarray) -> dict:
    mesh = object_root / object_name / "processed_data" / "mesh" / "simplified.obj"
    return {
        "mesh": {"target": {"pose": se32cart(pose).tolist(), "file_path": str(mesh)}},
        "cuboid": {"table": TABLE_CUBOID},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="precision_key_1p5mm")
    parser.add_argument("--candidate-id", default="78")
    parser.add_argument("--version", default="v8")
    parser.add_argument("--pose-idx", default="000")
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=float(np.pi))
    parser.add_argument("--close-samples", type=int, default=31)
    parser.add_argument(
        "--output",
        type=Path,
        default=(Path.home() / "shared_data" / "AutoDex" / "precision_insertion" /
                 "visualizations" / "common_grasp_78_planned_trajectory.npz"),
    )
    args = parser.parse_args()
    if args.close_samples < 2:
        parser.error("--close-samples must be at least 2")

    object_root = Path(get_obj_root(args.version))
    rest_pose = np.load(
        object_root / args.object / "processed_data" / "info" / "tabletop" /
        f"{args.pose_idx}.npy"
    )
    object_pose = _place(rest_pose, args.x, args.y, args.yaw)
    scene = _scene(object_root, args.object, object_pose)

    planner = GraspPlanner(hand="fr3_inspire")
    result = planner.plan(
        scene,
        args.object,
        args.version,
        hand="inspire",
        skip_done=False,
        scene_id="0",
        scene_type_filter="table",
        tabletop_pose_stem=args.pose_idx,
        candidate_order=[["table", "0", str(args.candidate_id)]],
    )
    if not result.success or result.traj is None or result.lift_preflight is None:
        raise RuntimeError("candidate did not pass approach and lift planning")

    approach = np.asarray(result.traj, dtype=np.float32)
    lift = np.asarray(result.lift_preflight.traj, dtype=np.float32)
    if approach.ndim != 2 or approach.shape[1] != 13:
        raise RuntimeError(f"expected FR3+Inspire (T,13), got {approach.shape}")
    if lift.ndim != 2 or lift.shape[1] != 13:
        raise RuntimeError(f"expected FR3+Inspire lift (T,13), got {lift.shape}")

    close = np.repeat(approach[-1][None, :], args.close_samples, axis=0)
    alpha = np.linspace(0.0, 1.0, args.close_samples, dtype=np.float32)[:, None]
    pregrasp = np.asarray(result.pregrasp_pose, dtype=np.float32).reshape(1, -1)
    grasp = np.asarray(result.grasp_pose, dtype=np.float32).reshape(1, -1)
    close[:, 7:] = (1.0 - alpha) * pregrasp + alpha * grasp

    approach_to_close_arm_jump = float(np.max(np.abs(approach[-1, :7] - close[0, :7])))
    close_to_lift_jump = float(np.max(np.abs(close[-1] - lift[0])))
    if approach_to_close_arm_jump > 1.0e-5 or close_to_lift_jump > 1.0e-4:
        raise RuntimeError(
            "phase boundary is discontinuous: "
            f"approach/close={approach_to_close_arm_jump:.3g}, "
            f"close/lift={close_to_lift_jump:.3g} rad"
        )

    candidate_dir = (
        Path.home() / "shared_data" / "AutoDex" / "candidates" / "inspire" /
        args.version / args.object / "table" / "0" / str(args.candidate_id)
    )
    object_mesh = object_root / args.object / "processed_data" / "mesh" / "simplified.obj"
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        approach_q=approach,
        close_q=close,
        lift_q=lift,
        object_world_se3=object_pose,
        lift_start_wrist_se3=np.asarray(result.lift_preflight.start_wrist_se3),
        lift_target_wrist_se3=np.asarray(result.lift_preflight.target_wrist_se3),
        object_to_wrist_se3=np.asarray(result.wrist_se3),
        joint_names=np.asarray([
            "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
            "fr3_joint5", "fr3_joint6", "fr3_joint7",
            "right_thumb_1_joint", "right_thumb_2_joint",
            "right_index_1_joint", "right_middle_1_joint",
            "right_ring_1_joint", "right_little_1_joint",
        ]),
        object_mesh_path=np.asarray(str(object_mesh)),
        candidate_dir=np.asarray(str(candidate_dir)),
        object_name=np.asarray(args.object),
        candidate_id=np.asarray(str(args.candidate_id)),
    )

    metadata = {
        "schema_version": 1,
        "status": "hardware_free_plan_only",
        "object": args.object,
        "candidate": ["table", "0", str(args.candidate_id)],
        "tabletop_pose_stem": args.pose_idx,
        "object_pose": {"x_m": args.x, "y_m": args.y, "yaw_rad": args.yaw},
        "trajectory": {
            "joint_order": "7 FR3 joints followed by 6 Inspire actuators",
            "approach_waypoints": int(len(approach)),
            "close_waypoints": int(len(close)),
            "lift_waypoints": int(len(lift)),
            "lift_height_m": float(result.lift_preflight.height_m),
            "lift_time_s": (
                None if result.lift_preflight.time_s is None
                else float(result.lift_preflight.time_s[-1])
            ),
        },
        "continuity_max_rad": {
            "approach_to_close_arm": approach_to_close_arm_jump,
            "close_to_lift": close_to_lift_jump,
        },
        "warning": (
            "Visualization/planning artifact only. It has not been executed on "
            "the Franka, and it does not include reorientation or insertion."
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
