#!/usr/bin/env python3
"""Find one shared tabletop pose where every precision key plans on FR3.

The four keys deliberately use the same grasp candidate.  This tool searches
one small, explicit object-pose grid and accepts a pose only when every key
passes the same checks: hand/table collision, IK, approach planning, and the
10 cm held-object lift preflight.  It then writes evidence beside each runtime
candidate plus a family-level report.

This is hardware-free validation.  It does not validate the physical cell,
fixture pose, Inspire tracking, or insertion.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID
from autodex.utils.conversion import se32cart
from autodex.utils.path import get_obj_root
from build_assets import KEY_SPECS


TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2) + "\n", encoding="utf-8")


def _place(pose: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rotation_z = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    transform = pose.copy()
    transform[:3, :3] = rotation_z @ pose[:3, :3]
    transform[0, 3] = x
    transform[1, 3] = y
    transform[2, 3] = float(pose[2, 3]) + TABLE_SURFACE_Z
    return transform


def _scene(object_root: Path, object_name: str, pose: np.ndarray) -> dict[str, Any]:
    mesh = object_root / object_name / "processed_data" / "mesh" / "simplified.obj"
    return {
        "mesh": {"target": {"pose": se32cart(pose).tolist(), "file_path": str(mesh)}},
        "cuboid": {"table": TABLE_CUBOID},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-id", default="78")
    parser.add_argument("--version", default="v8")
    parser.add_argument("--robot", default="fr3_inspire")
    parser.add_argument("--candidate-hand", default="inspire")
    parser.add_argument("--pose-idx", default="000")
    parser.add_argument("--x-grid", type=float, nargs="+", default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--y-grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--yaw-grid", type=float, nargs="+", default=[0.0])
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    args = parser.parse_args()

    shared_root = args.shared_root.expanduser().resolve()
    object_root = Path(get_obj_root(args.version))
    object_names = [spec[0] for spec in KEY_SPECS]
    tabletop = {
        name: np.load(
            object_root / name / "processed_data" / "info" / "tabletop" /
            f"{args.pose_idx}.npy"
        )
        for name in object_names
    }
    candidate_order = [["table", "0", str(args.candidate_id)]]
    attempts: list[dict[str, Any]] = []
    selected: tuple[float, float, float, dict[str, Any]] | None = None

    for x in args.x_grid:
        for y in args.y_grid:
            for yaw in args.yaw_grid:
                results: dict[str, Any] = {}
                all_passed = True
                for object_name in object_names:
                    # cuRobo's current batched world updater does not safely
                    # replace a prior mesh+cube world with another object's
                    # world.  A fresh planner also makes each result
                    # independent of the order in which keys are checked.
                    planner = GraspPlanner(hand=args.robot)
                    transform = _place(tabletop[object_name], x, y, yaw)
                    result = planner.plan(
                        _scene(object_root, object_name, transform),
                        object_name,
                        args.version,
                        hand=args.candidate_hand,
                        skip_done=False,
                        scene_id="0",
                        scene_type_filter="table",
                        tabletop_pose_stem=args.pose_idx,
                        candidate_order=candidate_order,
                    )
                    del planner
                    gc.collect()
                    torch.cuda.empty_cache()
                    record = {
                        "success": bool(result.success),
                        "scene_info": list(result.scene_info),
                        "trajectory_waypoints": (
                            None if result.traj is None else len(result.traj)
                        ),
                        "timing": result.timing,
                    }
                    results[object_name] = record
                    if not result.success:
                        all_passed = False
                        # A common pose requires every object. Stop evaluating
                        # this pose as soon as one object fails.
                        break
                attempt = {
                    "pose": {"x_m": x, "y_m": y, "yaw_rad": yaw},
                    "all_passed": all_passed,
                    "objects": results,
                }
                attempts.append(attempt)
                if all_passed:
                    selected = (x, y, yaw, results)
                    break
            if selected is not None:
                break
        if selected is not None:
            break

    report_path = (
        shared_root / "AutoDex" / "precision_insertion" /
        "common_grasp_franka_plan_validation.json"
    )
    report = {
        "schema_version": 1,
        "status": "passed" if selected is not None else "failed",
        "scope": "hardware_free_FR3_Inspire_approach_and_10cm_lift",
        "candidate_id": str(args.candidate_id),
        "objects": object_names,
        "attempts": attempts,
        "warning": "does not establish physical grasp or insertion success",
    }
    if selected is None:
        _write(report_path, report)
        print(f"no common pose passed; report: {report_path}")
        return 2

    x, y, yaw, results = selected
    report["selected_pose"] = {"x_m": x, "y_m": y, "yaw_rad": yaw}
    _write(report_path, report)
    for object_name, result in results.items():
        candidate_dir = (
            shared_root / "AutoDex" / "candidates" / args.candidate_hand /
            args.version / object_name / "table" / "0" / str(args.candidate_id)
        )
        _write(
            candidate_dir / "franka_plan_validation.json",
            {
                "schema_version": 1,
                "status": "passed",
                "scope": "hardware_free_FR3_Inspire_approach_and_10cm_lift",
                "planner_robot": args.robot,
                "candidate_hand": args.candidate_hand,
                "object": object_name,
                "candidate": ["table", "0", str(args.candidate_id)],
                "tabletop_pose_stem": args.pose_idx,
                "test_object_pose": {
                    "x_m": x,
                    "y_m": y,
                    "yaw_rad": yaw,
                    "z_policy": "AutoDex TABLE_SURFACE_Z",
                },
                "checks": {
                    "world_collision": "passed",
                    "self_collision": "passed",
                    "ik": "passed",
                    "approach_trajectory": "passed",
                    "vertical_lift_preflight": "passed",
                },
                "trajectory_waypoints": result["trajectory_waypoints"],
                "planner_timing_and_lift_evidence": result["timing"],
                "family_report": str(report_path),
                "warning": "hardware-free planning result; physical cell and insertion remain unvalidated",
            },
        )
    print(f"common pose passed at x={x:.3f}, y={y:.3f}, yaw={yaw:.3f}")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
