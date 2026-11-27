#!/usr/bin/env python3
"""Replay one saved post-squeeze lift without opening robot hardware."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from autodex.planner import GraspPlanner
from autodex.utils.conversion import cart2se3
from precision_insertion.assets import selected_robot_urdf
from precision_insertion.end_effector import end_effector_spec
from precision_insertion.planner_factory import selected_fr3_planner_config_options


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured-report", type=Path, required=True)
    parser.add_argument("--trial-scene", type=Path, required=True)
    parser.add_argument("--planned-trajectories", type=Path)
    parser.add_argument("--sweep-arm", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    measured = json.loads(args.measured_report.read_text(encoding="utf-8"))
    scene = json.loads(args.trial_scene.read_text(encoding="utf-8"))
    q = np.asarray(measured["measured_start"]["full_q"], dtype=np.float32)
    spec = end_effector_spec()
    urdf = selected_robot_urdf()
    planner = GraspPlanner(
        **selected_fr3_planner_config_options(spec),
        hand=spec.planner_hand,
        robot_urdf_path=str(urdf),
    )
    planner.warmup(scene, warmup_js_trajopt=False)
    target = scene["mesh"]["target"]
    object_pose = cart2se3(np.asarray(target["pose"], dtype=np.float64))

    variants = {"measured_full": (q, scene, object_pose)}
    no_socket = copy.deepcopy(scene)
    no_socket["mesh"] = {"target": no_socket["mesh"]["target"]}
    variants["measured_without_socket"] = (q, no_socket, object_pose)
    no_table = copy.deepcopy(scene)
    no_table["cuboid"] = {}
    variants["measured_without_table"] = (q, no_table, None)
    if args.planned_trajectories is not None:
        trajectories = np.load(args.planned_trajectories, allow_pickle=False)
        nominal = np.asarray(trajectories["held_lift"][0], dtype=np.float32)
        measured_arm_nominal_hand = q.copy()
        measured_arm_nominal_hand[7:] = nominal[7:]
        nominal_arm_measured_hand = nominal.copy()
        nominal_arm_measured_hand[7:] = q[7:]
        variants.update({
            "planned_full": (nominal, scene, object_pose),
            "measured_arm_nominal_hand_full": (
                measured_arm_nominal_hand, scene, object_pose),
            "nominal_arm_measured_hand_full": (
                nominal_arm_measured_hand, scene, object_pose),
        })
        if args.sweep_arm:
            for alpha in np.linspace(0.0, 1.0, 11):
                swept = nominal.copy()
                swept[:7] = (nominal[:7] + alpha * (q[:7] - nominal[:7]))
                swept[7:] = q[7:]
                variants[f"arm_residual_alpha_{alpha:.1f}"] = (
                    swept, scene, object_pose)

    results = {}
    for name, (state, world, payload) in variants.items():
        start = planner.fk_wrist(state)
        goal = start.copy()
        goal[2, 3] += 0.10
        result = planner.plan_vertical_stroke(
            state,
            start,
            goal,
            expected_travel_m=0.10,
            travel_tolerance_m=1.0e-5,
            scene_cfg=world,
            include_obj_obstacle=False,
            label=f"saved lift diagnostic: {name}",
            attached_object_pose_at_start=payload,
            return_result=True,
        )
        results[name] = {
            "success": bool(result.success),
            "failure_code": result.failure_code,
            "failure_detail": result.failure_detail,
            "direction": result.direction,
            "distance_m": result.distance_m,
            "completed_steps": sum(
                row.get("success") is True for row in result.step_records),
            "validation": result.validation,
            "timing": result.timing,
            "q": np.asarray(state).tolist(),
        }
    record = {
        "variants": results,
        "measured_q": q.tolist(),
        "robot_urdf": str(urdf),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
