#!/usr/bin/env python3
"""Search successful per-grasp records for a visible, table-safe legacy wobble."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validation.planning.pipeline_lift_reachability.core import (
    build_scene, evaluate_candidate_group, load_candidate_catalogue,
    load_tabletop_transform, read_jsonl, tabletop_files, write_json)
from src.validation.planning.pipeline_lift_reachability.recompute_legacy_lift_comparison import (
    LEGACY_LIFT_HOLD_MASK, _fk, _metrics, _same_full_path_validation)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-cases", type=int, default=40)
    parser.add_argument("--stop-lateral-mm", type=float, default=20.0)
    parser.add_argument("--seed-attempts", type=int, default=4)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  run_dir / "wobbly_legacy_lift_search")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [row for row in read_jsonl(run_dir / "per_grasp.jsonl")
            if row.get("pipeline_success")]
    # Near-singular successful Jacobian cases are the most useful probes for
    # endpoint-IK branch/path wobble.  Keep one record per grasp+cell.
    rows.sort(key=lambda row: (
        float((row.get("stroke") or {}).get("max_condition_number", 0.0)),
        float(row["r_m"])), reverse=True)
    rows = rows[:args.max_cases]

    catalogue = load_candidate_catalogue(
        obj=manifest["object"], version=manifest["version"], hand=manifest["hand"],
        pose_stem=manifest["tabletop_pose_stem"], pool="all",
        clean_state_root=output_dir / "candidate_state_clean")
    group_by_key = {"/".join(group["candidate_key"]): group
                    for group in catalogue.groups}
    pose_file = next(path for path in tabletop_files(
        manifest["object"], manifest["version"])
        if path.stem == manifest["tabletop_pose_stem"])

    from autodex.planner import GraspPlanner
    from autodex.planner.planner import _to_curobo_world, _without_target_mesh
    import trimesh
    planner = GraspPlanner(hand=manifest["planner_robot"], use_cuda_graph=True)
    first = rows[0]
    raw = load_tabletop_transform(
        pose_file, first["r_m"], first["theta_deg"],
        manifest["table_surface_z_m"])
    scene, _ = build_scene(manifest["object"], manifest["version"], raw,
                           manifest["table_surface_z_m"])
    planner.warmup(scene)
    object_vertices = np.asarray(trimesh.load(
        scene["mesh"]["target"]["file_path"], force="mesh",
        process=False).vertices)

    results = []
    best = None
    for order, source in enumerate(rows, 1):
        key = str(source["candidate_key_str"])
        raw = load_tabletop_transform(
            pose_file, source["r_m"], source["theta_deg"],
            manifest["table_surface_z_m"])
        scene, object_pose = build_scene(
            manifest["object"], manifest["version"], raw,
            manifest["table_surface_z_m"])
        cell = {name: source[name] for name in (
            "cell_id", "r_m", "theta_deg", "r_index", "theta_index",
            "nominal_x_m", "nominal_y_m")}
        cell.update({
            "object_x_m": float(object_pose[0, 3]),
            "object_y_m": float(object_pose[1, 3]),
            "object_z_m": float(object_pose[2, 3]),
        })
        current_record = None
        for offset in range(args.seed_attempts):
            current_record = evaluate_candidate_group(
                planner, catalogue=catalogue, group=group_by_key[key],
                scene_cfg=scene, object_pose=object_pose, cell=cell,
                trial=offset, seed=int(source["seed"]) + offset,
                trajectory_dir=output_dir / "current_trajectories")
            if current_record["pipeline_success"]:
                break
        if not current_record or not current_record["pipeline_success"]:
            print(f"[{order}/{len(rows)}] current replay failed {key}", flush=True)
            continue
        current_raw = np.load(
            output_dir / "current_trajectories"
            / Path(current_record["trajectory_file"]).name)
        current_qpos = np.asarray(current_raw["lift_qpos"], dtype=np.float32)
        start = current_qpos[0]
        planner._set_motion_world(_without_target_mesh(_to_curobo_world(scene)))
        start_pose = planner.fk_wrist(start)
        target_pose = start_pose.copy()
        target_pose[2, 3] += 0.10
        legacy = planner._plan_endpoint_approximation(
            start, target_pose, LEGACY_LIFT_HOLD_MASK,
            scene_cfg=scene, include_obj_obstacle=False, debug_dump_dir=None,
            return_result=True, timing_parent_id=None, timing_phase="validation")
        if not legacy.success or legacy.trajectory is None:
            print(f"[{order}/{len(rows)}] legacy failed {key}", flush=True)
            continue
        legacy_qpos = np.asarray(legacy.trajectory, dtype=np.float32)
        legacy_position, legacy_poses = _fk(planner, legacy_qpos)
        current_position, current_poses = _fk(planner, current_qpos)
        relative = np.linalg.inv(current_poses[0]) @ object_pose
        legacy_object = legacy_poses @ relative
        current_object = current_poses @ relative
        legacy_validation = _same_full_path_validation(
            planner, legacy_qpos, legacy_poses, legacy_object, object_vertices)
        current_validation = _same_full_path_validation(
            planner, current_qpos, current_poses, current_object, object_vertices)
        metrics = _metrics(legacy_poses)
        result = {
            "candidate_key": key.split("/"), "candidate_key_str": key,
            "cell_id": source["cell_id"], "r_m": source["r_m"],
            "theta_deg": source["theta_deg"],
            "source_condition_number": (source.get("stroke") or {}).get(
                "max_condition_number"),
            "legacy_metrics": metrics,
            "current_metrics": _metrics(current_poses),
            "legacy_same_full_path_validation": legacy_validation,
            "current_same_full_path_validation": current_validation,
        }
        eligible = bool(
            legacy_validation["joint_limits_pass"]
            and legacy_validation["world_self_collision_pass"]
            and legacy_validation["object_table_clearance_pass"]
            and current_validation["passed"])
        result["eligible"] = eligible
        results.append(result)
        lateral_mm = metrics["max_lateral_deviation_m"] * 1000.0
        print(f"[{order}/{len(rows)}] {key} {source['cell_id']} "
              f"lateral={lateral_mm:.1f}mm table={int(legacy_validation['object_table_clearance_pass'])}",
              flush=True)
        if eligible and (best is None or metrics["max_lateral_deviation_m"]
                         > best[0]["legacy_metrics"]["max_lateral_deviation_m"]):
            comparison = output_dir / "selected_legacy_vs_jacobian.npz"
            np.savez_compressed(
                comparison,
                legacy_qpos=legacy_qpos, legacy_ee_position=legacy_position,
                legacy_object_poses=legacy_object,
                current_qpos=current_qpos, current_ee_position=current_position,
                current_object_poses=current_object,
                start_wrist_pose=current_poses[0], target_wrist_pose=target_pose,
                object_pose_initial=object_pose,
                candidate_key=np.asarray(key.split("/")),
                cell_id=source["cell_id"], r_m=source["r_m"],
                theta_deg=source["theta_deg"])
            result["comparison_npz"] = comparison.name
            best = (result, comparison)
        if best is not None and best[0]["legacy_metrics"]["max_lateral_deviation_m"] * 1000 >= args.stop_lateral_mm:
            break
    if best is None:
        raise RuntimeError("no table-safe legacy comparison found")
    report = {
        "definition": {
            "selection": "maximum legacy lateral deviation after all non-vertical safety checks pass",
            "historical_acceptance": (
                "endpoint IK success followed by cuRobo plan_single_js success "
                "with hold-mask [rx,ry,rz,x,y]=1, z=0; no wrist-path or "
                "attached-object clearance constraint"),
            "posthoc_selection_note": (
                "joint/world/self/object-table checks below are stricter "
                "comparison controls and were not all enforced by legacy"),
            "required": ["joint_limits", "world_self_collision",
                         "object_table_clearance", "current_full_path_pass"],
        },
        "selected": best[0], "all_cases": results,
    }
    write_json(output_dir / "comparison_summary.json", report)
    print(f"[selected] {best[0]['candidate_key_str']} {best[0]['cell_id']} "
          f"{best[0]['legacy_metrics']['max_lateral_deviation_m'] * 1000:.1f}mm",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
