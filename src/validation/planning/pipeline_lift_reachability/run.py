#!/usr/bin/env python3
"""Run production-path lift reachability without perception or robot execution."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.validation.planning.pipeline_lift_reachability.analysis import analyze_run
from src.validation.planning.pipeline_lift_reachability.core import (
    available_objects, build_scene, evaluate_pose, load_candidate_catalogue,
    load_tabletop_transform, planner_robot_for, polar_cells, tabletop_files,
    write_json,
)


def _float_range(lower: float, upper: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("grid step must be positive")
    if upper < lower:
        raise ValueError("grid maximum must be >= minimum")
    count = int(np.floor((upper - lower) / step + 1.0e-9)) + 1
    return [float(round(lower + index * step, 9)) for index in range(count)]


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-grasp and full-pool production-path lift reachability")
    parser.add_argument("--obj", default=None,
                        help="candidate object; omit to process every available object")
    parser.add_argument("--arm", default="xarm", choices=["xarm", "franka"])
    parser.add_argument("--hand", default="inspire",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--version", default="v8")
    parser.add_argument("--tabletop-pose", default=None,
                        help="pose filename stem; omit to process every tabletop pose")
    parser.add_argument("--candidate-pool", default="all",
                        choices=["all", "training-remaining", "verified-only"])
    parser.add_argument("--evaluation", default="both",
                        choices=["per-grasp", "pipeline-replay", "both"])
    parser.add_argument("--r-min", type=float, default=0.20)
    parser.add_argument("--r-max", type=float, default=0.60)
    parser.add_argument("--r-step", type=float, default=0.05)
    parser.add_argument("--theta-step", type=float, default=30.0)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-grasps", type=int, default=0,
                        help=("per-grasp debug cap; pipeline-replay always uses the "
                              "complete pool; 0 evaluates every base grasp"))
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="save successful approach/lift trajectories (off by default)")
    parser.add_argument("--table-surface-z-m", type=float, default=0.040)
    parser.add_argument("--output-root", default=str(_REPO / "outputs" / "reachability"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.n_trials <= 0:
        raise SystemExit("--n-trials must be positive")
    if args.max_grasps < 0:
        raise SystemExit("--max-grasps must be >= 0")
    try:
        planner_robot = planner_robot_for(args.arm, args.hand)
        radii = _float_range(args.r_min, args.r_max, args.r_step)
        if args.theta_step <= 0.0 or args.theta_step > 360.0:
            raise ValueError("--theta-step must be in (0, 360]")
        thetas = [float(value) for value in np.arange(0.0, 360.0, args.theta_step)]
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    objects = [args.obj] if args.obj else available_objects(args.hand, args.version)
    if not objects:
        raise SystemExit(
            f"no candidate objects under hand={args.hand} version={args.version}")
    unknown = [obj for obj in objects
               if obj not in available_objects(args.hand, args.version)]
    if unknown:
        raise SystemExit(f"objects missing from candidate pool: {', '.join(unknown)}")

    from autodex.planner import GraspPlanner

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    planner = GraspPlanner(
        hand=planner_robot, use_cuda_graph=args.cuda_graph == "on")
    setup_started = perf_counter()
    warmed = False
    planner_setup_s = None
    completed_dirs: list[str] = []

    for obj in objects:
        poses = tabletop_files(obj, args.version)
        if args.tabletop_pose is not None:
            poses = [path for path in poses if path.stem == args.tabletop_pose]
        if not poses:
            print(f"[skip] {obj}: requested/no tabletop pose", flush=True)
            continue
        for pose_file in poses:
            output_dir = (Path(args.output_root) / args.hand / obj / "pipeline_lift"
                          / args.arm / args.version / pose_file.stem / run_id)
            if output_dir.exists() and not args.resume:
                raise SystemExit(
                    f"output exists: {output_dir}\nUse --resume or choose another --run-id.")
            output_dir.mkdir(parents=True, exist_ok=True)
            catalogue = load_candidate_catalogue(
                obj=obj, version=args.version, hand=args.hand,
                pose_stem=pose_file.stem, pool=args.candidate_pool,
                clean_state_root=output_dir / "candidate_state_clean")
            snapshot = catalogue.snapshot()
            snapshot_path = output_dir / "candidate_snapshot.json"
            if args.resume and snapshot_path.is_file():
                with snapshot_path.open() as stream:
                    saved_snapshot = json.load(stream)
                if saved_snapshot != snapshot:
                    raise SystemExit(
                        f"candidate snapshot changed; refusing an unsafe resume: {output_dir}")
            else:
                write_json(snapshot_path, snapshot)
            cells = polar_cells(radii, thetas)
            manifest = {
                "schema_version": 1,
                "git_revision": _git_revision(),
                "object": obj, "arm": args.arm, "hand": args.hand,
                "planner_robot": planner_robot, "version": args.version,
                "tabletop_pose_stem": pose_file.stem,
                "tabletop_pose_file": str(pose_file),
                "candidate_pool": args.candidate_pool,
                "evaluation": args.evaluation,
                "grid": {"type": "polar", "radii_m": radii,
                         "thetas_deg": thetas, "cell_count": len(cells)},
                "lift": {"height_m": 0.10, "jacobian_node_step_m": 0.005,
                         "direction": "+Z"},
                "n_trials": args.n_trials, "base_seed": args.seed,
                "max_grasps": args.max_grasps,
                "cuda_graph": args.cuda_graph,
                "table_surface_z_m": args.table_surface_z_m,
                "authoritative_success": "GraspPlanner.plan",
                "endpoint_ik_role": "diagnostic_only",
            }
            manifest_path = output_dir / "manifest.json"
            if args.resume and manifest_path.is_file():
                with manifest_path.open() as stream:
                    saved_manifest = json.load(stream)
                if saved_manifest != manifest:
                    raise SystemExit(
                        f"run contract changed; refusing an unsafe resume: {output_dir}")
            else:
                write_json(manifest_path, manifest)
            print(
                f"[run] {obj} pose={pose_file.stem} arm={args.arm} hand={args.hand} "
                f"pool={args.candidate_pool} base_grasps={len(catalogue.groups)} "
                f"expanded={len(catalogue.wrist_object)} cells={len(cells)}",
                flush=True)
            if not catalogue.groups:
                write_json(output_dir / "summary.json", {
                    "schema_version": 1, "status": "empty_candidate_pool",
                    "object": obj, "tabletop_pose_stem": pose_file.stem})
                continue
            if not warmed:
                raw_pose = load_tabletop_transform(
                    pose_file, radii[0], thetas[0], args.table_surface_z_m)
                warm_scene, _ = build_scene(
                    obj, args.version, raw_pose, args.table_surface_z_m)
                print(f"[planner] warming up {planner_robot}...", flush=True)
                planner.warmup(warm_scene)
                warmed = True
                planner_setup_s = perf_counter() - setup_started
            timing = evaluate_pose(
                planner=planner, catalogue=catalogue, pose_file=pose_file,
                cells=cells, output_dir=output_dir, evaluation=args.evaluation,
                n_trials=args.n_trials, base_seed=args.seed,
                max_grasps=(None if args.max_grasps == 0 else args.max_grasps),
                save_trajectories=args.save_trajectories,
                table_surface_z_m=args.table_surface_z_m)
            summary = analyze_run(output_dir)
            summary["run_wall_s"] = timing["wall_s"]
            summary["planner_setup_s"] = planner_setup_s
            write_json(output_dir / "summary.json", summary)
            completed_dirs.append(str(output_dir))
            print(f"[result] {output_dir}", flush=True)
            print(f"[plot] {output_dir / 'plots' / 'greedy_coverage.png'}", flush=True)
    if not completed_dirs:
        raise SystemExit("no object/tabletop-pose run was completed")
    print("[complete]")
    for directory in completed_dirs:
        print(f"  {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
