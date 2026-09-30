#!/usr/bin/env python3
"""Replan representative trajectories for a run's greedy base grasps."""
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
    build_scene,
    evaluate_candidate_group,
    load_candidate_catalogue,
    load_tabletop_transform,
    read_jsonl,
    tabletop_files,
    write_json,
)


def _pose_matrix(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    transforms = np.repeat(np.eye(4)[None], len(position), axis=0)
    transforms[:, :3, 3] = position
    transforms[:, :3, :3] = Rotation.from_quat(
        quaternion_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    return transforms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-grasps", type=int, default=6)
    parser.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    parser.add_argument("--seed-attempts", type=int, default=16,
                        help="Deterministic seeds tried for each marginal cell")
    parser.add_argument("--ranks", type=int, nargs="*", default=None,
                        help="Only recompute these 1-based greedy ranks")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  run_dir / "greedy_trajectory_replays")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "manifest.json").open() as stream:
        manifest = json.load(stream)
    with (run_dir / "summary.json").open() as stream:
        summary = json.load(stream)
    with (run_dir / "candidate_snapshot.json").open() as stream:
        snapshot = json.load(stream)
    matrix = np.load(run_dir / "coverage_matrix.npz")
    rows = read_jsonl(run_dir / "per_grasp.jsonl")
    row_lookup = {(str(row["candidate_key_str"]), str(row["cell_id"])): row
                  for row in rows}
    grasp_keys = [str(value) for value in matrix["grasp_keys"]]
    cell_ids = [str(value) for value in matrix["cell_ids"]]
    success = np.asarray(matrix["pipeline_success"], dtype=bool)
    indices = summary["greedy"]["pipeline_success"]["candidate_indices"][
        :args.max_grasps]
    keys = summary["greedy"]["pipeline_success"]["candidate_keys"][
        :args.max_grasps]
    if len(indices) != args.max_grasps:
        raise SystemExit(
            f"requested {args.max_grasps} greedy grasps, found {len(indices)}")

    representatives = []
    covered = np.zeros(len(cell_ids), dtype=bool)
    for rank, (index, key) in enumerate(zip(indices, keys), start=1):
        gained = success[index] & ~covered
        if not gained.any():
            raise SystemExit(f"greedy grasp has no marginal cell: {key}")
        cell_index = int(np.flatnonzero(gained)[0])
        cell_id = cell_ids[cell_index]
        source = row_lookup[(key, cell_id)]
        representatives.append({
            "rank": rank,
            "candidate_key": key.split("/"),
            "candidate_key_str": key,
            "cell_id": cell_id,
            "r_m": float(source["r_m"]),
            "theta_deg": float(source["theta_deg"]),
            "seed": int(source["seed"]),
            "marginal_gain": int(gained.sum()),
            "cumulative_coverage": int((covered | success[index]).sum()),
            # A representative must come from the cells this grasp adds at
            # its greedy step.  Do not silently fall back to an already
            # covered cell merely because replay numerics differ by GPU.
            "candidate_cell_ids": [cell_ids[i] for i in np.flatnonzero(gained)],
        })
        covered |= success[index]

    contract = {
        "source_run": str(run_dir),
        "object": manifest["object"],
        "pose": manifest["tabletop_pose_stem"],
        "stage": "pipeline_success",
        "representative_policy": (
            "first replayable newly-covered cell in matrix order; no fallback "
            "outside the greedy marginal set"),
        "greedy_grasps": representatives,
    }
    from autodex.planner import GraspPlanner

    catalogue = load_candidate_catalogue(
        obj=manifest["object"], version=manifest["version"], hand=manifest["hand"],
        pose_stem=manifest["tabletop_pose_stem"], pool="all",
        clean_state_root=output_dir / "candidate_state_clean")
    current_signature = [
        (group["candidate_key"], group["variant_count"])
        for group in catalogue.snapshot()["groups"]]
    source_signature = [
        (group["candidate_key"], group["variant_count"])
        for group in snapshot["groups"]]
    if current_signature != source_signature:
        raise SystemExit("current candidate catalogue differs from source snapshot")
    group_by_key = {"/".join(group["candidate_key"]): group
                    for group in catalogue.groups}
    planner = GraspPlanner(hand=manifest["planner_robot"],
                           use_cuda_graph=args.cuda_graph == "on")
    pose_file = next(
        (path for path in tabletop_files(manifest["object"], manifest["version"])
         if path.stem == manifest["tabletop_pose_stem"]),
        None)
    if pose_file is None:
        raise SystemExit(
            f"tabletop pose asset not found on this host: "
            f"{manifest['object']}/{manifest['tabletop_pose_stem']}")
    first = representatives[0]
    raw = load_tabletop_transform(
        pose_file, first["r_m"], first["theta_deg"], manifest["table_surface_z_m"])
    warm_scene, _ = build_scene(
        manifest["object"], manifest["version"], raw, manifest["table_surface_z_m"])
    print(f"[warmup] {manifest['planner_robot']}", flush=True)
    planner.warmup(warm_scene)

    import torch

    records_path = output_dir / "replay_records.json"
    existing_records = (json.loads(records_path.read_text())
                        if records_path.is_file() else [])
    replay_records = {str(record["candidate_key_str"]): record
                      for record in existing_records}
    for rep in representatives:
        stem = f"{rep['rank']:02d}_{rep['candidate_key_str'].replace('/', '_')}"
        animation_path = output_dir / f"{stem}.npz"
        if animation_path.is_file():
            saved = np.load(animation_path)
            rep.update({
                "cell_id": str(saved["cell_id"]),
                "r_m": float(saved["r_m"]),
                "theta_deg": float(saved["theta_deg"]),
            })
        if args.ranks is not None and rep["rank"] not in args.ranks:
            continue
        if animation_path.exists() and not args.overwrite:
            print(f"[skip] {animation_path}", flush=True)
            continue
        record = None
        object_pose = None
        attempt_count = 0
        for cell_id in rep.pop("candidate_cell_ids"):
            source = row_lookup[(rep["candidate_key_str"], cell_id)]
            raw = load_tabletop_transform(
                pose_file, source["r_m"], source["theta_deg"],
                manifest["table_surface_z_m"])
            scene_cfg, candidate_object_pose = build_scene(
                manifest["object"], manifest["version"], raw,
                manifest["table_surface_z_m"])
            cell = {
                "cell_id": cell_id, "r_m": float(source["r_m"]),
                "theta_deg": float(source["theta_deg"]),
                "r_index": int(source["r_index"]),
                "theta_index": int(source["theta_index"]),
                "nominal_x_m": float(source["nominal_x_m"]),
                "nominal_y_m": float(source["nominal_y_m"]),
                "object_x_m": float(candidate_object_pose[0, 3]),
                "object_y_m": float(candidate_object_pose[1, 3]),
                "object_z_m": float(candidate_object_pose[2, 3]),
            }
            for seed_offset in range(args.seed_attempts):
                attempt_count += 1
                seed = int(source["seed"]) + seed_offset
                candidate_record = evaluate_candidate_group(
                    planner, catalogue=catalogue,
                    group=group_by_key[rep["candidate_key_str"]], scene_cfg=scene_cfg,
                    object_pose=candidate_object_pose, cell=cell, trial=seed_offset,
                    seed=seed, trajectory_dir=output_dir / "trajectories")
                print(
                    f"[replay] grasp={rep['rank']}/{len(representatives)} "
                    f"cell={cell_id} seed={seed} "
                    f"{'ok' if candidate_record['pipeline_success'] else candidate_record['failure_code']}",
                    flush=True)
                if candidate_record["pipeline_success"]:
                    record = candidate_record
                    object_pose = candidate_object_pose
                    rep.update({
                        "cell_id": cell_id, "r_m": float(source["r_m"]),
                        "theta_deg": float(source["theta_deg"]), "seed": seed,
                        "replay_attempt_count": attempt_count,
                    })
                    break
            if record is not None:
                break
        if record is None or object_pose is None:
            raise RuntimeError(
                f"could not reproduce any successful representative for "
                f"{rep['candidate_key_str']}")
        raw_path = output_dir / record["trajectory_file"]
        data = np.load(raw_path)
        approach = np.asarray(data["approach_qpos"], dtype=np.float32)
        lift = np.asarray(data["lift_qpos"], dtype=np.float32)
        squeeze_count = 16
        squeeze = np.linspace(approach[-1], lift[0], squeeze_count + 2,
                              dtype=np.float32)[1:-1]
        qpos = np.concatenate([approach, squeeze, lift], axis=0)
        phases = np.asarray(
            ["approach"] * len(approach) + ["squeeze"] * len(squeeze)
            + ["lift"] * len(lift))
        with torch.inference_mode():
            state = planner._ik_solver.fk(torch.as_tensor(
                qpos, dtype=torch.float32, device=planner._tensor_args.device))
        ee_position = state.ee_position.detach().cpu().numpy()
        ee_quaternion = state.ee_quaternion.detach().cpu().numpy()
        link_positions = state.links_position.detach().cpu().numpy()
        link_names = np.asarray(state.link_names)
        ee_poses = _pose_matrix(ee_position, ee_quaternion)
        object_poses = np.repeat(object_pose[None], len(qpos), axis=0)
        lift_start = len(approach) + len(squeeze)
        relative = np.linalg.inv(ee_poses[lift_start]) @ object_pose
        object_poses[lift_start:] = np.matmul(ee_poses[lift_start:], relative)
        np.savez_compressed(
            animation_path, qpos=qpos, phases=phases,
            ee_position=ee_position, ee_quaternion=ee_quaternion,
            link_positions=link_positions, link_names=link_names,
            object_poses=object_poses, object_pose_initial=object_pose,
            candidate_key=np.asarray(rep["candidate_key"]), cell_id=rep["cell_id"],
            r_m=rep["r_m"], theta_deg=rep["theta_deg"],
            seed=rep["seed"],
            marginal_gain=rep["marginal_gain"],
            cumulative_coverage=rep["cumulative_coverage"])
        record["animation_npz"] = animation_path.name
        replay_records[rep["candidate_key_str"]] = record
        print(f"[saved] {animation_path}", flush=True)
    contract["greedy_grasps"] = [
        {key: value for key, value in rep.items() if key != "candidate_cell_ids"}
        for rep in representatives]
    write_json(records_path, sorted(
        replay_records.values(), key=lambda record: int(
            next(rep["rank"] for rep in representatives
                 if rep["candidate_key_str"] == record["candidate_key_str"]))))
    write_json(output_dir / "replay_manifest.json", contract)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
