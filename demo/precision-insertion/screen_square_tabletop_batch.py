#!/usr/bin/env python3
"""Screen a complete square-key v8 grasp batch at 20 mm.

The original AutoDex cuRobo/squeeze/MuJoCo filter runs first. This command
checks its complete raw/pass output and adds the exact socket/key/whole-hand
endpoint test. Results are offline evidence, not installed runtime candidates
or claims about continuous Franka motion and contact insertion.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path

from precision_insertion.config import select_mode
from precision_insertion.endpoint import screen_grasp_endpoint
from precision_insertion.stock_tabletop import audit_tabletop_cell


GAPS_MM = (0.3, 0.5, 1.0, 1.5)
SCENES = tuple(str(index) for index in range(5))


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _screen_candidate(task: tuple) -> dict:
    """One independent exact endpoint check; safe for spawned CPU workers."""
    shared_root, mode, candidate, minimum_hand_clearance_m = task
    try:
        return screen_grasp_endpoint(
            shared_root=shared_root, mode=mode, candidate_dir=candidate,
            minimum_hand_clearance_m=minimum_hand_clearance_m)
    except Exception as exc:
        return {
            "endpoint_pass": False,
            "error": f"{type(exc).__name__}: {exc}",
            "candidate_dir": str(candidate),
        }


def run(*, shared_root: Path, raw_root: Path, pass_root: Path,
        output_root: Path, expected_per_scene: int,
        minimum_hand_clearance_m: float,
        gap_mm: tuple[float, ...] = GAPS_MM,
        scene_ids: tuple[str, ...] = SCENES,
        workers: int = 1) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite screen: {output_root}")
    if not 0 < minimum_hand_clearance_m < 0.001:
        raise ValueError("give an explicit positive numerical clearance below 1 mm")
    if not gap_mm or len(set(gap_mm)) != len(gap_mm) or any(
            gap not in GAPS_MM for gap in gap_mm):
        raise ValueError("select one or more distinct configured square gaps")
    if (type(expected_per_scene) is not int or expected_per_scene < 1 or
            type(workers) is not int or workers < 1 or workers > 16 or
            not scene_ids or len(set(scene_ids)) != len(scene_ids) or
            any(scene not in SCENES for scene in scene_ids)):
        raise ValueError("select distinct v8 tabletop scenes and a positive count")
    audited = {}
    for gap in gap_mm:
        mode = select_mode("square", gap)
        rows = []
        scene_counts = {}
        for scene in scene_ids:
            scene_counts[scene], passed = audit_tabletop_cell(
                raw_root=raw_root, pass_root=pass_root,
                key_object=mode.key_object, scene_id=scene,
                expected_seed_count=expected_per_scene)
            rows.extend((scene, seed_id, candidate) for seed_id, candidate in passed)
        audited[str(gap)] = (mode, scene_counts, rows)

    output_root.mkdir(parents=True, exist_ok=False)
    modes = {}
    for gap in gap_mm:
        mode, scene_counts, passed = audited[str(gap)]
        eligible = []
        errors = []
        endpoint_counts = {scene: {"tested": 0, "endpoint_pass": 0}
                           for scene in scene_ids}
        tasks = ((shared_root, mode, candidate, minimum_hand_clearance_m)
                 for _scene, _seed_id, candidate in passed)
        if workers == 1:
            results = map(_screen_candidate, tasks)
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers, mp_context=get_context("spawn"))
            results = executor.map(_screen_candidate, tasks)
        try:
            screened = zip(passed, results)
            for (scene, seed_id, candidate), result in screened:
                endpoint_counts[scene]["tested"] += 1
                if "error" in result:
                    errors.append(f"table/{scene}/{seed_id}: {result['error']}")
                result["original_autodex_sim_filter_pass"] = True
                result["original_autodex_sim_eval"] = str(
                    raw_root / mode.key_object / "table" / scene /
                    seed_id / "sim_eval.json")
                result["scope_note"] = (
                    "Aligned 20 mm endpoint only; no full Franka path, contact "
                    "dynamics, or physical insertion success")
                _write(output_root / mode.key_object / "table" / scene /
                       f"{seed_id}.json", result)
                if result["endpoint_pass"]:
                    endpoint_counts[scene]["endpoint_pass"] += 1
                    eligible.append(f"table/{scene}/{seed_id}")
        finally:
            if workers > 1:
                executor.shutdown(wait=True, cancel_futures=True)
        modes[mode.key_object] = {
            "gap_mm": gap,
            "socket_object": mode.socket_object,
            "scene_counts": scene_counts,
            "endpoint_counts": endpoint_counts,
            "eligible_offline_grasp_ids": eligible,
            "errors": errors,
        }
        print(f"{mode.key_object}: {len(eligible)}/{len(passed)} stable "
              "grasps have a clear exact 20 mm endpoint", flush=True)
    summary = {
        "schema": ("precision_insertion_square_tabletop_1000_screen_v1"
                   if expected_per_scene == 1000 and scene_ids == SCENES else
                   "precision_insertion_square_tabletop_batch_screen_v2"),
        "status": "offline_grasp_and_20mm_endpoint_only_not_robot_ready",
        "expected_proposals_per_scene": expected_per_scene,
        "expected_tabletop_scenes_per_key": len(scene_ids),
        "scene_ids": list(scene_ids),
        "raw_root": str(raw_root),
        "original_autodex_sim_candidate_root": str(pass_root),
        "minimum_hand_clearance_m": minimum_hand_clearance_m,
        "clearance_commissioned": False,
        "original_autodex_filter": [
            "cuRobo tabletop pregrasp scene and self collision",
            "cuRobo full-key squeeze contact",
            "MuJoCo full-key squeeze/gravity stability",
        ],
        "additional_filter": (
            "centered 20 mm exact-CAD key/socket fit and whole Inspire "
            "hand/socket endpoint non-collision"),
        "modes": modes,
        "robot_ready": False,
    }
    _write(output_root / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pass-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-per-scene", type=int, default=1000)
    parser.add_argument("--minimum-hand-clearance-m", type=float, required=True)
    parser.add_argument("--gap-mm", type=float, nargs="+",
                        default=list(GAPS_MM), choices=GAPS_MM,
                        help="screen selected modes; defaults to all four")
    parser.add_argument("--scene-id", dest="scene_ids", nargs="+",
                        default=list(SCENES), choices=SCENES,
                        help="screen only these complete v8 tabletop cells")
    parser.add_argument("--workers", type=int, default=1,
                        help="spawn independent CPU endpoint screens (default 1)")
    args = parser.parse_args()
    report = run(**{**vars(args), "gap_mm": tuple(args.gap_mm),
                    "scene_ids": tuple(args.scene_ids)})
    print(json.dumps({name: {
        "mujoco_stable": sum(row["mujoco_stable"] for row in mode["scene_counts"].values()),
        "endpoint_eligible": len(mode["eligible_offline_grasp_ids"]),
        "errors": len(mode["errors"]),
    } for name, mode in report["modes"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
