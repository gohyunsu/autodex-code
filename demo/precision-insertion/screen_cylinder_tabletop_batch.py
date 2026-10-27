#!/usr/bin/env python3
"""Audit AutoDex full-key sim filtering, then screen every socket at 20 mm.

Reads the staged 1,000-per-scene results and the original AutoDex
``run_sim_filter`` candidate output. Writes *offline* evidence only, never
robot commands or the live v8 candidate tree. Every socket gap is evaluated
independently against its exact CAD collision mesh.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from precision_insertion.assets import AssetPaths
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.endpoint import screen_grasp_endpoint
from precision_insertion.stock_tabletop import audit_tabletop_cell

KEY = "precision_key_cylinder_r15_h80"
SCENES = ("0", "1")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def _screen_gap(shared_root: Path, output_root: Path,
                all_passed: list[tuple[str, str, Path]],
                stage_root: Path, gap: float, clearance_m: float,
                scene_ids: tuple[str, ...]) -> tuple[str, dict]:
    mode = select_mode("cylinder", gap)
    label = f"{int(gap):02d}mm"
    gap_dir = output_root / f"gap_{label}"
    eligible = []
    errors = []
    by_scene = {scene: {"tested": 0, "endpoint_pass": 0} for scene in scene_ids}
    for scene_id, name, directory in all_passed:
        by_scene[scene_id]["tested"] += 1
        target = gap_dir / "table" / scene_id / f"{name}.json"
        try:
            result = screen_grasp_endpoint(
                shared_root=shared_root, mode=mode, candidate_dir=directory,
                minimum_hand_clearance_m=clearance_m)
        except Exception as exc:
            result = {"endpoint_pass": False,
                      "error": f"{type(exc).__name__}: {exc}",
                      "candidate_dir": str(directory), "gap_mm": gap}
            errors.append(f"table/{scene_id}/{name}: {result['error']}")
        result["original_autodex_sim_filter_pass"] = True
        result["original_autodex_sim_eval"] = str(
            stage_root / KEY / "table" / scene_id / name / "sim_eval.json")
        result["note"] = (
            "Endpoint-only geometry; no Franka path or physical insertion was tested.")
        _write(target, result)
        if result["endpoint_pass"]:
            by_scene[scene_id]["endpoint_pass"] += 1
            eligible.append(f"table/{scene_id}/{name}")
    return label, {
        "socket_object": mode.socket_object,
        "by_scene": by_scene,
        "eligible_offline_grasp_ids": eligible,
        "errors": errors,
    }


def run(shared_root: Path, stage_root: Path, candidate_root: Path,
        output_root: Path, expected: int, clearance_m: float,
        workers: int = 1) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite results: {output_root}")
    manifest = json.loads((stage_root / "stage_manifest.json").read_text(encoding="utf-8"))
    scene_ids = tuple(row["scene_id"] for row in manifest["scenes"])
    if (not scene_ids or len(set(scene_ids)) != len(scene_ids) or
            not set(scene_ids) <= set(SCENES) or
            manifest["seed_count_per_scene"] != expected or
            manifest["seed_count_total"] != len(scene_ids) * expected):
        raise ValueError("stage manifest does not match requested 1,000-per-scene run")
    if not 0 <= clearance_m < 0.001:
        raise ValueError("provide a nonnegative geometric margin below 1 mm")
    if type(workers) is not int or not 1 <= workers <= 6:
        raise ValueError("workers must be between one and six socket gaps")
    robot_urdf = AssetPaths(shared_root, select_mode("cylinder", 20)).robot_urdf
    print(f"screening robot URDF: {robot_urdf}", flush=True)
    all_passed = []
    stages = {}
    for scene_id in scene_ids:
        stages[scene_id], passed = audit_tabletop_cell(
            raw_root=stage_root, pass_root=candidate_root,
            key_object=KEY, scene_id=scene_id,
            expected_seed_count=expected)
        all_passed.extend((scene_id, name, directory) for name, directory in passed)
    output_root.mkdir(parents=True)
    gaps = {}
    jobs = [(shared_root, output_root, all_passed, stage_root,
             gap, clearance_m, scene_ids) for gap in CYLINDER_RADIAL_GAPS_MM]
    if workers == 1:
        results = [_screen_gap(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_screen_gap, *zip(*jobs)))
    for label, result in results:
        gaps[label] = result
        print(f"gap {label}: {len(result['eligible_offline_grasp_ids'])}/"
              f"{len(all_passed)} exact endpoints clear", flush=True)
    summary = {
        "schema": "precision_insertion_cylinder_1000_per_scene_screen_v1",
        "status": "offline_grasp_and_20mm_endpoint_only_not_robot_ready",
        "proposal_config": "sim_inspire/precision_insertion.yml",
        "expected_proposals_per_scene": expected,
        "total_raw_proposals": len(scene_ids) * expected,
        "scene_ids": list(scene_ids),
        "raw_proxy_object": manifest["proxy_object"],
        "physical_key_object": KEY,
        "stage_root": str(stage_root),
        "original_autodex_sim_candidate_root": str(candidate_root),
        "minimum_hand_clearance_m": clearance_m,
        "screening_robot_urdf": str(robot_urdf),
        "screening_robot_urdf_sha256": hashlib.sha256(
            robot_urdf.read_bytes()).hexdigest(),
        "clearance_note": "This numerical screen tolerance is not a calibrated real-world safety margin.",
        "original_autodex_filter": [
            "cuRobo tabletop pregrasp hand/world and self collision",
            "cuRobo squeeze-pose contact with the full key",
            "MuJoCo full-key closure/squeeze and tabletop-gravity stability",
        ],
        "additional_filter": "centered, axis-aligned 20 mm exact-CAD key/socket fit and whole Inspire hand/socket non-collision",
        "scene_counts": stages,
        "socket_gaps": gaps,
        "not_tested": ["Franka arm IK, collisions, or continuous path",
                       "contact-rich insertion dynamics or physical success"],
    }
    _write(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-per-scene", type=int, default=1000)
    parser.add_argument("--minimum-hand-clearance-m", type=float, required=True)
    parser.add_argument("--workers", type=int, default=1,
                        help="screen distinct socket gaps in parallel (1..6)")
    args = parser.parse_args()
    report = run(args.shared_root.resolve(), args.stage_root.resolve(),
                 args.candidate_root.resolve(), args.output_root.resolve(),
                 args.expected_per_scene, args.minimum_hand_clearance_m,
                 workers=args.workers)
    print(json.dumps({"scene_counts": report["scene_counts"],
                      "socket_gaps": {gap: len(info["eligible_offline_grasp_ids"])
                                      for gap, info in report["socket_gaps"].items()},
                      "summary": str(args.output_root / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
