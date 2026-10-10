#!/usr/bin/env python3
"""Audit AutoDex full-key sim filtering, then screen every socket at 20 mm.

Reads the staged 1,000-per-scene results and the original AutoDex
``run_sim_filter`` candidate output. Writes *offline* evidence only, never
robot commands or the live v8 candidate tree. Every socket gap is evaluated
independently against its exact CAD collision mesh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.endpoint import screen_grasp_endpoint

KEY = "precision_key_cylinder_r15_h80"
SCENES = ("0", "1")


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def _scene_rows(stage_root: Path, candidate_root: Path, scene_id: str,
                expected: int) -> tuple[dict, list[tuple[str, Path]]]:
    raw_scene = stage_root / KEY / "table" / scene_id
    passed_scene = candidate_root / KEY / "table" / scene_id
    raw_dirs = {p.name: p for p in raw_scene.iterdir() if p.is_dir()}
    if set(raw_dirs) != {str(i) for i in range(expected)}:
        raise ValueError(f"incomplete staged scene {scene_id}: {len(raw_dirs)}/{expected}")
    counts = {"generated": expected, "scene_clear": 0, "squeeze_contact": 0,
              "mujoco_stable": 0, "failed_scene_collision": 0,
              "failed_no_object_contact": 0, "failed_mujoco_or_error": 0}
    passed = []
    for name, directory in sorted(raw_dirs.items(), key=lambda row: int(row[0])):
        coll = bool(np.load(directory / "coll_valid.npy", allow_pickle=False))
        evaluation = json.loads((directory / "sim_eval.json").read_text(encoding="utf-8"))
        if coll:
            counts["scene_clear"] += 1
        if not coll:
            counts["failed_scene_collision"] += 1
            if evaluation.get("reason") != "scene_collision":
                raise ValueError(f"inconsistent scene collision result: {directory}")
            continue
        if evaluation.get("reason") == "no_object_contact":
            counts["failed_no_object_contact"] += 1
            continue
        counts["squeeze_contact"] += 1
        if evaluation.get("success") is True:
            candidate = passed_scene / name
            required = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy", "bodex_info.npy")
            if not all((candidate / file).is_file() for file in required):
                raise FileNotFoundError(f"AutoDex did not copy stable candidate {candidate}")
            counts["mujoco_stable"] += 1
            passed.append((name, candidate))
        else:
            counts["failed_mujoco_or_error"] += 1
    observed = {p.name for p in passed_scene.iterdir() if p.is_dir()} if passed_scene.is_dir() else set()
    if observed != {name for name, _ in passed}:
        raise ValueError(f"AutoDex candidate pool disagrees with sim results: scene {scene_id}")
    return counts, passed


def run(shared_root: Path, stage_root: Path, candidate_root: Path,
        output_root: Path, expected: int, clearance_m: float) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite results: {output_root}")
    manifest = json.loads((stage_root / "stage_manifest.json").read_text(encoding="utf-8"))
    if manifest["seed_count_per_scene"] != expected or manifest["seed_count_total"] != 2 * expected:
        raise ValueError("stage manifest does not match requested 1,000-per-scene run")
    if not 0 < clearance_m < 0.001:
        raise ValueError("provide a positive geometric tolerance below 1 mm")
    all_passed = []
    stages = {}
    for scene_id in SCENES:
        stages[scene_id], passed = _scene_rows(stage_root, candidate_root,
                                              scene_id, expected)
        all_passed.extend((scene_id, name, directory) for name, directory in passed)
    output_root.mkdir(parents=True)
    gaps = {}
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        gap_dir = output_root / f"gap_{int(gap):02d}mm"
        eligible = []
        errors = []
        by_scene = {scene: {"tested": 0, "endpoint_pass": 0} for scene in SCENES}
        for scene_id, name, directory in all_passed:
            by_scene[scene_id]["tested"] += 1
            target = gap_dir / "table" / scene_id / f"{name}.json"
            try:
                result = screen_grasp_endpoint(
                    shared_root=shared_root, mode=mode, candidate_dir=directory,
                    minimum_hand_clearance_m=clearance_m)
            except Exception as exc:
                result = {"endpoint_pass": False, "error": f"{type(exc).__name__}: {exc}",
                          "candidate_dir": str(directory), "gap_mm": gap}
                errors.append(f"table/{scene_id}/{name}: {result['error']}")
            result["original_autodex_sim_filter_pass"] = True
            result["original_autodex_sim_eval"] = str(stage_root / KEY / "table" /
                                                       scene_id / name / "sim_eval.json")
            result["note"] = "Endpoint-only geometry; no Franka path or physical insertion was tested."
            _write(target, result)
            if result["endpoint_pass"]:
                by_scene[scene_id]["endpoint_pass"] += 1
                eligible.append(f"table/{scene_id}/{name}")
        gaps[f"{int(gap):02d}mm"] = {
            "socket_object": mode.socket_object,
            "by_scene": by_scene,
            "eligible_offline_grasp_ids": eligible,
            "errors": errors,
        }
        print(f"gap {int(gap):02d}mm: {len(eligible)}/{len(all_passed)} exact endpoints clear", flush=True)
    summary = {
        "schema": "precision_insertion_cylinder_1000_per_scene_screen_v1",
        "status": "offline_grasp_and_20mm_endpoint_only_not_robot_ready",
        "proposal_config": "sim_inspire/precision_insertion.yml",
        "expected_proposals_per_scene": expected,
        "total_raw_proposals": 2 * expected,
        "raw_proxy_object": manifest["proxy_object"],
        "physical_key_object": KEY,
        "stage_root": str(stage_root),
        "original_autodex_sim_candidate_root": str(candidate_root),
        "minimum_hand_clearance_m": clearance_m,
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
    args = parser.parse_args()
    report = run(args.shared_root.resolve(), args.stage_root.resolve(),
                 args.candidate_root.resolve(), args.output_root.resolve(),
                 args.expected_per_scene, args.minimum_hand_clearance_m)
    print(json.dumps({"scene_counts": report["scene_counts"],
                      "socket_gaps": {gap: len(info["eligible_offline_grasp_ids"])
                                      for gap, info in report["socket_gaps"].items()},
                      "summary": str(args.output_root / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()
