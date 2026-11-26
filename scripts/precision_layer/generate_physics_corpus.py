#!/usr/bin/env python3
"""Write an auditable R1-alpha MuJoCo precision corpus."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

import mujoco

from precision_layer.audit import audit_cases
from precision_layer.generator import GeneratorConfig, generate_cases
from precision_layer.procedural_mujoco import build_model, evaluate_scene, make_scene


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=6)
    parser.add_argument("--physics-steps-per-waypoint", type=int, default=5)
    args = parser.parse_args()

    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        name: output / name
        for name in ("programs", "scenes", "worlds", "trajectories", "labels")
    }
    for directory in directories.values():
        directory.mkdir()

    created_at = datetime.now(timezone.utc).isoformat()
    config = GeneratorConfig(
        seed=args.seed,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    )
    cases = generate_cases(config, created_at=created_at)
    violations = audit_cases(cases)
    if violations:
        raise RuntimeError(f"split leakage audit failed: {violations}")

    index_cases = []
    physics_cases = []
    for case in cases:
        case_id = case["case_id"]
        program_path = directories["programs"] / f"{case_id}.json"
        scene_path = directories["scenes"] / f"{case_id}.json"
        world_path = directories["worlds"] / f"{case_id}.xml"
        trajectory_path = directories["trajectories"] / f"{case_id}.json"
        labels_path = directories["labels"] / f"{case_id}.json"

        scene = make_scene(case, seed=args.seed)
        model, _ = build_model(scene)
        mujoco.mj_saveLastXML(str(world_path), model)
        labels = evaluate_scene(
            scene,
            physics_steps_per_waypoint=args.physics_steps_per_waypoint)

        _write_json(program_path, case["program"])
        scene_payload = asdict(scene)
        scene_payload.pop("program")
        _write_json(scene_path, scene_payload)
        _write_json(trajectory_path, {
            "case_id": case_id,
            "base_policy_class": case["base_policy_class"],
            "positions_m": scene.trajectory_positions,
        })
        _write_json(labels_path, {
            "case_id": case_id,
            "privileged_evaluation_only": True,
            "waypoints": labels,
        })

        index_cases.append({
            key: case[key]
            for key in (
                "case_id", "partition", "geometry_family",
                "layout_family", "base_policy_class")
        } | {"program_path": str(program_path.relative_to(output))})
        physics_cases.append({
            "case_id": case_id,
            "scene_path": str(scene_path.relative_to(output)),
            "world_path": str(world_path.relative_to(output)),
            "trajectory_path": str(trajectory_path.relative_to(output)),
            "privileged_labels_path": str(labels_path.relative_to(output)),
        })

    _write_json(output / "index.json", {
        "schema": "precision_benchmark_index_v1",
        "cases": index_cases,
    })
    _write_json(output / "physics_manifest.json", {
        "schema": "precision_physics_corpus_alpha_v1",
        "created_at": created_at,
        "generator_seed": args.seed,
        "physics_steps_per_waypoint": args.physics_steps_per_waypoint,
        "case_count": len(cases),
        "split_leakage_audit_passed": True,
        "claim_scope": "procedural_primitive_mujoco_margin_corpus",
        "passes_R1": False,
        "R1_missing": [
            "validated_mesh_geometry_generators",
            "observation_and_calibration_corruptions",
            "multiple_independent_nominal_policy_implementations",
            "contact_rich_trajectory_diversity",
        ],
        "cases": physics_cases,
    })
    print(output / "physics_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
