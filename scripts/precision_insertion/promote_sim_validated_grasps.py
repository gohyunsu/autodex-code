#!/usr/bin/env python3
"""Promote contact-safe, collision-safe, MuJoCo-stable grasps to a runtime pool.

Promotion never marks a grasp physically trusted. It copies only candidates
whose contact screen, cuRobo collision result, and MuJoCo result all pass, then
writes explicit simulation provenance and a physical-validation template.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screened-scene", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--replace-backup", type=Path, required=True)
    parser.add_argument("--full-object-mesh", type=Path, required=True)
    args = parser.parse_args()

    source = args.screened_scene.expanduser().resolve()
    output = args.output_scene.expanduser().resolve()
    backup = args.replace_backup.expanduser().resolve()
    full_mesh = args.full_object_mesh.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"screened scene does not exist: {source}")
    if not full_mesh.is_file():
        parser.error(f"full object mesh does not exist: {full_mesh}")
    if output.exists():
        if backup.exists():
            parser.error(f"backup already exists: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        output.rename(backup)

    output.mkdir(parents=True)
    promoted: list[str] = []
    rejected: dict[str, str] = {}
    for candidate in sorted(path for path in source.iterdir() if path.is_dir()):
        name = candidate.name
        screen_path = candidate / "contact_screen.json"
        collision_path = candidate / "coll_valid.npy"
        simulation_path = candidate / "sim_eval.json"
        if not screen_path.is_file() or not _json(screen_path).get("accepted"):
            rejected[name] = "contact_screen"
            continue
        if not collision_path.is_file() or not bool(np.load(collision_path)):
            rejected[name] = "full_key_collision"
            continue
        if not simulation_path.is_file() or not _json(simulation_path).get("success"):
            rejected[name] = "mujoco_stability"
            continue

        destination = output / name
        shutil.copytree(candidate, destination)
        validation = {
            "schema_version": 1,
            "status": "passed",
            "scope": "contact_policy_plus_curobo_full_key_collision_plus_mujoco_stability",
            "physical_validation": False,
            "full_object_mesh": str(full_mesh),
            "full_object_mesh_sha256": _sha256(full_mesh),
            "source_candidate": str(candidate),
            "warning": "simulation validation is not evidence of physical grasp or insertion success",
        }
        _write(destination / "simulation_validation.json", validation)
        promoted.append(name)

    report = {
        "schema_version": 1,
        "status": "sim_validated_not_physical",
        "screened_scene": str(source),
        "output_scene": str(output),
        "backup_scene": str(backup),
        "promoted": promoted,
        "rejected": rejected,
    }
    _write(output / "promotion_report.json", report)
    _write(
        output / "PHYSICAL_VALIDATION_REQUIRED.json",
        {
            "status": "required",
            "candidate_ids": promoted,
            "required_checks": [
                "visual clearance of shaft and socket-facing shoulder",
                "Franka reachability and slow collision preflight on the physical cell",
                "repeatable grasp and lift without shaft contact",
                "slow 1.5 mm socket insertion with abort thresholds enabled",
            ],
            "completion_file": "physical_validation.json",
            "do_not_claim": "trusted grasp until physical_validation.json records status=passed",
        },
    )
    print(json.dumps(report, indent=2))
    return 0 if promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
