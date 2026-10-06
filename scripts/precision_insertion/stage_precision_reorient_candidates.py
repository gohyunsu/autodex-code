#!/usr/bin/env python3
"""Stage precision-key reorientation grasps behind the whole-hand policy gate.

The input is the output of ``filter_contact_safe_grasps.py`` for one or more
AutoDex reorientation cells (for example ``0_4``).  Every numerical BODex
candidate is rechecked against the complete Inspire visual geometry and the
full key mesh.  Only sampled passes are copied into the runtime candidate
layout expected by ``src/experiment/reset/reorient.py``.

This stage deliberately does not label a candidate executable: cuRobo's full
Franka approach/lift/reorient/descent chain and physical validation remain
separate downstream gates.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from validate_whole_hand_contact_policy import inspect_whole_hand


SHARED = Path.home() / "shared_data"
DEFAULT_INPUT = (
    SHARED / "AutoDex/contact_screen_staging/inspire/"
    "precision_insertion_reset_12_handle_proxy_v3_frame_fixed/"
    "precision_key_1p5mm/reorient_12"
)
DEFAULT_OUTPUT = (
    SHARED / "AutoDex/candidates/inspire/reset_12/"
    "precision_key_1p5mm/reorient_12"
)
DEFAULT_OBJECT_MESH = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/mesh/"
    "simplified.obj"
)
DEFAULT_POLICY = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/info/"
    "contact_regions.json"
)
DEFAULT_ROBOT = (
    SHARED / "AutoDex/content/assets/robot/fr3_inspire_description/"
    "fr3_inspire.urdf"
)


def _declared_face_opposition(declared: dict[str, Any]) -> dict[str, Any]:
    """Require contacts on both signs of one handle side axis.

    Merely touching two permitted faces is not enough for a pinch.  This gate
    rejects the common BODex failure in which thumb and fingers are all placed
    on the same side of the handle.  Rear-face contacts are allowed by the
    insertion policy, but do not by themselves establish opposition.
    """
    points = declared.get("transformed_object_contacts_m", [])
    regions = declared.get("regions", [])
    x_values = [float(point[0]) for point, region in zip(points, regions)
                if region == "handle_x_side"]
    y_values = [float(point[1]) for point, region in zip(points, regions)
                if region == "handle_y_side"]
    x_opposed = bool(x_values and min(x_values) < 0.0 < max(x_values))
    y_opposed = bool(y_values and min(y_values) < 0.0 < max(y_values))
    return {
        "passed": x_opposed or y_opposed,
        "x_side_signs_opposed": x_opposed,
        "y_side_signs_opposed": y_opposed,
        "criterion": "contacts occur on both signs of x-side or y-side",
    }


def _numeric_dirs(path: Path) -> list[Path]:
    return sorted(
        (item for item in path.iterdir() if item.is_dir() and item.name.isdigit()),
        key=lambda item: int(item.name),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--object-mesh", type=Path, default=DEFAULT_OBJECT_MESH)
    parser.add_argument("--contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--samples-per-link", type=int, default=12000)
    parser.add_argument("--penetration-threshold-mm", type=float, default=0.2)
    parser.add_argument(
        "--replace-backup", type=Path,
        help="move an existing output root to this explicit audit path first",
    )
    args = parser.parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    object_mesh = args.object_mesh.expanduser().resolve()
    policy = args.contact_policy.expanduser().resolve()
    robot = args.robot_urdf.expanduser().resolve()
    missing = [
        str(path) for path in (input_root, object_mesh, policy, robot)
        if not path.exists()
    ]
    if missing:
        parser.error("missing input: " + ", ".join(missing))
    if args.samples_per_link < 100:
        parser.error("--samples-per-link must be at least 100")
    if output_root.exists():
        if args.replace_backup is None:
            parser.error(
                f"output exists; pass --replace-backup PATH: {output_root}"
            )
        backup = args.replace_backup.expanduser().resolve()
        if backup.exists():
            parser.error(f"backup already exists: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        output_root.rename(backup)
    output_root.mkdir(parents=True)

    cells: dict[str, Any] = {}
    total_input = 0
    total_passed = 0
    for cell_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        candidates = _numeric_dirs(cell_dir)
        if not candidates:
            cells[cell_dir.name] = {
                "input_candidates": 0,
                "sampled_passes": 0,
                "passed_candidates": [],
                "candidates": [],
            }
            continue
        destination_cell = output_root / cell_dir.name
        destination_cell.mkdir(parents=True)
        records = []
        passed_ids = []
        for candidate_dir in candidates:
            report, _ = inspect_whole_hand(
                candidate_dir=candidate_dir,
                object_mesh_path=object_mesh,
                policy_path=policy,
                robot_urdf=robot,
                symmetry_mode="none",
                samples_per_link=args.samples_per_link,
                penetration_threshold_m=args.penetration_threshold_mm / 1000.0,
            )
            opposition = _declared_face_opposition(report["declared_contacts"])
            staged_pass = (
                report["status"] == "sampled_pass" and opposition["passed"]
            )
            records.append({
                "candidate": candidate_dir.name,
                "status": "sampled_pass" if staged_pass else "rejected",
                "declared_contacts": report["declared_contacts"],
                "declared_face_opposition": opposition,
                "whole_hand": report["whole_hand"],
            })
            if not staged_pass:
                continue
            passed_ids.append(candidate_dir.name)
            destination = destination_cell / candidate_dir.name
            shutil.copytree(candidate_dir, destination)
            (destination / "whole_hand_contact_policy.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
        cell_report = {
            "schema_version": 1,
            "status": "sampled_whole_hand_gate_not_motion_or_physical_validation",
            "cell": cell_dir.name,
            "input_candidates": len(candidates),
            "sampled_passes": len(passed_ids),
            "passed_candidates": passed_ids,
            "candidates": records,
        }
        (destination_cell / "stage_report.json").write_text(
            json.dumps(cell_report, indent=2) + "\n", encoding="utf-8"
        )
        cells[cell_dir.name] = cell_report
        total_input += len(candidates)
        total_passed += len(passed_ids)

    manifest = {
        "schema_version": 1,
        "status": "sampled_whole_hand_gate_not_motion_or_physical_validation",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "object_mesh": str(object_mesh),
        "contact_policy": str(policy),
        "robot_urdf": str(robot),
        "sampling": {
            "samples_per_link": args.samples_per_link,
            "penetration_threshold_mm": args.penetration_threshold_mm,
        },
        "total_input_candidates": total_input,
        "total_sampled_passes": total_passed,
        "cells": cells,
        "required_next_gates": [
            "cuRobo full Franka approach/lift/reorient/descent preflight",
            "physical grasp and release-height validation",
            "post-release perception verification of the target tabletop pose",
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "input_candidates": total_input,
        "sampled_passes": total_passed,
    }))
    return 0 if total_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
