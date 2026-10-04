#!/usr/bin/env python3
"""Validate geometry assets and report remaining runtime blockers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_assets import KEY_SPECS, FIXTURE_NAME, contact_face_partition, read_binary_stl


def validate(shared_root: Path, source_dir: Path, require_runtime: bool = False) -> int:
    shared_root = shared_root.expanduser().resolve()
    source_dir = source_dir.resolve()
    project = shared_root / "AutoDex"
    failures: list[str] = []
    blockers: list[str] = []

    for object_name, _gap, source_name in KEY_SPECS:
        source_mesh = read_binary_stl(source_dir / source_name)
        allowed, forbidden = contact_face_partition(source_mesh)
        object_dir = shared_root / "object_processing" / object_name
        required = [
            object_dir / "raw_mesh" / f"{object_name}.obj",
            object_dir / "processed_data" / "mesh" / "simplified.obj",
            object_dir / "processed_data" / "mesh" / "contact_allowed.obj",
            object_dir / "processed_data" / "mesh" / "contact_forbidden.obj",
            object_dir / "processed_data" / "info" / "simplified.json",
            object_dir / "processed_data" / "info" / "contact_regions.json",
            object_dir / "processed_data" / "info" / "tabletop" / "000.npy",
            project / "scene" / "inspire" / object_name / "table" / "0.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.extend(f"missing: {path}" for path in missing)
            continue

        pose = np.load(object_dir / "processed_data" / "info" / "tabletop" / "000.npy")
        if pose.shape != (4, 4) or not np.allclose(pose, np.eye(4), atol=1e-9):
            failures.append(f"{object_name}: baseline tabletop pose must be identity")
        policy = json.loads(
            (object_dir / "processed_data" / "info" / "contact_regions.json").read_text()
        )
        if policy["allowed"]["face_count"] != len(allowed):
            failures.append(f"{object_name}: allowed contact face count mismatch")
        if policy["forbidden"]["face_count"] != len(forbidden):
            failures.append(f"{object_name}: forbidden contact face count mismatch")

        repre = project / "foundpose_assets" / object_name / "object_repre" / "v1" / object_name / "1" / "repre.pth"
        if not repre.is_file():
            blockers.append(f"{object_name}: FoundPose repre.pth not generated")
        candidate_root = project / "candidates" / "inspire" / "v8" / object_name
        if not any(candidate_root.rglob("wrist_se3.npy")):
            blockers.append(f"{object_name}: no contact-safe Inspire grasp candidate")

    fixture = project / "precision_insertion" / "fixtures" / FIXTURE_NAME
    if not (fixture / "socket_shared_bore_1p5.obj").is_file():
        failures.append("socket collision mesh is missing")
    if not (fixture / "task_geometry.json").is_file():
        failures.append("socket task geometry is missing")
    fixture_pose = fixture / "fixture_pose.json"
    if not fixture_pose.is_file():
        blockers.append("fixture_pose.json is not calibrated")
    else:
        payload = json.loads(fixture_pose.read_text())
        transform = np.asarray(payload.get("T_robot_socket"), dtype=float)
        if not payload.get("calibrated") or transform.shape != (4, 4):
            failures.append("fixture_pose.json exists but is not a calibrated 4x4 transform")

    snapshots = sorted((project / "precision_insertion" / "calibration").glob("zerodex_*"))
    complete_snapshots = [
        path
        for path in snapshots
        if (path / "cam_param" / "intrinsics.json").is_file()
        and (path / "cam_param" / "extrinsics.json").is_file()
        and (path / "C2R.npy").is_file()
        and (path / "provenance.json").is_file()
    ]
    if not complete_snapshots:
        blockers.append("no frozen ZeroDex camera + Franka hand-eye snapshot")
    else:
        provenance = json.loads((complete_snapshots[-1] / "provenance.json").read_text())
        if provenance.get("status") != "verified_on_physical_rig":
            blockers.append("frozen ZeroDex calibration exists but physical four-camera set is not confirmed")

    print("geometry validation:", "PASS" if not failures else "FAIL")
    for failure in failures:
        print("  ERROR:", failure)
    print("runtime blockers:", len(blockers))
    for blocker in blockers:
        print("  BLOCKED:", blocker)

    if failures or (require_runtime and blockers):
        return 1
    return 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "assets" / "precision_insertion" / "source",
    )
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate(args.shared_root, args.source_dir, args.require_runtime))


if __name__ == "__main__":
    main()
