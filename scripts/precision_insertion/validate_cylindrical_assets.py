#!/usr/bin/env python3
"""Validate generated cylindrical AutoDex geometry and symmetry contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_cylindrical_assets import (
    KEY_HEIGHT_M,
    KEY_OBJECT,
    KEY_PROXY_OBJECT,
    SOCKET_BORE_DEPTH_M,
    SOCKET_SPECS,
    VERIFICATION_DEPTH_M,
)


def validate(shared_root: Path, *, require_learned: bool = False) -> int:
    shared_root = shared_root.expanduser().resolve()
    object_root = shared_root / "object_processing"
    project = shared_root / "AutoDex"
    failures: list[str] = []
    blockers: list[str] = []

    key = object_root / KEY_OBJECT
    key_required = [
        key / "raw_mesh" / f"{KEY_OBJECT}.obj",
        key / "processed_data" / "mesh" / "simplified.obj",
        key / "processed_data" / "mesh" / "contact_allowed.obj",
        key / "processed_data" / "mesh" / "contact_forbidden.obj",
        key / "processed_data" / "info" / "symmetry.json",
        key / "processed_data" / "info" / "tabletop" / "000.npy",
        key / "processed_data" / "info" / "tabletop" / "001.npy",
        object_root / KEY_PROXY_OBJECT / "processed_data" / "mesh" / "simplified.obj",
    ]
    failures.extend(f"missing: {p}" for p in key_required if not p.is_file())
    tabletop_end = key / "processed_data" / "info" / "tabletop" / "000.npy"
    if tabletop_end.is_file():
        pose = np.load(tabletop_end)
        grasp_end_z = float((pose @ np.array([0.0, 0.0, 0.0, 1.0]))[2])
        insertion_end_z = float(
            (pose @ np.array([0.0, 0.0, KEY_HEIGHT_M, 1.0]))[2])
        if not (np.isclose(grasp_end_z, KEY_HEIGHT_M) and
                np.isclose(insertion_end_z, 0.0)):
            failures.append(
                "end-down tabletop pose must expose z=0 grasp end and place "
                "z=80 mm insertion end on the table"
            )
    symmetry_path = key / "processed_data" / "info" / "symmetry.json"
    if symmetry_path.is_file():
        symmetry = json.loads(symmetry_path.read_text(encoding="utf-8"))
        axes = symmetry.get("axes", [])
        if symmetry.get("type") != "Dinf":
            failures.append("key symmetry must be Dinf")
        if not any(a.get("axis") == [0.0, 0.0, 1.0] and a.get("fold") == "inf" for a in axes):
            failures.append("key must declare continuous local-z symmetry")
        if sum(a.get("fold") == 2 for a in axes) < 2:
            failures.append("key must declare end-exchange flip symmetry")

    for object_name, gap_mm, bore_radius_mm, _source in SOCKET_SPECS:
        root = object_root / object_name
        required = [
            root / "raw_mesh" / f"{object_name}.obj",
            root / "processed_data" / "mesh" / "simplified.obj",
            root / "processed_data" / "mesh" / "static_collision.obj",
            root / "processed_data" / "info" / "symmetry.json",
            root / "processed_data" / "info" / "frame_contract.json",
            project / "precision_insertion" / "cylindrical" / "fixtures" /
            object_name / "task_geometry.json",
        ]
        failures.extend(f"missing: {p}" for p in required if not p.is_file())
        sym_path = root / "processed_data" / "info" / "symmetry.json"
        if sym_path.is_file():
            sym = json.loads(sym_path.read_text(encoding="utf-8"))
            if sym.get("type") != "Cinf" or len(sym.get("axes", [])) != 1:
                failures.append(f"{object_name}: socket symmetry must be Cinf only")
        task_path = required[-1]
        if task_path.is_file():
            task = json.loads(task_path.read_text(encoding="utf-8"))
            if task.get("radial_clearance_mm") != gap_mm:
                failures.append(f"{object_name}: radial clearance mismatch")
            if task.get("socket_bore_radius_m") != bore_radius_mm * 1e-3:
                failures.append(f"{object_name}: bore radius mismatch")
            entry = np.asarray(task["T_socket_key_entry"], dtype=float)
            verify = np.asarray(task["T_socket_key_verification"], dtype=float)
            seated = np.asarray(task["T_socket_key_seated"], dtype=float)
            if not np.isclose(entry[2, 3] - verify[2, 3], VERIFICATION_DEPTH_M):
                failures.append(f"{object_name}: verification stroke is not 20 mm")
            if not np.isclose(entry[2, 3] - seated[2, 3], SOCKET_BORE_DEPTH_M):
                failures.append(f"{object_name}: seated stroke does not equal bore depth")
            tip_entry = (entry @ np.array([0.0, 0.0, KEY_HEIGHT_M, 1.0]))[2]
            if not np.isclose(tip_entry, 0.055):
                failures.append(f"{object_name}: key tip is not on the rim at entry")

        repre = (
            project / "foundpose_assets" / object_name / "object_repre" /
            "v1" / object_name / "1" / "repre.pth"
        )
        if not repre.is_file():
            blockers.append(f"{object_name}: FoundPose repre.pth required")

    key_repre = (
        project / "foundpose_assets" / KEY_OBJECT / "object_repre" /
        "v1" / KEY_OBJECT / "1" / "repre.pth"
    )
    if not key_repre.is_file():
        blockers.append(f"{KEY_OBJECT}: FoundPose repre.pth required")
    candidate_root = project / "candidates" / "inspire" / "v8" / KEY_OBJECT
    if not any(candidate_root.rglob("wrist_se3.npy")):
        blockers.append(f"{KEY_OBJECT}: no generated Inspire grasp candidates")

    if failures:
        print("CYLINDRICAL ASSET VALIDATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("CYLINDRICAL GEOMETRY VALIDATION PASSED")
    for blocker in blockers:
        print(f"BLOCKED RUNTIME: {blocker}")
    return 1 if require_learned and blockers else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument("--require-learned", action="store_true")
    args = parser.parse_args()
    raise SystemExit(validate(args.shared_root, require_learned=args.require_learned))


if __name__ == "__main__":
    main()
