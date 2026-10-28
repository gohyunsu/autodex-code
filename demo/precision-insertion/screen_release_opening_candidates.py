#!/usr/bin/env python3
"""Rank v8 cylinder grasps by sampled hand-opening geometry at 20 mm.

This is an offline diagnostic for candidate order. It does not change the
v8/20 mm held-grasp acceptance rule or claim that a Franka path is feasible.
Use the same mount URDF selected by the ZeroDex launcher.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.endpoint import (
    _coal_mesh, _hand_link_meshes, _load_mesh, _mesh_pair_report,
    validate_task_geometry,
)
from precision_insertion.solid_occupancy import CylinderSocketOccupancy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--pose-stem", default="000")
    parser.add_argument("--candidate-id", action="append",
                        help="optional scene_type/scene_id/grasp_id subset")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mode = select_mode("cylinder", args.gap_mm)
    paths = AssetPaths(args.shared_root.resolve(), mode)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog["mode"]["socket_object"] != mode.socket_object:
        raise ValueError("catalog and socket gap differ")
    geometry = json.loads(paths.task_geometry.read_text(encoding="utf-8"))
    T_socket_key = validate_task_geometry(geometry, mode)
    # Coal must be loaded before trimesh on this host.
    import coal  # noqa: F401

    socket = _load_mesh(paths.socket_collision_mesh)
    fixed = _coal_mesh(socket)
    occupancy = CylinderSocketOccupancy(socket, geometry)
    fractions = np.linspace(0.0, 1.0, 21)
    rows = []
    started = time.perf_counter()
    for candidate in catalog["candidates"]:
        candidate_id = "/".join(candidate["key"])
        if (not candidate["eligible"] or
                candidate["tabletop_pose_stem"] != args.pose_stem or
                args.candidate_id and candidate_id not in args.candidate_id):
            continue
        T_key_hand = np.asarray(candidate["endpoint_report"]["T_key_hand"],
                                dtype=np.float64)
        T_socket_hand = T_socket_key @ T_key_hand
        held = np.asarray(candidate["endpoint_report"]["hold_pose_screens"]
                          ["autodex_default_controller_hold"]["hand_q"],
                          dtype=np.float64)
        minimum = float("inf")
        collisions = []
        clearance_by_fraction = []
        for fraction in fractions:
            hand = _hand_link_meshes(paths.robot_urdf, held * (1.0 - fraction))
            checks = {
                name: _mesh_pair_report(mesh, fixed, T_socket_hand, occupancy)
                for name, mesh in hand.items()
            }
            sampled_clearance = min(
                entry["minimum_surface_distance_m"]
                for entry in checks.values())
            clearance_by_fraction.append(sampled_clearance)
            minimum = min(minimum, sampled_clearance)
            for name, entry in checks.items():
                if entry["colliding"]:
                    collisions.append({"opening_fraction": float(fraction),
                                       "link": name})
        row = {
            "candidate_id": candidate_id,
            "sampled_opening_clear": not collisions,
            "minimum_surface_distance_m": minimum,
            "clearance_by_fraction_m": clearance_by_fraction,
            "collisions": collisions,
        }
        rows.append(row)
        print(row["candidate_id"], "clear" if not collisions else "collision",
              f"{1000 * minimum:.2f} mm", flush=True)
    rows.sort(key=lambda row: (
        not row["sampled_opening_clear"],
        -row["minimum_surface_distance_m"], row["candidate_id"]))
    output = {
        "schema": "precision_insertion_nominal_release_opening_sweep_v1",
        "scope": "sampled_nominal_hand_socket_geometry_only_not_arm_path_or_live_hold",
        "mode": catalog["mode"],
        "pose_stem": args.pose_stem,
        "mount_urdf": str(paths.robot_urdf),
        "sample_fractions": fractions.tolist(),
        "candidate_count": len(rows),
        "elapsed_s": time.perf_counter() - started,
        "candidates": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
