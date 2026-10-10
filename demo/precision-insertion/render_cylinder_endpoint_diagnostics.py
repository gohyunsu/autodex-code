#!/usr/bin/env python3
"""Build exact-mesh, 20 mm endpoint still bundles from exploratory raw seeds.

This diagnostic deliberately does NOT promote reorientation BODex seeds to the
tabletop v8 pool. A rendered pass means only that the nominal CAD key fits and
the posed Inspire visual links avoid the socket at the 20 mm endpoint. Native
BODex success, MuJoCo stability, Franka reachability, continuous insertion,
and physical success remain separate gates.

Example:
  python demo/precision-insertion/render_cylinder_endpoint_diagnostics.py \
    --shared-root ~/shared_data --raw-root RAW_REORIENT_ROOT \
    --output-root NEW_DIAGNOSTIC_DIRECTORY --min-hand-clearance-mm 0.2

Render each generated ``endpoint_bundle.npz`` with the existing original-mesh
Blender renderer using ``--still-frame 1 --view key-socket`` or ``task``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.endpoint import _hand_link_meshes, screen_grasp_endpoint


def _raw_numeric_pass(candidate: Path) -> tuple[bool, dict]:
    path = candidate / "bodex_info.npy"
    if not path.is_file():
        return False, {"reason": "missing_bodex_info"}
    # BODex saves this local artifact as a pickled dict; never load untrusted
    # external NPY files through this diagnostic tool.
    data = np.load(path, allow_pickle=True).item()
    error = np.asarray(data["grasp_error"], dtype=float)
    distance = np.asarray(data["dist_error"], dtype=float)
    if not (error.size and distance.size and np.all(np.isfinite(error)) and
            np.all(np.isfinite(distance))):
        return False, {"reason": "nonfinite_or_empty_numeric_evidence"}
    maximum = float(np.max(np.abs(error)))
    mean_distance = float(np.mean(np.abs(distance)))
    native = bool(np.asarray(data["success"]).all())
    return maximum <= 0.2 and mean_distance <= 0.01, {
        "grasp_error_max_abs": maximum,
        "dist_error_mean_abs_m": mean_distance,
        "bodex_native_success": native,
        "relaxed_numeric_pass": maximum <= 0.2 and mean_distance <= 0.01,
    }


def _candidate_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        path.parent for path in raw_root.glob("*/*/bodex_info.npy")
        if (path.parent / "wrist_se3.npy").is_file()
        and (path.parent / "grasp_pose.npy").is_file()
    )


def _bundle(
    *, candidate: Path, paths: AssetPaths, endpoint: dict,
    output: Path,
) -> Path:
    """Export posed actual Inspire, key and socket meshes for existing Blender."""
    import trimesh

    output.mkdir(parents=True, exist_ok=False)
    mesh_dir = output / "mesh_cache"
    mesh_dir.mkdir()
    key_mesh = trimesh.load(paths.raw_mesh(paths.mode.key_object),
                            force="mesh", process=False)
    socket_mesh = trimesh.load(paths.raw_mesh(paths.mode.socket_object),
                               force="mesh", process=False)
    key_path = mesh_dir / "precision_key.ply"
    socket_path = mesh_dir / "precision_socket.ply"
    key_mesh.export(key_path)
    socket_mesh.export(socket_path)

    hand_q = np.load(candidate / "grasp_pose.npy", allow_pickle=False)
    links = _hand_link_meshes(paths.robot_urdf, hand_q)
    names = sorted(links)
    for index, name in enumerate(names):
        links[name].export(mesh_dir / f"geometry_{index:03d}.ply")

    # Put the exact centered endpoint into the established task-view frame.
    world_socket = np.eye(4)
    world_socket[:3, 3] = [0.45, -0.10, 0.04]
    socket_key = np.asarray(endpoint["T_socket_key_verification"], dtype=float)
    key_hand = np.asarray(endpoint["T_key_hand"], dtype=float)
    world_key = world_socket @ socket_key
    world_hand = world_key @ key_hand
    bundle = output / "endpoint_bundle.npz"
    np.savez_compressed(
        bundle,
        robot_geometry_transforms=np.repeat(
            world_hand[None, None, :, :], len(names), axis=1).astype(np.float32),
        object_poses=world_key[None].astype(np.float32),
        socket_pose=world_socket.astype(np.float32),
        robot_mesh_dir=np.asarray(str(mesh_dir)),
        geometry_names=np.asarray(names),
        object_mesh_path=np.asarray(str(key_path)),
        socket_mesh_path=np.asarray(str(socket_path)),
    )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-hand-clearance-mm", type=float, required=True)
    args = parser.parse_args()
    root = args.shared_root.expanduser().resolve()
    raw = args.raw_root.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    clearance = float(args.min_hand_clearance_mm) / 1000.0
    if not raw.is_dir():
        parser.error(f"raw candidate root is missing: {raw}")
    if not math.isfinite(clearance) or clearance <= 0:
        parser.error("minimum hand clearance must be positive")
    if output.exists():
        parser.error(f"output already exists; choose a new versioned path: {output}")
    candidates = _candidate_dirs(raw)
    if not candidates:
        parser.error("no raw BODex candidate files found")
    relaxed = []
    for candidate in candidates:
        passed, numeric = _raw_numeric_pass(candidate)
        if passed:
            relaxed.append((candidate, numeric))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "cylinder_raw_reorient_20mm_endpoint_diagnostics_v1",
        "scope": "exploratory_raw_reorient_seeds_not_runtime_grasps",
        "shared_root": str(root),
        "raw_root": str(raw),
        "raw_candidate_count": len(candidates),
        "relaxed_numeric_candidate_count": len(relaxed),
        "relaxed_grasp_error_max_abs": 0.2,
        "relaxed_mean_abs_distance_error_m": 0.01,
        "minimum_hand_clearance_m": clearance,
        "minimum_hand_clearance_commissioned": False,
        "socket_variants": [],
        "runtime_eligible_count": 0,
        "not_validated": [
            "native BODex success", "full-key MuJoCo grasp stability",
            "tabletop v8 pose-conditioned grasp", "Franka IK or path",
            "continuous insertion, contact, or physical success",
        ],
    }
    for gap in CYLINDER_RADIAL_GAPS_MM:
        mode = select_mode("cylinder", gap)
        paths = AssetPaths(root, mode)
        socket_dir = output / mode.socket_object
        socket_dir.mkdir()
        rows = []
        for candidate, numeric in relaxed:
            endpoint = screen_grasp_endpoint(
                shared_root=root, mode=mode, candidate_dir=candidate,
                minimum_hand_clearance_m=clearance,
            )
            candidate_id = "raw_reorient_" + "_".join(
                candidate.relative_to(raw).parts)
            row = {
                "candidate_id": candidate_id,
                "source_dir": str(candidate),
                "numeric": numeric,
                "endpoint": endpoint,
                "rendered_images": [],
            }
            if endpoint["endpoint_pass"]:
                bundle = _bundle(
                    candidate=candidate, paths=paths,
                    endpoint=endpoint, output=socket_dir / candidate_id,
                )
                row["bundle"] = str(bundle)
                row["rendered_images"] = [
                    str(bundle.parent / "key_socket.png"),
                    str(bundle.parent / "task.png"),
                ]
            rows.append(row)
        report = {
            "socket_object": mode.socket_object,
            "radial_gap_mm": gap,
            "verification_depth_mm": 20,
            "key_object": mode.key_object,
            "numeric_candidates_screened": len(rows),
            "endpoint_geometry_pass_count": sum(
                row["endpoint"]["endpoint_pass"] for row in rows),
            "runtime_eligible_count": 0,
            "rows": rows,
        }
        (socket_dir / "screen_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        manifest["socket_variants"].append({
            "socket_object": mode.socket_object,
            "radial_gap_mm": gap,
            "endpoint_geometry_pass_count": report["endpoint_geometry_pass_count"],
            "report": str(socket_dir / "screen_report.json"),
        })
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_root": str(output),
        "raw_count": len(candidates),
        "relaxed_count": len(relaxed),
        "per_socket": manifest["socket_variants"],
        "runtime_eligible_count": 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
