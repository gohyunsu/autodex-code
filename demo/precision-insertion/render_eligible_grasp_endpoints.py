#!/usr/bin/env python3
"""Render every *fully offline-filtered* v8 grasp at the aligned 20 mm pose.

The evidence gate is AutoDex v8/MuJoCo grasp stability followed by a fresh
exact-mesh key/socket and whole-Inspire-hand endpoint screen. This does not
claim Franka reachability, continuous insertion, or physical success. No
robot or camera is connected. Outputs are versioned by choosing a NEW root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.candidates import _grasp_evidence, select_pose_candidates
from precision_insertion.config import select_mode
from precision_insertion.endpoint import (
    _hand_link_meshes, _load_mesh, nominal_inspire_hold_poses,
    screen_grasp_endpoint, validate_task_geometry,
)
from precision_insertion.geometry import validate_se3


REPO = Path(__file__).resolve().parents[2]
RENDERER = REPO / "scripts/precision_insertion/render_blender_actual_mesh_animation.py"
VIEWS = ("endpoint-oblique", "endpoint-side")
HOLDS = ("mujoco_squeeze", "autodex_default_controller_hold")


def eligible_rows(catalog: dict) -> tuple[object, list[dict]]:
    """Reject incomplete/stale catalogues and return all pose-valid rows."""
    if catalog.get("schema") != "precision_insertion_endpoint_catalog_v1":
        raise ValueError("unknown endpoint catalogue schema")
    identity = catalog.get("mode", {})
    mode = select_mode(identity.get("family"), identity.get("gap_mm"))
    expected = {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
        "target_depth_m": mode.target_depth_m,
    }
    if identity != expected:
        raise ValueError("catalogue key/socket/depth does not match a configured mode")
    if not catalog.get("complete_scan"):
        raise ValueError("catalogue is not a complete v8 scan")
    rows = catalog.get("candidates")
    if not isinstance(rows, list) or catalog.get("eligible_count") != sum(
            row.get("eligible") is True for row in rows):
        raise ValueError("catalogue candidate count disagrees with rows")
    paths = AssetPaths(Path(catalog["shared_root"]).expanduser().resolve(), mode)
    for row in rows:
        if not row.get("eligible"):
            continue
        expected_dir = paths.candidate_dir.joinpath(*row["key"]).resolve()
        if (Path(row["candidate_dir"]).resolve() != expected_dir or
                row.get("grasp_stability_pass") is not True or
                row.get("endpoint_pass") is not True or
                not isinstance(row.get("endpoint_report"), dict) or
                row["endpoint_report"].get("endpoint_pass") is not True):
            raise ValueError("eligible row lacks matching v8/MuJoCo/endpoint evidence")
    stems = {row.get("tabletop_pose_stem") for row in rows}
    if any(not isinstance(stem, str) for stem in stems):
        raise ValueError("candidate has no tabletop pose stem")
    selected = []
    for stem in sorted(stems):
        result = select_pose_candidates(
            catalog, expected_mode=mode, tabletop_pose_stem=stem)
        if result["status"] not in (
                "candidates_available", "no_eligible_in_screened_pool"):
            raise ValueError(f"catalogue is stale: {result['reason']}")
        selected.extend({**row, "tabletop_pose_stem": stem}
                        for row in result["candidates"])
    if len(selected) != catalog["eligible_count"]:
        raise ValueError("eligible rows are missing from pose-conditioned selection")
    return mode, selected


def _write_bundle(
    *, output: Path, shared_root: Path, mode, candidate: Path,
    endpoint: dict, hold_name: str, key_mesh_path: Path,
    socket_mesh_path: Path,
) -> Path:
    """Export exactly the held URDF links, CAD key, and exact socket mesh."""
    paths = AssetPaths(shared_root, mode)
    pre = np.load(candidate / "pregrasp_pose.npy", allow_pickle=False)
    grasp = np.load(candidate / "grasp_pose.npy", allow_pickle=False)
    hold_q = nominal_inspire_hold_poses(pre, grasp)[hold_name]
    saved_q = np.asarray(endpoint["hold_pose_screens"][hold_name]["hand_q"])
    if not np.allclose(hold_q, saved_q, atol=1e-10):
        raise ValueError("fresh screen and hold joint pose disagree")
    hand = _hand_link_meshes(paths.robot_urdf, hold_q)
    names = sorted(hand)
    mesh_dir = output / "hand_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=False)
    for index, name in enumerate(names):
        hand[name].export(mesh_dir / f"geometry_{index:03d}.ply")

    # The presentation renderer's fixed table/socket viewpoint is only a
    # display frame; no session calibration or robot pose is invented here.
    T_world_socket = np.eye(4)
    T_world_socket[:3, 3] = (0.45, -0.10, 0.04)
    T_socket_key = validate_se3(
        endpoint["T_socket_key_tested"], name="screened T_socket_key")
    T_key_hand = validate_se3(endpoint["T_key_hand"], name="screened T_key_hand")
    T_world_key = T_world_socket @ T_socket_key
    T_world_hand = T_world_key @ T_key_hand
    bundle = output / "endpoint_bundle.npz"
    np.savez_compressed(
        bundle,
        robot_geometry_transforms=np.repeat(
            T_world_hand[None, None, :, :], len(names), axis=1).astype(np.float32),
        object_poses=T_world_key[None].astype(np.float32),
        socket_pose=T_world_socket.astype(np.float32),
        robot_mesh_dir=np.asarray(str(mesh_dir)),
        geometry_names=np.asarray(names),
        object_mesh_path=np.asarray(str(key_mesh_path)),
        socket_mesh_path=np.asarray(str(socket_mesh_path)),
    )
    return bundle


def render_catalog(
    *, catalog_path: Path, output_root: Path, blender: Path | None = None,
    width: int = 1600, height: int = 900,
) -> dict:
    """Create one audited folder per grasp and one PNG per hold/view pair."""
    source = Path(catalog_path).expanduser().resolve()
    catalog = json.loads(source.read_text(encoding="utf-8"))
    mode, rows = eligible_rows(catalog)
    root = Path(catalog["shared_root"]).expanduser().resolve()
    paths = AssetPaths(root, mode)
    target = Path(output_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"output already exists: {target}")
    if width < 320 or height < 180:
        raise ValueError("render dimensions are too small")
    if rows:
        executable = Path(blender or shutil.which("blender") or "").expanduser()
        if not executable.is_file() or not RENDERER.is_file():
            raise FileNotFoundError("Blender or existing actual-mesh renderer missing")
        key_mesh = _load_mesh(paths.raw_mesh(mode.key_object))
        socket_mesh = _load_mesh(paths.socket_collision_mesh)
        geometry = json.loads(paths.task_geometry.read_text(encoding="utf-8"))
        T_nominal = validate_task_geometry(geometry, mode)
    else:
        executable = None

    # Re-run the exact screen on every eligible row before making *any* image.
    # A catalogue's hashes reject stale source files; this additionally
    # rejects a fabricated or inconsistent endpoint report.
    screens = []
    for row in rows:
        candidate = Path(row["candidate_dir"])
        grasp_ok, reason = _grasp_evidence(candidate, paths.key_planning_mesh)
        if not grasp_ok:
            raise ValueError(f"current v8/MuJoCo grasp evidence failed: {reason}")
        report = screen_grasp_endpoint(
            shared_root=root, mode=mode, candidate_dir=candidate,
            minimum_hand_clearance_m=catalog["minimum_hand_clearance_m"],
        )
        if (report.get("endpoint_pass") is not True or
                not np.allclose(report["T_socket_key_tested"], T_nominal,
                                atol=1e-9) or
                any(report["hold_pose_screens"][name]["clear"] is not True
                    for name in HOLDS)):
            raise ValueError(f"current 20 mm screen failed: {candidate}")
        screens.append(report)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    manifest = {
        "schema": "precision_insertion_eligible_endpoint_renders_v1",
        "catalog": str(source), "mode": catalog["mode"],
        "catalog_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "verification_depth_mm": 20.0,
        "minimum_hand_clearance_m": catalog["minimum_hand_clearance_m"],
        "eligible_count": len(rows), "rendered_grasp_count": 0,
        "scope": "offline_v8_mujoco_grasp_plus_exact_20mm_endpoint_only",
        "not_validated": catalog["not_validated"],
        "robot_ready": False, "grasps": [],
    }
    if rows:
        shared_meshes = target / "mesh_cache"
        shared_meshes.mkdir()
        key_path = shared_meshes / "key.ply"
        socket_path = shared_meshes / "socket_exact_collision.ply"
        key_mesh.export(key_path)
        socket_mesh.export(socket_path)
        for row, report in zip(rows, screens):
            pose = row["tabletop_pose_stem"]
            key = row["key"]
            grasp_dir = target / f"pose_{pose}" / "_".join(key)
            grasp_dir.mkdir(parents=True, exist_ok=False)
            (grasp_dir / "fresh_endpoint_screen.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8")
            images = []
            for hold in HOLDS:
                hold_dir = grasp_dir / hold
                hold_dir.mkdir()
                bundle = _write_bundle(
                    output=hold_dir, shared_root=root, mode=mode,
                    candidate=Path(row["candidate_dir"]), endpoint=report,
                    hold_name=hold, key_mesh_path=key_path,
                    socket_mesh_path=socket_path,
                )
                for view in VIEWS:
                    image = hold_dir / f"{view}.png"
                    subprocess.run([
                        str(executable), "--background", "--python", str(RENDERER),
                        "--", str(bundle), "--output", str(image),
                        "--width", str(width), "--height", str(height),
                        "--view", view, "--still-frame", "1",
                    ], check=True)
                    if not image.is_file():
                        raise RuntimeError(f"Blender produced no image: {image}")
                    images.append(str(image.relative_to(target)))
            manifest["grasps"].append({
                "key": key, "tabletop_pose_stem": pose,
                "candidate_dir": row["candidate_dir"],
                "fresh_endpoint_screen": str(
                    (grasp_dir / "fresh_endpoint_screen.json").relative_to(target)),
                "minimum_observed_hand_clearance_m":
                    report["minimum_observed_hand_clearance_m"],
                "images": images,
            })
            manifest["rendered_grasp_count"] += 1
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=900)
    args = parser.parse_args()
    try:
        result = render_catalog(
            catalog_path=args.catalog, output_root=args.output_root,
            blender=args.blender, width=args.width, height=args.height)
    except (OSError, KeyError, TypeError, ValueError,
            subprocess.CalledProcessError) as exc:
        parser.exit(2, f"render rejected: {exc}\n")
    print(json.dumps({
        "output_root": str(args.output_root.expanduser().resolve()),
        "eligible_count": result["eligible_count"],
        "rendered_grasp_count": result["rendered_grasp_count"],
        "scope": result["scope"],
    }, indent=2))
    return 0 if result["rendered_grasp_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
