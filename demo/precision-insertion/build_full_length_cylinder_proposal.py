#!/usr/bin/env python3
"""Build a full-length, low-polygon cylinder for unrestricted BODex proposals.

The 64-sided mesh avoids the known high-resolution convex-hull seeder error.
It covers the key's entire 80 mm height; it is never used as the physical
key in AutoDex collision, squeeze, MuJoCo or insertion screening. Existing
25 mm grip-proxy assets and their historical evidence are left untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


KEY = "precision_key_cylinder_r15_h80"
PROPOSAL = f"{KEY}_full_length_proposal"
RADIUS_M = 0.015
HEIGHT_M = 0.080
SEGMENTS = 64
SCENES = {"table": ("0", "1"), "reorient_12": ("0_1", "1_0")}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def build(*, shared_root: Path) -> dict:
    """Install a distinct proposal object without replacing any old asset."""
    root = Path(shared_root).expanduser().resolve()
    objects = root / "object_processing"
    scenes = root / "AutoDex" / "scene" / "inspire"
    source = objects / KEY
    target = objects / PROPOSAL
    scene_target = scenes / PROPOSAL
    manifest = (root / "AutoDex" / "precision_insertion" / "cylindrical" /
                "full_length_proposal_asset_manifest.json")
    if target.exists() or scene_target.exists() or manifest.exists():
        raise FileExistsError("full-length proposal object or scene already exists")
    source_info = source / "processed_data" / "info"
    source_urdf = source / "processed_data" / "urdf" / "coacd.urdf"
    source_files = [source_info / "simplified.json", source_urdf]
    source_files.extend(source_info / "tabletop" / f"{stem:03d}.npy"
                        for stem in (0, 1))
    source_files.extend(scenes / KEY / kind / f"{cell}.json"
                        for kind, cells in SCENES.items() for cell in cells)
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("full-key proposal inputs missing: " + ", ".join(missing))

    mesh = trimesh.creation.cylinder(
        radius=RADIUS_M, height=HEIGHT_M, sections=SEGMENTS)
    mesh.apply_translation([0.0, 0.0, HEIGHT_M / 2.0])
    if not mesh.is_watertight or not np.allclose(
            mesh.bounds, [[-RADIUS_M, -RADIUS_M, 0.0],
                          [RADIUS_M, RADIUS_M, HEIGHT_M]], atol=1e-9):
        raise ValueError("full-length proposal mesh dimensions are invalid")
    asset_files = {
        "raw_mesh": target / "raw_mesh" / f"{PROPOSAL}.obj",
        "raw": target / "processed_data" / "mesh" / "raw.obj",
        "manifold": target / "processed_data" / "mesh" / "manifold.obj",
        "simplified": target / "processed_data" / "mesh" / "simplified.obj",
        "coacd": target / "processed_data" / "mesh" / "coacd.obj",
        "convex": target / "processed_data" / "urdf" / "meshes" /
                  "convex_piece_000.obj",
    }
    for path in asset_files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path, file_type="obj")
    urdf_path = target / "processed_data" / "urdf" / "coacd.urdf"
    urdf_path.write_text(
        source_urdf.read_text(encoding="utf-8").replace(
            f'<robot name="{KEY}">', f'<robot name="{PROPOSAL}">'),
        encoding="utf-8")
    info = json.loads((source_info / "simplified.json").read_text(
        encoding="utf-8"))
    info.update({
        "generation_proxy": True, "runtime_object": KEY,
        "proposal_surface_z_range_m": [0.0, HEIGHT_M],
        "proposal_radial_segments": SEGMENTS,
        "proposal_only": True,
    })
    _write_json(target / "processed_data" / "info" / "simplified.json", info)
    _write_json(target / "processed_data" / "info" / "symmetry.json", {
        "type": "none", "center": [0.0, 0.0, HEIGHT_M / 2.0],
        "scale": float(np.linalg.norm(mesh.extents)), "axes": [],
        "rel_tol": 0.001,
        "reason": "proposal-only polygonal cylinder; full-key Dinf symmetry is checked after proposal",
    })
    (target / "processed_data" / "info" / "tabletop").mkdir(parents=True)
    for stem in (0, 1):
        pose = np.load(source_info / "tabletop" / f"{stem:03d}.npy",
                       allow_pickle=False)
        np.save(target / "processed_data" / "info" / "tabletop" /
                f"{stem:03d}.npy", pose)
    scene_hashes = {}
    for kind, cells in SCENES.items():
        for cell in cells:
            original = scenes / KEY / kind / f"{cell}.json"
            record = json.loads(original.read_text(encoding="utf-8"))
            part = record["scene"]["mesh"]["target"]
            part["file_path"] = str(asset_files["simplified"])
            part["urdf_path"] = str(urdf_path)
            record["meta"].update({
                "generation_proxy": True, "runtime_object": KEY,
                "proposal_surface_z_range_m": [0.0, HEIGHT_M],
            })
            destination = scene_target / kind / f"{cell}.json"
            _write_json(destination, record)
            scene_hashes[f"{kind}/{cell}"] = {
                "source_sha256": _sha(original),
                "proposal_sha256": _sha(destination),
            }
    report = {
        "schema": "precision_insertion_full_length_cylinder_proposal_v1",
        "proposal_object": PROPOSAL, "runtime_object": KEY,
        "shared_root": str(root), "proposal_height_m": HEIGHT_M,
        "proposal_radius_m": RADIUS_M,
        "radial_segments": SEGMENTS,
        "max_radial_sagitta_m": RADIUS_M *
        (1.0 - np.cos(np.pi / SEGMENTS)),
        "proposal_mesh_sha256": _sha(asset_files["simplified"]),
        "scene_hashes": scene_hashes,
        "proposal_contact_region_restricted": False,
        "validation_object": KEY,
        "scope": "unrestricted_proposals_only_full_key_filter_still_required",
        "robot_ready": False,
    }
    _write_json(manifest, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(shared_root=args.shared_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
