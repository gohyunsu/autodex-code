#!/usr/bin/env python3
"""Onboard the supplied millimetre cylinder STLs into AutoDex's v8 asset tree.

This writes geometry and metadata only. It never fabricates FoundPose weights,
grasp candidates, robot trajectories, or physical validation records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_assets import (
    _json_dump,
    _pose7,
    _write_static_fixture_urdf,
    _write_urdf,
    convex_hull_mesh,
    read_binary_stl,
    validate_watertight,
    write_mtl,
    write_obj,
)


KEY_NAME = "precision_key_cylinder_r15_h80"
SOCKET_PREFIX = "precision_socket_cylinder_gap_"
GAPS_MM = (1, 3, 5, 10, 15, 20)
KEY_HEIGHT_M = 0.080
KEY_RADIUS_M = 0.015
BORE_DEPTH_M = 0.050
RIM_Z_M = 0.055
INSERTION_DEPTH_M = 0.020
GRASP_SIDE_MIN_Z_M = 0.050


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometry_info(mesh, *, role: str) -> dict:
    bounds = mesh.bounds
    centre = (bounds[0] + bounds[1]) / 2
    transform = np.eye(4)
    transform[:3, 3] = centre
    return {
        "gravity_center": mesh.center_mass.tolist(),
        "obb": (bounds[1] - bounds[0]).tolist(),
        "obb_transform": transform.tolist(),
        "scale": 1.0,
        "density": 1.0,
        "mass": mesh.volume,
        "role": role,
    }


def _foundpose_marker(project: Path, name: str, raw_mesh: Path) -> None:
    _json_dump(project / "foundpose_assets" / name / "GENERATION_REQUIRED.json", {
        "status": "required",
        "object": name,
        "mesh": str(raw_mesh),
        "mesh_sha256": _sha256(raw_mesh),
        "expected": f"object_repre/v1/{name}/1/repre.pth",
        "generator": "src/process/batch_onboard_foundpose.py",
        "reason": "Learned representation must use this exact metric mesh and frame.",
    })


def _write_common_meshes(mesh, object_dir: Path, name: str, color: tuple[float, ...],
                         *, socket: bool) -> tuple[Path, Path, Path]:
    raw = object_dir / "raw_mesh" / f"{name}.obj"
    mesh_dir = object_dir / "processed_data" / "mesh"
    urdf = object_dir / "processed_data" / "urdf"
    write_mtl(raw.parent / "material.mtl", color)
    write_obj(raw, mesh, material_file="material.mtl")
    write_mtl(mesh_dir / "material.mtl", color)
    for filename in ("raw.obj", "manifold.obj", "simplified.obj"):
        write_obj(mesh_dir / filename, mesh, material_file="material.mtl")
    if socket:
        # A convex hull seals the bore. The static world needs its exact cavity.
        write_obj(mesh_dir / "static_collision.obj", mesh, material_file="material.mtl")
        urdf_path = urdf / "socket_static_exact.urdf"
        _write_static_fixture_urdf(urdf_path, name, mesh.volume, mesh.center_mass)
    else:
        hull = convex_hull_mesh(mesh)
        write_obj(mesh_dir / "coacd.obj", hull)
        write_obj(urdf / "meshes" / "convex_piece_000.obj", hull)
        urdf_path = urdf / "coacd.urdf"
        _write_urdf(urdf_path, name, mesh.volume, mesh.center_mass)
    return raw, mesh_dir / "simplified.obj", urdf_path


def _key(source: Path, shared: Path) -> dict:
    name = KEY_NAME
    mesh = read_binary_stl(source)
    validate_watertight(mesh)
    if not np.allclose(mesh.bounds, [[-KEY_RADIUS_M, -KEY_RADIUS_M, 0],
                                      [KEY_RADIUS_M, KEY_RADIUS_M, KEY_HEIGHT_M]],
                       atol=1e-5):
        raise ValueError(f"unexpected cylinder key bounds: {mesh.bounds}")
    obj_dir = shared / "object_processing" / name
    project = shared / "AutoDex"
    raw, planning, urdf = _write_common_meshes(mesh, obj_dir, name,
                                               (0.03, 0.22, 0.83), socket=False)
    info = obj_dir / "processed_data" / "info"
    _json_dump(info / "simplified.json", _geometry_info(mesh, role="dynamic_key"))
    _json_dump(info / "symmetry.json", {
        "type": "Dinf", "center": mesh.center_mass.tolist(),
        "axes": [{"axis": [0, 0, 1], "fold": "inf"},
                 {"axis": [1, 0, 0], "fold": 2}],
        "reason": "Identical flat ends and circular shaft; insertion uses the free end after grasp.",
    })
    normals = mesh.face_normals
    triangles = mesh.vertices[mesh.faces]
    allowed = [i for i, (normal, tri) in enumerate(zip(normals, triangles))
               if abs(float(normal[2])) < 0.05
               and float(tri[:, 2].min()) >= GRASP_SIDE_MIN_Z_M - 1e-8]
    allowed_set = set(allowed)
    forbidden = [i for i in range(len(mesh.faces)) if i not in allowed_set]
    write_obj(obj_dir / "processed_data" / "mesh" / "contact_allowed.obj",
              mesh, face_indices=allowed)
    write_obj(obj_dir / "processed_data" / "mesh" / "contact_forbidden.obj",
              mesh, face_indices=forbidden)
    _json_dump(info / "contact_regions.json", {
        "schema_version": 1, "units": "m", "frame": "object",
        "allowed": {"surface": "cylindrical_side", "z_range_m": [0.050, 0.080],
                    "mesh": "../mesh/contact_allowed.obj", "face_count": len(allowed)},
        "forbidden": {"description": "lower 50 mm and end faces",
                      "mesh": "../mesh/contact_forbidden.obj", "face_count": len(forbidden)},
        "note": "Proposal rule only; check every Inspire link and full 20 mm insertion path.",
    })
    poses = []
    upright = np.eye(4)
    poses.append(("000", "upright_flat_end", upright))
    side = np.eye(4)
    side[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    side[2, 3] = KEY_RADIUS_M
    poses.append(("001", "side_lying", side))
    for scene_index, (stem, label, pose) in enumerate(poses):
        path = info / "tabletop" / f"{stem}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, pose)
        _json_dump(project / "scene" / "inspire" / name / "table" /
                   f"{scene_index}.json", {
            "scene": {
                "mesh": {"target": {"scale": [1, 1, 1], "pose": _pose7(pose),
                                    "file_path": str(planning), "urdf_path": str(urdf)}},
                "cuboid": {"table": {"dims": [2, 2, 0.2],
                                      "pose": [0, 0, -0.1, 1, 0, 0, 0]}},
            },
            "meta": {"pose_idx": stem, "param": {"placement": label},
                     "precision_insertion": {"mode": "cylinder"}},
        })
    _json_dump(info / "tabletop_policy.json", {
        "poses": [{"stem": stem, "label": label} for stem, label, _ in poses],
        "baseline_pose_stem": "000",
        "note": "Side-lying azimuth remains continuous and must be measured per trial.",
    })
    _foundpose_marker(project, name, raw)
    _json_dump(project / "candidates" / "inspire" / "v8" / name /
               "GENERATION_REQUIRED.json", {
        "status": "required", "mode": "cylinder",
        "reason": "No BODex/MuJoCo/FR3 insertion-valid candidate is supplied.",
    })
    return {"name": name, "source": str(source), "source_sha256": _sha256(source),
            "raw_mesh": str(raw), "planning_mesh": str(planning),
            "tabletop_pose_stems": [stem for stem, _, _ in poses]}


def _socket(source: Path, gap_mm: int, shared: Path) -> dict:
    name = f"{SOCKET_PREFIX}{gap_mm:02d}mm"
    mesh = read_binary_stl(source)
    validate_watertight(mesh)
    inner_radius = KEY_RADIUS_M + gap_mm / 1000
    obj_dir = shared / "object_processing" / name
    project = shared / "AutoDex"
    raw, planning, urdf = _write_common_meshes(mesh, obj_dir, name,
                                               (0.82, 0.05, 0.08), socket=True)
    info = obj_dir / "processed_data" / "info"
    _json_dump(info / "simplified.json", {
        **_geometry_info(mesh, role="static_socket"),
        "dynamic_simulation_supported": False,
        "collision_mesh": "../mesh/static_collision.obj",
    })
    _json_dump(info / "symmetry.json", {
        "type": "Cinf", "center": mesh.center_mass.tolist(),
        "axes": [{"axis": [0, 0, 1], "fold": "inf"}],
        "reason": "Circular blind bore; yaw is unobservable and task-irrelevant.",
    })
    _json_dump(info / "frame_contract.json", {
        "schema_version": 1, "units": "m", "frame": "source_STL",
        "T_socket_raw_mesh": np.eye(4).tolist(),
        "bore_center_xy_m": [0, 0], "rim_z_m": RIM_Z_M,
        "bore_depth_m": BORE_DEPTH_M, "bore_radius_m": inner_radius,
        "insertion_direction_socket": [0, 0, -1],
        "socket_axis_local": [0, 0, 1],
    })
    _foundpose_marker(project, name, raw)
    return {"name": name, "gap_mm": gap_mm, "source": str(source),
            "source_sha256": _sha256(source), "raw_mesh": str(raw),
            "planning_mesh": str(planning),
            "collision_mesh": str(obj_dir / "processed_data" / "mesh" /
                                  "static_collision.obj"), "static_urdf": str(urdf)}


def build(source_dir: Path, shared_root: Path) -> dict:
    source_dir = source_dir.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    source_mesh_dir = source_dir / "stl" if (source_dir / "stl").is_dir() else source_dir / "source"
    key = _key(source_mesh_dir / "key_r15_h80.stl", shared_root)
    sockets = [_socket(source_mesh_dir /
                       f"socket_gap_{gap:02d}_r{15 + gap}.stl", gap, shared_root)
               for gap in GAPS_MM]
    result = {"schema_version": 1, "mode": "cylinder", "units": "m",
              "key": key, "sockets": sockets,
              "verification_depth_m": INSERTION_DEPTH_M,
              "status": "geometry_only_foundpose_grasps_and_insertion_validation_required"}
    _json_dump(shared_root / "AutoDex" / "precision_insertion" / "cylindrical_assets.json",
               result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "assets" /
                        "precision_insertion" / "cylindrical_fixed_key")
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.shared_root), indent=2))


if __name__ == "__main__":
    main()
