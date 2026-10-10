#!/usr/bin/env python3
"""Build symmetric cylindrical key/socket assets for AutoDex.

The canonical STL inputs are millimetre meshes.  Every generated OBJ and pose
uses metres.  The key is a finite cylinder with D-infinity symmetry: continuous
rotation about local +z plus a 180 degree flip that exchanges its identical
ends.  A socket has C-infinity symmetry about +z, but no flip symmetry because
its open rim and closed base are different.

The task contract chooses key z=80 mm as the representative insertion end.
That convention removes ambiguity from saved transforms without denying the
physical end-for-end symmetry of the key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from build_assets import (
    Mesh,
    _json_dump,
    _pose7,
    _rotation_y,
    _write_static_fixture_urdf,
    _write_urdf,
    convex_hull_mesh,
    read_binary_stl,
    validate_watertight,
    write_mtl,
    write_obj,
)


KEY_OBJECT = "precision_key_cylinder_r15_h80"
KEY_PROXY_OBJECT = f"{KEY_OBJECT}_grip_proxy"
KEY_SOURCE = "key_r15_h80.stl"
KEY_RADIUS_M = 0.015
KEY_HEIGHT_M = 0.080
GRIP_ZONE_HEIGHT_M = 0.025
SOCKET_BASE_RADIUS_M = 0.060
SOCKET_BASE_HEIGHT_M = 0.005
SOCKET_RIM_Z_M = 0.055
SOCKET_BORE_DEPTH_M = 0.050
VERIFICATION_DEPTH_M = 0.020

SOCKET_SPECS = (
    ("precision_socket_cylinder_gap_01mm", 1, 16, "socket_gap_01_r16.stl"),
    ("precision_socket_cylinder_gap_03mm", 3, 18, "socket_gap_03_r18.stl"),
    ("precision_socket_cylinder_gap_05mm", 5, 20, "socket_gap_05_r20.stl"),
    ("precision_socket_cylinder_gap_10mm", 10, 25, "socket_gap_10_r25.stl"),
    ("precision_socket_cylinder_gap_15mm", 15, 30, "socket_gap_15_r30.stl"),
    ("precision_socket_cylinder_gap_20mm", 20, 35, "socket_gap_20_r35.stl"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounds_info(mesh: Mesh, **extra) -> dict:
    bounds = mesh.bounds
    transform = np.eye(4)
    transform[:3, 3] = (bounds[0] + bounds[1]) / 2.0
    return {
        "gravity_center": mesh.center_mass.tolist(),
        "obb": (bounds[1] - bounds[0]).tolist(),
        "obb_transform": transform.tolist(),
        "scale": 1.0,
        "density": 1.0,
        "mass": mesh.volume,
        **extra,
    }


def _cylinder_side_partition(mesh: Mesh) -> tuple[list[int], list[int]]:
    """Allow the top grip-zone side and z=0 cap; forbid the insertion body."""
    allowed: list[int] = []
    forbidden: list[int] = []
    triangles = mesh.vertices[mesh.faces]
    for index, (triangle, normal) in enumerate(zip(triangles, mesh.face_normals)):
        z_min = float(triangle[:, 2].min())
        z_max = float(triangle[:, 2].max())
        top_cap = normal[2] < -0.95 and z_max <= 1e-9
        grip_side = abs(float(normal[2])) < 0.05 and z_max <= GRIP_ZONE_HEIGHT_M + 1e-9
        (allowed if top_cap or grip_side else forbidden).append(index)
    if not allowed or not forbidden:
        raise ValueError("cylindrical contact partition produced an empty set")
    return allowed, forbidden


def _key_tabletop_poses(mesh: Mesh) -> list[tuple[str, str, np.ndarray]]:
    """Two stable classes after quotienting by D-infinity symmetry."""
    # Keep the task-frame convention useful for grasp generation: z=0 is the
    # exposed grasp end and z=80 mm is the end resting on the table.  Dinf says
    # the physical ends are equivalent; it does not make this saved task-frame
    # choice arbitrary once contact regions have been assigned.
    end_down = np.eye(4)
    end_down[:3, :3] = np.diag([1.0, -1.0, -1.0])
    end_down[2, 3] = KEY_HEIGHT_M
    side_down = np.eye(4)
    side_down[:3, :3] = _rotation_y(math.pi / 2.0)
    transformed = (side_down[:3, :3] @ mesh.vertices.T).T
    side_down[2, 3] = -float(transformed[:, 2].min())
    return [
        ("000", "end_down", end_down),
        ("001", "side_down", side_down),
    ]


def _write_generation_marker(path: Path, object_name: str, reason: str) -> None:
    _json_dump(
        path,
        {
            "status": "required",
            "expected": f"object_repre/v1/{object_name}/1/repre.pth",
            "generator": "src/process/batch_onboard_foundpose.py",
            "reason": reason,
            "do_not_substitute": "a representation generated for another mesh or frame",
        },
    )


def _build_key(source: Path, object_root: Path, project_root: Path) -> dict:
    mesh = read_binary_stl(source)
    validate_watertight(mesh)
    hull = convex_hull_mesh(mesh)
    allowed, forbidden = _cylinder_side_partition(mesh)
    bounds = mesh.bounds
    if not np.allclose(bounds, [[-0.015, -0.015, 0.0], [0.015, 0.015, 0.080]], atol=2e-5):
        raise ValueError(f"unexpected key bounds: {bounds.tolist()}")

    root = object_root / KEY_OBJECT
    raw = root / "raw_mesh"
    mesh_dir = root / "processed_data" / "mesh"
    info = root / "processed_data" / "info"
    urdf = root / "processed_data" / "urdf"
    for directory in (raw, mesh_dir, info / "tabletop", urdf / "meshes"):
        directory.mkdir(parents=True, exist_ok=True)

    write_mtl(raw / "material.mtl", (0.05, 0.22, 0.85))
    write_obj(raw / f"{KEY_OBJECT}.obj", mesh, material_file="material.mtl")
    write_mtl(mesh_dir / "material.mtl", (0.05, 0.22, 0.85))
    for name in ("raw.obj", "manifold.obj", "simplified.obj"):
        write_obj(mesh_dir / name, mesh, material_file="material.mtl")
    write_obj(mesh_dir / "contact_allowed.obj", mesh, face_indices=allowed)
    write_obj(mesh_dir / "contact_forbidden.obj", mesh, face_indices=forbidden)
    write_obj(mesh_dir / "coacd.obj", hull)
    write_obj(urdf / "meshes" / "convex_piece_000.obj", hull)
    _write_urdf(urdf / "coacd.urdf", KEY_OBJECT, mesh.volume, mesh.center_mass)

    _json_dump(info / "simplified.json", _bounds_info(mesh))
    _json_dump(
        info / "symmetry.json",
        {
            "type": "Dinf",
            "center": [0.0, 0.0, KEY_HEIGHT_M / 2.0],
            "scale": float(np.linalg.norm(bounds[1] - bounds[0])),
            "axes": [
                {"axis": [0.0, 0.0, 1.0], "fold": "inf", "residual": 0.0},
                {"axis": [1.0, 0.0, 0.0], "fold": 2, "residual": 0.0},
                {"axis": [0.0, 1.0, 0.0], "fold": 2, "residual": 0.0},
            ],
            "rel_tol": 0.001,
            "task_convention": (
                "z=80 mm is the representative insertion end; the z=0 end is "
                "physically equivalent under the declared Dinf symmetry"
            ),
        },
    )
    _json_dump(
        info / "pose_symmetry.json",
        {
            "schema_version": 1,
            "equivalence_source": "processed_data/info/symmetry.json",
            "tabletop_classes": {
                "000": {"label": "end_down", "equivalent_end": "either_flat_end"},
                "001": {"label": "side_down", "equivalent_roll": "continuous_about_axis"},
            },
        },
    )
    _json_dump(
        info / "contact_regions.json",
        {
            "schema_version": 3,
            "frame": "object",
            "units": "m",
            "representative_insertion_axis": [0.0, 0.0, -1.0],
            "representative_inserted_end_local": [0.0, 0.0, KEY_HEIGHT_M],
            "physical_end_exchange_symmetry": True,
            "allowed": {
                "description": "cylinder side in z=[0,25] mm and the z=0 cap",
                "z_range_m": [0.0, GRIP_ZONE_HEIGHT_M],
                "mesh": "../mesh/contact_allowed.obj",
                "face_count": len(allowed),
            },
            "forbidden": {
                "description": "body below the grip zone, including the representative inserted end",
                "z_range_m": [GRIP_ZONE_HEIGHT_M, KEY_HEIGHT_M],
                "mesh": "../mesh/contact_forbidden.obj",
                "face_count": len(forbidden),
            },
            "reason": (
                "at full 50 mm seating, local z>30 mm enters below the socket rim; "
                "the 25 mm grip zone preserves a nominal 5 mm axial margin"
            ),
        },
    )

    poses = _key_tabletop_poses(mesh)
    scenes = project_root / "scene" / "inspire" / KEY_OBJECT / "table"
    scenes.mkdir(parents=True, exist_ok=True)
    records = []
    for scene_id, (stem, label, transform) in enumerate(poses):
        np.save(info / "tabletop" / f"{stem}.npy", transform)
        records.append({"stem": stem, "label": label})
        _json_dump(
            scenes / f"{scene_id}.json",
            {
                "scene": {
                    "mesh": {
                        "target": {
                            "scale": [1.0, 1.0, 1.0],
                            "pose": _pose7(transform),
                            "file_path": str(mesh_dir / "simplified.obj"),
                            "urdf_path": str(urdf / "coacd.urdf"),
                        }
                    },
                    "cuboid": {
                        "table": {
                            "dims": [2.0, 2.0, 0.2],
                            "pose": [0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],
                        }
                    },
                },
                "meta": {
                    "pose_idx": stem,
                    "param": {"placement": label},
                    "symmetry_reduced": True,
                },
            },
        )
    _json_dump(
        info / "tabletop_policy.json",
        {
            "poses": records,
            "symmetry_reduced_pose_count": 2,
            "equivalence": "Dinf: axial yaw and identical-end flip are folded",
            "recommended_bringup_pose": "000",
        },
    )

    _write_generation_marker(
        project_root / "foundpose_assets" / KEY_OBJECT / "GENERATION_REQUIRED.json",
        KEY_OBJECT,
        "FoundPose/DINOv2 representation requires GPU onboarding from the generated metric mesh",
    )
    _json_dump(
        project_root / "candidates" / "inspire" / "v8" / KEY_OBJECT / "GENERATION_REQUIRED.json",
        {
            "status": "required",
            "proposal_object": KEY_PROXY_OBJECT,
            "runtime_object": KEY_OBJECT,
            "tabletop_pose_stems": [stem for stem, _label, _pose in poses],
            "required_validation": [
                "BODex numerical quality",
                "full-key contact policy",
                "cuRobo tabletop and fixed-socket planning",
                "MuJoCo squeeze and gravity stability",
                "physical grasp and insertion",
            ],
        },
    )
    return {
        "object": KEY_OBJECT,
        "source": str(source),
        "source_sha256": _sha256(source),
        "bounds_m": bounds.tolist(),
        "volume_m3": mesh.volume,
        "center_mass_m": mesh.center_mass.tolist(),
        "symmetry": "Dinf",
        "tabletop_pose_classes": [record["label"] for record in records],
        "allowed_contact_faces": len(allowed),
        "forbidden_contact_faces": len(forbidden),
    }


def _build_key_proxy(source: Path, object_root: Path, project_root: Path) -> dict:
    full = read_binary_stl(source)
    # The source side quads cross the 25 mm plane, so build an exact short
    # cylinder from the source generator instead of clipping triangles.
    from generate_cylindrical_mesh import cylinder_mesh

    proxy = cylinder_mesh(KEY_RADIUS_M, GRIP_ZONE_HEIGHT_M, segments=256)
    hull = convex_hull_mesh(proxy)
    root = object_root / KEY_PROXY_OBJECT
    raw = root / "raw_mesh"
    mesh_dir = root / "processed_data" / "mesh"
    info = root / "processed_data" / "info"
    urdf = root / "processed_data" / "urdf"
    for directory in (raw, mesh_dir, info / "tabletop", urdf / "meshes"):
        directory.mkdir(parents=True, exist_ok=True)
    write_mtl(raw / "material.mtl", (0.20, 0.55, 0.95))
    write_obj(raw / f"{KEY_PROXY_OBJECT}.obj", proxy, material_file="material.mtl")
    write_mtl(mesh_dir / "material.mtl", (0.20, 0.55, 0.95))
    for name in ("raw.obj", "manifold.obj", "simplified.obj", "coacd.obj"):
        write_obj(mesh_dir / name, proxy, material_file="material.mtl")
    write_obj(urdf / "meshes" / "convex_piece_000.obj", hull)
    _write_urdf(urdf / "coacd.urdf", KEY_PROXY_OBJECT, full.volume, full.center_mass)
    _json_dump(
        info / "simplified.json",
        _bounds_info(
            proxy,
            gravity_center=full.center_mass.tolist(),
            mass=full.volume,
            generation_proxy=True,
            runtime_object=KEY_OBJECT,
        ),
    )
    shutil.copy2(
        object_root / KEY_OBJECT / "processed_data" / "info" / "symmetry.json",
        info / "symmetry.json",
    )
    poses = _key_tabletop_poses(full)
    scenes = project_root / "scene" / "inspire" / KEY_PROXY_OBJECT / "table"
    scenes.mkdir(parents=True, exist_ok=True)
    for scene_id, (stem, label, transform) in enumerate(poses):
        np.save(info / "tabletop" / f"{stem}.npy", transform)
        _json_dump(
            scenes / f"{scene_id}.json",
            {
                "scene": {
                    "mesh": {
                        "target": {
                            "scale": [1.0, 1.0, 1.0],
                            "pose": _pose7(transform),
                            "file_path": str(mesh_dir / "simplified.obj"),
                            "urdf_path": str(urdf / "coacd.urdf"),
                        }
                    },
                    "cuboid": {
                        "table": {
                            "dims": [2.0, 2.0, 0.2],
                            "pose": [0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],
                        }
                    },
                },
                "meta": {
                    "pose_idx": stem,
                    "param": {"placement": label},
                    "generation_proxy": True,
                    "runtime_object": KEY_OBJECT,
                },
            },
        )
    return {
        "object": KEY_PROXY_OBJECT,
        "runtime_object": KEY_OBJECT,
        "grip_zone_height_m": GRIP_ZONE_HEIGHT_M,
    }


def _build_socket(
    source: Path,
    object_name: str,
    gap_mm: int,
    bore_radius_mm: int,
    object_root: Path,
    project_root: Path,
) -> dict:
    mesh = read_binary_stl(source)
    validate_watertight(mesh)
    bounds = mesh.bounds
    root = object_root / object_name
    raw = root / "raw_mesh"
    mesh_dir = root / "processed_data" / "mesh"
    info = root / "processed_data" / "info"
    urdf = root / "processed_data" / "urdf"
    for directory in (raw, mesh_dir, info, urdf):
        directory.mkdir(parents=True, exist_ok=True)
    write_mtl(raw / "material.mtl", (0.82, 0.05, 0.08))
    write_obj(raw / f"{object_name}.obj", mesh, material_file="material.mtl")
    write_mtl(mesh_dir / "material.mtl", (0.82, 0.05, 0.08))
    for name in ("raw.obj", "manifold.obj", "simplified.obj", "static_collision.obj"):
        write_obj(mesh_dir / name, mesh, material_file="material.mtl")
    _write_static_fixture_urdf(
        urdf / "socket_static_exact.urdf", object_name, mesh.volume, mesh.center_mass
    )
    _json_dump(info / "simplified.json", _bounds_info(mesh, static_fixture=True))
    _json_dump(
        info / "symmetry.json",
        {
            "type": "Cinf",
            "center": [0.0, 0.0, 0.0],
            "scale": float(np.linalg.norm(bounds[1] - bounds[0])),
            "axes": [{"axis": [0.0, 0.0, 1.0], "fold": "inf", "residual": 0.0}],
            "rel_tol": 0.001,
            "reason": "open rim and closed base forbid end-for-end flip symmetry",
        },
    )
    _json_dump(
        info / "frame_contract.json",
        {
            "schema_version": 1,
            "frame": "socket",
            "units": "m",
            "T_socket_raw_mesh": np.eye(4).tolist(),
            "origin": "centre of circular base bottom plane",
            "axis": [0.0, 0.0, 1.0],
            "rim_z_m": SOCKET_RIM_Z_M,
            "bore_floor_z_m": SOCKET_BASE_HEIGHT_M,
            "yaw_observable_from_geometry": False,
        },
    )
    _write_generation_marker(
        project_root / "foundpose_assets" / object_name / "GENERATION_REQUIRED.json",
        object_name,
        "circular geometry cannot determine yaw; onboarding still supplies centre and axis pose evidence",
    )

    T_entry = np.eye(4)
    T_entry[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_entry[2, 3] = SOCKET_RIM_Z_M + KEY_HEIGHT_M
    T_verify = T_entry.copy()
    T_verify[2, 3] -= VERIFICATION_DEPTH_M
    T_seated = T_entry.copy()
    T_seated[2, 3] -= SOCKET_BORE_DEPTH_M
    fixture = project_root / "precision_insertion" / "cylindrical" / "fixtures" / object_name
    fixture.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh_dir / "static_collision.obj", fixture / "static_collision.obj")
    shutil.copy2(raw / "material.mtl", fixture / "material.mtl")
    _json_dump(
        fixture / "task_geometry.json",
        {
            "schema_version": 1,
            "units": "m",
            "key_object": KEY_OBJECT,
            "socket_pose_object": object_name,
            "radial_clearance_mm": gap_mm,
            "diametral_clearance_mm": 2 * gap_mm,
            "key_radius_m": KEY_RADIUS_M,
            "socket_bore_radius_m": bore_radius_mm * 1e-3,
            "socket_rim_z_m": SOCKET_RIM_Z_M,
            "socket_bore_depth_m": SOCKET_BORE_DEPTH_M,
            "verification_depth_m": VERIFICATION_DEPTH_M,
            "T_socket_pose_object": np.eye(4).tolist(),
            "T_socket_key_entry": T_entry.tolist(),
            "T_socket_key_verification": T_verify.tolist(),
            "T_socket_key_seated": T_seated.tolist(),
            "task_success_target": "T_socket_key_verification",
            "symmetry_contract": {
                "key": "Dinf",
                "socket": "Cinf",
                "insertion_yaw_dof": "quotiented_out",
                "controlled_alignment_dofs": ["x", "y", "axis_tilt_x", "axis_tilt_y"],
            },
        },
    )
    _json_dump(
        fixture / "fixture_pose.template.json",
        {
            "schema": "autodex_frozen_fixture_v1",
            "name": f"fixture_{object_name}",
            "object": object_name,
            "fixed_for_session": True,
            "pose_robot": None,
            "pose_source": "foundpose_multiview_session_preflight",
            "collision_mesh": str(mesh_dir / "static_collision.obj"),
            "note": "runtime pose must be measured; this template is not a calibration",
        },
    )
    return {
        "object": object_name,
        "source": str(source),
        "source_sha256": _sha256(source),
        "radial_clearance_mm": gap_mm,
        "diametral_clearance_mm": 2 * gap_mm,
        "bore_radius_mm": bore_radius_mm,
        "bounds_m": bounds.tolist(),
        "volume_m3": mesh.volume,
        "center_mass_m": mesh.center_mass.tolist(),
        "symmetry": "Cinf",
        "task_geometry": str(fixture / "task_geometry.json"),
    }


def build(source_dir: Path, shared_root: Path) -> dict:
    source_dir = source_dir.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    object_root = shared_root / "object_processing"
    project_root = shared_root / "AutoDex"
    required = [source_dir / KEY_SOURCE] + [source_dir / spec[3] for spec in SOCKET_SPECS]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing cylindrical source STL: " + ", ".join(missing))

    key = _build_key(source_dir / KEY_SOURCE, object_root, project_root)
    proxy = _build_key_proxy(source_dir / KEY_SOURCE, object_root, project_root)
    sockets = [
        _build_socket(source_dir / source_name, name, gap, bore, object_root, project_root)
        for name, gap, bore, source_name in SOCKET_SPECS
    ]
    out = project_root / "precision_insertion" / "cylindrical"
    out.mkdir(parents=True, exist_ok=True)
    _json_dump(
        out / "socket_family.json",
        {
            "schema_version": 1,
            "key_object": KEY_OBJECT,
            "clearance_definition": "radial one-sided nominal CAD gap",
            "recommended_progression_mm": [20, 15, 10, 5, 3, 1],
            "sockets": [
                {
                    "object": item["object"],
                    "radial_clearance_mm": item["radial_clearance_mm"],
                    "diametral_clearance_mm": item["diametral_clearance_mm"],
                    "task_geometry": item["task_geometry"],
                }
                for item in sockets
            ],
        },
    )
    manifest = {
        "schema_version": 1,
        "units": "m",
        "source_units": "mm",
        "key": key,
        "grasp_generation_proxy": proxy,
        "sockets": sockets,
        "generated_root": str(shared_root),
        "learned_assets_generated": False,
        "remaining_required": [
            "FoundPose representation for key and each socket used on robot",
            "BODex proposals and full-key contact/simulation screening",
            "continuous FR3/Inspire pick-lift-transfer-insertion planning",
            "physical controller commissioning and validation",
        ],
    }
    _json_dump(out / "asset_manifest.json", manifest)
    return manifest


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo / "assets" / "precision_insertion" / "cylindrical_fixed_key" / "source",
    )
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    args = parser.parse_args()
    manifest = build(args.source_dir, args.shared_root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
