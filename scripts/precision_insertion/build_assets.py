#!/usr/bin/env python3
"""Build AutoDex assets for the unified precision-insertion key family.

The source STL files are in millimetres and use this object frame::

    +z: handle rear -> insertion tip
    z=0 mm: handle rear face
    z=45 mm: handle/shaft shoulder (socket-facing handle face)
    z=85.5 mm: key tip

The generated AutoDex meshes are in metres.  Only the four lateral faces of
the handle and its rear face are contact-allowed.  The shaft, shoulder, bevel,
and tip remain part of the collision mesh but are marked contact-forbidden.

This builder deliberately does not fabricate learned FoundPose descriptors or
robot grasps.  It writes machine-readable ``GENERATION_REQUIRED.json`` markers
for those GPU/physical-validation stages.  It does create one handle-only
grasp-proposal proxy per key so each optimization uses the actual full key's
centre of mass while keeping the insertion shaft off-limits to contacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


MM_TO_M = 1e-3
HANDLE_FRONT_Z_M = 45.0 * MM_TO_M
CONTACT_EDGE_MARGIN_M = 2.0 * MM_TO_M
CONTACT_PLANE_TOLERANCE_M = 1.0 * MM_TO_M
DEFAULT_PREINSERT_CLEARANCE_M = 30.0 * MM_TO_M

KEY_SPECS = (
    ("precision_key_1p5mm", "1.5", "plug_gap_1p5.stl"),
    ("precision_key_1p0mm", "1.0", "plug_gap_1p0.stl"),
    ("precision_key_0p5mm", "0.5", "plug_gap_0p5.stl"),
    ("precision_key_0p3mm", "0.3", "plug_gap_0p3.stl"),
)
SOCKET_SOURCE = "socket_shared_bore_1p5.stl"
FIXTURE_NAME = "unified_socket"
HANDLE_PROXY_NAME = "precision_key_handle_contact_proxy"

STAGE_SPECS = {
    "precision_key_1p5mm": {
        "purpose": "full pipeline bring-up and safe open-loop insertion",
        "control_mode": "open_loop_cartesian",
        "xy_yaw_search": False,
        "promotion_prerequisites": [],
    },
    "precision_key_1p0mm": {
        "purpose": "measure cumulative perception, calibration, and grasp-repeatability error",
        "control_mode": "open_loop_accuracy_measurement",
        "xy_yaw_search": False,
        "promotion_prerequisites": ["precision_key_1p5mm"],
    },
    "precision_key_0p5mm": {
        "purpose": "introduce force/contact-based XY and yaw search",
        "control_mode": "force_contact_xy_yaw_search",
        "xy_yaw_search": True,
        "promotion_prerequisites": ["precision_key_1p5mm", "precision_key_1p0mm"],
    },
    "precision_key_0p3mm": {
        "purpose": "final precision condition",
        "control_mode": "force_contact_xy_yaw_search",
        "xy_yaw_search": True,
        "promotion_prerequisites": [
            "precision_key_1p5mm",
            "precision_key_1p0mm",
            "precision_key_0p5mm",
        ],
    },
}


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray

    @property
    def bounds(self) -> np.ndarray:
        return np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))

    @property
    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(normal, axis=1)
        if np.any(length <= 1e-14):
            raise ValueError("mesh contains a degenerate triangle")
        return normal / length[:, None]

    @property
    def volume(self) -> float:
        tri = self.vertices[self.faces]
        return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)

    @property
    def center_mass(self) -> np.ndarray:
        tri = self.vertices[self.faces]
        six_v = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
        volume = six_v.sum() / 6.0
        if abs(volume) <= 1e-14:
            raise ValueError("mesh has zero signed volume")
        return (six_v[:, None] * tri.sum(axis=1)).sum(axis=0) / (24.0 * volume)


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _passed_validation(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
    except (OSError, AttributeError, json.JSONDecodeError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_binary_stl(path: Path, scale: float = MM_TO_M) -> Mesh:
    """Read a binary STL and merge identical vertices."""
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"invalid STL header: {path}")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + triangle_count * 50
    if len(data) != expected:
        raise ValueError(
            f"only binary STL is supported: {path} has {len(data)} bytes, expected {expected}"
        )

    dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attr", "<u2")]
    )
    records = np.frombuffer(data, dtype=dtype, offset=84, count=triangle_count)
    triangles = records["vertices"].astype(np.float64) * scale

    vertex_index: dict[tuple[float, float, float], int] = {}
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    for triangle in triangles:
        face: list[int] = []
        for vertex in triangle:
            key = tuple(float(value) for value in np.round(vertex, decimals=12))
            if key not in vertex_index:
                vertex_index[key] = len(vertices)
                vertices.append(vertex)
            face.append(vertex_index[key])
        faces.append(face)
    return Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))


def validate_watertight(mesh: Mesh) -> None:
    edge_counts: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for index in range(3):
            edge = tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    bad = {edge: count for edge, count in edge_counts.items() if count != 2}
    if bad:
        sample = list(bad.items())[:5]
        raise ValueError(f"mesh is not watertight; bad edges include {sample}")
    if mesh.volume <= 0:
        raise ValueError(f"mesh winding has non-positive signed volume {mesh.volume}")


def write_obj(
    path: Path,
    mesh: Mesh,
    *,
    face_indices: Iterable[int] | None = None,
    material_name: str = "precision_part",
    material_file: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(range(len(mesh.faces))) if face_indices is None else list(face_indices)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Generated by scripts/precision_insertion/build_assets.py\n")
        if material_file:
            handle.write(f"mtllib {material_file}\nusemtl {material_name}\n")
        for vertex in mesh.vertices:
            handle.write("v {:.10f} {:.10f} {:.10f}\n".format(*vertex))
        for face_index in selected:
            a, b, c = mesh.faces[face_index] + 1
            handle.write(f"f {a} {b} {c}\n")


def write_mtl(path: Path, rgb: Sequence[float], material_name: str = "precision_part") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Generated material\n"
        f"newmtl {material_name}\n"
        "Ka 0.050000 0.050000 0.050000\n"
        f"Kd {rgb[0]:.6f} {rgb[1]:.6f} {rgb[2]:.6f}\n"
        "Ks 0.100000 0.100000 0.100000\n"
        "Ns 20.000000\n"
        "d 1.000000\n",
        encoding="utf-8",
    )


def convex_hull_mesh(mesh: Mesh) -> Mesh:
    try:
        from scipy.spatial import ConvexHull
    except ImportError as exc:
        raise RuntimeError("scipy is required to generate the conservative collision hull") from exc
    hull = ConvexHull(mesh.vertices)
    faces = np.asarray(hull.simplices, dtype=np.int64).copy()
    center = mesh.vertices.mean(axis=0)
    for index, face in enumerate(faces):
        triangle = mesh.vertices[face]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if np.dot(normal, triangle.mean(axis=0) - center) < 0:
            faces[index, [1, 2]] = faces[index, [2, 1]]
    result = Mesh(mesh.vertices.copy(), faces)
    if result.volume <= 0:
        raise ValueError("failed to orient convex hull")
    return result


def contact_face_partition(mesh: Mesh) -> tuple[list[int], list[int]]:
    """Split key faces according to the handle-only contact rule."""
    normals = mesh.face_normals
    triangles = mesh.vertices[mesh.faces]
    allowed: list[int] = []
    forbidden: list[int] = []
    for index, (triangle, normal) in enumerate(zip(triangles, normals)):
        max_z = float(triangle[:, 2].max())
        on_rear = normal[2] < -0.95 and max_z <= 1e-8
        on_handle_side = (
            max_z <= HANDLE_FRONT_Z_M + 1e-8
            and abs(normal[2]) < 0.05
            and max(abs(normal[0]), abs(normal[1])) > 0.95
        )
        if on_rear or on_handle_side:
            allowed.append(index)
        else:
            forbidden.append(index)
    if not allowed or not forbidden:
        raise ValueError("contact partition unexpectedly produced an empty face set")
    return allowed, forbidden


def _box_mesh(bounds_min: Sequence[float], bounds_max: Sequence[float]) -> Mesh:
    """Return an outward-wound rectangular box mesh."""
    x0, y0, z0 = (float(value) for value in bounds_min)
    x1, y1, z1 = (float(value) for value in bounds_max)
    vertices = np.asarray(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    mesh = Mesh(vertices, faces)
    validate_watertight(mesh)
    return mesh


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def tabletop_poses(mesh: Mesh) -> list[tuple[str, str, np.ndarray]]:
    """Controlled placements: rear-down plus four handle-side placements.

    Tip-down is intentionally omitted to protect the insertion shaft.
    """
    rotations = (
        ("000", "handle_rear_down", np.eye(3)),
        ("001", "handle_x_positive_side_down", _rotation_y(math.pi / 2.0)),
        ("002", "handle_x_negative_side_down", _rotation_y(-math.pi / 2.0)),
        ("003", "handle_y_positive_side_down", _rotation_x(-math.pi / 2.0)),
        ("004", "handle_y_negative_side_down", _rotation_x(math.pi / 2.0)),
    )
    result: list[tuple[str, str, np.ndarray]] = []
    for stem, label, rotation in rotations:
        transformed = (rotation @ mesh.vertices.T).T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[2, 3] = -float(transformed[:, 2].min())
        result.append((stem, label, transform))
    return result


def _pose7(transform: np.ndarray) -> list[float]:
    """Return [x,y,z,qw,qx,qy,qz] for the limited rotations used here."""
    rotation = transform[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    return [
        *[float(value) for value in transform[:3, 3]],
        float(qw),
        float(qx),
        float(qy),
        float(qz),
    ]


def _write_urdf(path: Path, object_name: str, mass: float, center_mass: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version=\"1.0\"?>
<robot name=\"{object_name}\">
  <link name=\"object\">
    <inertial>
      <origin xyz=\"{center_mass[0]:.9f} {center_mass[1]:.9f} {center_mass[2]:.9f}\" rpy=\"0 0 0\"/>
      <mass value=\"{mass:.12f}\"/>
      <inertia ixx=\"1e-6\" ixy=\"0\" ixz=\"0\" iyy=\"1e-6\" iyz=\"0\" izz=\"1e-6\"/>
    </inertial>
    <visual>
      <geometry><mesh filename=\"../mesh/simplified.obj\" scale=\"1 1 1\"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename=\"meshes/convex_piece_000.obj\" scale=\"1 1 1\"/></geometry>
    </collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )


def _build_key(
    source: Path,
    object_name: str,
    gap_mm: str,
    object_root: Path,
    project_root: Path,
) -> dict:
    mesh = read_binary_stl(source)
    validate_watertight(mesh)
    hull = convex_hull_mesh(mesh)
    allowed, forbidden = contact_face_partition(mesh)

    object_dir = object_root / object_name
    raw_dir = object_dir / "raw_mesh"
    processed = object_dir / "processed_data"
    mesh_dir = processed / "mesh"
    info_dir = processed / "info"
    urdf_dir = processed / "urdf"

    for directory in (raw_dir, mesh_dir, info_dir / "tabletop", urdf_dir / "meshes"):
        directory.mkdir(parents=True, exist_ok=True)

    write_mtl(raw_dir / "material.mtl", (0.03, 0.12, 0.80))
    write_obj(
        raw_dir / f"{object_name}.obj",
        mesh,
        material_file="material.mtl",
    )
    write_mtl(mesh_dir / "material.mtl", (0.03, 0.12, 0.80))
    for name in ("raw.obj", "manifold.obj", "simplified.obj"):
        write_obj(mesh_dir / name, mesh, material_file="material.mtl")
    write_obj(mesh_dir / "contact_allowed.obj", mesh, face_indices=allowed)
    write_obj(mesh_dir / "contact_forbidden.obj", mesh, face_indices=forbidden)
    write_obj(mesh_dir / "coacd.obj", hull)
    write_obj(urdf_dir / "meshes" / "convex_piece_000.obj", hull)

    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    obb_transform = np.eye(4)
    obb_transform[:3, 3] = center
    _json_dump(
        info_dir / "simplified.json",
        {
            "gravity_center": mesh.center_mass.tolist(),
            "obb": (bounds[1] - bounds[0]).tolist(),
            "obb_transform": obb_transform.tolist(),
            "scale": 1.0,
            "density": 1.0,
            "mass": mesh.volume,
        },
    )
    _json_dump(
        info_dir / "symmetry.json",
        {
            "type": "none",
            "center": mesh.center_mass.tolist(),
            "scale": float(np.linalg.norm(bounds[1] - bounds[0])),
            "axes": [],
            "rel_tol": 0.01,
            "reason": "keyed cross-section intentionally breaks rotational symmetry",
        },
    )
    _json_dump(
        info_dir / "contact_regions.json",
        {
            "schema_version": 1,
            "frame": "object",
            "units": "m",
            "insertion_axis": [0.0, 0.0, 1.0],
            "handle_z_range": [0.0, HANDLE_FRONT_Z_M],
            "handle_half_extents_xy_m": [
                float(max(abs(bounds[0, 0]), abs(bounds[1, 0]))),
                float(max(abs(bounds[0, 1]), abs(bounds[1, 1]))),
            ],
            "allowed": {
                "description": "four lateral handle faces and rear handle face only",
                "mesh": "../mesh/contact_allowed.obj",
                "face_count": len(allowed),
                "edge_margin_m": CONTACT_EDGE_MARGIN_M,
                "plane_tolerance_m": CONTACT_PLANE_TOLERANCE_M,
            },
            "forbidden": {
                "description": "shaft, tip, bevel, and socket-facing handle shoulder",
                "mesh": "../mesh/contact_forbidden.obj",
                "face_count": len(forbidden),
                "z_rule": f"all shaft/transition geometry at z >= {HANDLE_FRONT_Z_M:.6f} m is forbidden",
            },
            "candidate_acceptance": {
                "all_declared fingertip contacts must be in allowed region": True,
                "all hand links must clear forbidden mesh": True,
                "physical insertion validation required": True,
            },
        },
    )

    pose_records = []
    scenes_dir = project_root / "scene" / "inspire" / object_name / "table"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    simplified_path = mesh_dir / "simplified.obj"
    urdf_path = urdf_dir / "coacd.urdf"
    for scene_id, (stem, label, transform) in enumerate(tabletop_poses(mesh)):
        np.save(info_dir / "tabletop" / f"{stem}.npy", transform)
        pose_records.append({"stem": stem, "label": label, "baseline": stem == "000"})
        _json_dump(
            scenes_dir / f"{scene_id}.json",
            {
                "scene": {
                    "mesh": {
                        "target": {
                            "scale": [1.0, 1.0, 1.0],
                            "pose": _pose7(transform),
                            "file_path": str(simplified_path),
                            "urdf_path": str(urdf_path),
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
                    "precision_insertion": {"gap_mm": gap_mm, "baseline": stem == "000"},
                },
            },
        )
    _json_dump(
        info_dir / "tabletop_policy.json",
        {
            "poses": pose_records,
            "baseline_pose_stem": "000",
            "excluded": ["tip_down"],
            "reason": "tip-down placement can damage or contaminate the insertion shaft",
        },
    )
    _write_urdf(urdf_path, object_name, mesh.volume, mesh.center_mass)

    foundpose_dir = project_root / "foundpose_assets" / object_name
    candidate_dir = project_root / "candidates" / "inspire" / "v8" / object_name
    _json_dump(
        foundpose_dir / "GENERATION_REQUIRED.json",
        {
            "status": "required",
            "expected": f"object_repre/v1/{object_name}/1/repre.pth",
            "generator": "src/process/batch_onboard_foundpose.py",
            "required_source": (
                "autodex/perception/thirdparty/MV-GoTrack/scripts/"
                "onboard_custom_mesh_for_foundpose.py"
            ),
            "reason": (
                "learned DINOv2/FoundPose representation requires object-specific "
                "GPU onboarding; the required MV-GoTrack source is not present"
            ),
            "do_not_substitute": "a representation generated for another mesh or frame",
        },
    )
    simulation_records = sorted(candidate_dir.rglob("simulation_validation.json"))
    has_simulated_candidate = any(
        _passed_validation(path) for path in simulation_records
    )
    planned_simulation_records = [
        path
        for path in simulation_records
        if _passed_validation(path)
        and _passed_validation(path.parent / "franka_plan_validation.json")
    ]
    _json_dump(
        candidate_dir / "GENERATION_REQUIRED.json",
        {
            "status": (
                "simulation_and_franka_plan_candidate_available_physical_validation_required"
                if planned_simulation_records
                else "simulation_candidate_available_franka_plan_and_physical_validation_required"
                if has_simulated_candidate
                else "required"
            ),
            "expected_files_per_grasp": [
                "wrist_se3.npy",
                "pregrasp_pose.npy",
                "grasp_pose.npy",
                "bodex_info.npy",
            ],
            "optional_files_per_grasp": ["openpose_000.npy"],
            "contact_policy": str(info_dir / "contact_regions.json"),
            "simulation_validation_records": [
                str(path) for path in simulation_records
            ],
            "simulation_and_franka_plan_records": [
                str(path) for path in planned_simulation_records
            ],
            "reason": (
                "a simulated and FR3-planned candidate exists but remains physically unvalidated"
                if planned_simulation_records
                else "a simulated candidate still needs FR3 planning and physical validation"
                if has_simulated_candidate
                else "a robot grasp must be optimized, contact-filtered, and physically validated"
            ),
        },
    )

    return {
        "object": object_name,
        "gap_mm": gap_mm,
        "source": str(source),
        "source_sha256": _sha256(source),
        "bounds_m": bounds.tolist(),
        "volume_m3": mesh.volume,
        "center_mass_m": mesh.center_mass.tolist(),
        "faces": int(len(mesh.faces)),
        "allowed_contact_faces": len(allowed),
        "forbidden_contact_faces": len(forbidden),
    }


def handle_proxy_name(runtime_object: str) -> str:
    """Return the proposal object name, preserving the original 1.5 mm name."""
    if runtime_object == "precision_key_1p5mm":
        return HANDLE_PROXY_NAME
    return f"{runtime_object}_handle_contact_proxy"


def _build_handle_proxy(
    reference_source: Path,
    runtime_object: str,
    object_root: Path,
    project_root: Path,
) -> dict:
    """Build the handle-only BODex proposal proxy in the real key frame.

    Its geometry deliberately omits the shaft. Candidates must subsequently
    be checked against the full key; this object is never a runtime target.
    """
    reference = read_binary_stl(reference_source)
    bounds = reference.bounds
    proxy = _box_mesh(
        [bounds[0, 0], bounds[0, 1], 0.0],
        [bounds[1, 0], bounds[1, 1], HANDLE_FRONT_Z_M],
    )
    proxy_name = handle_proxy_name(runtime_object)
    object_dir = object_root / proxy_name
    raw_dir = object_dir / "raw_mesh"
    mesh_dir = object_dir / "processed_data" / "mesh"
    info_dir = object_dir / "processed_data" / "info"
    urdf_dir = object_dir / "processed_data" / "urdf"
    for directory in (raw_dir, mesh_dir, info_dir / "tabletop", urdf_dir / "meshes"):
        directory.mkdir(parents=True, exist_ok=True)

    write_mtl(raw_dir / "material.mtl", (0.20, 0.55, 0.95))
    write_obj(raw_dir / f"{proxy_name}.obj", proxy, material_file="material.mtl")
    write_mtl(mesh_dir / "material.mtl", (0.20, 0.55, 0.95))
    for name in ("raw.obj", "manifold.obj", "simplified.obj", "coacd.obj"):
        write_obj(mesh_dir / name, proxy, material_file="material.mtl")
    write_obj(urdf_dir / "meshes" / "convex_piece_000.obj", proxy)

    obb_transform = np.eye(4)
    obb_transform[:3, 3] = (bounds[0] + bounds[1]) / 2.0
    _json_dump(
        info_dir / "simplified.json",
        {
            # Keep the actual full-key wrench reference while sampling only
            # the handle surface.
            "gravity_center": reference.center_mass.tolist(),
            "obb": (bounds[1] - bounds[0]).tolist(),
            "obb_transform": obb_transform.tolist(),
            "scale": 1.0,
            "density": 1.0,
            "mass": reference.volume,
            "generation_proxy": True,
            "runtime_object": runtime_object,
        },
    )
    _json_dump(
        info_dir / "symmetry.json",
        {
            "type": "none",
            "center": reference.center_mass.tolist(),
            "scale": float(np.linalg.norm(bounds[1] - bounds[0])),
            "axes": [],
            "rel_tol": 0.01,
            "reason": "proposal-only proxy; candidate is rechecked against keyed full mesh",
        },
    )
    pose = np.eye(4)
    np.save(info_dir / "tabletop" / "000.npy", pose)
    _json_dump(
        info_dir / "tabletop_policy.json",
        {
            "poses": [{"stem": "000", "label": "handle_rear_down", "baseline": True}],
            "baseline_pose_stem": "000",
            "generation_proxy": True,
        },
    )
    urdf_path = urdf_dir / "coacd.urdf"
    _write_urdf(urdf_path, proxy_name, reference.volume, reference.center_mass)

    scene_dir = project_root / "scene" / "inspire" / proxy_name / "table"
    _json_dump(
        scene_dir / "0.json",
        {
            "scene": {
                "mesh": {
                    "target": {
                        "scale": [1.0, 1.0, 1.0],
                        "pose": _pose7(pose),
                        "file_path": str(mesh_dir / "simplified.obj"),
                        "urdf_path": str(urdf_path),
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
                "pose_idx": "000",
                "param": {"placement": "handle_rear_down"},
                "precision_insertion": {
                    "generation_proxy": True,
                    "runtime_object": runtime_object,
                },
            },
        },
    )
    return {
        "object": proxy_name,
        "runtime_object": runtime_object,
        "reference_source": str(reference_source),
        "reference_source_sha256": _sha256(reference_source),
        "reference_center_mass_m": reference.center_mass.tolist(),
        "reference_volume_m3": reference.volume,
        "bounds_m": proxy.bounds.tolist(),
        "warning": "proposal only; recheck every candidate against the full key mesh",
    }


def _build_stage_profiles(project_root: Path) -> list[dict]:
    """Write declarative, fail-closed experiment assets for all four gaps.

    These files describe inputs and runtime gates.  They do not claim that an
    insertion controller, a physical calibration, or a physical grasp has
    been completed.
    """
    task_root = project_root / "precision_insertion"
    fixture_root = task_root / "fixtures" / FIXTURE_NAME
    calibration_root = task_root / "calibration"
    profiles: list[dict] = []
    for object_name, gap_mm, _source_name in KEY_SPECS:
        spec = STAGE_SPECS[object_name]
        profile = {
            "schema_version": 1,
            "task": "precision_insertion",
            "object": object_name,
            "nominal_per_side_gap_mm": float(gap_mm),
            "purpose": spec["purpose"],
            "frames": {
                "key": "object",
                "socket": "source STL frame",
                "robot": "fr3_link0",
            },
            "assets": {
                "object_root": str(project_root.parent / "object_processing" / object_name),
                "scene": str(project_root / "scene" / "inspire" / object_name / "table" / "0.json"),
                "candidate_scene": str(project_root / "candidates" / "inspire" / "v8" / object_name / "table" / "0"),
                "foundpose_representation": str(
                    project_root / "foundpose_assets" / object_name / "object_repre" /
                    "v1" / object_name / "1" / "repre.pth"
                ),
                "socket_geometry": str(fixture_root / "task_geometry.json"),
                "fixture_pose": str(fixture_root / "fixture_pose.json"),
                "camera_calibration_root": str(calibration_root),
            },
            "grasp": {
                "proposal_proxy": handle_proxy_name(object_name),
                "scene_type": "table",
                "scene_id": "0",
                "tabletop_pose_stem": "000",
                "contact_policy": (
                    "four lateral handle faces and rear face only; shaft, tip, "
                    "bevel, and socket-facing shoulder forbidden"
                ),
                "physical_validation_required": True,
            },
            "controller": {
                "mode": spec["control_mode"],
                "implementation_status": "required",
                "use_xy_yaw_search": spec["xy_yaw_search"],
                "force_torque_limits": None,
                "search_step_sizes": None,
                "note": (
                    "Safety and search parameters must be commissioned on the "
                    "physical Franka; this asset intentionally supplies no invented defaults."
                ),
            },
            "promotion_prerequisites": spec["promotion_prerequisites"],
            "runtime_gates": [
                "FoundPose representation exists for this exact full-key mesh and frame",
                "candidate passed full-key simulation and FR3 planning",
                "candidate passed supervised physical grasp/lift validation",
                "fixture_pose.json contains a measured T_robot_socket",
                "ZeroDex four-camera and hand-eye snapshot is verified on the physical rig",
                "insertion controller and abort thresholds for this stage are commissioned",
            ],
        }
        _json_dump(task_root / "stages" / f"{object_name}.json", profile)
        profiles.append(profile)
    return profiles


def _build_socket(source: Path, project_root: Path) -> dict:
    mesh = read_binary_stl(source)
    validate_watertight(mesh)
    fixture_dir = project_root / "precision_insertion" / "fixtures" / FIXTURE_NAME
    fixture_dir.mkdir(parents=True, exist_ok=True)
    write_mtl(fixture_dir / "material.mtl", (0.80, 0.03, 0.03))
    write_obj(
        fixture_dir / "socket_shared_bore_1p5.obj",
        mesh,
        material_file="material.mtl",
    )

    bounds = mesh.bounds
    socket_entry_z = float(bounds[1, 2])
    key_tip_z = 85.5 * MM_TO_M
    seated = np.eye(4)
    seated[:3, :3] = _rotation_x(math.pi)
    seated[2, 3] = socket_entry_z + HANDLE_FRONT_Z_M
    preinsert = seated.copy()
    preinsert[2, 3] += DEFAULT_PREINSERT_CLEARANCE_M

    task_geometry = {
        "schema_version": 1,
        "units": "m",
        "socket_frame": "source STL frame",
        "socket_mesh": "socket_shared_bore_1p5.obj",
        "socket_entry_plane_z_m": socket_entry_z,
        "socket_bore_bottom_z_m": 13.5 * MM_TO_M,
        "insertion_direction_socket": [0.0, 0.0, -1.0],
        "key_frame": {
            "insertion_axis": [0.0, 0.0, 1.0],
            "handle_rear_z_m": 0.0,
            "handle_front_z_m": HANDLE_FRONT_Z_M,
            "tip_z_m": key_tip_z,
        },
        "T_socket_key_seated": seated.tolist(),
        "T_socket_key_preinsert": preinsert.tolist(),
        "nominal_insertion_depth_m": key_tip_z - HANDLE_FRONT_Z_M,
        "preinsert_clearance_m": DEFAULT_PREINSERT_CLEARANCE_M,
        "alignment_note": (
            "Rx(pi) is required: after flipping the printed plug for insertion, "
            "its chamfered profile matches the socket's mirrored bore profile."
        ),
    }
    _json_dump(fixture_dir / "task_geometry.json", task_geometry)
    _json_dump(
        fixture_dir / "fixture_pose.template.json",
        {
            "schema_version": 1,
            "calibrated": False,
            "frame_from": "socket",
            "frame_to": "fr3_link0",
            "T_robot_socket": None,
            "required_method": "measure the rigidly mounted socket pose in the Franka base frame",
            "do_not_run_reason": "sub-millimetre insertion targets cannot use an invented fixture pose",
        },
    )
    return {
        "fixture": FIXTURE_NAME,
        "source": str(source),
        "source_sha256": _sha256(source),
        "bounds_m": bounds.tolist(),
        "volume_m3": mesh.volume,
        "faces": int(len(mesh.faces)),
        "task_geometry": task_geometry,
    }


def build(source_dir: Path, shared_root: Path) -> dict:
    source_dir = source_dir.resolve()
    shared_root = shared_root.expanduser().resolve()
    object_root = shared_root / "object_processing"
    project_root = shared_root / "AutoDex"
    object_root.mkdir(parents=True, exist_ok=True)
    project_root.mkdir(parents=True, exist_ok=True)

    missing = [name for name in [*(spec[2] for spec in KEY_SPECS), SOCKET_SOURCE] if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source STL files under {source_dir}: {missing}")

    keys = [
        _build_key(source_dir / source_name, object_name, gap_mm, object_root, project_root)
        for object_name, gap_mm, source_name in KEY_SPECS
    ]
    handle_proxies = [
        _build_handle_proxy(
            source_dir / source_name,
            object_name,
            object_root,
            project_root,
        )
        for object_name, _gap_mm, source_name in KEY_SPECS
    ]
    socket = _build_socket(source_dir / SOCKET_SOURCE, project_root)
    stage_profiles = _build_stage_profiles(project_root)

    task_root = project_root / "precision_insertion"
    manifest = {
        "schema_version": 1,
        "family": "unified_single_socket",
        "units": "m",
        "shared_root": str(shared_root),
        "objects": keys,
        # Singular field retained for readers created before per-key proxies.
        "grasp_generation_proxy": handle_proxies[0],
        "grasp_generation_proxies": handle_proxies,
        "socket": socket,
        "progression": [
            {
                "object": profile["object"],
                "gap_mm": str(profile["nominal_per_side_gap_mm"]),
                "purpose": profile["purpose"],
                "stage_profile": str(
                    task_root / "stages" / f"{profile['object']}.json"
                ),
            }
            for profile in stage_profiles
        ],
        "baseline": {
            "object": "precision_key_1p5mm",
            "tabletop_pose_stem": "000",
            "candidate_scene_type": "table",
            "candidate_scene_id": "0",
            "socket_pose": str(task_root / "fixtures" / FIXTURE_NAME / "fixture_pose.json"),
        },
        "zerodex_camera_profile": (
            "assets/precision_insertion/zerodex_camera_profile.json"
        ),
        "incomplete_runtime_assets": [
            "FoundPose repre.pth for each key",
            "supervised physical validation for each simulated and FR3-planned grasp",
            "measured fixture_pose.json",
            "ZeroDex camera/hand-eye calibration snapshot selected for the physical rig",
            "stage-appropriate insertion controller and safety parameters",
        ],
    }
    _json_dump(task_root / "asset_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=repo_root / "assets" / "precision_insertion" / "source",
    )
    parser.add_argument(
        "--shared-root",
        type=Path,
        default=Path.home() / "shared_data",
        help="Writable AutoDex shared-data overlay (default: ~/shared_data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args.source_dir, args.shared_root)
    print(f"built {len(manifest['objects'])} key objects under {manifest['shared_root']}")
    print(Path(manifest["shared_root"]) / "AutoDex" / "precision_insertion" / "asset_manifest.json")


if __name__ == "__main__":
    main()
