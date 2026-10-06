#!/usr/bin/env python3
"""Sample the complete Inspire visual mesh against the precision-key policy.

The BODex contact screen checks only the declared object-side contact points.
This second, deliberately separate gate checks every Inspire visual link.  A
candidate fails when a sampled hand point penetrates a shaft, bevel, tip,
socket-facing shoulder, or diagonal handle chamfer.  Penetration through an
axis-aligned handle side or the handle rear is recorded as permitted contact.

The optional ``rear_x``/``rear_y`` symmetries are proposal tools for an
insertion grasp.  They rotate an existing handle grasp by 180 degrees about
the handle centre so that the palm is behind the key instead of beside the
shaft.  Such a derived grasp is not a BODex, cuRobo, MuJoCo, or physical pass.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation
from yourdfpy import URDF

from filter_contact_safe_grasps import _point_region


SHARED = Path.home() / "shared_data"
DEFAULT_CANDIDATE = (
    SHARED / "AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy/"
    "precision_key_handle_contact_proxy/table/0/84"
)
DEFAULT_OBJECT_MESH = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/mesh/"
    "simplified.obj"
)
DEFAULT_POLICY = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/info/"
    "contact_regions.json"
)
DEFAULT_ROBOT_URDF = (
    SHARED / "AutoDex/content/assets/robot/fr3_inspire_description/"
    "fr3_inspire.urdf"
)


def handle_symmetry(mode: str, handle_top_m: float = 0.045) -> np.ndarray:
    """Return an object-frame handle symmetry about the handle centre."""
    transform = np.eye(4, dtype=np.float64)
    if mode == "none":
        return transform
    axis = {"rear_x": "x", "rear_y": "y"}.get(mode)
    if axis is None:
        raise ValueError(f"unknown handle symmetry: {mode}")
    transform[:3, :3] = Rotation.from_euler(axis, 180.0, degrees=True).as_matrix()
    centre = np.asarray([0.0, 0.0, handle_top_m / 2.0], dtype=np.float64)
    transform[:3, 3] = centre - transform[:3, :3] @ centre
    return transform


def allowed_contact_faces(mesh: trimesh.Trimesh, handle_top_m: float) -> np.ndarray:
    """Classify faces whose surfaces may touch any part of the hand.

    The 2 mm edge margin applies to BODex's *declared contact locations*.
    For the whole-hand gate, an axis-aligned side/rear triangle remains an
    allowed surface all the way to an edge shared by another allowed surface.
    Diagonal chamfers and the front shoulder remain forbidden.
    """
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
    normals = np.asarray(mesh.face_normals)
    maximum_z = triangles[:, :, 2].max(axis=1)
    rear = (normals[:, 2] < -0.95) & (maximum_z <= 1.0e-8)
    side = (
        (maximum_z <= handle_top_m + 1.0e-8)
        & (np.abs(normals[:, 2]) < 0.05)
        & (np.maximum(np.abs(normals[:, 0]), np.abs(normals[:, 1])) > 0.95)
    )
    return rear | side


def _raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    return scene


def _hand_link_meshes(
    robot_urdf: Path,
    grasp_pose: np.ndarray,
) -> dict[str, trimesh.Trimesh]:
    robot = URDF.load(str(robot_urdf), build_scene_graph=True, load_meshes=True)
    names = [joint.name for joint in robot.actuated_joints]
    robot.update_cfg(dict(zip(names, np.concatenate([np.zeros(7), grasp_pose]))))
    hand_base = np.asarray(robot.get_transform("base_link", robot.base_link))
    hand_base_inverse = np.linalg.inv(hand_base)
    result: dict[str, trimesh.Trimesh] = {}
    for name, mesh in robot.scene.geometry.items():
        if not (name.startswith("base_link") or name.startswith("right_")):
            continue
        geometry_world, _ = robot.scene.graph.get(name)
        moved = mesh.copy()
        moved.apply_transform(hand_base_inverse @ geometry_world)
        result[name] = moved
    if not result:
        raise RuntimeError("Inspire visual meshes were not found in the robot URDF")
    return result


def _declared_contact_check(
    candidate_dir: Path,
    policy: dict[str, Any],
    symmetry: np.ndarray,
) -> dict[str, Any]:
    data = np.load(candidate_dir / "bodex_info.npy", allow_pickle=True).item()
    contacts = np.asarray(data["contact_point"], dtype=np.float64)
    contacts = contacts.reshape(-1, contacts.shape[-1])[:, :3]
    transformed = (
        (symmetry[:3, :3] @ contacts.T).T + symmetry[:3, 3]
    )
    half_x, half_y = policy["handle_half_extents_xy_m"]
    regions = [
        _point_region(
            point,
            half_x=float(half_x),
            half_y=float(half_y),
            handle_top=float(policy["handle_z_range"][1]),
            margin=float(policy["allowed"]["edge_margin_m"]),
            tolerance=float(policy["allowed"]["plane_tolerance_m"]),
        )
        for point in transformed
    ]
    return {
        "passed": all(region is not None for region in regions),
        "transformed_object_contacts_m": transformed.tolist(),
        "regions": regions,
        "edge_margin_mm": float(policy["allowed"]["edge_margin_m"]) * 1000.0,
    }


def inspect_whole_hand(
    *,
    candidate_dir: Path,
    object_mesh_path: Path,
    policy_path: Path,
    robot_urdf: Path,
    symmetry_mode: str,
    samples_per_link: int,
    penetration_threshold_m: float,
    seed: int = 84,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return the sampled gate report and the evaluated ``T_key_hand``."""
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    handle_top = float(policy["handle_z_range"][1])
    symmetry = handle_symmetry(symmetry_mode, handle_top)
    key_to_hand = symmetry @ np.load(candidate_dir / "wrist_se3.npy")
    grasp_pose = np.load(candidate_dir / "grasp_pose.npy").reshape(-1)

    object_mesh = trimesh.load(object_mesh_path, force="mesh", process=False)
    if not isinstance(object_mesh, trimesh.Trimesh) or not object_mesh.is_watertight:
        raise ValueError("object mesh must be a watertight triangle mesh")
    allowed_faces = allowed_contact_faces(object_mesh, handle_top)
    object_scene = _raycast_scene(object_mesh)
    link_meshes = _hand_link_meshes(robot_urdf, grasp_pose)

    link_reports: dict[str, dict[str, Any]] = {}
    forbidden_total = 0
    permitted_total = 0
    near_contact_links: list[str] = []
    minimum_signed_distance = float("inf")
    for index, (name, mesh) in enumerate(sorted(link_meshes.items())):
        points, _ = trimesh.sample.sample_surface(
            mesh, samples_per_link, seed=seed + index
        )
        object_points = (
            (key_to_hand[:3, :3] @ points.T).T + key_to_hand[:3, 3]
        )
        query = o3d.core.Tensor(object_points.astype(np.float32))
        signed = np.asarray(object_scene.compute_signed_distance(query).numpy())
        nearest = object_scene.compute_closest_points(query)
        face_ids = np.asarray(nearest["primitive_ids"].numpy(), dtype=np.int64)
        penetrates = signed < -penetration_threshold_m
        forbidden = penetrates & ~allowed_faces[face_ids]
        permitted = penetrates & allowed_faces[face_ids]
        near = (np.abs(signed) <= 1.2e-3) & allowed_faces[face_ids]
        minimum_signed_distance = min(minimum_signed_distance, float(signed.min()))
        forbidden_count = int(np.count_nonzero(forbidden))
        permitted_count = int(np.count_nonzero(permitted))
        near_count = int(np.count_nonzero(near))
        forbidden_total += forbidden_count
        permitted_total += permitted_count
        if near_count or permitted_count:
            near_contact_links.append(name)
        if forbidden_count or permitted_count or near_count:
            link_reports[name] = {
                "forbidden_penetrating_samples": forbidden_count,
                "permitted_penetrating_samples": permitted_count,
                "permitted_near_surface_samples": near_count,
                "minimum_signed_distance_mm": float(signed.min() * 1000.0),
                "forbidden_minimum_signed_distance_mm": (
                    float(signed[forbidden].min() * 1000.0)
                    if forbidden_count else None
                ),
            }

    declared = _declared_contact_check(candidate_dir, policy, symmetry)
    contact_links = sorted(set(near_contact_links))
    opposing_digits = (
        any("thumb" in name for name in contact_links)
        and any("thumb" not in name for name in contact_links)
    )
    passed = (
        declared["passed"]
        and forbidden_total == 0
        and opposing_digits
    )
    report = {
        "schema_version": 1,
        "status": "sampled_pass" if passed else "rejected",
        "scope": "declared_contacts_plus_whole_inspire_visual_mesh_contact_region",
        "candidate_dir": str(candidate_dir),
        "object_mesh": str(object_mesh_path),
        "contact_policy": str(policy_path),
        "robot_urdf": str(robot_urdf),
        "symmetry_mode": symmetry_mode,
        "T_key_hand": key_to_hand.tolist(),
        "sampling": {
            "samples_per_link": samples_per_link,
            "seed": seed,
            "penetration_threshold_mm": penetration_threshold_m * 1000.0,
            "absence_of_sampled_penetration_is_not_a_continuous_collision_proof": True,
        },
        "declared_contacts": declared,
        "whole_hand": {
            "forbidden_penetrating_samples": forbidden_total,
            "permitted_penetrating_samples": permitted_total,
            "permitted_contact_links": contact_links,
            "opposing_thumb_and_finger_contact": opposing_digits,
            "minimum_signed_distance_mm": minimum_signed_distance * 1000.0,
            "links": link_reports,
        },
        "interpretation": (
            "sampled hand/key contacts are confined to handle sides/rear"
            if passed else
            "do not use this grasp: declared contacts or sampled hand geometry violate policy"
        ),
        "not_validated": [
            "continuous exact collision",
            "self collision",
            "table or fixture clearance",
            "grasp stability after symmetry",
            "Franka motion planning",
            "physical execution",
        ],
    }
    return report, key_to_hand


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--object-mesh", type=Path, default=DEFAULT_OBJECT_MESH)
    parser.add_argument("--contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument(
        "--symmetry", choices=("none", "rear_x", "rear_y"), default="none"
    )
    parser.add_argument("--samples-per-link", type=int, default=20000)
    parser.add_argument("--penetration-threshold-mm", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate_dir = args.candidate_dir.expanduser().resolve()
    required = [
        candidate_dir / "wrist_se3.npy",
        candidate_dir / "grasp_pose.npy",
        candidate_dir / "bodex_info.npy",
        args.object_mesh.expanduser(),
        args.contact_policy.expanduser(),
        args.robot_urdf.expanduser(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing input: " + ", ".join(missing))
    if args.samples_per_link < 100:
        parser.error("--samples-per-link must be at least 100")

    report, _ = inspect_whole_hand(
        candidate_dir=candidate_dir,
        object_mesh_path=args.object_mesh.expanduser().resolve(),
        policy_path=args.contact_policy.expanduser().resolve(),
        robot_urdf=args.robot_urdf.expanduser().resolve(),
        symmetry_mode=args.symmetry,
        samples_per_link=args.samples_per_link,
        penetration_threshold_m=args.penetration_threshold_mm / 1000.0,
    )
    payload = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
        print(output)
    print(payload, end="")
    return 0 if report["status"] == "sampled_pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
