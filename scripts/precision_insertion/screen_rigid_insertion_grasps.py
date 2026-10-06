#!/usr/bin/env python3
"""Screen tabletop grasps before expensive Franka trajectory planning.

The screen keeps the candidate's native ``T_key_hand`` fixed and evaluates
three independent gates with the exact key, socket, and Inspire visual meshes:

1. declared and complete-hand key contact policy;
2. pregrasp/grasp clearance above the candidate's tabletop stable pose; and
3. complete-hand/socket clearance at CAD pre-insertion and seated poses.

Passing this sampled screen is necessary but not sufficient.  It is not a
continuous collision proof, arm IK/trajectory plan, MuJoCo validation, or a
physical success label.  Its purpose is to avoid spending those expensive
checks on grasps that are geometrically impossible for insertion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh

from validate_whole_hand_contact_policy import (
    _hand_link_meshes,
    _raycast_scene,
    inspect_whole_hand,
)


SHARED = Path.home() / "shared_data"
DEFAULT_SCENE = (
    SHARED / "AutoDex/contact_screen_staging/inspire/"
    "precision_insertion_tabletop_v1/precision_key_1p5mm/table/0"
)
DEFAULT_KEY_MESH = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/mesh/"
    "simplified.obj"
)
DEFAULT_POLICY = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/info/"
    "contact_regions.json"
)
DEFAULT_TABLETOP = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/info/"
    "tabletop/000.npy"
)
DEFAULT_GEOMETRY = (
    SHARED / "AutoDex/precision_insertion/fixtures/unified_socket/"
    "task_geometry.json"
)
DEFAULT_SOCKET = (
    SHARED / "object_processing/precision_socket_unified/processed_data/mesh/"
    "static_collision.obj"
)
DEFAULT_ROBOT = (
    SHARED / "AutoDex/content/assets/robot/fr3_inspire_description/"
    "fr3_inspire.urdf"
)
DEFAULT_OUTPUT = (
    SHARED / "AutoDex/precision_insertion/"
    "rigid_insertion_grasp_screen_table_000.json"
)


def _candidate_dirs(scene: Path) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        return (
            (int(path.name), path.name)
            if path.name.isdigit()
            else (10**12, path.name)
        )

    return sorted(
        (
            path for path in scene.iterdir()
            if path.is_dir() and (path / "wrist_se3.npy").is_file()
        ),
        key=key,
    )


def _transform(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (transform[:3, :3] @ points.T).T + transform[:3, 3]


def _signed_distance(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return np.asarray(scene.compute_signed_distance(query).numpy())


def _sample_hand(
    robot_urdf: Path,
    hand_q: np.ndarray,
    samples_per_link: int,
    seed: int,
) -> np.ndarray:
    meshes = _hand_link_meshes(robot_urdf, hand_q)
    samples = []
    for index, (_name, mesh) in enumerate(sorted(meshes.items())):
        points, _ = trimesh.sample.sample_surface(
            mesh, samples_per_link, seed=seed + index
        )
        samples.append(points)
    return np.concatenate(samples)


def _environment_report(
    *,
    candidate: Path,
    stable_pose: np.ndarray,
    geometry: dict[str, Any],
    socket_scene: o3d.t.geometry.RaycastingScene,
    robot_urdf: Path,
    samples_per_link: int,
    penetration_threshold_m: float,
) -> dict[str, Any]:
    key_to_hand = np.load(candidate / "wrist_se3.npy")
    q_pregrasp = np.load(candidate / "pregrasp_pose.npy").reshape(-1)
    q_grasp = np.load(candidate / "grasp_pose.npy").reshape(-1)
    hand_points = {
        "pregrasp": _sample_hand(
            robot_urdf, q_pregrasp, samples_per_link, 1100
        ),
        "grasp": _sample_hand(
            robot_urdf, q_grasp, samples_per_link, 2100
        ),
    }

    table: dict[str, Any] = {}
    for phase, points in hand_points.items():
        world_points = _transform(stable_pose @ key_to_hand, points)
        minimum_z = float(world_points[:, 2].min())
        table[phase] = {
            "minimum_z_mm": minimum_z * 1000.0,
            "below_table_samples": int(np.count_nonzero(
                world_points[:, 2] < -penetration_threshold_m
            )),
        }

    socket: dict[str, Any] = {}
    for phase in ("preinsert", "seated"):
        socket_to_key = np.asarray(
            geometry[f"T_socket_key_{phase}"], dtype=np.float64
        )
        socket_to_hand = socket_to_key @ key_to_hand
        socket_points = _transform(socket_to_hand, hand_points["grasp"])
        distance = _signed_distance(socket_scene, socket_points)
        socket[phase] = {
            "minimum_signed_distance_mm": float(distance.min() * 1000.0),
            "penetrating_samples": int(np.count_nonzero(
                distance < -penetration_threshold_m
            )),
        }

    passed = (
        all(item["below_table_samples"] == 0 for item in table.values())
        and all(item["penetrating_samples"] == 0 for item in socket.values())
    )
    return {
        "passed": passed,
        "table": table,
        "socket": socket,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--key-mesh", type=Path, default=DEFAULT_KEY_MESH)
    parser.add_argument("--contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--tabletop-pose", type=Path, default=DEFAULT_TABLETOP)
    parser.add_argument("--task-geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--socket-mesh", type=Path, default=DEFAULT_SOCKET)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-link", type=int, default=3000)
    parser.add_argument("--penetration-threshold-mm", type=float, default=0.2)
    args = parser.parse_args()

    for field in (
        "scene", "key_mesh", "contact_policy", "tabletop_pose",
        "task_geometry", "socket_mesh", "robot_urdf", "output",
    ):
        setattr(args, field, getattr(args, field).expanduser().resolve())
    required_files = (
        args.key_mesh, args.contact_policy, args.tabletop_pose,
        args.task_geometry, args.socket_mesh, args.robot_urdf,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if not args.scene.is_dir():
        missing.append(str(args.scene))
    if missing:
        parser.error("missing input: " + ", ".join(missing))
    if args.samples_per_link < 100:
        parser.error("--samples-per-link must be at least 100")

    stable_pose = np.load(args.tabletop_pose)
    geometry = json.loads(args.task_geometry.read_text(encoding="utf-8"))
    socket_mesh = trimesh.load(args.socket_mesh, force="mesh", process=False)
    socket_scene = _raycast_scene(socket_mesh)
    threshold = args.penetration_threshold_mm / 1000.0

    rows = []
    for candidate in _candidate_dirs(args.scene):
        policy, _key_to_hand = inspect_whole_hand(
            candidate_dir=candidate,
            object_mesh_path=args.key_mesh,
            policy_path=args.contact_policy,
            robot_urdf=args.robot_urdf,
            symmetry_mode="none",
            samples_per_link=args.samples_per_link,
            penetration_threshold_m=threshold,
            seed=3100 + (int(candidate.name) if candidate.name.isdigit() else 0),
        )
        environment = _environment_report(
            candidate=candidate,
            stable_pose=stable_pose,
            geometry=geometry,
            socket_scene=socket_scene,
            robot_urdf=args.robot_urdf,
            samples_per_link=args.samples_per_link,
            penetration_threshold_m=threshold,
        )
        passed = policy["status"] == "sampled_pass" and environment["passed"]
        rows.append({
            "candidate": candidate.name,
            "passed": passed,
            "contact_policy": {
                "status": policy["status"],
                "declared_contacts_passed": policy["declared_contacts"]["passed"],
                "forbidden_penetrating_samples": (
                    policy["whole_hand"]["forbidden_penetrating_samples"]
                ),
                "permitted_contact_links": (
                    policy["whole_hand"]["permitted_contact_links"]
                ),
            },
            "environment": environment,
        })
        print(
            f"{candidate.name}: {'PASS' if passed else 'reject'} "
            f"policy={policy['status']} env={environment['passed']}"
        )

    report = {
        "schema_version": 1,
        "status": "sampled_prefilter_not_trajectory_or_physical_validation",
        "scene": str(args.scene),
        "tabletop_pose": str(args.tabletop_pose),
        "task_geometry": str(args.task_geometry),
        "socket_mesh": str(args.socket_mesh),
        "sampling": {
            "samples_per_link": args.samples_per_link,
            "penetration_threshold_mm": args.penetration_threshold_mm,
            "zero_samples_is_not_a_continuous_collision_proof": True,
        },
        "candidate_count": len(rows),
        "passed_candidates": [row["candidate"] for row in rows if row["passed"]],
        "candidates": rows,
        "required_next_checks": [
            "continuous full-key and full-hand collision checking",
            "Franka approach, lift, transfer, pre-insertion, and reset planning",
            "MuJoCo grasp validation",
            "physical validation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    print(json.dumps({
        "candidate_count": len(rows),
        "passed_candidates": report["passed_candidates"],
    }))
    return 0 if report["passed_candidates"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
