#!/usr/bin/env python3
"""Build actual-mesh tabletop-pick-to-insertion animation trajectories.

The five object-processing stable poses are handled explicitly.  Each starts
with a pose-specific table-clear grasp and lifts the key.  Poses whose pickup
grasp remains insertion-clear use a fixed-grasp wrist reorientation.  The
remaining poses show the intended transition to a common insertion-safe grasp;
that transition is explicitly marked unvalidated.

The output is a sampled geometric IK preview, not a cuRobo plan or evidence of
physical success.  The report keeps that distinction machine-readable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import trimesh
from yourdfpy import URDF

from build_fixture_to_fixture_success_preview import (
    _interpolate_pose,
    _socket_pose,
)
from build_insertion_reachability_preview import (
    FrankaHandKinematics,
    _hand_samples_in_base,
    _raycast_scene,
    _signed_distance,
    _transform_points,
)
from filter_contact_safe_grasps import _point_region
from validate_whole_hand_contact_policy import (
    _hand_link_meshes,
    allowed_contact_faces,
    handle_symmetry,
)


SHARED = Path.home() / "shared_data"
DEFAULT_TABLETOP_CANDIDATE_ROOT = (
    SHARED / "AutoDex/contact_screen_staging/inspire/"
    "precision_insertion_tabletop_v1/precision_key_1p5mm/table"
)
DEFAULT_INSERTION_CANDIDATE = (
    SHARED / "AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy/"
    "precision_key_handle_contact_proxy/table/0/84"
)
DEFAULT_TABLETOP_DIR = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/info/tabletop"
)
DEFAULT_GEOMETRY = (
    SHARED / "AutoDex/precision_insertion/fixtures/unified_socket/task_geometry.json"
)
DEFAULT_KEY_MESH = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/mesh/"
    "simplified.obj"
)
DEFAULT_POLICY = (
    SHARED / "object_processing/precision_key_1p5mm/processed_data/info/"
    "contact_regions.json"
)
DEFAULT_SOCKET_MESH = (
    SHARED / "object_processing/precision_socket_unified/processed_data/mesh/"
    "static_collision.obj"
)
DEFAULT_ROBOT_URDF = (
    SHARED / "AutoDex/content/assets/robot/fr3_inspire_description/"
    "fr3_inspire.urdf"
)
DEFAULT_OUTPUT_DIR = (
    SHARED / "AutoDex/precision_insertion/visualizations/tabletop_pose_set"
)


@dataclass(frozen=True)
class Grasp:
    name: str
    candidate_dir: Path
    key_to_hand: np.ndarray
    pregrasp: np.ndarray
    grasp: np.ndarray


@dataclass
class Segment:
    phase: str
    hand_targets: list[np.ndarray]
    hand_q: np.ndarray
    object_poses: np.ndarray
    sample_mode: str


def _tabletop_grasp(candidate_root: Path, pose_id: int) -> Grasp:
    candidate_ids = {0: "346", 1: "403", 2: "511", 3: "27", 4: "290"}
    candidate_dir = candidate_root / str(pose_id) / candidate_ids[pose_id]
    pregrasp = np.load(candidate_dir / "pregrasp_pose.npy").reshape(-1)
    grasp = np.load(candidate_dir / "grasp_pose.npy").reshape(-1).copy()
    if pose_id == 0:
        grasp[0] = 0.821
        grasp[1] = 0.220
    elif pose_id == 1:
        grasp[2] = 1.10
    elif pose_id == 2:
        grasp[2] = 0.80
    elif pose_id == 3:
        grasp[:2] = 0.2 * pregrasp[:2] + 0.8 * grasp[:2]
        grasp[2] = 1.15
    return Grasp(
        name=f"tabletop_pose_{pose_id:03d}",
        candidate_dir=candidate_dir,
        key_to_hand=np.load(candidate_dir / "wrist_se3.npy"),
        pregrasp=pregrasp,
        grasp=grasp,
    )


def _insertion_grasp(candidate_dir: Path) -> Grasp:
    return Grasp(
        name="common_insertion_grasp",
        candidate_dir=candidate_dir,
        key_to_hand=(
            handle_symmetry("rear_x", 0.045)
            @ np.load(candidate_dir / "wrist_se3.npy")
        ),
        pregrasp=np.load(candidate_dir / "pregrasp_pose.npy").reshape(-1),
        grasp=np.load(candidate_dir / "grasp_pose.npy").reshape(-1),
    )


def _translated_world_z(pose: np.ndarray, distance: float) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float64).copy()
    result[2, 3] += distance
    return result


def _fixed_pose(pose: np.ndarray, count: int) -> np.ndarray:
    return np.repeat(np.asarray(pose)[None, :, :], count, axis=0)


def _hand_interpolation(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return (1.0 - alpha[:, None]) * start + alpha[:, None] * end


def _object_path(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(_interpolate_pose(start, end, count), dtype=np.float64)


def _solve_path_robust(
    kinematics: FrankaHandKinematics,
    targets: list[np.ndarray],
    seed: np.ndarray,
    fallback_seeds: list[np.ndarray],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Solve a waypoint path and retry local-minimum failures at every point."""
    q_previous = np.asarray(seed, dtype=np.float64)
    solved: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        q_arm, diagnostic = kinematics.solve(
            target, q_previous, max_evaluations=1200
        )
        if (
            diagnostic["translation_error_mm"] > 0.5
            or diagnostic["rotation_error_deg"] > 0.5
        ):
            candidates = [(q_arm, diagnostic)]
            for fallback in fallback_seeds:
                candidates.append(kinematics.solve(
                    target, fallback, max_evaluations=1200
                ))
            q_arm, diagnostic = min(
                candidates,
                key=lambda item: (
                    float(item[1]["translation_error_mm"])
                    + float(item[1]["rotation_error_deg"])
                ),
            )
        if (
            diagnostic["translation_error_mm"] > 0.5
            or diagnostic["rotation_error_deg"] > 0.5
        ):
            raise RuntimeError(
                f"IK failed at waypoint {index}: "
                f"{diagnostic['translation_error_mm']:.3f} mm, "
                f"{diagnostic['rotation_error_deg']:.3f} deg"
            )
        q_previous = q_arm
        solved.append(q_arm.copy())
        diagnostics.append(diagnostic)
    return np.asarray(solved), diagnostics


def _attached_segment(
    phase: str,
    start: np.ndarray,
    end: np.ndarray,
    count: int,
    grasp: Grasp,
    sample_mode: str,
) -> Segment:
    objects = _object_path(start, end, count)
    return Segment(
        phase=phase,
        hand_targets=[pose @ grasp.key_to_hand for pose in objects],
        hand_q=np.repeat(grasp.grasp[None, :], count, axis=0),
        object_poses=objects,
        sample_mode=sample_mode,
    )


def _direct_segments(
    start_key: np.ndarray,
    target_preinsert: np.ndarray,
    target_seated: np.ndarray,
    pickup: Grasp,
    insertion: Grasp,
    lift_height: float,
    needs_grasp_transition: bool,
) -> list[Segment]:
    grasp_hand = start_key @ pickup.key_to_hand
    approach_hand = _translated_world_z(grasp_hand, 0.08)
    lifted = _translated_world_z(start_key, lift_height)
    lifted_hand = lifted @ pickup.key_to_hand
    reoriented = lifted_hand @ np.linalg.inv(insertion.key_to_hand)
    target_lifted = _translated_world_z(target_preinsert, lift_height)
    segments = [
        Segment(
            "approach tabletop key",
            _interpolate_pose(approach_hand, grasp_hand, 16),
            np.repeat(pickup.pregrasp[None, :], 16, axis=0),
            _fixed_pose(start_key, 16),
            "pickup_pre",
        ),
        Segment(
            "close on handle sides and rear",
            [grasp_hand] * 10,
            _hand_interpolation(pickup.pregrasp, pickup.grasp, 10),
            _fixed_pose(start_key, 10),
            "pickup_close",
        ),
        _attached_segment(
            "lift key from table", start_key, lifted, 16, pickup, "pickup_grasp"
        ),
    ]
    if needs_grasp_transition:
        segments.append(
            Segment(
                "unvalidated in-hand transition to insertion grasp",
                [lifted_hand] * 24,
                _hand_interpolation(pickup.grasp, insertion.grasp, 24),
                _object_path(lifted, reoriented, 24),
                "in_hand",
            )
        )
    segments.extend(
        [
            _attached_segment(
                "reorient and transfer above socket",
                reoriented,
                target_lifted,
                26,
                insertion,
                "insertion_grasp",
            ),
            _attached_segment(
                "descend to preinsert",
                target_lifted,
                target_preinsert,
                16,
                insertion,
                "insertion_grasp",
            ),
            _attached_segment(
                "insert to CAD seated pose",
                target_preinsert,
                target_seated,
                14,
                insertion,
                "insertion_grasp",
            ),
            Segment(
                "final hold",
                [target_seated @ insertion.key_to_hand] * 10,
                np.repeat(insertion.grasp[None, :], 10, axis=0),
                _fixed_pose(target_seated, 10),
                "insertion_grasp",
            ),
        ]
    )
    return segments


def _declared_contacts(candidate_dir: Path, key_to_hand: np.ndarray) -> list[list[float]]:
    data = np.load(candidate_dir / "bodex_info.npy", allow_pickle=True).item()
    raw = np.asarray(data["contact_point"])
    contacts = np.asarray(raw, dtype=np.float64).reshape(-1, raw.shape[-1])[:, :3]
    source = np.load(candidate_dir / "wrist_se3.npy")
    derived = key_to_hand @ np.linalg.inv(source)
    return _transform_points(derived, contacts).tolist()


def _inspect_grasp(
    grasp: Grasp,
    key_mesh: trimesh.Trimesh,
    policy: dict[str, Any],
    robot_urdf: Path,
    samples_per_link: int,
) -> dict[str, Any]:
    allowed_faces = allowed_contact_faces(
        key_mesh, float(policy["handle_z_range"][1])
    )
    scene = _raycast_scene(key_mesh)
    declared_points = np.asarray(
        _declared_contacts(grasp.candidate_dir, grasp.key_to_hand)
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
        for point in declared_points
    ]
    forbidden_total = 0
    permitted_total = 0
    contact_links: list[str] = []
    minimum = float("inf")
    links = _hand_link_meshes(robot_urdf, grasp.grasp)
    for index, (name, mesh) in enumerate(sorted(links.items())):
        points, _ = trimesh.sample.sample_surface(
            mesh, samples_per_link, seed=503 + index
        )
        object_points = _transform_points(grasp.key_to_hand, points)
        query = o3d.core.Tensor(object_points.astype(np.float32))
        signed = np.asarray(scene.compute_signed_distance(query).numpy())
        face_ids = np.asarray(
            scene.compute_closest_points(query)["primitive_ids"].numpy(),
            dtype=np.int64,
        )
        forbidden = (signed < -2.0e-4) & ~allowed_faces[face_ids]
        permitted = (signed < -2.0e-4) & allowed_faces[face_ids]
        near = (np.abs(signed) <= 1.2e-3) & allowed_faces[face_ids]
        forbidden_total += int(np.count_nonzero(forbidden))
        permitted_total += int(np.count_nonzero(permitted))
        minimum = min(minimum, float(signed.min()))
        if np.any(permitted | near):
            contact_links.append(name)
    contact_links = sorted(set(contact_links))
    opposing_digits = (
        any("thumb" in name for name in contact_links)
        and any("thumb" not in name for name in contact_links)
    )
    # An in-plane transform of a rectangular handle changes the actual visual
    # contact locations; BODex's original sparse contact points therefore do
    # not remain authoritative.  Accept/reject the derived proposal from the
    # complete hand visual mesh, and retain the transformed declarations only
    # as a warning/audit field.
    declared_reusable = all(region is not None for region in regions)
    passed = forbidden_total == 0 and opposing_digits
    return {
        "status": "sampled_pass" if passed else "rejected",
        "source_candidate": str(grasp.candidate_dir),
        "derived_T_key_hand": grasp.key_to_hand.tolist(),
        "declared_contact_points_m": declared_points.tolist(),
        "declared_contact_regions": regions,
        "declared_contacts_reusable_after_transform": declared_reusable,
        "acceptance_basis": "sampled complete hand visual mesh",
        "whole_hand_forbidden_penetrating_samples": forbidden_total,
        "whole_hand_permitted_penetrating_samples": permitted_total,
        "permitted_contact_links": contact_links,
        "opposing_thumb_and_finger_contact": opposing_digits,
        "minimum_signed_distance_mm": minimum * 1000.0,
        "samples_per_link": samples_per_link,
        "derived_grasp_not_bodex_or_mujoco_validated": True,
    }


def _sample_sets(
    robot_urdf: Path,
    joint_names: list[str],
    arm_seed: np.ndarray,
    pickup: Grasp,
    rear: Grasp,
    count: int,
) -> dict[str, np.ndarray]:
    def points(q_hand: np.ndarray) -> np.ndarray:
        result, _ = _hand_samples_in_base(
            robot_urdf, joint_names, arm_seed, q_hand, count
        )
        return result

    pickup_pre = points(pickup.pregrasp)
    pickup_grasp = points(pickup.grasp)
    insertion_pre = points(rear.pregrasp)
    insertion_grasp = points(rear.grasp)
    return {
        "pickup_pre": pickup_pre,
        "pickup_close": np.concatenate([pickup_pre, pickup_grasp]),
        "pickup_grasp": pickup_grasp,
        "insertion_grasp": insertion_grasp,
        "in_hand": np.concatenate([pickup_grasp, insertion_grasp]),
    }


def _build_one(
    pose_id: int,
    args: argparse.Namespace,
    geometry: dict[str, Any],
    robot: URDF,
    kinematics: FrankaHandKinematics,
    key_mesh: trimesh.Trimesh,
    socket_mesh: trimesh.Trimesh,
    policy: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    pickup = _tabletop_grasp(args.tabletop_candidate_root, pose_id)
    needs_grasp_transition = pose_id in {0, 1, 3}
    insertion = (
        _insertion_grasp(args.insertion_candidate)
        if needs_grasp_transition
        else pickup
    )
    stable = np.load(args.tabletop_dir / f"{pose_id:03d}.npy")
    start_key = stable.copy()
    start_key[:3, 3] += np.asarray(
        [args.key_x, args.key_y, args.table_z]
    )
    socket_pose = _socket_pose(
        args.socket_x, args.socket_y, args.table_z, args.socket_yaw_deg
    )
    target_preinsert = socket_pose @ np.asarray(
        geometry["T_socket_key_preinsert"], dtype=np.float64
    )
    target_seated = socket_pose @ np.asarray(
        geometry["T_socket_key_seated"], dtype=np.float64
    )
    segments = _direct_segments(
        start_key,
        target_preinsert,
        target_seated,
        pickup,
        insertion,
        args.lift_height,
        needs_grasp_transition,
    )

    fallback_seeds = [
        np.asarray([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8]),
        np.asarray([0.0, 0.8, 0.3, -1.2, 0.4, 0.8, 1.4]),
        np.asarray([0.1, 1.0, -0.2, -1.2, 0.4, 0.8, 1.2]),
    ]
    q_seed = fallback_seeds[1]
    q_frames: list[np.ndarray] = []
    object_frames: list[np.ndarray] = []
    phases: list[str] = []
    sample_modes: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    for segment in segments:
        arm, segment_diagnostics = _solve_path_robust(
            kinematics, segment.hand_targets, q_seed, fallback_seeds
        )
        q_seed = arm[-1]
        q_frames.append(np.concatenate([arm, segment.hand_q], axis=1))
        object_frames.append(segment.object_poses)
        phases.extend([segment.phase] * len(arm))
        sample_modes.extend([segment.sample_mode] * len(arm))
        diagnostics.extend(segment_diagnostics)
    qpos = np.concatenate(q_frames)
    objects = np.concatenate(object_frames)
    joint_names = [joint.name for joint in robot.actuated_joints]

    grasp_reports = {
        "insertion_grasp": _inspect_grasp(
            insertion,
            key_mesh,
            policy,
            args.robot_urdf,
            args.policy_samples_per_link,
        ),
        "tabletop_pickup_grasp": _inspect_grasp(
            pickup,
            key_mesh,
            policy,
            args.robot_urdf,
            args.policy_samples_per_link,
        ),
    }
    if any(item["status"] != "sampled_pass" for item in grasp_reports.values()):
        raise RuntimeError(
            f"pose {pose_id:03d} has a rejected derived grasp: {grasp_reports}"
        )

    samples = _sample_sets(
        args.robot_urdf,
        joint_names,
        qpos[0, :7],
        pickup,
        insertion,
        args.collision_samples,
    )
    socket_world = socket_mesh.copy()
    socket_world.apply_transform(socket_pose)
    socket_scene = _raycast_scene(socket_world)
    key_points, _ = trimesh.sample.sample_surface(
        key_mesh, args.collision_samples, seed=701
    )
    hand_socket = np.zeros(len(qpos), dtype=np.int32)
    hand_table = np.zeros(len(qpos), dtype=np.int32)
    key_socket = np.zeros(len(qpos), dtype=np.int32)
    key_table = np.zeros(len(qpos), dtype=np.int32)
    for index, (q, object_pose, mode) in enumerate(
        zip(qpos, objects, sample_modes)
    ):
        hand_world = kinematics.fk(q[:7])
        hand_points = _transform_points(hand_world, samples[mode])
        hand_distance = _signed_distance(socket_scene, hand_points)
        object_points = _transform_points(object_pose, key_points)
        object_distance = _signed_distance(socket_scene, object_points)
        hand_socket[index] = int(np.count_nonzero(hand_distance < -2.0e-4))
        hand_table[index] = int(np.count_nonzero(hand_points[:, 2] < args.table_z - 2.0e-4))
        key_socket[index] = int(np.count_nonzero(object_distance < -2.0e-4))
        key_table[index] = int(np.count_nonzero(object_points[:, 2] < args.table_z - 5.0e-4))
    collision_counts = hand_socket + hand_table + key_socket + key_table
    if np.any(collision_counts):
        status = "rejected_by_sampled_environment_check"
    elif needs_grasp_transition:
        status = "concept_preview_requires_in_hand_reorientation_validation"
    else:
        status = "sampled_geometric_preview_passed_not_physical_validation"
    strategy = (
        "pose-specific tabletop grasp, lift, unvalidated in-hand transition "
        "to common insertion grasp, transfer, insert"
        if needs_grasp_transition
        else "pose-specific tabletop grasp, lift, fixed-grasp wrist "
        "reorientation, transfer, insert"
    )
    not_validated = [
        "cuRobo continuous collision planning",
        "robot self-collision and dynamics",
        "BODex/MuJoCo optimization of the transformed grasps",
        "grasp force closure and physical stability",
        "contact-search insertion control",
        "physical execution",
    ]
    if needs_grasp_transition:
        not_validated.insert(
            0, "in-hand transition between tabletop and insertion grasps"
        )
    report = {
        "schema_version": 1,
        "status": status,
        "tabletop_pose_id": f"{pose_id:03d}",
        "starts_with_key_on_table": True,
        "uses_actual_meshes": True,
        "strategy": strategy,
        "requires_unvalidated_grasp_transition": needs_grasp_transition,
        "grasp_reports": grasp_reports,
        "ik": {
            "method": "numerical Cartesian waypoint IK",
            "maximum_translation_error_mm": max(
                float(item["translation_error_mm"]) for item in diagnostics
            ),
            "maximum_rotation_error_deg": max(
                float(item["rotation_error_deg"]) for item in diagnostics
            ),
        },
        "sampled_environment_check": {
            "hand_socket_penetrating_samples_max": int(hand_socket.max()),
            "hand_below_table_samples_max": int(hand_table.max()),
            "key_socket_penetrating_samples_max": int(key_socket.max()),
            "key_below_table_samples_max": int(key_table.max()),
            "samples": args.collision_samples,
            "zero_samples_is_not_a_continuous_collision_proof": True,
        },
        "not_validated": not_validated,
        "sequence_phases": [segment.phase for segment in segments],
    }

    output = args.output_dir / f"tabletop_{pose_id:03d}_to_insertion_preview.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=qpos.astype(np.float32),
        phase=np.asarray(phases),
        object_pose=objects,
        socket_pose=socket_pose,
        desired_preinsert_key_pose=target_preinsert,
        desired_seated_key_pose=target_seated,
        collision_counts=collision_counts,
        hand_socket_collision_counts=hand_socket,
        hand_table_collision_counts=hand_table,
        key_socket_collision_counts=key_socket,
        key_table_collision_counts=key_table,
        joint_names=np.asarray(joint_names),
        object_mesh_path=np.asarray(str(args.key_mesh)),
        socket_mesh_path=np.asarray(str(args.socket_mesh)),
        robot_urdf_path=np.asarray(str(args.robot_urdf)),
        preview_status=np.asarray(status),
        preview_kind=np.asarray("sampled_geometric_success"),
        tabletop_pose_id=np.asarray(f"{pose_id:03d}"),
    )
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pose-id",
        choices=["all", "000", "001", "002", "003", "004"],
        default="all",
    )
    parser.add_argument(
        "--tabletop-candidate-root",
        type=Path,
        default=DEFAULT_TABLETOP_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--insertion-candidate", type=Path, default=DEFAULT_INSERTION_CANDIDATE
    )
    parser.add_argument("--tabletop-dir", type=Path, default=DEFAULT_TABLETOP_DIR)
    parser.add_argument("--task-geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--key-mesh", type=Path, default=DEFAULT_KEY_MESH)
    parser.add_argument("--contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--socket-mesh", type=Path, default=DEFAULT_SOCKET_MESH)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--key-x", type=float, default=0.40)
    parser.add_argument("--key-y", type=float, default=0.18)
    parser.add_argument("--socket-x", type=float, default=0.45)
    parser.add_argument("--socket-y", type=float, default=-0.10)
    parser.add_argument("--table-z", type=float, default=0.04)
    parser.add_argument("--socket-yaw-deg", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.12)
    parser.add_argument("--collision-samples", type=int, default=12000)
    parser.add_argument("--policy-samples-per-link", type=int, default=12000)
    args = parser.parse_args()

    path_fields = [
        "tabletop_candidate_root",
        "insertion_candidate",
        "tabletop_dir",
        "task_geometry",
        "key_mesh",
        "contact_policy",
        "socket_mesh",
        "robot_urdf",
        "output_dir",
    ]
    for field in path_fields:
        setattr(args, field, getattr(args, field).expanduser().resolve())
    required = [
        *(
            args.tabletop_candidate_root / str(scene_id) / candidate_id /
            "wrist_se3.npy"
            for scene_id, candidate_id in {
                0: "346", 1: "403", 2: "511", 3: "27", 4: "290"
            }.items()
        ),
        args.insertion_candidate / "wrist_se3.npy",
        args.task_geometry,
        args.key_mesh,
        args.contact_policy,
        args.socket_mesh,
        args.robot_urdf,
        *(args.tabletop_dir / f"{index:03d}.npy" for index in range(5)),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("missing input: " + ", ".join(missing))

    geometry = json.loads(args.task_geometry.read_text(encoding="utf-8"))
    policy = json.loads(args.contact_policy.read_text(encoding="utf-8"))
    key_mesh = trimesh.load(args.key_mesh, force="mesh", process=False)
    socket_mesh = trimesh.load(args.socket_mesh, force="mesh", process=False)
    robot = URDF.load(
        str(args.robot_urdf), build_scene_graph=False, load_meshes=False
    )
    kinematics = FrankaHandKinematics(robot)
    pose_ids = range(5) if args.pose_id == "all" else [int(args.pose_id)]
    failed = False
    for pose_id in pose_ids:
        output, report = _build_one(
            pose_id,
            args,
            geometry,
            robot,
            kinematics,
            key_mesh,
            socket_mesh,
            policy,
        )
        print(output)
        print(output.with_suffix(".json"))
        print(json.dumps({
            "pose": f"{pose_id:03d}",
            "status": report["status"],
            "strategy": report["strategy"],
        }))
        failed |= report["status"].startswith("rejected")
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
