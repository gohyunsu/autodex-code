#!/usr/bin/env python3
"""Build a sampled geometric success preview for an insertion-safe grasp.

The key begins seated in a staging socket with its rear face exposed.  The
Inspire hand approaches from behind the key, closes only on the handle
sides/rear, extracts the key, lifts it by 10 cm, transfers it to the target
socket, and reaches the CAD seated pose.

This artifact demonstrates that an insertion-compatible grasp and endpoint
IK exist.  It is not a cuRobo plan, a stability result for the symmetry-derived
grasp, or evidence of robot success.  In particular, it does not solve the
preceding tabletop-pick-to-staging step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation, Slerp
from yourdfpy import URDF

from build_insertion_reachability_preview import (
    FrankaHandKinematics,
    _hand_samples_in_base,
    _pose_error,
    _raycast_scene,
    _signed_distance,
    _transform_points,
)
from validate_whole_hand_contact_policy import inspect_whole_hand


SHARED = Path.home() / "shared_data"
DEFAULT_CANDIDATE = (
    SHARED / "AutoDex/bodex_raw/inspire/precision_insertion_v3_proxy/"
    "precision_key_handle_contact_proxy/table/0/84"
)
DEFAULT_GEOMETRY = (
    SHARED / "AutoDex/precision_insertion/fixtures/unified_socket/"
    "task_geometry.json"
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
DEFAULT_OUTPUT = (
    SHARED / "AutoDex/precision_insertion/visualizations/"
    "insertion_safe_rear_grasp_success_preview.npz"
)


def _socket_pose(x: float, y: float, z: float, yaw_deg: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    pose[:3, 3] = [x, y, z]
    return pose


def _interpolate_pose(start: np.ndarray, end: np.ndarray, count: int) -> list[np.ndarray]:
    if count < 2:
        raise ValueError("pose path needs at least two samples")
    alpha = np.linspace(0.0, 1.0, count)
    rotations = Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]]))
    slerp = Slerp([0.0, 1.0], rotations)
    result: list[np.ndarray] = []
    for value, rotation in zip(alpha, slerp(alpha)):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = rotation.as_matrix()
        pose[:3, 3] = (1.0 - value) * start[:3, 3] + value * end[:3, 3]
        result.append(pose)
    return result


def _solve_path(
    kinematics: FrankaHandKinematics,
    targets: list[np.ndarray],
    seed: np.ndarray,
    fallback_seeds: list[np.ndarray],
) -> tuple[np.ndarray, list[dict]]:
    q_previous = np.asarray(seed, dtype=np.float64)
    solved: list[np.ndarray] = []
    diagnostics: list[dict] = []
    for index, target in enumerate(targets):
        attempts = [q_previous, *fallback_seeds] if index == 0 else [q_previous]
        candidates: list[tuple[float, np.ndarray, dict]] = []
        for attempt in attempts:
            q_arm, diagnostic = kinematics.solve(
                target, attempt, max_evaluations=1200
            )
            score = (
                float(diagnostic["translation_error_mm"])
                + float(diagnostic["rotation_error_deg"])
            )
            candidates.append((score, q_arm, diagnostic))
        _, q_previous, diagnostic = min(candidates, key=lambda item: item[0])
        if (
            diagnostic["translation_error_mm"] > 0.5
            or diagnostic["rotation_error_deg"] > 0.5
        ):
            raise RuntimeError(
                f"IK failed at waypoint {index}: "
                f"{diagnostic['translation_error_mm']:.3f} mm, "
                f"{diagnostic['rotation_error_deg']:.3f} deg"
            )
        solved.append(q_previous.copy())
        diagnostics.append(diagnostic)
    return np.asarray(solved), diagnostics


def _transform_mesh(mesh: trimesh.Trimesh, pose: np.ndarray) -> trimesh.Trimesh:
    moved = mesh.copy()
    moved.apply_transform(pose)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--task-geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--key-mesh", type=Path, default=DEFAULT_KEY_MESH)
    parser.add_argument("--contact-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--socket-mesh", type=Path, default=DEFAULT_SOCKET_MESH)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--staging-x", type=float, default=0.42)
    parser.add_argument("--staging-y", type=float, default=0.20)
    parser.add_argument("--target-x", type=float, default=0.45)
    parser.add_argument("--target-y", type=float, default=-0.10)
    parser.add_argument("--socket-z", type=float, default=0.04)
    parser.add_argument("--socket-yaw-deg", type=float, default=0.0)
    parser.add_argument("--lift-height", type=float, default=0.10)
    parser.add_argument("--approach-height", type=float, default=0.08)
    parser.add_argument("--approach-frames", type=int, default=20)
    parser.add_argument("--close-frames", type=int, default=12)
    parser.add_argument("--extract-frames", type=int, default=12)
    parser.add_argument("--lift-frames", type=int, default=18)
    parser.add_argument("--transfer-frames", type=int, default=28)
    parser.add_argument("--target-descent-frames", type=int, default=18)
    parser.add_argument("--insertion-frames", type=int, default=16)
    parser.add_argument("--final-hold-frames", type=int, default=12)
    parser.add_argument("--collision-samples", type=int, default=50000)
    parser.add_argument("--policy-samples-per-link", type=int, default=20000)
    args = parser.parse_args()

    paths = [
        args.candidate_dir / "wrist_se3.npy",
        args.candidate_dir / "pregrasp_pose.npy",
        args.candidate_dir / "grasp_pose.npy",
        args.candidate_dir / "bodex_info.npy",
        args.task_geometry,
        args.key_mesh,
        args.contact_policy,
        args.socket_mesh,
        args.robot_urdf,
    ]
    missing = [str(path.expanduser()) for path in paths if not path.expanduser().is_file()]
    if missing:
        parser.error("missing input: " + ", ".join(missing))

    candidate_dir = args.candidate_dir.expanduser().resolve()
    geometry_path = args.task_geometry.expanduser().resolve()
    key_mesh_path = args.key_mesh.expanduser().resolve()
    policy_path = args.contact_policy.expanduser().resolve()
    socket_mesh_path = args.socket_mesh.expanduser().resolve()
    robot_urdf = args.robot_urdf.expanduser().resolve()
    output = args.output.expanduser().resolve()
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

    policy_report, key_to_hand = inspect_whole_hand(
        candidate_dir=candidate_dir,
        object_mesh_path=key_mesh_path,
        policy_path=policy_path,
        robot_urdf=robot_urdf,
        symmetry_mode="rear_x",
        samples_per_link=args.policy_samples_per_link,
        penetration_threshold_m=2.0e-4,
    )
    if policy_report["status"] != "sampled_pass":
        raise RuntimeError("derived rear grasp failed the whole-hand contact policy")

    pregrasp_hand = np.load(candidate_dir / "pregrasp_pose.npy").reshape(-1)
    grasp_hand = np.load(candidate_dir / "grasp_pose.npy").reshape(-1)
    robot = URDF.load(str(robot_urdf), build_scene_graph=False, load_meshes=False)
    kinematics = FrankaHandKinematics(robot)

    staging_socket = _socket_pose(
        args.staging_x, args.staging_y, args.socket_z, args.socket_yaw_deg
    )
    target_socket = _socket_pose(
        args.target_x, args.target_y, args.socket_z, args.socket_yaw_deg
    )
    socket_to_preinsert = np.asarray(
        geometry["T_socket_key_preinsert"], dtype=np.float64
    )
    socket_to_seated = np.asarray(
        geometry["T_socket_key_seated"], dtype=np.float64
    )
    staging_seated = staging_socket @ socket_to_seated
    staging_preinsert = staging_socket @ socket_to_preinsert
    staging_lifted = staging_preinsert.copy()
    staging_lifted[2, 3] += args.lift_height
    target_preinsert = target_socket @ socket_to_preinsert
    target_lifted = target_preinsert.copy()
    target_lifted[2, 3] += args.lift_height
    target_seated = target_socket @ socket_to_seated

    grasp_hand_world = staging_seated @ key_to_hand
    approach_hand_world = grasp_hand_world.copy()
    approach_hand_world[2, 3] += args.approach_height
    hand_target_paths: list[tuple[str, list[np.ndarray], np.ndarray]] = [
        (
            "sampled-policy approach to staging grasp",
            _interpolate_pose(
                approach_hand_world, grasp_hand_world, args.approach_frames
            ),
            pregrasp_hand,
        ),
    ]
    key_paths = [
        (
            "sampled-policy extraction from staging socket",
            _interpolate_pose(
                staging_seated, staging_preinsert, args.extract_frames
            ),
        ),
        (
            "sampled-policy 10 cm lift",
            _interpolate_pose(
                staging_preinsert, staging_lifted, args.lift_frames
            )[1:],
        ),
        (
            "sampled-policy high transfer",
            _interpolate_pose(
                staging_lifted, target_lifted, args.transfer_frames
            )[1:],
        ),
        (
            "sampled-policy descent to target preinsert",
            _interpolate_pose(
                target_lifted, target_preinsert, args.target_descent_frames
            )[1:],
        ),
        (
            "sampled-policy insertion to CAD seated pose",
            _interpolate_pose(
                target_preinsert, target_seated, args.insertion_frames
            )[1:],
        ),
    ]
    hand_target_paths.extend(
        (phase, [key_pose @ key_to_hand for key_pose in poses], grasp_hand)
        for phase, poses in key_paths
    )

    fallback_seeds = [
        np.asarray([0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8]),
        np.asarray([0.0, 0.8, 0.3, -1.2, 0.4, 0.8, 1.4]),
        np.asarray([0.1, 1.0, -0.2, -1.2, 0.4, 0.8, 1.2]),
    ]
    q_seed = fallback_seeds[1]
    arm_segments: list[np.ndarray] = []
    hand_segments: list[np.ndarray] = []
    phase_segments: list[str] = []
    ik_diagnostics: list[dict] = []
    for phase, targets, hand_pose in hand_target_paths:
        arm, diagnostics = _solve_path(
            kinematics, targets, q_seed, fallback_seeds
        )
        q_seed = arm[-1]
        arm_segments.append(arm)
        hand_segments.append(np.repeat(hand_pose[None, :], len(arm), axis=0))
        phase_segments.extend([phase] * len(arm))
        ik_diagnostics.extend(diagnostics)

    approach_arm = arm_segments[0]
    close_arm = np.repeat(approach_arm[-1][None, :], args.close_frames, axis=0)
    close_hand = np.linspace(pregrasp_hand, grasp_hand, args.close_frames)
    qpos = [
        np.concatenate([approach_arm, hand_segments[0]], axis=1),
        np.concatenate([close_arm, close_hand], axis=1),
    ]
    phases = (
        phase_segments[:len(approach_arm)]
        + ["sampled-policy close on handle sides/rear"] * args.close_frames
    )
    for arm, hand, (phase, _) in zip(
        arm_segments[1:], hand_segments[1:], key_paths
    ):
        qpos.append(np.concatenate([arm, hand], axis=1))
        phases.extend([phase] * len(arm))
    final = np.repeat(qpos[-1][-1][None, :], args.final_hold_frames, axis=0)
    qpos.append(final)
    phases.extend(["sampled-policy final seated hold"] * len(final))
    q_frames = np.concatenate(qpos)
    phase_array = np.asarray(phases)

    attach_index = len(approach_arm) + args.close_frames
    object_poses = np.repeat(staging_seated[None, :, :], len(q_frames), axis=0)
    for index in range(attach_index, len(q_frames)):
        object_poses[index] = (
            kinematics.fk(q_frames[index, :7]) @ np.linalg.inv(key_to_hand)
        )

    joint_names = [joint.name for joint in robot.actuated_joints]
    socket_local = trimesh.load(socket_mesh_path, force="mesh", process=False)
    socket_world = trimesh.util.concatenate([
        _transform_mesh(socket_local, staging_socket),
        _transform_mesh(socket_local, target_socket),
    ])
    socket_scene = _raycast_scene(socket_world)
    grasp_hand_points, _ = _hand_samples_in_base(
        robot_urdf,
        joint_names,
        q_frames[attach_index, :7],
        grasp_hand,
        args.collision_samples,
    )
    pregrasp_hand_points, _ = _hand_samples_in_base(
        robot_urdf,
        joint_names,
        q_frames[0, :7],
        pregrasp_hand,
        args.collision_samples,
    )
    key_local = trimesh.load(key_mesh_path, force="mesh", process=False)
    key_points, _ = trimesh.sample.sample_surface(
        key_local, args.collision_samples, seed=151
    )

    hand_socket_counts = np.zeros(len(q_frames), dtype=np.int32)
    key_socket_counts = np.zeros(len(q_frames), dtype=np.int32)
    table_counts = np.zeros(len(q_frames), dtype=np.int32)
    minimum_clearance = np.zeros(len(q_frames), dtype=np.float64)
    for index, q in enumerate(q_frames):
        hand_world = kinematics.fk(q[:7])
        if index < len(approach_arm):
            local_hand_points = pregrasp_hand_points
        elif index < attach_index:
            # Checking both endpoint geometries is conservative for the short
            # finger-closing interpolation without resampling every link.
            local_hand_points = np.concatenate([
                pregrasp_hand_points, grasp_hand_points,
            ])
        else:
            local_hand_points = grasp_hand_points
        hand_world_points = _transform_points(hand_world, local_hand_points)
        hand_distance = _signed_distance(socket_scene, hand_world_points)
        key_distance = _signed_distance(
            socket_scene, _transform_points(object_poses[index], key_points)
        )
        hand_socket_counts[index] = int(np.count_nonzero(hand_distance < -2.0e-4))
        key_socket_counts[index] = int(np.count_nonzero(key_distance < -2.0e-4))
        table_counts[index] = int(np.count_nonzero(hand_world_points[:, 2] < 0.0398))
        minimum_clearance[index] = float(hand_distance.min())
    collision_counts = hand_socket_counts + key_socket_counts + table_counts

    final_hand_actual = kinematics.fk(q_frames[-1, :7])
    final_hand_target = target_seated @ key_to_hand
    final_translation_error, final_rotation_error = _pose_error(
        final_hand_actual, final_hand_target
    )
    sampled_checks_passed = not np.any(collision_counts)
    status = (
        "geometric_ik_preview_passed_not_curobo_or_physical"
        if sampled_checks_passed else
        "rejected_by_sampled_environment_check"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "scope": "fixture_to_fixture_insertion_safe_grasp_geometric_preview",
        "source_candidate": str(candidate_dir),
        "derived_grasp": {
            "operation": "180 degree rear_x handle-centre symmetry",
            "T_key_hand": key_to_hand.tolist(),
            "whole_hand_policy_report": policy_report,
            "bodex_optimized_after_symmetry": False,
            "mujoco_validated_after_symmetry": False,
        },
        "sequence": [
            "approach and close on a key seated in a staging socket",
            "extract to 30 mm preinsert clearance",
            f"lift an additional {args.lift_height * 1000.0:.1f} mm",
            "transfer above the target socket",
            "descend to target preinsert and reach the CAD seated pose",
        ],
        "fixture_poses": {
            "staging_T_robot_socket": staging_socket.tolist(),
            "target_T_robot_socket": target_socket.tolist(),
            "illustrative_not_session_measured": True,
        },
        "ik": {
            "method": "numerical Cartesian waypoint IK",
            "maximum_translation_error_mm": max(
                float(item["translation_error_mm"]) for item in ik_diagnostics
            ),
            "maximum_rotation_error_deg": max(
                float(item["rotation_error_deg"]) for item in ik_diagnostics
            ),
            "final_translation_error_mm": final_translation_error * 1000.0,
            "final_rotation_error_deg": float(np.degrees(final_rotation_error)),
        },
        "sampled_environment_check": {
            "hand_socket_penetrating_samples_max": int(hand_socket_counts.max()),
            "key_socket_penetrating_samples_max": int(key_socket_counts.max()),
            "hand_below_table_samples_max": int(table_counts.max()),
            "minimum_hand_socket_signed_distance_mm": float(
                minimum_clearance.min() * 1000.0
            ),
            "negative_sample_proves_collision": True,
            "zero_negative_samples_is_not_a_continuous_collision_proof": True,
        },
        "not_solved": [
            "tabletop pickup and transfer into the staging socket",
            "continuous attached-object cuRobo planning",
            "robot self-collision and dynamics",
            "symmetry-derived grasp stability",
            "physical execution and success",
        ],
        "conclusion": (
            "The actual meshes admit a sampled-clear, IK-reachable insertion segment "
            "when the key is first presented rear-face-up in a staging socket."
            if sampled_checks_passed else
            "The proposed sequence must not be presented as a geometric success."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=q_frames.astype(np.float32),
        phase=phase_array,
        object_pose=object_poses,
        socket_pose=target_socket,
        additional_socket_poses=np.asarray([staging_socket]),
        desired_preinsert_key_pose=target_preinsert,
        desired_seated_key_pose=target_seated,
        collision_counts=collision_counts,
        hand_socket_collision_counts=hand_socket_counts,
        key_socket_collision_counts=key_socket_counts,
        table_collision_counts=table_counts,
        min_hand_socket_signed_distance_m=minimum_clearance,
        joint_names=np.asarray(joint_names),
        object_mesh_path=np.asarray(str(key_mesh_path)),
        socket_mesh_path=np.asarray(str(socket_mesh_path)),
        robot_urdf_path=np.asarray(str(robot_urdf)),
        candidate_dir=np.asarray(str(candidate_dir)),
        preview_status=np.asarray(status),
        preview_kind=np.asarray("sampled_geometric_success"),
    )
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report_path)
    print(json.dumps({
        "status": status,
        "frames": len(q_frames),
        "maximum_collision_samples": int(collision_counts.max()),
        "final_translation_error_mm": final_translation_error * 1000.0,
        "final_rotation_error_deg": float(np.degrees(final_rotation_error)),
    }, indent=2))
    return 0 if sampled_checks_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
