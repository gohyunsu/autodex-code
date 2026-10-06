#!/usr/bin/env python3
"""Build an honest lift-to-insertion reachability preview.

The saved AutoDex approach/close/lift trajectory is reused exactly.  After the
validated lift, this script solves endpoint IK for the CAD pre-insertion and
seated poses, keeps the selected key-to-hand transform rigid, and diagnoses
Inspire-hand penetration into the exact socket mesh.

The post-lift portion is deliberately *not* presented as a cuRobo plan.  It is
a numerical-IK/joint-interpolation diagnostic used to decide whether a grasp
deserves a future attached-object transfer plan.  A negative signed-distance
sample proves collision; the absence of one is not a collision-free proof.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


SHARED = Path.home() / "shared_data"
DEFAULT_TRAJECTORY = (
    SHARED / "AutoDex/precision_insertion/visualizations/"
    "common_grasp_78_planned_trajectory.npz"
)
DEFAULT_GEOMETRY = (
    SHARED / "AutoDex/precision_insertion/fixtures/unified_socket/"
    "task_geometry.json"
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
    "common_grasp_78_lift_to_insertion_preview.npz"
)


def _rows(array: np.ndarray, count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("phase sample count must be at least two")
    indices = np.linspace(0, len(array) - 1, count).round().astype(int)
    return np.asarray(array[indices], dtype=np.float64)


def _smooth_joint_path(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float64)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return (1.0 - alpha[:, None]) * start + alpha[:, None] * end


def _pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(np.linalg.norm(Rotation.from_matrix(
        target[:3, :3].T @ actual[:3, :3]
    ).as_rotvec()))
    return translation, rotation


@dataclass
class FrankaHandKinematics:
    """Fast seven-joint FK matching the combined FR3+Inspire URDF."""

    urdf: URDF

    def __post_init__(self) -> None:
        joints = {joint.name: joint for joint in self.urdf.robot.joints}
        self._joints = joints
        self._chain = [
            ("fr3_base_joint", None),
            *((f"fr3_joint{index}", index - 1) for index in range(1, 8)),
            ("fr3_joint8", None),
            ("flange_to_hand", None),
        ]
        self.lower = np.asarray([
            joints[f"fr3_joint{index}"].limit.lower for index in range(1, 8)
        ], dtype=np.float64)
        self.upper = np.asarray([
            joints[f"fr3_joint{index}"].limit.upper for index in range(1, 8)
        ], dtype=np.float64)

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        q_arm = np.asarray(q_arm, dtype=np.float64).reshape(7)
        transform = np.eye(4, dtype=np.float64)
        for joint_name, q_index in self._chain:
            joint = self._joints[joint_name]
            transform = transform @ np.asarray(joint.origin, dtype=np.float64)
            if q_index is not None:
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = Rotation.from_rotvec(
                    np.asarray(joint.axis, dtype=np.float64) * q_arm[q_index]
                ).as_matrix()
                transform = transform @ motion
        return transform

    def solve(
        self,
        target: np.ndarray,
        seed: np.ndarray,
        *,
        max_evaluations: int = 1000,
    ) -> tuple[np.ndarray, dict]:
        target = np.asarray(target, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64).reshape(7)

        def residual(q_arm: np.ndarray) -> np.ndarray:
            actual = self.fk(q_arm)
            translation = (actual[:3, 3] - target[:3, 3]) * 30.0
            rotation = Rotation.from_matrix(
                target[:3, :3].T @ actual[:3, :3]
            ).as_rotvec() * 3.0
            return np.concatenate([translation, rotation])

        result = least_squares(
            residual,
            np.clip(seed, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=max_evaluations,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        actual = self.fk(result.x)
        translation, rotation = _pose_error(actual, target)
        return np.asarray(result.x, dtype=np.float64), {
            "translation_error_mm": translation * 1000.0,
            "rotation_error_deg": float(np.degrees(rotation)),
            "active_joint_limits": [int(value) for value in result.active_mask],
            "function_evaluations": int(result.nfev),
        }


def _raycast_scene(mesh: trimesh.Trimesh) -> o3d.t.geometry.RaycastingScene:
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    return scene


def _hand_samples_in_base(
    robot_urdf: Path,
    joint_names: list[str],
    q_arm: np.ndarray,
    q_hand: np.ndarray,
    count: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    urdf = URDF.load(str(robot_urdf), build_scene_graph=True, load_meshes=True)
    urdf.update_cfg(dict(zip(joint_names, np.concatenate([q_arm, q_hand]))))
    base_world = np.asarray(urdf.get_transform("base_link", urdf.base_link))
    base_world_inverse = np.linalg.inv(base_world)
    hand_meshes: list[trimesh.Trimesh] = []
    link_meshes: dict[str, trimesh.Trimesh] = {}
    for name, mesh in urdf.scene.geometry.items():
        if not (name.startswith("base_link") or name.startswith("right_")):
            continue
        geometry_world, _ = urdf.scene.graph.get(name)
        moved = mesh.copy()
        moved.apply_transform(base_world_inverse @ geometry_world)
        hand_meshes.append(moved)
        link_meshes[name] = moved
    if not hand_meshes:
        raise RuntimeError("Inspire visual meshes were not found in the robot URDF")
    combined = trimesh.util.concatenate(hand_meshes)
    points, _ = trimesh.sample.sample_surface(combined, count, seed=17)
    per_link: dict[str, np.ndarray] = {}
    per_link_count = max(1000, count // max(len(link_meshes), 1))
    for name, mesh in link_meshes.items():
        per_link[name], _ = trimesh.sample.sample_surface(
            mesh, per_link_count, seed=19)
    return np.asarray(points), per_link


def _transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return ((transform[:3, :3] @ points.T).T + transform[:3, 3])


def _signed_distance(
    scene: o3d.t.geometry.RaycastingScene,
    points: np.ndarray,
) -> np.ndarray:
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return np.asarray(scene.compute_signed_distance(query).numpy())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lift-trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--task-geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--socket-mesh", type=Path, default=DEFAULT_SOCKET_MESH)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--socket-x", type=float, default=0.55)
    parser.add_argument("--socket-y", type=float, default=-0.25)
    parser.add_argument("--socket-z", type=float, default=0.04)
    parser.add_argument("--socket-yaw-deg", type=float, default=0.0)
    parser.add_argument("--approach-frames", type=int, default=45)
    parser.add_argument("--close-frames", type=int, default=18)
    parser.add_argument("--lift-frames", type=int, default=35)
    parser.add_argument("--transfer-frames", type=int, default=50)
    parser.add_argument("--insertion-frames", type=int, default=24)
    parser.add_argument("--final-hold-frames", type=int, default=18)
    parser.add_argument("--collision-samples", type=int, default=40000)
    args = parser.parse_args()

    paths = [
        args.lift_trajectory, args.task_geometry, args.socket_mesh, args.robot_urdf
    ]
    missing = [str(path.expanduser()) for path in paths if not path.expanduser().is_file()]
    if missing:
        parser.error("missing input: " + ", ".join(missing))

    trajectory_path = args.lift_trajectory.expanduser().resolve()
    geometry_path = args.task_geometry.expanduser().resolve()
    socket_mesh_path = args.socket_mesh.expanduser().resolve()
    robot_urdf = args.robot_urdf.expanduser().resolve()
    output = args.output.expanduser().resolve()
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

    with np.load(trajectory_path, allow_pickle=False) as source:
        approach = _rows(source["approach_q"], args.approach_frames)
        close = _rows(source["close_q"], args.close_frames)
        lift = _rows(source["lift_q"], args.lift_frames)
        object_world = np.asarray(source["object_world_se3"], dtype=np.float64)
        joint_names = [str(value) for value in source["joint_names"].tolist()]
        object_mesh_path = Path(str(source["object_mesh_path"].item())).resolve()
        candidate_dir = Path(str(source["candidate_dir"].item())).resolve()
        object_name = str(source["object_name"].item())
        candidate_id = str(source["candidate_id"].item())

    candidate_wrist = np.load(candidate_dir / "wrist_se3.npy")
    grasp_hand = np.load(candidate_dir / "grasp_pose.npy").reshape(-1)
    if not np.allclose(lift[-1, 7:], grasp_hand, atol=1.0e-4):
        raise RuntimeError("exported lift does not hold the selected grasp pose")

    kinematic_urdf = URDF.load(
        str(robot_urdf), build_scene_graph=False, load_meshes=False
    )
    kinematics = FrankaHandKinematics(kinematic_urdf)

    yaw = np.radians(args.socket_yaw_deg)
    socket_pose = np.eye(4, dtype=np.float64)
    socket_pose[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    socket_pose[:3, 3] = [args.socket_x, args.socket_y, args.socket_z]
    preinsert_key = socket_pose @ np.asarray(
        geometry["T_socket_key_preinsert"], dtype=np.float64
    )
    seated_key = socket_pose @ np.asarray(
        geometry["T_socket_key_seated"], dtype=np.float64
    )
    preinsert_hand = preinsert_key @ candidate_wrist
    seated_hand = seated_key @ candidate_wrist

    q_lift_end = np.asarray(lift[-1, :7], dtype=np.float64)
    q_preinsert, preinsert_ik = kinematics.solve(preinsert_hand, q_lift_end)
    transfer_arm = _smooth_joint_path(
        q_lift_end, q_preinsert, args.transfer_frames
    )[1:]

    insertion_arm: list[np.ndarray] = []
    insertion_ik: list[dict] = []
    q_seed = q_preinsert.copy()
    for alpha in np.linspace(0.0, 1.0, args.insertion_frames)[1:]:
        key_target = preinsert_key.copy()
        key_target[:3, 3] = (
            (1.0 - alpha) * preinsert_key[:3, 3]
            + alpha * seated_key[:3, 3]
        )
        q_seed, diagnostic = kinematics.solve(
            key_target @ candidate_wrist, q_seed, max_evaluations=300
        )
        insertion_arm.append(q_seed.copy())
        diagnostic["stroke_fraction"] = float(alpha)
        insertion_ik.append(diagnostic)
    insertion_arm_array = np.asarray(insertion_arm, dtype=np.float64)

    transfer = np.concatenate([
        transfer_arm,
        np.tile(grasp_hand, (len(transfer_arm), 1)),
    ], axis=1)
    insertion = np.concatenate([
        insertion_arm_array,
        np.tile(grasp_hand, (len(insertion_arm_array), 1)),
    ], axis=1)
    final = np.repeat(
        insertion[-1][None, :], args.final_hold_frames, axis=0
    )
    q_frames = np.concatenate([approach, close, lift, transfer, insertion, final])
    phases = np.asarray(
        ["validated approach"] * len(approach)
        + ["validated close"] * len(close)
        + ["validated 10 cm lift"] * len(lift)
        + ["diagnostic transfer to preinsert"] * len(transfer)
        + ["diagnostic insertion IK"] * len(insertion)
        + ["diagnostic final hold"] * len(final)
    )

    attach_index = len(approach) + len(close)
    object_poses = np.repeat(object_world[None, :, :], len(q_frames), axis=0)
    for index in range(attach_index, len(q_frames)):
        object_poses[index] = (
            kinematics.fk(q_frames[index, :7]) @ np.linalg.inv(candidate_wrist)
        )

    socket_mesh = trimesh.load(socket_mesh_path, force="mesh", process=False)
    if not isinstance(socket_mesh, trimesh.Trimesh):
        raise TypeError("socket collision asset is not a triangle mesh")
    socket_world = socket_mesh.copy()
    socket_world.apply_transform(socket_pose)
    socket_scene = _raycast_scene(socket_world)
    hand_points, hand_links = _hand_samples_in_base(
        robot_urdf, joint_names, q_lift_end, grasp_hand,
        args.collision_samples,
    )

    collision_counts = np.zeros(len(q_frames), dtype=np.int32)
    min_signed_distance = np.zeros(len(q_frames), dtype=np.float64)
    for index, qpos in enumerate(q_frames):
        hand_world = kinematics.fk(qpos[:7])
        distances = _signed_distance(
            socket_scene, _transform_points(hand_world, hand_points)
        )
        collision_counts[index] = int(np.count_nonzero(distances < -2.0e-4))
        min_signed_distance[index] = float(np.min(distances))

    colliding_links: dict[str, dict] = {}
    preinsert_base = kinematics.fk(q_preinsert)
    for name, points in hand_links.items():
        distances = _signed_distance(
            socket_scene, _transform_points(preinsert_base, points)
        )
        count = int(np.count_nonzero(distances < -2.0e-4))
        if count:
            colliding_links[name] = {
                "penetrating_samples": count,
                "minimum_signed_distance_mm": float(np.min(distances) * 1000.0),
            }

    final_actual_hand = kinematics.fk(insertion[-1, :7])
    seated_translation_error, seated_rotation_error = _pose_error(
        final_actual_hand, seated_hand
    )
    preinsert_collision = bool(colliding_links)
    seated_reached = (
        seated_translation_error <= 5.0e-4
        and seated_rotation_error <= np.radians(0.5)
    )
    status = "rejected_for_insertion" if (
        preinsert_collision or not seated_reached
    ) else "diagnostic_only_requires_collision_planner"

    report = {
        "schema_version": 1,
        "status": status,
        "scope": "actual_mesh_lift_to_insertion_reachability_diagnostic",
        "object": object_name,
        "candidate": ["table", "0", candidate_id],
        "fixture_pose": {
            "source": "illustrative_only_not_session_measurement",
            "T_robot_socket": socket_pose.tolist(),
        },
        "validated_input": {
            "trajectory": str(trajectory_path),
            "phases": ["approach", "close", "10 cm lift"],
        },
        "diagnostic_extension": {
            "method": "endpoint numerical IK plus joint interpolation",
            "collision_planned": False,
            "executable_on_robot": False,
            "preinsert_ik": preinsert_ik,
            "seated_ik": {
                "translation_error_mm": seated_translation_error * 1000.0,
                "rotation_error_deg": float(np.degrees(seated_rotation_error)),
                "last_step": insertion_ik[-1],
            },
        },
        "hand_socket_clearance": {
            "sample_method": "deterministic Inspire visual-surface signed distance",
            "negative_distance_proves_collision": True,
            "no_negative_sample_does_not_prove_clearance": True,
            "threshold_mm": -0.2,
            "preinsert_collision": preinsert_collision,
            "colliding_links": colliding_links,
            "minimum_over_animation_mm": float(
                np.min(min_signed_distance) * 1000.0
            ),
        },
        "conclusion": (
            "Candidate is not insertion-compatible at the CAD goal; keep its "
            "lift evidence separate and reject it from insertion execution."
            if status == "rejected_for_insertion" else
            "Candidate remains diagnostic-only until an attached-object cuRobo "
            "transfer plan and guarded physical insertion are validated."
        ),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=q_frames.astype(np.float32),
        phase=phases,
        object_pose=object_poses,
        socket_pose=socket_pose,
        desired_preinsert_key_pose=preinsert_key,
        desired_seated_key_pose=seated_key,
        collision_counts=collision_counts,
        min_hand_socket_signed_distance_m=min_signed_distance,
        joint_names=np.asarray(joint_names),
        object_mesh_path=np.asarray(str(object_mesh_path)),
        socket_mesh_path=np.asarray(str(socket_mesh_path)),
        robot_urdf_path=np.asarray(str(robot_urdf)),
        candidate_dir=np.asarray(str(candidate_dir)),
        preview_status=np.asarray(status),
    )
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(report_path)
    print(json.dumps({
        "status": status,
        "preinsert_collision": preinsert_collision,
        "colliding_links": sorted(colliding_links),
        "seated_translation_error_mm": seated_translation_error * 1000.0,
        "seated_rotation_error_deg": float(np.degrees(seated_rotation_error)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
