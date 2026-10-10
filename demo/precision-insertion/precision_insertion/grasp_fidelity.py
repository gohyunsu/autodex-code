"""Read-only audit of the grasp relation hidden by the stock MuJoCo pass.

This is diagnostic evidence, not a new candidate filter: AutoDex's gravity
test begins *after* squeeze and does not require the key to retain its
pre-squeeze hand-relative pose. A rigid endpoint rendering must not be
interpreted as a physically achieved grasp without this check.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .endpoint import _hand_link_meshes, _load_mesh, nominal_inspire_hold_poses


def pose7_to_se3(pose: Any) -> np.ndarray:
    """Convert MuJoCo ``[x, y, z, qw, qx, qy, qz]`` to an SE(3) matrix."""
    from scipy.spatial.transform import Rotation

    values = np.asarray(pose, dtype=np.float64)
    if (values.shape != (7,) or not np.all(np.isfinite(values)) or
            not np.isclose(np.linalg.norm(values[3:]), 1.0, atol=1e-3)):
        raise ValueError("invalid MuJoCo position/quaternion pose")
    result = np.eye(4)
    result[:3, 3] = values[:3]
    result[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return result


def cylinder_pose_change(
    *, initial_key: np.ndarray, final_key: np.ndarray,
    initial_hand: np.ndarray, final_hand: np.ndarray,
    key_height_m: float,
) -> dict[str, float]:
    """Measure physical cylinder center/axis and key-in-hand displacement.

    Axial yaw and end-for-end flip are unobservable for this symmetric key,
    so neither is counted as a rotation error. The key's physical center is
    used instead of its end-cap origin, avoiding a spurious 80 mm shift on
    an end-for-end flip.
    """
    if not math.isfinite(key_height_m) or key_height_m <= 0:
        raise ValueError("key height must be positive")
    poses = (initial_key, final_key, initial_hand, final_hand)
    if any(np.asarray(T).shape != (4, 4) or
           not np.all(np.isfinite(T)) for T in poses):
        raise ValueError("expected finite SE(3) matrices")
    center = np.array([0.0, 0.0, key_height_m / 2.0])
    before_world = initial_key[:3, :3] @ center + initial_key[:3, 3]
    after_world = final_key[:3, :3] @ center + final_key[:3, 3]
    before_hand = initial_hand[:3, :3].T @ (before_world - initial_hand[:3, 3])
    after_hand = final_hand[:3, :3].T @ (after_world - final_hand[:3, 3])
    axis_dot = float(np.dot(initial_key[:3, 2], final_key[:3, 2]))
    return {
        "center_world_displacement_m": float(np.linalg.norm(after_world - before_world)),
        "center_in_hand_displacement_m": float(np.linalg.norm(after_hand - before_hand)),
        "symmetry_reduced_axis_tilt_deg": math.degrees(
            math.acos(float(np.clip(abs(axis_dot), -1.0, 1.0)))),
    }


def trajectory_closure_audit(trajectory: dict, *, key_height_m: float) -> dict:
    """Compare the recorded first closure with the stock gravity-test replay."""
    phases = trajectory.get("phase")
    objects = trajectory.get("object_pose")
    robots = trajectory.get("robot_qpos")
    if (not isinstance(phases, list) or not phases or
            not isinstance(objects, list) or not isinstance(robots, list) or
            len(phases) != len(objects) or len(phases) != len(robots) or
            phases[0] != "pregrasp"):
        raise ValueError("incomplete MuJoCo trajectory")
    squeeze = [i for i, phase in enumerate(phases) if phase == "squeeze"]
    gravity = [i for i, phase in enumerate(phases) if phase == "force_gravity"]
    if not squeeze or not gravity or squeeze[-1] >= gravity[0]:
        raise ValueError("trajectory lacks squeeze followed by gravity replay")

    def pose_pair(index: int) -> tuple[np.ndarray, np.ndarray]:
        robot = np.asarray(robots[index], dtype=np.float64)
        if robot.ndim != 1 or robot.size < 7:
            raise ValueError("trajectory has no floating-hand qpos")
        return pose7_to_se3(objects[index]), pose7_to_se3(robot[:7])

    initial_key, initial_hand = pose_pair(0)
    result = {}
    for label, index in (("end_squeeze", squeeze[-1]),
                         ("first_gravity_step", gravity[0]),
                         ("end_gravity", gravity[-1])):
        final_key, final_hand = pose_pair(index)
        result[label] = {
            "trajectory_index": index,
            **cylinder_pose_change(
                initial_key=initial_key, final_key=final_key,
                initial_hand=initial_hand, final_hand=final_hand,
                key_height_m=key_height_m),
        }
    return result


def nominal_visual_penetration_audit(
    *, candidate_dir: Path, key_mesh_path: Path, robot_urdf: Path,
    points_per_link: int = 300, depth_threshold_m: float = 0.0002,
) -> dict:
    """Sample actual hand visual surfaces against the *nominal* fixed key.

    This is a deterministic diagnostic, not a certified exact penetration
    bound or a contact-physics test. Samples deeper than the stated tolerance
    indicate that a fixed-pose illustration cannot depict rigid contact.
    """
    if points_per_link < 10 or not 0 < depth_threshold_m < 0.01:
        raise ValueError("invalid sampling resolution or penetration tolerance")
    directory = Path(candidate_dir)
    _verify_cylinder_mesh(Path(key_mesh_path))
    T_key_hand = np.load(directory / "wrist_se3.npy", allow_pickle=False)
    if T_key_hand.shape != (4, 4) or not np.all(np.isfinite(T_key_hand)):
        raise ValueError("invalid candidate T_key_hand")
    poses = nominal_inspire_hold_poses(
        np.load(directory / "pregrasp_pose.npy", allow_pickle=False),
        np.load(directory / "grasp_pose.npy", allow_pickle=False))
    return {
        hold_name: _visual_penetration_at_pose(
            robot_urdf=robot_urdf, hand_q=hand_q,
            T_key_hand=T_key_hand, points_per_link=points_per_link,
            depth_threshold_m=depth_threshold_m)
        for hold_name, hand_q in poses.items()
    }


def _verify_cylinder_mesh(key_mesh_path: Path) -> None:
    """Refuse a different key; the fast interior test assumes this CAD size."""
    mesh = _load_mesh(key_mesh_path)
    expected = np.array([[-0.015, -0.015, 0.0],
                         [0.015, 0.015, 0.08]])
    if not mesh.is_watertight or not np.allclose(mesh.bounds, expected, atol=1e-6):
        raise ValueError("expected watertight r15-h80 cylinder mesh")


def _cylinder_inside_depth(points: np.ndarray) -> np.ndarray:
    """Positive interior depth for the known r15-h80 key, zero outside.

    The exact CAD uses 256 planar sides. This analytic circle encloses each
    facet by at most 1.13 micrometers, far below the 0.2 mm diagnostic
    threshold; it avoids intermittent rtree failures on large batches.
    """
    points = np.asarray(points, dtype=np.float64)
    radial = np.linalg.norm(points[:, :2], axis=1)
    return np.maximum(0.0, np.minimum.reduce((
        np.full(len(points), 0.015) - radial,
        points[:, 2], 0.08 - points[:, 2],
    )))


def _visual_penetration_at_pose(
    *, robot_urdf: Path, hand_q: np.ndarray,
    T_key_hand: np.ndarray, points_per_link: int, depth_threshold_m: float,
) -> dict:
    import trimesh

    links = {}
    for name, mesh in _hand_link_meshes(Path(robot_urdf), hand_q).items():
        # Stable seed per link: the report is reproducible across runs.
        seed = int.from_bytes(name.encode("utf-8"), "little") % (2**32)
        points, _ = trimesh.sample.sample_surface(
            mesh, points_per_link, seed=seed)
        in_key = trimesh.transform_points(points, T_key_hand)
        depth = _cylinder_inside_depth(in_key)
        penetrating = depth > depth_threshold_m
        if np.any(penetrating):
            links[name] = {
                "sample_points_over_threshold": int(np.sum(penetrating)),
                "maximum_sampled_depth_m": float(np.max(depth[penetrating])),
            }
    return {
        "sampled_points_per_link": points_per_link,
        "depth_threshold_m": depth_threshold_m,
        "penetrating_links": links,
        "sample_points_over_threshold": sum(
            item["sample_points_over_threshold"] for item in links.values()),
    }


def simulated_visual_penetration_audit(
    *, trajectory: dict, key_mesh_path: Path, robot_urdf: Path,
    points_per_link: int = 300, depth_threshold_m: float = 0.0002,
) -> dict:
    """Audit *achieved* MuJoCo joints and object pose, not squeeze commands.

    The Inspire MuJoCo qpos has a floating wrist followed by 12 hand joints;
    six independent joints occupy indices 0, 1, 4, 6, 8, 10 in that suffix.
    The remaining joints are mimic followers. This distinction is central:
    contact can keep actual joints from reaching the commanded squeeze pose.
    """
    if points_per_link < 10 or not 0 < depth_threshold_m < 0.01:
        raise ValueError("invalid sampling resolution or penetration tolerance")
    phases = trajectory.get("phase")
    robots = trajectory.get("robot_qpos")
    objects = trajectory.get("object_pose")
    if (not isinstance(phases, list) or not isinstance(robots, list) or
            not isinstance(objects, list) or
            not len(phases) == len(robots) == len(objects)):
        raise ValueError("incomplete MuJoCo trajectory")
    squeeze = [i for i, phase in enumerate(phases) if phase == "squeeze"]
    gravity = [i for i, phase in enumerate(phases) if phase == "force_gravity"]
    if not squeeze or not gravity or squeeze[-1] >= gravity[0]:
        raise ValueError("trajectory lacks squeeze followed by gravity")
    _verify_cylinder_mesh(Path(key_mesh_path))
    result = {}
    for label, index in (("end_squeeze", squeeze[-1]),
                         ("end_gravity", gravity[-1])):
        robot = np.asarray(robots[index], dtype=np.float64)
        if robot.shape != (19,) or not np.all(np.isfinite(robot)):
            raise ValueError("expected 7 floating-wrist and 12 Inspire joint qpos")
        hand_q = robot[7:][[0, 1, 4, 6, 8, 10]]
        T_key_hand = (np.linalg.inv(pose7_to_se3(objects[index])) @
                      pose7_to_se3(robot[:7]))
        result[label] = {
            "trajectory_index": index,
            "achieved_hand_q": hand_q.tolist(),
            **_visual_penetration_at_pose(
                robot_urdf=robot_urdf, hand_q=hand_q,
                T_key_hand=T_key_hand, points_per_link=points_per_link,
                depth_threshold_m=depth_threshold_m),
        }
    return result
