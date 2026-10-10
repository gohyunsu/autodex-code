"""Exact-mesh, grasp-only 20 mm insertion endpoint screen.

This offline check excludes the Franka arm, trajectories, contact dynamics,
release, and physical success. It tests the *fixed* v8 candidate grasp against
the socket at the nominal centered 20 mm CAD pose. No robot or camera API is
imported. The result is an endpoint screen, not motion authorization.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path):
    import trimesh

    mesh = trimesh.load(str(path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"invalid triangle mesh: {path}")
    if not (np.all(np.isfinite(mesh.vertices)) and
            np.all(np.isfinite(mesh.faces))):
        raise ValueError(f"non-finite triangle mesh: {path}")
    return mesh


def _hand_link_meshes(robot_urdf: Path, grasp_q: np.ndarray) -> dict:
    """Evaluate Inspire visual links in the URDF's `base_link` hand frame.

    This narrow adapter follows the frame convention of the existing
    `scripts/precision_insertion/validate_whole_hand_contact_policy.py`
    helper, without importing that script's unrelated contact-policy and
    Open3D dependencies. `wrist_se3.npy` must therefore be T_key_hand.
    """
    from yourdfpy import URDF

    hand_q = np.asarray(grasp_q, dtype=np.float64).reshape(-1)
    if hand_q.shape != (6,) or not np.all(np.isfinite(hand_q)):
        raise ValueError("Inspire grasp_pose.npy must contain six finite joints")
    robot = URDF.load(str(robot_urdf), build_scene_graph=True, load_meshes=True)
    joints = [joint.name for joint in robot.actuated_joints]
    if len(joints) != 13 or not all(name.startswith("fr3_") for name in joints[:7]):
        raise ValueError("unexpected Franka/Inspire URDF joint contract")
    robot.update_cfg(dict(zip(joints, np.concatenate([np.zeros(7), hand_q]))))
    hand_base = np.asarray(robot.get_transform("base_link", robot.base_link),
                           dtype=np.float64)
    hand_base_inverse = np.linalg.inv(hand_base)
    result = {}
    for name, mesh in robot.scene.geometry.items():
        if not (name.startswith("base_link") or name.startswith("right_")):
            continue
        geometry_world, _ = robot.scene.graph.get(name)
        local = mesh.copy()
        local.apply_transform(hand_base_inverse @ geometry_world)
        result[name] = local
    if not result or not any(name.startswith("right_") for name in result):
        raise ValueError("Inspire hand visual links are missing from the URDF")
    return result


def nominal_inspire_hold_poses(pregrasp_q: np.ndarray,
                               grasp_q: np.ndarray) -> dict[str, np.ndarray]:
    """Return simulated squeeze and AutoDex's default executed hold pose.

    ``run_sim_filter.eval_single_grasp`` uses ``2*grasp-pregrasp``. The
    unchanged ``RealExecutor.execute`` defaults to squeeze_level=2 and its
    final loop index is 9/5, so its Inspire controller command corresponds
    to ``grasp + 1.8*(grasp-pregrasp)`` after joint-limit clipping. This is a
    *nominal* controller pose, not measured hand feedback.
    """
    pre = np.asarray(pregrasp_q, dtype=np.float64).reshape(-1)
    grasp = np.asarray(grasp_q, dtype=np.float64).reshape(-1)
    if (pre.shape != (6,) or grasp.shape != (6,) or
            not np.all(np.isfinite(pre)) or not np.all(np.isfinite(grasp))):
        raise ValueError("pregrasp and grasp must be six finite Inspire joints")
    limits = np.asarray([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
    pre_action_equivalent = np.clip(pre, 0.0, limits)
    grasp_action_equivalent = np.clip(grasp, 0.0, limits)
    return {
        "mujoco_squeeze": np.clip(2.0 * grasp - pre, 0.0, limits),
        "autodex_default_controller_hold": np.clip(
            grasp_action_equivalent + 1.8 *
            (grasp_action_equivalent - pre_action_equivalent), 0.0, limits),
    }


# Preserve the original private helper name for existing offline tools/tests.
_nominal_inspire_hold_poses = nominal_inspire_hold_poses


def _coal_mesh(mesh):
    import coal

    vertices = coal.StdVec_Vec3s()
    triangles = coal.StdVec_Triangle()
    for point in mesh.vertices:
        vertices.append(np.asarray(point, dtype=np.float64))
    for face in mesh.faces:
        triangles.append(coal.Triangle(*(int(index) for index in face)))
    model = coal.BVHModelOBBRSS()
    if (model.beginModel(len(vertices), len(triangles)) != 0 or
            model.addSubModel(vertices, triangles) != 0 or
            model.endModel() != 0):
        raise RuntimeError("Coal failed to construct a triangle-mesh BVH")
    return model


def _coal_models_report(moving, fixed_model, T_fixed_moving: np.ndarray) -> dict:
    import coal

    transform = coal.Transform3s(T_fixed_moving[:3, :3],
                                 T_fixed_moving[:3, 3])
    collision = coal.CollisionResult()
    colliding = bool(coal.collide(
        moving, transform, fixed_model, coal.Transform3s(),
        coal.CollisionRequest(), collision))
    distance = coal.distance(
        moving, transform, fixed_model, coal.Transform3s(),
        coal.DistanceRequest(), coal.DistanceResult())
    return {"colliding": colliding, "minimum_surface_distance_m": float(distance)}


def _mesh_pair_report(moving_mesh, fixed_model, T_fixed_moving: np.ndarray) -> dict:
    return _coal_models_report(_coal_mesh(moving_mesh), fixed_model,
                               T_fixed_moving)


def validate_task_geometry(geometry: dict[str, Any], mode: TaskMode) -> np.ndarray:
    if geometry.get("units") != "m":
        raise ValueError("task geometry must use meters")
    if geometry.get("socket_pose_object") != mode.socket_object:
        raise ValueError("task geometry socket object does not match mode")
    if "key_object" in geometry and geometry["key_object"] != mode.key_object:
        raise ValueError("task geometry key object does not match mode")
    socket_pose_object = validate_se3(
        geometry["T_socket_pose_object"], name="T_socket_pose_object")
    if not np.allclose(socket_pose_object, np.eye(4), atol=1e-8):
        raise ValueError("nonidentity socket mesh/pose frame needs an explicit adapter")
    entry = validate_se3(geometry["T_socket_key_entry"], name="T_socket_key_entry")
    target = validate_se3(geometry["T_socket_key_verification"],
                          name="T_socket_key_verification")
    depth = float(geometry["verification_insertion_depth_m"])
    direction = np.asarray(geometry["insertion_direction_socket"], dtype=float)
    key_axis = np.asarray(geometry["key_frame"]["insertion_axis"], dtype=float)
    if (not math.isclose(depth, mode.target_depth_m, abs_tol=1e-8) or
            direction.shape != (3,) or key_axis.shape != (3,) or
            not np.all(np.isfinite(direction)) or
            not np.all(np.isfinite(key_axis)) or
            not np.isclose(np.linalg.norm(direction), 1.0, atol=1e-6) or
            not np.isclose(np.linalg.norm(key_axis), 1.0, atol=1e-6)):
        raise ValueError("invalid 20 mm insertion depth or axis contract")
    if (not np.allclose(target[:3, :3], entry[:3, :3], atol=1e-7) or
            not np.allclose(target[:3, 3] - entry[:3, 3],
                            depth * direction, atol=1e-7) or
            not np.allclose(target[:2, 3], [0.0, 0.0], atol=1e-7) or
            not np.allclose(target[:3, :3] @ key_axis, direction, atol=1e-7)):
        raise ValueError("verification pose is not centered and axially aligned")
    return target


# Retain the private name for existing offline screen/tests while the public
# validator is reused by the live-target builder in this demo.
_validate_geometry = validate_task_geometry


def screen_grasp_endpoint(
    *, shared_root: Path, mode: TaskMode, candidate_dir: Path,
    minimum_hand_clearance_m: float,
    xy_offset_socket_m: tuple[float, float] = (0.0, 0.0),
) -> dict[str, Any]:
    """Screen CAD key/socket fit and Inspire links at one aligned 20 mm pose.

    `candidate_dir` may be a staged BODex candidate or v8 candidate. This
    function does not verify simulated grasp stability; callers must combine
    this result with independently checked v8/MuJoCo evidence. The offset is
    an absolute target in the *socket* XY frame, not a robot or camera delta.
    """
    if not math.isfinite(minimum_hand_clearance_m) or minimum_hand_clearance_m <= 0:
        raise ValueError("minimum_hand_clearance_m must be positive and calibrated")
    offset = np.asarray(xy_offset_socket_m, dtype=np.float64)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("xy_offset_socket_m must be two finite metric values")
    paths = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    candidate = Path(candidate_dir).expanduser().resolve()
    files = {
        "key_mesh": paths.raw_mesh(mode.key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
        "wrist_se3": candidate / "wrist_se3.npy",
        "pregrasp_pose": candidate / "pregrasp_pose.npy",
        "grasp_pose": candidate / "grasp_pose.npy",
    }
    missing = [f"{name}: {path}" for name, path in files.items()
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing endpoint input: " + ", ".join(missing))
    geometry = json.loads(files["task_geometry"].read_text(encoding="utf-8"))
    T_socket_key_nominal = validate_task_geometry(geometry, mode)
    T_socket_key = T_socket_key_nominal.copy()
    T_socket_key[:2, 3] += offset
    # On this host, a fresh process must load Coal before trimesh/yourdfpy
    # to avoid binding the older system libstdc++. The CLI starts fresh.
    import coal  # noqa: F401

    T_key_hand = validate_se3(np.load(files["wrist_se3"], allow_pickle=False),
                              name="T_key_hand")
    hand_poses = nominal_inspire_hold_poses(
        np.load(files["pregrasp_pose"], allow_pickle=False),
        np.load(files["grasp_pose"], allow_pickle=False))
    key_mesh = _load_mesh(files["key_mesh"])
    socket_mesh = _load_mesh(files["socket_mesh"])
    if not key_mesh.is_watertight or not socket_mesh.is_watertight:
        raise ValueError("key and socket CAD meshes must be watertight")
    fixed_socket = _coal_mesh(socket_mesh)
    key_fit = _mesh_pair_report(key_mesh, fixed_socket, T_socket_key)
    T_socket_hand = T_socket_key @ T_key_hand
    hold_screens = {}
    for hold_name, hand_q in hand_poses.items():
        hand = _hand_link_meshes(files["robot_urdf"], hand_q)
        links = {
            name: _mesh_pair_report(mesh, fixed_socket, T_socket_hand)
            for name, mesh in sorted(hand.items())
        }
        minimum_hold = min(row["minimum_surface_distance_m"]
                           for row in links.values())
        hold_screens[hold_name] = {
            "hand_q": hand_q.tolist(),
            "hand_links": links,
            "minimum_observed_hand_clearance_m": minimum_hold,
            "clear": all(not row["colliding"] and
                         row["minimum_surface_distance_m"] >=
                         minimum_hand_clearance_m for row in links.values()),
        }
    minimum = min(row["minimum_observed_hand_clearance_m"]
                  for row in hold_screens.values())
    hand_pass = all(row["clear"] for row in hold_screens.values())
    return {
        "schema": "precision_insertion_endpoint_screen_v1",
        "scope": "aligned_20mm_key_fit_and_whole_inspire_hand_endpoint",
        "candidate_dir": str(candidate),
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "verification_depth_m": mode.target_depth_m,
        "xy_offset_socket_m": offset.tolist(),
        "minimum_required_hand_clearance_m": minimum_hand_clearance_m,
        "minimum_observed_hand_clearance_m": minimum,
        "key_socket_fit": key_fit,
        "hold_pose_screens": hold_screens,
        "hold_pose_contract": "AutoDex default Inspire squeeze_level=2; measured hardware pose must be checked online",
        "hand_socket_clear_at_20mm": hand_pass,
        "endpoint_pass": not key_fit["colliding"] and hand_pass,
        "method": "Coal triangle-mesh surface collision and minimum distance",
        "T_key_hand": T_key_hand.tolist(),
        "T_socket_key_verification": T_socket_key_nominal.tolist(),
        "T_socket_key_tested": T_socket_key.tolist(),
        "input_sha256": {name: _sha256(path) for name, path in files.items()},
        "not_validated": [
            "simulated or physical grasp stability",
            "Franka arm IK or collisions",
            "continuous pick, lift, transfer, insertion or retreat motion",
            "contact forces, slip, or physical insertion success",
            "measured post-grasp hand joint state and controller tracking",
        ],
        "robot_ready": False,
    }
