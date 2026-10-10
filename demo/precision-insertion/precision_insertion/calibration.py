"""Read-only ChArUco-then-socket calibration for one insertion session.

The caller acquires synchronized AutoDex images and quality-checked FoundPose
observations. This module reuses AutoDex's board measurement, transforms all
socket observations into the robot frame, freezes an *observed* medoid, and
adds the exact socket mesh to a copy of the collision scene. It neither
captures images nor connects to, commands, or authorizes the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import TaskMode
from .geometry import freeze_fixture_pose, validate_se3
from .symmetry import load_axial_symmetry
from .world import add_fixed_mesh_fixtures


@dataclass(frozen=True)
class SocketObservation:
    capture_id: str
    camera_id: str
    timestamp_s: float
    pose_world: np.ndarray


@dataclass(frozen=True)
class SessionCalibration:
    board: dict
    socket_pose_robot: np.ndarray
    socket_diagnostics: dict
    collision_scene: dict
    record: dict


def write_session_calibration(calibration: SessionCalibration, path: Path) -> Path:
    """Save the versioned evidence record without replacing a prior session.

    Captured images and masks are separate evidence artifacts; the caller
    must retain them under the capture IDs in ``socket_observations``.
    """
    target = Path(path).expanduser().resolve()
    payload = json.dumps(calibration.record, indent=2, sort_keys=True) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return target


def _finite_time(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite_time(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibrate_session(
    *,
    mode: TaskMode,
    object_root: Path,
    board_images_bgr: Mapping[str, np.ndarray],
    board_timestamps_s: Mapping[str, float],
    socket_observations: Sequence[SocketObservation],
    intrinsics_full: Mapping,
    extrinsics_full: Mapping,
    c2r: np.ndarray,
    base_scene: dict,
    socket_collision_mesh: Path,
    max_capture_skew_s: float,
    max_socket_translation_mm: float,
    max_socket_angle_deg: float,
    min_socket_captures: int = 2,
    min_views_per_capture: int = 2,
) -> SessionCalibration:
    """Build one frozen calibration from supplied, already-captured evidence.

    A ``capture_id`` is one simultaneous multi-camera socket snapshot; at
    least two different captures are required to test fixture repeatability.
    ``pose_world`` uses the same world frame as ``c2r`` and AutoDex's board
    measurement. Bounds are explicit commissioning inputs, not defaults.

    Geometric agreement alone does not establish absolute sub-mm accuracy or
    FoundPose photometric quality. The caller must reject bad masks/poses
    before supplying them and must separately validate hand-eye calibration.
    """
    skew = _positive(max_capture_skew_s, "max_capture_skew_s")
    translation_limit = _positive(
        max_socket_translation_mm, "max_socket_translation_mm")
    angle_limit = _positive(max_socket_angle_deg, "max_socket_angle_deg")
    if min_socket_captures < 2 or min_views_per_capture < 2:
        raise ValueError("require at least two socket captures and two views per capture")

    camera_ids = set(intrinsics_full) & set(extrinsics_full)
    if not camera_ids or set(intrinsics_full) != set(extrinsics_full):
        raise ValueError("intrinsic and extrinsic camera IDs must match and be nonempty")
    board_ids = set(board_images_bgr)
    if not board_ids or board_ids != set(board_timestamps_s):
        raise ValueError("board images and timestamps must have identical camera IDs")
    if not board_ids <= camera_ids:
        raise ValueError("board snapshot contains an uncalibrated camera")
    board_times = {
        camera: _finite_time(board_timestamps_s[camera], f"board timestamp {camera}")
        for camera in board_ids
    }
    if max(board_times.values()) - min(board_times.values()) > skew:
        raise ValueError("board snapshot is not synchronized")

    c2r_matrix = validate_se3(c2r, name="c2r")
    world_to_robot = np.linalg.inv(c2r_matrix)
    groups: dict[str, list[SocketObservation]] = {}
    for index, observation in enumerate(socket_observations):
        if not isinstance(observation, SocketObservation):
            raise TypeError(f"socket observation {index} must be SocketObservation")
        if not observation.capture_id or not observation.camera_id:
            raise ValueError("socket capture ID and camera ID must be nonempty")
        if observation.camera_id not in camera_ids:
            raise ValueError(f"uncalibrated socket camera {observation.camera_id}")
        _finite_time(observation.timestamp_s, "socket timestamp")
        validate_se3(observation.pose_world, name="socket pose_world")
        groups.setdefault(observation.capture_id, []).append(observation)
    if len(groups) < min_socket_captures:
        raise ValueError("too few separate socket captures")

    ordered_groups = sorted(
        groups.items(), key=lambda item: min(float(o.timestamp_s) for o in item[1]))
    prior_capture_end = max(board_times.values())
    for capture_id, observations in ordered_groups:
        ids = [o.camera_id for o in observations]
        if len(observations) < min_views_per_capture or len(ids) != len(set(ids)):
            raise ValueError(f"socket capture {capture_id}: too few or duplicate views")
        times = [float(o.timestamp_s) for o in observations]
        if max(times) - min(times) > skew:
            raise ValueError(f"socket capture {capture_id} is not synchronized")
        if min(times) <= prior_capture_end:
            raise ValueError("socket captures must follow board capture in time")
        prior_capture_end = max(times)

    mesh = Path(socket_collision_mesh).expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f"exact socket collision mesh not found: {mesh}")
    if not isinstance(base_scene, dict):
        raise TypeError("base_scene must be a scene dictionary")
    if not isinstance(base_scene.get("mesh", {}), dict):
        raise TypeError("base_scene mesh must be a dictionary")
    if not isinstance(base_scene.get("cuboid", {}), dict):
        raise TypeError("base_scene cuboid must be a dictionary")
    if "target" in base_scene.get("mesh", {}):
        raise ValueError("session fixed collision world must not contain a key target")
    axis = None
    if mode.family == "cylinder":
        symmetry = load_axial_symmetry(Path(object_root), mode.socket_object)
        if symmetry.end_exchange:
            raise ValueError("cylindrical socket may not exchange open and closed ends")
        axis = symmetry.axis_local
    elif mode.family != "square":
        raise ValueError(f"unsupported insertion family: {mode.family}")

    # This is intentionally called only after temporal and asset validation:
    # measuring an empty board is the first geometric operation in a session.
    from src.execution.charuco_tabletop import measure_tabletop_from_images
    from autodex.utils.tabletop_geometry import table_cuboid

    board = measure_tabletop_from_images(
        board_images_bgr, intrinsics_full, extrinsics_full, c2r_matrix)
    measured_base_scene = copy.deepcopy(base_scene)
    measured_base_scene.setdefault("cuboid", {})["table"] = table_cuboid(board)
    observations_robot: list[np.ndarray] = []
    source_rows: list[dict] = []
    for capture_id, observations in ordered_groups:
        for observation in sorted(observations, key=lambda o: o.camera_id):
            pose_world = validate_se3(
                observation.pose_world, name="socket pose_world")
            pose_robot = validate_se3(
                world_to_robot @ pose_world,
                name="socket pose_robot")
            observations_robot.append(pose_robot)
            source_rows.append({
                "capture_id": capture_id,
                "camera_id": observation.camera_id,
                "timestamp_s": float(observation.timestamp_s),
                "pose_world": pose_world.tolist(),
                "pose_robot": pose_robot.tolist(),
            })
    selected, diagnostics = freeze_fixture_pose(
        observations_robot,
        translation_limit_mm=translation_limit,
        angle_limit_deg=angle_limit,
        continuous_axis_local=axis,
    )
    scene = add_fixed_mesh_fixtures(measured_base_scene, {
        "fixture_socket": {
            "pose_robot": selected,
            "collision_mesh": mesh,
        },
    })
    record = {
        "schema": "precision_insertion_session_calibration_v1",
        "mode": {
            "family": mode.family,
            "gap_mm": mode.gap_mm,
            "key_object": mode.key_object,
            "socket_object": mode.socket_object,
        },
        "board": board,
        "board_timestamps_s": board_times,
        "socket_observations": source_rows,
        "socket_pose_robot": selected.tolist(),
        "socket_diagnostics": diagnostics,
        "socket_collision_mesh": str(mesh),
        "socket_collision_mesh_sha256": _file_sha256(mesh),
        "c2r": c2r_matrix.tolist(),
        "scope": "read_only_session_calibration_not_robot_authorization",
        "robot_ready": False,
    }
    return SessionCalibration(board, selected, diagnostics, scene, record)
