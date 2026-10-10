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
    timestamp_source: str


@dataclass(frozen=True)
class SessionCalibration:
    board: dict
    socket_pose_robot: np.ndarray
    socket_diagnostics: dict
    collision_scene: dict
    record: dict


def validate_session_camera_calibration(
    calibration: SessionCalibration, *,
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_full: Mapping[str, np.ndarray],
    calibrated_camera_ids: set[str],
) -> None:
    """Reject a key/retry camera rig different from the frozen session rig.

    Older offline records without a camera snapshot remain loadable but cannot
    satisfy this live-perception gate. A digest detects changed saved bytes;
    it does not certify physical camera stability or hand-eye accuracy.
    """
    record = calibration.record
    snapshot = record.get("camera_calibration")
    if (not isinstance(snapshot, dict) or
            record.get("camera_calibration_sha256") !=
            _canonical_sha256(snapshot)):
        raise ValueError("session lacks a valid frozen camera calibration")
    stored_intrinsics = snapshot.get("intrinsics_full")
    stored_extrinsics = snapshot.get("extrinsics_full")
    if (not isinstance(stored_intrinsics, dict) or
            not isinstance(stored_extrinsics, dict) or
            set(stored_intrinsics) != calibrated_camera_ids or
            set(stored_extrinsics) != calibrated_camera_ids or
            set(intrinsics_undist) != calibrated_camera_ids or
            set(extrinsics_full) != calibrated_camera_ids):
        raise ValueError("live camera IDs differ from frozen session calibration")
    for serial in calibrated_camera_ids:
        if not isinstance(stored_intrinsics[serial], Mapping):
            raise ValueError(f"invalid saved camera intrinsics: {serial}")
        stored_K = np.asarray(
            stored_intrinsics[serial].get("K_undist"), dtype=float)
        live_K = np.asarray(intrinsics_undist[serial], dtype=float)
        stored_E = validate_se3(stored_extrinsics[serial],
                                name=f"saved {serial} extrinsic")
        live_E = validate_se3(extrinsics_full[serial],
                              name=f"live {serial} extrinsic")
        if (stored_K.shape != (3, 3) or live_K.shape != (3, 3) or
                not np.all(np.isfinite(stored_K)) or
                not np.all(np.isfinite(live_K)) or
                not np.allclose(stored_K, live_K, rtol=0, atol=1e-10) or
                not np.allclose(stored_E, live_E, rtol=0, atol=1e-10)):
            raise ValueError(f"live camera calibration changed: {serial}")


def _json_native(value):
    """Snapshot a cuRobo scene without stringifying unknown Python objects."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("collision scene keys must be strings")
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("collision scene contains a non-finite number")
        return value
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError(f"collision scene contains unsupported {type(value).__name__}")


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


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


def load_session_calibration(
    path: Path, *, mode: TaskMode, shared_root: Path,
) -> SessionCalibration:
    """Reconstruct and revalidate a saved frozen world for offline replay.

    Older records without a collision-world snapshot cannot be replayed:
    rebuilding from a current default scene would silently change obstacles.
    This is read-only and does not establish live physical fixture stability.
    """
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != (
            "precision_insertion_session_calibration_v1"):
        raise ValueError("unknown session calibration record schema")
    scene = record.get("collision_scene")
    if not isinstance(scene, dict) or not isinstance(scene.get("mesh"), dict) or (
            not isinstance(scene.get("cuboid"), dict)):
        raise ValueError("saved session has no frozen collision world snapshot")
    if record.get("collision_scene_sha256") != _canonical_sha256(scene):
        raise ValueError("saved collision world descriptor hash changed")
    camera_calibration = record.get("camera_calibration")
    camera_calibration_hash = record.get("camera_calibration_sha256")
    if camera_calibration is not None or camera_calibration_hash is not None:
        if (not isinstance(camera_calibration, dict) or
                camera_calibration_hash != _canonical_sha256(
                    camera_calibration)):
            raise ValueError("saved camera calibration changed or is incomplete")
    board = record.get("board")
    if not isinstance(board, dict):
        raise ValueError("saved ChArUco board measurement is missing")
    if (record.get("board_timestamp_source") != "camera_acquisition" or
            not isinstance(record.get("socket_observations"), list) or
            not record["socket_observations"] or
            any(not isinstance(row, dict) or
                row.get("timestamp_source") != "camera_acquisition"
                for row in record["socket_observations"])):
        raise ValueError("saved session lacks camera acquisition timestamps")
    from autodex.utils.tabletop_geometry import table_cuboid

    if scene["cuboid"].get("table") != table_cuboid(board):
        raise ValueError("saved table cuboid differs from ChArUco board")
    hashes = record.get("fixed_mesh_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(scene["mesh"]):
        raise ValueError("saved fixed mesh hashes are missing")
    for name, entry in scene["mesh"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("file_path"), str):
            raise ValueError(f"saved fixed mesh {name} has no file path")
        mesh_path = Path(entry["file_path"]).expanduser().resolve()
        if not mesh_path.is_file() or _file_sha256(mesh_path) != hashes[name]:
            raise ValueError(f"saved fixed mesh {name} changed or is missing")
    validate_se3(record.get("c2r"), name="saved C2R")
    socket_pose = validate_se3(record.get("socket_pose_robot"),
                               name="saved socket pose")
    diagnostics = record.get("socket_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("accepted") is not True:
        raise ValueError("saved socket calibration was not accepted")
    session = SessionCalibration(board, socket_pose, diagnostics, scene, record)
    from .world import validated_frozen_socket_pose

    validated_frozen_socket_pose(
        mode=mode, shared_root=shared_root, calibration=session)
    return session


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
    board_timestamp_source: str,
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
    if board_timestamp_source != "camera_acquisition":
        raise ValueError("ChArUco images require camera acquisition timestamps")

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
        if observation.timestamp_source != "camera_acquisition":
            raise ValueError("socket observation requires camera acquisition timestamp")
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
                "timestamp_source": observation.timestamp_source,
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
    frozen_scene = _json_native(scene)
    camera_calibration = {
        "intrinsics_full": _json_native(intrinsics_full),
        "extrinsics_full": _json_native(extrinsics_full),
    }
    fixed_mesh_hashes = {}
    for name, entry in frozen_scene["mesh"].items():
        path = Path(entry["file_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"fixed collision mesh missing: {path}")
        fixed_mesh_hashes[name] = _file_sha256(path)
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
        "board_timestamp_source": board_timestamp_source,
        "socket_observations": source_rows,
        "socket_pose_robot": selected.tolist(),
        "socket_diagnostics": diagnostics,
        "socket_collision_mesh": str(mesh),
        "socket_collision_mesh_sha256": _file_sha256(mesh),
        "collision_scene": frozen_scene,
        "collision_scene_sha256": _canonical_sha256(frozen_scene),
        "camera_calibration": camera_calibration,
        "camera_calibration_sha256": _canonical_sha256(camera_calibration),
        "fixed_mesh_sha256": fixed_mesh_hashes,
        "c2r": c2r_matrix.tolist(),
        "scope": "read_only_session_calibration_not_robot_authorization",
        "robot_ready": False,
    }
    return SessionCalibration(board, selected, diagnostics, scene, record)
