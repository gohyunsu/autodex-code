"""Build the session's fixed-fixture collision world without legacy hooks."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import numpy as np

from autodex.utils.conversion import se32cart

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3
from .symmetry import snap_axisymmetric_tabletop_pose

if TYPE_CHECKING:
    from .calibration import SessionCalibration


def add_fixed_mesh_fixtures(
    scene_cfg: dict,
    fixed_fixtures: Mapping[str, Mapping] | None,
) -> dict:
    """Return a new cuRobo scene with validated socket fixture meshes.

    The original scene is never mutated. The fixture name cannot shadow the
    target object or another mesh. The caller owns the free-key versus
    attached-key scene transition after grasp.
    """
    result = copy.deepcopy(scene_cfg)
    if not fixed_fixtures:
        return result
    meshes = result.setdefault("mesh", {})
    for name, fixture in sorted(fixed_fixtures.items()):
        if not isinstance(name, str) or not name or name == "target":
            raise ValueError(f"invalid fixed fixture name: {name!r}")
        if name in meshes:
            raise ValueError(f"fixed fixture would replace scene mesh {name!r}")
        if not isinstance(fixture, Mapping):
            raise ValueError(f"fixed fixture {name!r} must be a mapping")
        pose_robot = validate_se3(
            fixture.get("pose_robot"), name=f"fixed fixture {name} pose_robot")
        mesh_value = fixture.get("collision_mesh")
        if not isinstance(mesh_value, (str, Path)) or not str(mesh_value):
            raise FileNotFoundError(f"fixed fixture {name!r} collision mesh missing")
        mesh_path = Path(mesh_value).expanduser().resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"fixed fixture {name!r} collision mesh not found: {mesh_path}")
        meshes[name] = {
            "pose": se32cart(pose_robot).tolist(),
            "file_path": str(mesh_path),
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_frozen_socket_pose(
    *, mode: TaskMode, shared_root: Path, calibration: SessionCalibration,
) -> np.ndarray:
    """Verify the session's socket identity, CAD bytes and collision pose."""
    root = Path(shared_root).expanduser().resolve()
    paths = AssetPaths(root, mode)
    identity = calibration.record.get("mode")
    if identity != {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
    }:
        raise ValueError("session calibration key/socket mode does not match trial")
    T_robot_socket = validate_se3(
        calibration.socket_pose_robot, name="frozen T_robot_socket")
    fixed_world = calibration.collision_scene
    if not isinstance(fixed_world, dict):
        raise TypeError("session collision scene must be a dictionary")
    meshes = fixed_world.get("mesh", {})
    if not isinstance(meshes, dict):
        raise TypeError("session collision meshes must be a dictionary")
    if "target" in meshes:
        raise ValueError("session fixed world contains a stale key target")
    fixture = meshes.get("fixture_socket")
    if not isinstance(fixture, dict):
        raise ValueError("session fixed world has no socket fixture")
    socket_mesh = paths.socket_collision_mesh.resolve()
    if not socket_mesh.is_file():
        raise FileNotFoundError(f"frozen socket collision mesh missing: {socket_mesh}")
    if (Path(fixture.get("file_path", "")).resolve() != socket_mesh or
            calibration.record.get("socket_collision_mesh") != str(socket_mesh) or
            calibration.record.get("socket_collision_mesh_sha256") !=
            _sha256(socket_mesh)):
        raise ValueError("socket collision mesh differs from frozen session asset")
    record_socket = validate_se3(
        calibration.record.get("socket_pose_robot"),
        name="session record T_robot_socket")
    if not np.allclose(record_socket, T_robot_socket, atol=1e-8):
        raise ValueError("session record socket pose differs from frozen pose")
    fixture_pose = np.asarray(fixture.get("pose"), dtype=np.float64)
    if (fixture_pose.shape != (7,) or not np.all(np.isfinite(fixture_pose)) or
            not np.allclose(fixture_pose, se32cart(T_robot_socket), atol=1e-8)):
        raise ValueError("session socket collision pose differs from frozen pose")
    return T_robot_socket


def build_trial_scene_from_session(
    *, mode: TaskMode, shared_root: Path, calibration: SessionCalibration,
    key_pose_world: np.ndarray, max_axis_error_deg: float = 20.0,
) -> dict:
    """Create a fresh v8 key scene while retaining the frozen session socket.

    Reuses AutoDex's unchanged pose-to-scene conversion, with only the
    cylindrical local-z symmetry adaptation owned by this demo. The session
    base world contains measured table + socket, never a stale key target.
    This function is geometric preparation, not planning authorization.
    """
    root = Path(shared_root).expanduser().resolve()
    paths = AssetPaths(root, mode)
    validated_frozen_socket_pose(mode=mode, shared_root=root,
                                 calibration=calibration)
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    fixed_world = calibration.collision_scene

    pose_world = validate_se3(key_pose_world, name="fresh key pose_world")
    if mode.family == "cylinder":
        robot_pose = validate_se3(np.linalg.inv(c2r) @ pose_world,
                                  name="key pose_robot")
        robot_pose = snap_axisymmetric_tabletop_pose(
            robot_pose, object_root=paths.object_root,
            object_name=mode.key_object,
            max_axis_error_deg=max_axis_error_deg)
        pose_world = validate_se3(c2r @ robot_pose, name="snapped key pose_world")
    elif mode.family != "square":
        raise ValueError(f"unsupported key family: {mode.family}")

    from src.execution.scene_cfg import pose_world_to_scene_cfg
    live_key_scene = pose_world_to_scene_cfg(
        pose_world, c2r, mode.key_object, obj_root=str(paths.object_root),
        tabletop_geometry=calibration.board)
    if live_key_scene.get("cuboid", {}).get("table") != fixed_world.get(
            "cuboid", {}).get("table"):
        raise ValueError("live key scene table differs from measured session table")
    target = live_key_scene.get("mesh", {}).get("target")
    if not isinstance(target, dict):
        raise ValueError("AutoDex did not construct a v8 key target")
    result = copy.deepcopy(fixed_world)
    result.setdefault("mesh", {})["target"] = target
    return result
