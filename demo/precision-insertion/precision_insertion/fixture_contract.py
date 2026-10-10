"""Validate cylindrical socket CAD, frame and host-path consistency.

This is a read-only admission gate. File consistency is not FoundPose quality,
physical fixture stability or robot calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cylinder_socket_fixture(
    *, shared_root: Path, mode: TaskMode,
) -> dict:
    """Reject a stale/mismatched cylinder fixture before session capture.

    The socket pose estimator representation is deliberately *not* certified
    here. Session startup separately requires it to exist; mesh matching and
    pose quality require independent perception evidence.
    """
    if mode.family != "cylinder":
        raise ValueError("cylinder fixture contract needs a cylinder mode")
    paths = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    fixture = paths.task_geometry.parent
    socket = mode.socket_object
    files = {
        "raw_mesh": paths.raw_mesh(socket),
        "frame_contract": (paths.object_dir(socket) /
                           "processed_data/info/frame_contract.json"),
        "object_collision": paths.socket_collision_mesh,
        "fixture_collision": fixture / "static_collision.obj",
        "task_geometry": paths.task_geometry,
        "pose_template": fixture / "fixture_pose.template.json",
        "pose_measurement": fixture / "pose_measurement_asset.json",
    }
    for label, file in files.items():
        if not file.is_file():
            raise FileNotFoundError(f"cylinder fixture {label} is missing: {file}")
    if _sha(files["object_collision"]) != _sha(files["fixture_collision"]):
        raise ValueError("socket object and fixture collision meshes differ")

    frame = json.loads(files["frame_contract"].read_text(encoding="utf-8"))
    geometry = json.loads(files["task_geometry"].read_text(encoding="utf-8"))
    template = json.loads(files["pose_template"].read_text(encoding="utf-8"))
    measurement = json.loads(files["pose_measurement"].read_text(
        encoding="utf-8"))
    if any(not isinstance(row, dict) for row in
           (frame, geometry, template, measurement)):
        raise ValueError("cylinder fixture JSON payloads must be objects")

    # Local import avoids AssetPaths -> endpoint -> AssetPaths circularity.
    from .endpoint import validate_task_geometry
    validate_task_geometry(geometry, mode)
    if (geometry.get("socket_pose_mesh") != str(files["raw_mesh"]) or
            geometry.get("socket_mesh") != "static_collision.obj"):
        raise ValueError("task geometry has a stale socket mesh path")
    raw_frame = validate_se3(frame.get("T_socket_raw_mesh"),
                             name="T_socket_raw_mesh")
    if not np.allclose(raw_frame, np.eye(4), atol=1e-8):
        raise ValueError("nonidentity socket raw frame needs an explicit adapter")
    rim = float(frame.get("rim_z_m", float("nan")))
    geometry_rim = float(geometry.get("socket_rim_z_m", float("nan")))
    if not (math.isfinite(rim) and math.isfinite(geometry_rim) and
            math.isclose(rim, geometry_rim, abs_tol=1e-8)):
        raise ValueError("socket rim differs between CAD frame and task geometry")
    if (template.get("pose_object") != socket or
            template.get("calibrated") is not False or
            template.get("T_robot_socket") is not None or
            template.get("pose_object_mesh") != str(files["raw_mesh"]) or
            template.get("pose_object_frame_contract") !=
            str(files["frame_contract"]) or
            template.get("pose_estimator_asset") !=
            str(paths.foundpose_repre(socket))):
        raise ValueError("socket pose template is stale or falsely calibrated")
    template_frame = validate_se3(template.get("T_socket_pose_object"),
                                  name="template T_socket_pose_object")
    geometry_frame = validate_se3(geometry.get("T_socket_pose_object"),
                                  name="geometry T_socket_pose_object")
    if not np.allclose(template_frame, geometry_frame, atol=1e-8):
        raise ValueError("socket pose template and task frames differ")
    if measurement.get("pose_object") != socket:
        raise ValueError("socket pose measurement identity differs")
    return {
        "schema": "precision_insertion_cylinder_fixture_contract_v1",
        "socket_object": socket,
        "raw_mesh_sha256": _sha(files["raw_mesh"]),
        "collision_mesh_sha256": _sha(files["object_collision"]),
        "task_geometry_sha256": _sha(files["task_geometry"]),
        "pose_template_sha256": _sha(files["pose_template"]),
        "paths_bound_to_shared_root": True,
        "robot_ready": False,
    }
