"""Load a one-time measured support plane for board-free runtime sessions."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


TABLE_PLANE_SCHEMA = "precision_insertion_table_plane_v1"


def support_frame_from_plane(point_robot_m, normal_robot) -> np.ndarray:
    """Build a deterministic +Z-up support frame from a measured plane."""
    point = np.asarray(point_robot_m, dtype=float)
    normal = np.asarray(normal_robot, dtype=float)
    if (point.shape != (3,) or normal.shape != (3,) or
            not np.all(np.isfinite(point)) or
            not np.all(np.isfinite(normal)) or np.linalg.norm(normal) < 1e-9):
        raise ValueError("support plane needs a finite point and normal")
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    reference = np.array([1.0, 0.0, 0.0])
    x_axis = reference - np.dot(reference, normal) * normal
    if np.linalg.norm(x_axis) < 1e-6:
        reference = np.array([0.0, 1.0, 0.0])
        x_axis = reference - np.dot(reference, normal) * normal
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack((x_axis, y_axis, normal))
    frame[:3, 3] = point
    return frame


def load_table_surface(path: str | Path) -> dict:
    """Return the measured robot-frame support plane and an upright frame.

    Older records that contain only ``table_surface_z_m`` remain usable as a
    horizontal plane.  New capture4 records retain the fitted point and normal
    so board-free socket localization uses the measured surface tilt rather
    than silently replacing it with world Z.
    """
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    value = record.get("table_surface_z_m") if isinstance(record, dict) else None
    if (record.get("schema") != TABLE_PLANE_SCHEMA or
            type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError(f"invalid measured table-plane record: {source}")
    point = np.asarray(record.get(
        "plane_point_robot_m", [0.0, 0.0, float(value)]), dtype=float)
    normal = np.asarray(record.get(
        "normal_robot", [0.0, 0.0, 1.0]), dtype=float)
    try:
        frame = support_frame_from_plane(point, normal)
    except ValueError as exc:
        raise ValueError(
            f"invalid measured table-plane geometry: {source}") from exc
    normal = frame[:3, 2]
    return {
        "table_surface_z_m": float(value),
        "plane_point_robot_m": point,
        "normal_robot": normal,
        "T_robot_support": frame,
    }


def load_table_surface_z(path: str | Path) -> float:
    """Return the robot-frame support height from a table-plane record."""
    return load_table_surface(path)["table_surface_z_m"]


def session_support_plane_robot(calibration) -> tuple[np.ndarray, float]:
    """Return the frozen session support plane in the robot frame.

    ChArUco sessions describe the plane with ``T_robot_board``.  Board-free
    sessions store the commissioned plane point and normal directly.  Legacy
    board-free records that predate those fields remain a horizontal-plane
    fallback, but current records preserve their measured tilt.
    """
    support = getattr(calibration, "board", None)
    if not isinstance(support, dict):
        raise ValueError("session support surface is missing")
    if "T_robot_board" in support:
        from .geometry import validate_se3

        pose = validate_se3(support["T_robot_board"],
                            name="measured board pose")
        normal = -pose[:3, 2]
        return normal, float(np.dot(pose[:3, 3], normal))
    height = support.get("table_surface_z_m")
    if (support.get("source") != "configured_table_surface" or
            type(height) not in (int, float) or
            not math.isfinite(float(height))):
        raise ValueError("board-free session has no calibrated table plane")
    point = support.get("plane_point_robot_m", [0.0, 0.0, float(height)])
    normal = support.get("normal_robot", [0.0, 0.0, 1.0])
    frame = support_frame_from_plane(point, normal)
    up = frame[:3, 2]
    return up, float(np.dot(frame[:3, 3], up))
