"""Conditional key-tip depth from a measured wrist and bounded held grasp.

This is geometry, not observed penetration. A 20 mm wrist stroke alone is
never accepted as a key-depth source by the task-success checkpoint. The
interval is useful for identifying what independent calibration, grasp-slip
and camera evidence would still be needed to promote that hypothesis.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode
from .endpoint import validate_task_geometry
from .geometry import validate_se3
from .targets import InsertionTargets
from .uncertainty_margin import SurfaceDeviationBounds


def conditional_key_tip_depth(
    *, mode: TaskMode, targets: InsertionTargets,
    task_geometry_path: Path, T_robot_socket: np.ndarray,
    T_robot_hand_measured: np.ndarray,
    surface_bounds: SurfaceDeviationBounds,
) -> dict:
    """Bound axial depth *if* the physical key/hand relation still holds.

    The externally commissioned key-surface bound must already include
    grasp relation, FK/controller, socket registration and CAD errors for
    the complete relevant motion. It is not estimated from this pose.
    """
    surface_bounds.validate()
    if not isinstance(targets, InsertionTargets) or targets.mode != mode:
        raise ValueError("wrist depth needs matching rigid task targets")
    source = Path(task_geometry_path).expanduser().resolve()
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != targets.task_geometry_sha256:
        raise ValueError("wrist depth CAD differs from the planned target")
    geometry = json.loads(payload)
    if not isinstance(geometry, dict):
        raise ValueError("wrist depth CAD must be a JSON object")
    validate_task_geometry(geometry, mode)
    socket = validate_se3(T_robot_socket, name="frozen T_robot_socket")
    hand = validate_se3(T_robot_hand_measured, name="measured T_robot_hand")
    key_hand = validate_se3(targets.T_key_hand, name="bounded T_key_hand")
    entry = validate_se3(targets.T_robot_key_entry,
                         name="frozen key entry target")
    expected_entry = validate_se3(
        geometry["T_socket_key_entry"], name="CAD T_socket_key_entry").copy()
    yaw = targets.cylinder_yaw_gauge_socket_rad
    if mode.family != "cylinder" and yaw != 0:
        raise ValueError("square key cannot use a cylinder yaw gauge")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    gauge = np.array([[cosine, -sine, 0.],
                      [sine, cosine, 0.], [0., 0., 1.]])
    expected_entry[:3, :3] = gauge @ expected_entry[:3, :3]
    expected_entry = socket @ expected_entry
    expected_entry[:3, 3] += socket[:3, :2] @ np.asarray(
        targets.xy_offset_socket_m, dtype=float)
    if (not np.allclose(entry, expected_entry, atol=1e-8, rtol=0) or
            not np.allclose(targets.T_robot_hand_entry, entry @ key_hand,
                            atol=1e-8, rtol=0)):
        raise ValueError("entry target, frozen socket and held relation disagree")
    tip_z = geometry.get("key_frame", {}).get("tip_z_m")
    if (type(tip_z) not in (float, int) or
            not math.isfinite(tip_z) or tip_z <= 0):
        raise ValueError("CAD key insertion-tip centre is missing")
    local_tip = np.array([0., 0., float(tip_z)])
    entry_tip = entry[:3, :3] @ local_tip + entry[:3, 3]
    live_key = validate_se3(hand @ np.linalg.inv(key_hand),
                            name="conditional held key")
    live_tip = live_key[:3, :3] @ local_tip + live_key[:3, 3]
    axis = np.asarray(targets.insertion_axis_robot, dtype=float)
    direction = np.asarray(geometry["insertion_direction_socket"], dtype=float)
    if (axis.shape != (3,) or not np.allclose(
            axis, socket[:3, :3] @ direction, atol=1e-8, rtol=0)):
        raise ValueError("insertion axis differs from the frozen socket")
    entry_tip_socket = socket[:3, :3].T @ (entry_tip - socket[:3, 3])
    rim = geometry.get("socket_entry_plane_z_m")
    if (type(rim) not in (float, int) or not math.isfinite(rim) or
            abs(entry_tip_socket[2] - rim) > 1e-7):
        raise ValueError("CAD insertion tip does not start at the socket rim")
    difference = live_tip - entry_tip
    nominal_depth = float(np.dot(axis, difference))
    lateral = float(np.linalg.norm(difference - nominal_depth * axis))
    live_tip_socket = socket[:3, :3].T @ (live_tip - socket[:3, 3])
    socket_axis_lateral = float(np.linalg.norm(live_tip_socket[:2]))
    live_key_axis = live_key[:3, :3] @ np.asarray(
        geometry["key_frame"]["insertion_axis"], dtype=float)
    tilt = math.degrees(math.acos(float(np.clip(
        np.dot(live_key_axis, axis), -1., 1.))))
    error = surface_bounds.key_surface_m
    return {
        "schema": "precision_insertion_conditional_wrist_depth_v1",
        "nominal_key_tip_depth_m": nominal_depth,
        "conditional_depth_interval_m": [nominal_depth - error,
                                         nominal_depth + error],
        "tip_lateral_residual_to_planned_entry_m": lateral,
        "tip_offset_from_socket_axis_m": socket_axis_lateral,
        "key_axis_tilt_deg": tilt,
        "commissioned_key_surface_bound_m": error,
        "task_geometry_sha256": targets.task_geometry_sha256,
        "held_relation_assumption": "physical_key_remains_rigid_at_T_key_hand",
        "key_depth_source": None,
        "scope": "conditional_wrist_key_hypothesis_not_observed_penetration_or_task_label",
        "robot_ready": False,
    }
