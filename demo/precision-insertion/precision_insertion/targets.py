"""Read-only rigid key/hand goals above and 20 mm inside a frozen socket.

These SE(3) targets are not IK, collision, continuous-path, contact-control,
or robot-execution evidence. The caller supplies a validated held-key/hand
relation; this module never changes finger joints after the selected grasp.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import validate_task_geometry
from .geometry import validate_se3
from .world import validated_frozen_socket_pose

if TYPE_CHECKING:
    from .calibration import SessionCalibration


@dataclass(frozen=True)
class InsertionTargets:
    mode: TaskMode
    task_geometry_sha256: str
    socket_collision_mesh_sha256: str
    T_key_hand: np.ndarray
    T_robot_key_preinsert: np.ndarray
    T_robot_key_entry: np.ndarray
    T_robot_key_verification: np.ndarray
    T_robot_hand_preinsert: np.ndarray
    T_robot_hand_entry: np.ndarray
    T_robot_hand_verification: np.ndarray
    insertion_axis_robot: np.ndarray
    xy_offset_socket_m: tuple[float, float]
    preinsert_clearance_m: float
    cylinder_yaw_gauge_socket_rad: float = 0.0

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_rigid_targets_v1",
            "scope": "read_only_poses_not_trajectory_or_motion_authorization",
            "mode": {
                "family": self.mode.family, "gap_mm": self.mode.gap_mm,
                "key_object": self.mode.key_object,
                "socket_object": self.mode.socket_object,
                "target_depth_m": self.mode.target_depth_m,
            },
            "task_geometry_sha256": self.task_geometry_sha256,
            "socket_collision_mesh_sha256": self.socket_collision_mesh_sha256,
            "T_key_hand": self.T_key_hand.tolist(),
            "T_robot_key_preinsert": self.T_robot_key_preinsert.tolist(),
            "T_robot_key_entry": self.T_robot_key_entry.tolist(),
            "T_robot_key_verification": self.T_robot_key_verification.tolist(),
            "T_robot_hand_preinsert": self.T_robot_hand_preinsert.tolist(),
            "T_robot_hand_entry": self.T_robot_hand_entry.tolist(),
            "T_robot_hand_verification": self.T_robot_hand_verification.tolist(),
            "insertion_axis_robot": self.insertion_axis_robot.tolist(),
            "xy_offset_socket_m": list(self.xy_offset_socket_m),
            "cylinder_yaw_gauge_socket_rad": (
                self.cylinder_yaw_gauge_socket_rad),
            "preinsert_clearance_m": self.preinsert_clearance_m,
            "robot_ready": False,
        }


def build_rigid_insertion_targets(
    *, mode: TaskMode, shared_root: Path, calibration: SessionCalibration,
    T_key_hand: np.ndarray,
    xy_offset_socket_m: tuple[float, float] = (0.0, 0.0),
    cylinder_yaw_gauge_socket_rad: float = 0.0,
) -> InsertionTargets:
    """Compose CAD key targets with the frozen socket and one fixed grasp.

    ``T_key_hand`` is the BODex/v8 candidate's object-frame ``wrist_se3``
    or a separately checked post-lift held relation. XY offsets are absolute
    coordinates in the socket frame, not incremental image pixels.
    """
    root = Path(shared_root).expanduser().resolve()
    socket = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    hand_in_key = validate_se3(T_key_hand, name="fixed T_key_hand")
    offset = np.asarray(xy_offset_socket_m, dtype=np.float64)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("socket XY offset must contain two finite meters")
    yaw = float(cylinder_yaw_gauge_socket_rad)
    if (not math.isfinite(yaw) or
            (mode.family != "cylinder" and yaw != 0.0)):
        raise ValueError("only cylinder targets may use a finite yaw gauge")
    c, s = math.cos(yaw), math.sin(yaw)
    gauge_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])

    path = AssetPaths(root, mode).task_geometry
    geometry_bytes = path.read_bytes()
    geometry = json.loads(geometry_bytes)
    if not isinstance(geometry, dict):
        raise ValueError("task geometry must be a JSON object")
    verification = validate_task_geometry(geometry, mode)
    entry = validate_se3(geometry.get("T_socket_key_entry"),
                         name="T_socket_key_entry")
    preinsert = validate_se3(geometry.get("T_socket_key_preinsert"),
                             name="T_socket_key_preinsert")
    direction = np.asarray(geometry["insertion_direction_socket"],
                           dtype=np.float64)
    clearance = float(geometry.get("preinsert_clearance_m", float("nan")))
    if (not math.isfinite(clearance) or clearance <= 0 or
            not np.allclose(direction[:2], [0.0, 0.0], atol=1e-8) or
            not np.isclose(abs(direction[2]), 1.0, atol=1e-8)):
        raise ValueError("preinsert clearance or socket-local axial direction invalid")
    if (not np.allclose(preinsert[:3, :3], entry[:3, :3], atol=1e-8) or
            not np.allclose(preinsert[:3, 3] - entry[:3, 3],
                            -clearance * direction, atol=1e-8)):
        raise ValueError("preinsert CAD pose is not above entry on the socket axis")

    displacement = np.array([offset[0], offset[1], 0.0], dtype=np.float64)

    def in_robot(pose_socket_key: np.ndarray) -> np.ndarray:
        shifted = pose_socket_key.copy()
        shifted[:3, :3] = gauge_rotation @ shifted[:3, :3]
        shifted[:3, 3] += displacement
        return validate_se3(socket @ shifted, name="T_robot_key_goal")

    key_preinsert = in_robot(preinsert)
    key_entry = in_robot(entry)
    key_verification = in_robot(verification)
    axis_robot = socket[:3, :3] @ direction
    return InsertionTargets(
        mode=mode,
        task_geometry_sha256=hashlib.sha256(geometry_bytes).hexdigest(),
        socket_collision_mesh_sha256=calibration.record[
            "socket_collision_mesh_sha256"],
        T_key_hand=hand_in_key,
        T_robot_key_preinsert=key_preinsert,
        T_robot_key_entry=key_entry,
        T_robot_key_verification=key_verification,
        T_robot_hand_preinsert=key_preinsert @ hand_in_key,
        T_robot_hand_entry=key_entry @ hand_in_key,
        T_robot_hand_verification=key_verification @ hand_in_key,
        insertion_axis_robot=axis_robot,
        xy_offset_socket_m=(float(offset[0]), float(offset[1])),
        preinsert_clearance_m=clearance,
        cylinder_yaw_gauge_socket_rad=yaw,
    )
