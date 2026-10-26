"""Convert a previous trial's socket-XY error into a new wrist target.

The CAD key target stays on the socket axis. A learned translation changes
only the key-to-hand command relation, so endpoint screening still tests a
centered key and a correspondingly shifted hand.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3


def compensated_key_hand_relation(
    *, mode: TaskMode, shared_root: Path, T_key_hand: np.ndarray,
    wrist_correction_socket_m: tuple[float, float],
    cylinder_yaw_gauge_socket_rad: float = 0.0,
) -> np.ndarray:
    """Translate a new attempt's wrist by delta while keeping key CAD centered.

    With T_RH = T_RS T_SK T_KH, adding R_SK.T @ delta_S to the
    translation of T_KH moves the commanded wrist by delta_S in the socket
    plane. For a cylinder yaw gauge, T_SK's effective rotation is
    R_gauge @ R_SK. This is a *learned command correction*, not a claim that
    the key moved or that an occluded key pose was observed in this attempt.
    """
    relation = validate_se3(T_key_hand, name="key-to-hand relation").copy()
    offset = np.asarray(wrist_correction_socket_m, dtype=np.float64)
    if offset.shape != (2,) or not np.all(np.isfinite(offset)):
        raise ValueError("new-attempt wrist correction needs finite socket XY")
    yaw = float(cylinder_yaw_gauge_socket_rad)
    if not math.isfinite(yaw) or (mode.family != "cylinder" and yaw != 0.0):
        raise ValueError("only a cylinder may use a finite yaw gauge")
    if mode.family != "cylinder" and np.any(offset):
        raise ValueError("square mode has no XY wrist correction")
    if not np.any(offset):
        return relation
    geometry = json.loads(AssetPaths(Path(shared_root), mode).task_geometry.read_text(
        encoding="utf-8"))
    T_socket_key = validate_se3(
        geometry["T_socket_key_verification"],
        name="CAD verification key pose in socket")
    cosine, sine = math.cos(yaw), math.sin(yaw)
    gauge = np.array([[cosine, -sine, 0.],
                      [sine, cosine, 0.], [0., 0., 1.]])
    effective_rotation = gauge @ T_socket_key[:3, :3]
    relation[:3, 3] += effective_rotation.T @ np.array(
        [offset[0], offset[1], 0.0], dtype=np.float64)
    return validate_se3(relation, name="corrected key-to-hand command")
