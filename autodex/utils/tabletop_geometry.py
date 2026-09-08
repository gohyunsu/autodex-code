"""Session-scoped tabletop geometry helpers.

The normal collection setup has a fixed, level table model.  A Charuco
preflight can replace that model for one run with a measured plane in the
robot-base frame.  This module keeps that runtime value explicit instead of
mutating the legacy module-level table constants.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np


DEFAULT_TABLE_SURFACE_Z = 0.040
DEFAULT_TABLE_THICKNESS_Z = 0.20


def table_surface_z(tabletop_geometry: Mapping | None) -> float:
    """Return the session's level tabletop height in the robot-base frame.

    The empty-board preflight rejects a materially tilted board.  Once it is
    accepted, execution intentionally uses one constant height instead of
    following sub-millimetre fitted-plane slope across a placement stroke.
    ``plane_point_robot_m`` is retained as backward-compatible input for an
    older saved measurement that predates ``table_surface_z_m``.
    """
    if tabletop_geometry is None:
        return DEFAULT_TABLE_SURFACE_Z
    if "table_surface_z_m" in tabletop_geometry:
        return float(tabletop_geometry["table_surface_z_m"])
    point = np.asarray(tabletop_geometry["plane_point_robot_m"], dtype=np.float64)
    return float(point.reshape(3)[2])


def table_surface_z_at_xy(tabletop_geometry: Mapping | None, x: float, y: float) -> float:
    """Level tabletop height; ``x`` and ``y`` are accepted for API stability."""
    del x, y
    return table_surface_z(tabletop_geometry)


def table_cuboid(tabletop_geometry: Mapping | None,
                 *, thickness_m: float = DEFAULT_TABLE_THICKNESS_Z,
                 dims_xy: tuple[float, float] = (2.0, 3.0)) -> dict:
    """Return the cuRobo cuboid whose upper face is the measured level surface.

    The default branch deliberately reproduces the historical level-table
    cuboid byte-for-byte in meaning.  A successful Charuco preflight updates
    only its level top-face height; the measured normal is a quality check,
    not an execution-frame rotation.
    """
    table_z = table_surface_z(tabletop_geometry)
    return {
        "dims": [float(dims_xy[0]), float(dims_xy[1]), float(thickness_m)],
        "pose": [1.1, 0.0, table_z - float(thickness_m) / 2.0,
                 1.0, 0.0, 0.0, 0.0],
    }
