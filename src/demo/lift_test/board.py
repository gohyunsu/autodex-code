"""Charuco-board proxy geometry for the lift-test session.

The proxy is the complete checkerboard rectangle, not the smaller rectangle
joining the detected internal Charuco corners.  Board-11 has a 10 x 7 square
grid: its 9 x 6 detected internal-corner lattice is inset by one 5 cm square
from every physical checkerboard edge.  The board fit still uses only those
detected corners; the fitted board frame lets us recover the full rectangle
without extrapolating noisy world-frame extrema.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _as_ccw(vertices_xy: np.ndarray) -> np.ndarray:
    """Validate a convex quadrilateral and return it in counter-clockwise order."""
    v = np.asarray(vertices_xy, dtype=np.float64).reshape(4, 2)
    if not np.isfinite(v).all():
        raise ValueError("board proxy vertices contain a non-finite value")
    center = v.mean(axis=0)
    angle = np.arctan2(v[:, 1] - center[1], v[:, 0] - center[0])
    v = v[np.argsort(angle)]
    cross = []
    for i in range(4):
        a = v[(i + 1) % 4] - v[i]
        b = v[(i + 2) % 4] - v[(i + 1) % 4]
        cross.append(float(a[0] * b[1] - a[1] * b[0]))
    if min(cross) <= 1e-10:
        raise ValueError("outer Charuco corners do not form a convex quadrilateral")
    return v


def polygon_centroid(vertices_xy: np.ndarray) -> np.ndarray:
    """Area centroid, rather than an only-valid-for-parallelograms point mean."""
    v = _as_ccw(vertices_xy)
    nxt = np.roll(v, -1, axis=0)
    cross = v[:, 0] * nxt[:, 1] - nxt[:, 0] * v[:, 1]
    area2 = float(cross.sum())
    if abs(area2) < 1e-12:
        raise ValueError("board proxy has zero area")
    return np.array([
        ((v[:, 0] + nxt[:, 0]) * cross).sum() / (3.0 * area2),
        ((v[:, 1] + nxt[:, 1]) * cross).sum() / (3.0 * area2),
    ])


def point_in_polygon(point_xy: np.ndarray, vertices_xy: np.ndarray,
                     tol: float = 1e-9) -> bool:
    """Inclusive point-in-convex-polygon test for CCW or CW input."""
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    v = _as_ccw(vertices_xy)
    edges = np.roll(v, -1, axis=0) - v
    rel = p[None, :] - v
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    return bool(np.all(cross >= -tol))


def y_interval_at_x(vertices_xy: np.ndarray, x: float) -> tuple[float, float] | None:
    """Return the usable y interval of a convex polygon at a selected x."""
    v = _as_ccw(vertices_xy)
    ys: list[float] = []
    for a, b in zip(v, np.roll(v, -1, axis=0)):
        lo, hi = sorted((float(a[0]), float(b[0])))
        if x < lo - 1e-10 or x > hi + 1e-10:
            continue
        if abs(float(b[0] - a[0])) < 1e-10:
            if abs(x - float(a[0])) < 1e-10:
                ys.extend([float(a[1]), float(b[1])])
            continue
        t = (x - float(a[0])) / float(b[0] - a[0])
        if -1e-10 <= t <= 1.0 + 1e-10:
            ys.append(float(a[1] + t * (b[1] - a[1])))
    if len(ys) < 2:
        return None
    return min(ys), max(ys)


def _uniform_grid_step(values: np.ndarray, *, axis: str) -> float:
    """Return one Charuco lattice interval and reject malformed board models."""
    unique = np.unique(np.asarray(values, dtype=np.float64).reshape(-1))
    deltas = np.diff(unique)
    if len(deltas) == 0 or not np.isfinite(deltas).all() or np.any(deltas <= 0.0):
        raise ValueError(f"Charuco board has no usable {axis}-axis lattice spacing")
    step = float(np.median(deltas))
    if not np.allclose(deltas, step, rtol=1.0e-6, atol=1.0e-10):
        raise ValueError(f"Charuco board has non-uniform {axis}-axis lattice spacing")
    return step


def _charuco_local_geometry() -> tuple[np.ndarray, np.ndarray, list[int], dict[str, float]]:
    """Return board-11's fitted internal model and full outer checkerboard.

    OpenCV's Charuco model contains only the internal intersections.  A
    complete checker square surrounds that lattice on each side, so extending
    by one lattice interval gives the actual checkerboard boundary.  This is
    deliberately based on the configured board model rather than a hard-coded
    50 cm x 35 cm number.
    """
    from src.execution.charuco_tabletop import _board_corner_model

    model = np.asarray(_board_corner_model(), dtype=np.float64).reshape(-1, 3)
    umin, umax = float(model[:, 0].min()), float(model[:, 0].max())
    vmin, vmax = float(model[:, 1].min()), float(model[:, 1].max())
    du = _uniform_grid_step(model[:, 0], axis="x")
    dv = _uniform_grid_step(model[:, 1], axis="y")
    internal = ((umin, vmin), (umax, vmin), (umax, vmax), (umin, vmax))
    ids = [int(np.argmin(np.sum((model[:, :2] - np.asarray(uv)) ** 2, axis=1)))
           for uv in internal]
    outer = np.array([
        [umin - du, vmin - dv, 0.0],
        [umax + du, vmin - dv, 0.0],
        [umax + du, vmax + dv, 0.0],
        [umin - du, vmax + dv, 0.0],
    ], dtype=np.float64)
    return model, outer, ids, {
        "internal_u_span_m": umax - umin,
        "internal_v_span_m": vmax - vmin,
        "square_u_m": du,
        "square_v_m": dv,
        "board_u_m": float(outer[:, 0].max() - outer[:, 0].min()),
        "board_v_m": float(outer[:, 1].max() - outer[:, 1].min()),
    }


def proxy_from_charuco_measurement(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """Build the board proxy from a successful standard Charuco measurement.

    ``measure_tabletop_from_images`` estimates ``T_robot_board`` from all 54
    internal corners.  We extend the known Charuco *local* grid by one square
    on every side, then transform that full checkerboard boundary with the
    fitted transform.  Choosing extrema in noisy world XY could otherwise
    select the wrong neighbouring corner.
    """
    if "T_robot_board" not in measurement:
        raise ValueError("Charuco measurement lacks T_robot_board")
    _, outer_local, ids, geometry = _charuco_local_geometry()
    T = np.asarray(measurement["T_robot_board"], dtype=np.float64).reshape(4, 4)
    local_h = np.c_[outer_local, np.ones(4)]
    robot = (T @ local_h.T).T[:, :3]
    vertices = _as_ccw(robot[:, :2])
    return {
        "schema_version": 2,
        "kind": "charuco_full_checkerboard_proxy",
        "frame": "robot_base",
        "charuco_board": str(measurement.get("board", "11")),
        "internal_corner_ids": ids,
        "corner_ids": ids,  # Retained for consumers of schema-version 1 artifacts.
        "board_geometry_local_m": geometry,
        "vertices_xy_m": vertices.tolist(),
        "center_xy_m": polygon_centroid(vertices).tolist(),
        "xy_bounds_m": {
            "x": [float(vertices[:, 0].min()), float(vertices[:, 0].max())],
            "y": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
        },
        "table_surface_z_m": float(measurement["table_surface_z_m"]),
    }


def _expand_legacy_internal_proxy(vertices_xy: np.ndarray) -> np.ndarray:
    """Upgrade a board-11 internal-corner quadrilateral to full-board XY.

    Older saved ``board_proxy.json`` files did not retain ``T_robot_board``.
    Their quadrilateral is nevertheless the affine projection of the 9 x 6
    internal lattice.  Board-11's longer side is the 9-interval (x) direction,
    which identifies the expansion factors without a new camera measurement.
    """
    _, _, _, geometry = _charuco_local_geometry()
    u_factor = ((geometry["internal_u_span_m"] + 2.0 * geometry["square_u_m"])
                / geometry["internal_u_span_m"])
    v_factor = ((geometry["internal_v_span_m"] + 2.0 * geometry["square_v_m"])
                / geometry["internal_v_span_m"])
    v = _as_ccw(vertices_xy)
    center = v.mean(axis=0)
    half_first = (v[1] - v[0]) * 0.5
    half_second = (v[3] - v[0]) * 0.5
    if float(np.linalg.norm(half_first)) >= float(np.linalg.norm(half_second)):
        first_factor, second_factor = u_factor, v_factor
    else:
        first_factor, second_factor = v_factor, u_factor
    expanded = np.stack((
        center - first_factor * half_first - second_factor * half_second,
        center + first_factor * half_first - second_factor * half_second,
        center + first_factor * half_first + second_factor * half_second,
        center - first_factor * half_first + second_factor * half_second,
    ))
    return _as_ccw(expanded)


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def load_proxy(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    vertices = _as_ccw(np.asarray(data["vertices_xy_m"], dtype=np.float64))
    if data.get("kind") == "charuco_internal_corner_proxy":
        # Make an old session/grid proxy honour the same full-board contract as
        # a newly measured board.  Preserve the original polygon for audit.
        data["legacy_internal_vertices_xy_m"] = vertices.tolist()
        vertices = _expand_legacy_internal_proxy(vertices)
        data["schema_version"] = 2
        data["kind"] = "charuco_full_checkerboard_proxy"
        data["upgraded_from_kind"] = "charuco_internal_corner_proxy"
    data["vertices_xy_m"] = vertices.tolist()
    data["center_xy_m"] = polygon_centroid(vertices).tolist()
    data["xy_bounds_m"] = {
        "x": [float(vertices[:, 0].min()), float(vertices[:, 0].max())],
        "y": [float(vertices[:, 1].min()), float(vertices[:, 1].max())],
    }
    if "table_surface_z_m" not in data:
        raise ValueError("board proxy file lacks table_surface_z_m")
    return data
