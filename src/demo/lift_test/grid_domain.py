"""Geometry-only helpers for constructed tabletop lift-feasibility grids."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import numpy as np

from src.demo.lift_test.board import point_in_polygon, polygon_centroid


@dataclass(frozen=True)
class GridSpec:
    """Definition of an XY grid in the measured Charuco-proxy frame."""

    step_m: float = 0.01
    # The Charuco rectangle is an experimental position proxy, not a physical
    # object-support boundary.  Map every object-centre location in it by
    # default; full-footprint containment remains available for a stricter
    # edge-exclusion experiment.
    domain: str = "center-only"
    edge_clearance_m: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    xy_m: tuple[float, float]
    domain_valid: bool

    def as_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "x_m": self.xy_m[0],
            "y_m": self.xy_m[1],
            "domain_valid": self.domain_valid,
        }


def _as_hull(points_xy: np.ndarray) -> np.ndarray:
    """Return a small convex hull sufficient for convex-board containment."""
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        raise ValueError("object mesh has no vertices")
    # A 2D hull keeps per-cell board containment checks independent of mesh
    # tessellation density.  scipy is already required by jacobian_lift.
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        return pts[np.asarray(hull.vertices, dtype=int)]
    except Exception:
        # Degenerate/near-flat XY meshes are unusual but still have a useful
        # footprint.  Unique vertices preserve a correct, conservative test.
        return np.unique(pts, axis=0)


def tabletop_footprint_xy(mesh_vertices: np.ndarray,
                           tabletop_transform_zero_xy: np.ndarray) -> np.ndarray:
    """Projected object footprint relative to the requested grid origin.

    ``tabletop_transform_zero_xy`` is the selected tabletop pose with grid
    translation and table Z omitted.  Adding a cell XY to this returned hull
    produces the object's XY footprint at that cell.
    """
    vertices = np.asarray(mesh_vertices, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(tabletop_transform_zero_xy, dtype=np.float64).reshape(4, 4)
    world = (T @ np.c_[vertices, np.ones(len(vertices))].T).T[:, :2]
    return _as_hull(world)


def _point_to_segment_distance(point: np.ndarray, start: np.ndarray,
                               end: np.ndarray) -> float:
    edge = end - start
    denom = float(edge @ edge)
    if denom <= 1.0e-16:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip(((point - start) @ edge) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + alpha * edge)))


def _inside_with_clearance(point: np.ndarray, polygon: np.ndarray,
                           clearance_m: float) -> bool:
    if not point_in_polygon(point, polygon):
        return False
    if clearance_m <= 0.0:
        return True
    return all(
        _point_to_segment_distance(point, a, b) >= clearance_m - 1.0e-10
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0))
    )


def cell_is_in_domain(xy: np.ndarray, *, proxy_vertices_xy: np.ndarray,
                      footprint_xy: np.ndarray, spec: GridSpec) -> bool:
    """Check the configured centre-only or full-footprint domain contract."""
    point = np.asarray(xy, dtype=np.float64).reshape(2)
    polygon = np.asarray(proxy_vertices_xy, dtype=np.float64).reshape(4, 2)
    if spec.domain == "center-only":
        return _inside_with_clearance(point, polygon, spec.edge_clearance_m)
    if spec.domain != "footprint-inside":
        raise ValueError(f"unsupported grid domain: {spec.domain}")
    footprint = np.asarray(footprint_xy, dtype=np.float64).reshape(-1, 2)
    return all(_inside_with_clearance(point + vertex, polygon, spec.edge_clearance_m)
               for vertex in footprint)


def _axis_values(*, center: float, lower: float, upper: float,
                 step_m: float) -> list[float]:
    if step_m <= 0.0:
        raise ValueError("grid step must be positive")
    k_min = int(np.ceil((lower - center) / step_m - 1.0e-10))
    k_max = int(np.floor((upper - center) / step_m + 1.0e-10))
    return [float(center + k * step_m) for k in range(k_min, k_max + 1)]


def make_grid(proxy: Mapping[str, object], *, footprint_xy: np.ndarray,
              spec: GridSpec) -> list[GridCell]:
    """Create a centre-aligned rectangular grid covering the proxy bounds.

    Out-of-domain cells are retained so reports show the actual board shape,
    but only domain-valid cells should enter planning.
    """
    if spec.step_m <= 0.0:
        raise ValueError("--grid-step-m must be positive")
    if spec.edge_clearance_m < 0.0:
        raise ValueError("--edge-clearance-m must be non-negative")
    vertices = np.asarray(proxy["vertices_xy_m"], dtype=np.float64).reshape(4, 2)
    center = np.asarray(proxy.get("center_xy_m", polygon_centroid(vertices)), dtype=np.float64)
    xs = _axis_values(center=float(center[0]), lower=float(vertices[:, 0].min()),
                      upper=float(vertices[:, 0].max()), step_m=spec.step_m)
    ys = _axis_values(center=float(center[1]), lower=float(vertices[:, 1].min()),
                      upper=float(vertices[:, 1].max()), step_m=spec.step_m)
    if not xs or not ys:
        raise ValueError("grid has no cells inside proxy bounds")
    cells: list[GridCell] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            xy = np.array([x, y], dtype=np.float64)
            cells.append(GridCell(
                row=row, col=col, xy_m=(float(x), float(y)),
                domain_valid=cell_is_in_domain(
                    xy, proxy_vertices_xy=vertices, footprint_xy=footprint_xy, spec=spec)))
    return cells


def grid_summary(cells: Iterable[GridCell], spec: GridSpec) -> dict:
    items = list(cells)
    if not items:
        return {"grid_spec": spec.as_dict(), "cell_count": 0, "valid_cell_count": 0}
    return {
        "grid_spec": spec.as_dict(),
        "cell_count": len(items),
        "valid_cell_count": sum(cell.domain_valid for cell in items),
        "row_count": max(cell.row for cell in items) + 1,
        "col_count": max(cell.col for cell in items) + 1,
    }
