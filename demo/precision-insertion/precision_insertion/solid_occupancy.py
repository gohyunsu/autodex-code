"""Watertight-mesh solid containment for offline endpoint screening.

Coal's triangle BVH detects intersecting *surfaces* but can miss a small
entirely enclosed hand link. Trimesh's convenient ``mesh.contains`` has also
misclassified points in the blind cylindrical socket on this host because a
ray can hit the two triangles of one facet at the same distance. This module
counts unique ray-crossing distances along two oblique directions instead.
Any directional disagreement is treated as occupied (fail closed).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class OccupancyResult:
    inside_vertices: int
    ambiguous_vertices: int

    @property
    def intersects_solid(self) -> bool:
        return self.inside_vertices > 0 or self.ambiguous_vertices > 0


class SolidMeshOccupancy:
    """Classify points against a watertight fixed triangle mesh.

    This complements surface-intersection testing; it is not a substitute for
    the Coal triangle check, which catches crossings with all vertices out.
    """

    _DIRECTIONS = (
        np.array([0.973, 0.187, 0.123], dtype=np.float64),
        np.array([0.469, 0.361, 0.806], dtype=np.float64),
    )

    def __init__(self, mesh):
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError("solid occupancy requires a watertight socket mesh")
        self._triangles = np.asarray(mesh.triangles, dtype=np.float64)
        self._tree = mesh.triangles_tree

    def _parity(self, points: np.ndarray, direction: np.ndarray) -> np.ndarray:
        from trimesh.ray.ray_triangle import ray_triangle_id

        result = np.zeros(len(points), dtype=bool)
        direction = direction / np.linalg.norm(direction)
        for start in range(0, len(points), 10000):
            block = points[start:start + 10000]
            _, ray_id, locations = ray_triangle_id(
                self._triangles, block,
                np.tile(direction, (len(block), 1)), tree=self._tree)
            if len(ray_id) == 0:
                continue
            distance = np.einsum(
                "ij,j->i", locations - block[ray_id], direction)
            forward = distance > 1e-9
            ray_id = ray_id[forward]
            distance = distance[forward]
            if len(ray_id) == 0:
                continue
            order = np.lexsort((distance, ray_id))
            ray_id = ray_id[order]
            distance = distance[order]
            unique = np.ones(len(ray_id), dtype=bool)
            unique[1:] = ((ray_id[1:] != ray_id[:-1]) |
                          (distance[1:] - distance[:-1] > 1e-7))
            crossings = np.bincount(ray_id[unique], minlength=len(block))
            result[start:start + len(block)] = crossings % 2 == 1
        return result

    def classify(self, points: np.ndarray) -> OccupancyResult:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("solid occupancy expects finite Nx3 points")
        a = self._parity(values, self._DIRECTIONS[0])
        b = self._parity(values, self._DIRECTIONS[1])
        return OccupancyResult(
            inside_vertices=int(np.count_nonzero(a & b)),
            ambiguous_vertices=int(np.count_nonzero(a ^ b)),
        )


class CylinderSocketOccupancy:
    """Conservative analytic solid for the validated r15-h80 socket family.

    The printed CAD has a 5 mm circular base of radius 60 mm and a 5 mm wall
    from z=5 to 55 mm around its listed bore. Each circular cross-section is
    a 256-gon. The analytic radial margins include that polygon's sagitta,
    so a vertex close enough to the surface to be ambiguous fails closed.
    """

    def __init__(self, mesh, geometry: dict):
        bore = float(geometry["socket_bore_radius_m"])
        floor = float(geometry["socket_bore_bottom_z_m"])
        rim = float(geometry["socket_rim_z_m"])
        outer = bore + 0.005
        base = 0.06
        expected_volume = math.pi * (
            base**2 * floor + (outer**2 - bore**2) * (rim - floor))
        expected_bounds = np.array([[-base, -base, 0.0],
                                    [base, base, rim]])
        if (not mesh.is_watertight or not mesh.is_winding_consistent or
                len(mesh.faces) != 3072 or
                not np.allclose(mesh.bounds, expected_bounds, atol=1e-6) or
                not math.isclose(mesh.volume, expected_volume, rel_tol=0.001)):
            raise ValueError("socket mesh does not match validated cylinder CAD")
        self._bore = bore
        self._outer = outer
        self._base = base
        self._floor = floor
        self._rim = rim
        self._sagitta = outer * (1.0 - math.cos(math.pi / 256.0))

    def classify(self, points: np.ndarray) -> OccupancyResult:
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
            raise ValueError("solid occupancy expects finite Nx3 points")
        radius = np.linalg.norm(values[:, :2], axis=1)
        z = values[:, 2]
        epsilon = 1e-6
        base = ((z >= -epsilon) & (z <= self._floor + epsilon) &
                (radius <= self._base + epsilon))
        wall = ((z >= self._floor - epsilon) & (z <= self._rim + epsilon) &
                (radius >= self._bore - self._sagitta - epsilon) &
                (radius <= self._outer + epsilon))
        return OccupancyResult(
            inside_vertices=int(np.count_nonzero(base | wall)),
            ambiguous_vertices=0,
        )
