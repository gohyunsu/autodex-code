"""Small exact mesh helpers shared by the cylindrical asset builder."""

from __future__ import annotations

import math

import numpy as np

from build_assets import Mesh, validate_watertight


def cylinder_mesh(radius: float, height: float, *, segments: int = 256) -> Mesh:
    vertices = []
    for z in (0.0, float(height)):
        vertices.extend(
            (radius * math.cos(2.0 * math.pi * i / segments),
             radius * math.sin(2.0 * math.pi * i / segments), z)
            for i in range(segments)
        )
    vertices.extend([(0.0, 0.0, 0.0), (0.0, 0.0, float(height))])
    bottom_center = 2 * segments
    top_center = bottom_center + 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.extend([
            (i, j, segments + j),
            (i, segments + j, segments + i),
            (bottom_center, j, i),
            (top_center, segments + i, segments + j),
        ])
    mesh = Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    validate_watertight(mesh)
    return mesh
