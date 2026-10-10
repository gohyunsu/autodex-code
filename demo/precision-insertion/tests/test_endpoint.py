"""Exact-mesh endpoint and 20 mm CAD transform contracts."""

from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.endpoint import (  # noqa: E402
    _validate_geometry,
)


def _geometry():
    entry = np.diag([1.0, -1.0, -1.0, 1.0])
    entry[2, 3] = 0.144
    verification = entry.copy()
    verification[2, 3] -= 0.020
    return {
        "units": "m",
        "socket_pose_object": "precision_socket_unified",
        "T_socket_pose_object": np.eye(4).tolist(),
        "T_socket_key_entry": entry.tolist(),
        "T_socket_key_verification": verification.tolist(),
        "verification_insertion_depth_m": 0.020,
        "insertion_direction_socket": [0.0, 0.0, -1.0],
        "key_frame": {"insertion_axis": [0.0, 0.0, 1.0]},
    }


def test_verification_contract_accepts_centered_axial_20mm():
    target = _validate_geometry(_geometry(), select_mode("square", 1.5))
    assert target[2, 3] == pytest.approx(0.124)


def test_verification_contract_rejects_lateral_offset_and_wrong_depth():
    geometry = _geometry()
    geometry["T_socket_key_verification"][0][3] = 0.001
    with pytest.raises(ValueError, match="centered and axially aligned"):
        _validate_geometry(geometry, select_mode("square", 1.5))
    geometry = _geometry()
    geometry["verification_insertion_depth_m"] = 0.002
    with pytest.raises(ValueError, match="20 mm"):
        _validate_geometry(geometry, select_mode("square", 1.5))


def test_triangle_mesh_collision_and_clearance_are_distinct():
    code = "\n".join([
        "import coal, json, numpy as np, trimesh, sys",
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})",
        "from precision_insertion.endpoint import _coal_mesh, _mesh_pair_report",
        "fixed = trimesh.creation.box(extents=(1.0, 1.0, 1.0))",
        "moving = trimesh.creation.box(extents=(0.1, 0.1, 0.1))",
        "model = _coal_mesh(fixed)",
        "T = np.eye(4); T[2, 3] = 0.60",
        "clear = _mesh_pair_report(moving, model, T)",
        "T[2, 3] = 0.52",
        "overlap = _mesh_pair_report(moving, model, T)",
        "print(json.dumps({'clear': clear, 'overlap': overlap}))",
    ])
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    data = json.loads(run.stdout)
    clear = data["clear"]
    assert clear["colliding"] is False
    assert clear["minimum_surface_distance_m"] == pytest.approx(0.05, abs=1e-6)
    overlap = data["overlap"]
    assert overlap["colliding"] is True
    assert overlap["minimum_surface_distance_m"] == pytest.approx(0.0, abs=1e-8)
