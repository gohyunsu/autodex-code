"""The socket bore is free space; its walls and blind floor are solid."""

import numpy as np
import coal  # noqa: F401  # Load its newer libstdc++ before trimesh on this host.
import trimesh

from precision_insertion.solid_occupancy import (
    CylinderSocketOccupancy, SolidMeshOccupancy,
)


def test_blind_socket_bore_wall_and_floor():
    profile = np.array([
        [0.0, 0.0], [0.06, 0.0], [0.06, 0.005],
        [0.021, 0.005], [0.021, 0.055], [0.016, 0.055],
        [0.016, 0.005], [0.0, 0.005],
    ])
    socket = trimesh.creation.revolve(profile, sections=32)
    occupancy = SolidMeshOccupancy(socket)
    free = np.array([[0.0, 0.0, 0.03], [0.035, 0.0, 0.03]])
    solid = np.array([[0.018, 0.0, 0.03], [0.0, 0.0, 0.002]])
    assert occupancy.classify(free).intersects_solid is False
    report = occupancy.classify(solid)
    assert report.inside_vertices == 2
    assert report.ambiguous_vertices == 0


def test_validated_cylinder_analytic_occupancy():
    profile = np.array([
        [0.0, 0.0], [0.06, 0.0], [0.06, 0.005],
        [0.021, 0.005], [0.021, 0.055], [0.016, 0.055],
        [0.016, 0.005], [0.0, 0.005],
    ])
    socket = trimesh.creation.revolve(profile, sections=256)
    occupancy = CylinderSocketOccupancy(socket, {
        "socket_bore_radius_m": 0.016,
        "socket_bore_bottom_z_m": 0.005,
        "socket_rim_z_m": 0.055,
    })
    free = np.array([[0.0, 0.0, 0.03], [0.035, 0.0, 0.03]])
    solid = np.array([[0.018, 0.0, 0.03], [0.0, 0.0, 0.002]])
    assert occupancy.classify(free).intersects_solid is False
    assert occupancy.classify(solid).inside_vertices == 2
