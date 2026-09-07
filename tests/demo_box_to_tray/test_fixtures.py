"""Box measurement tests: a synthetic open box stands in for the real asset."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

trimesh = pytest.importorskip("trimesh")

from src.demo.box_to_tray.fixtures import (  # noqa: E402
    ContainerFixture,
    add_container_cuboids,
    build_fixture,
    container_obstacles,
    footprint_rect,
    measure_container,
    point_in_region,
)

# Outer 0.30 x 0.24 x 0.20 m, 1 cm walls, 1.5 cm floor slab: the mesh origin is
# its centre, like the processed AutoDex assets.
OUTER = (0.30, 0.24, 0.20)
WALL = 0.01
FLOOR = 0.015


def _open_box_mesh():
    ox, oy, oz = OUTER
    parts = [
        trimesh.creation.box(extents=(ox, oy, FLOOR),
                             transform=trimesh.transformations.translation_matrix(
                                 (0, 0, -oz / 2 + FLOOR / 2))),
        trimesh.creation.box(extents=(WALL, oy, oz),
                             transform=trimesh.transformations.translation_matrix(
                                 ((ox - WALL) / 2, 0, 0))),
        trimesh.creation.box(extents=(WALL, oy, oz),
                             transform=trimesh.transformations.translation_matrix(
                                 (-(ox - WALL) / 2, 0, 0))),
        trimesh.creation.box(extents=(ox - 2 * WALL, WALL, oz),
                             transform=trimesh.transformations.translation_matrix(
                                 (0, (oy - WALL) / 2, 0))),
        trimesh.creation.box(extents=(ox - 2 * WALL, WALL, oz),
                             transform=trimesh.transformations.translation_matrix(
                                 (0, -(oy - WALL) / 2, 0))),
    ]
    return trimesh.util.concatenate(parts)


def _pose(x=0.60, y=0.0, yaw=0.0, table_z=0.04):
    """Box standing on the table at ``table_z`` with its base flat on it."""
    T = np.eye(4)
    T[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    T[:3, 3] = [x, y, table_z + OUTER[2] / 2]
    return T


def test_measure_recovers_the_outer_and_interior_footprints():
    m = measure_container(_open_box_mesh(), _pose(), interior_inset=0.0)
    assert m["outer_size_xy"][0] == pytest.approx(OUTER[0], abs=1e-6)
    assert m["outer_size_xy"][1] == pytest.approx(OUTER[1], abs=1e-6)
    # The interior is sampled on a probe grid, so it is at most the true cavity
    # and never reaches into a wall.
    assert m["interior_size_xy"][0] <= OUTER[0] - 2 * WALL + 1e-9
    assert m["interior_size_xy"][1] <= OUTER[1] - 2 * WALL + 1e-9
    assert m["interior_size_xy"][0] > OUTER[0] - 2 * WALL - 0.02
    assert m["interior_size_xy"][1] > OUTER[1] - 2 * WALL - 0.02
    assert m["base_z"] == pytest.approx(0.04, abs=1e-6)
    assert m["rim_z"] == pytest.approx(0.04 + OUTER[2], abs=1e-6)
    assert m["floor_z"] == pytest.approx(0.04 + FLOOR, abs=1e-6)


def test_measure_follows_a_rotated_box():
    yaw = math.radians(25.0)
    m = measure_container(_open_box_mesh(), _pose(yaw=yaw), interior_inset=0.0)
    assert abs(m["yaw"] - yaw) < 1e-6 or abs(abs(m["yaw"] - yaw) - math.pi / 2) < 1e-6
    # Footprints are stated in the box frame, so a rotation does not inflate them.
    assert m["outer_size_xy"][0] == pytest.approx(OUTER[0], abs=1e-6)
    assert m["outer_size_xy"][1] == pytest.approx(OUTER[1], abs=1e-6)
    # The frame comes from the posed geometry, not from the pose rotation, so
    # an asset whose local up axis is not z is measured just as correctly.
    tipped = _pose(yaw=yaw) @ trimesh.transformations.rotation_matrix(0.0, (1, 0, 0))
    assert measure_container(_open_box_mesh(), tipped,
                             interior_inset=0.0)["outer_size_xy"][0] == pytest.approx(
        OUTER[0], abs=1e-6)


def _stepped_bowl_mesh():
    """A curved cavity, approximated by concentric steps.

    A real bowl's floor rises towards its wall, so the probe heights inside it
    are spread over centimetres.  This is the shape that exposed the median
    floor estimate: the middle probe height sits halfway up the curve.
    """
    outer, height, base = 0.20, 0.09, 0.010
    parts = [trimesh.creation.box(
        extents=(outer, outer, base),
        transform=trimesh.transformations.translation_matrix((0, 0, base / 2)))]
    # Three rings: the outermost is the tallest, so the cavity gets deeper
    # towards its centre.
    for i, (side, top) in enumerate(((0.20, 0.09), (0.16, 0.055), (0.12, 0.03))):
        inner = side - 0.03
        for sign, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            dims = [0.0, 0.0, top - base]
            dims[axis] = 0.015
            dims[1 - axis] = side
            offset = [0.0, 0.0, base + (top - base) / 2]
            offset[axis] = sign * (side - 0.015) / 2
            parts.append(trimesh.creation.box(
                extents=tuple(dims),
                transform=trimesh.transformations.translation_matrix(tuple(offset))))
    return trimesh.util.concatenate(parts)


def test_curved_bowl_floor_is_its_bottom_not_the_middle_of_the_curve():
    mesh = _stepped_bowl_mesh()
    pose = np.eye(4)
    pose[:3, 3] = [0.6, 0.0, 0.04]
    m = measure_container(mesh, pose, interior_inset=0.0, floor_percentile=5.0)
    # The cavity bottom is the base slab's top face: 0.04 + 0.010.
    assert m["floor_z"] == pytest.approx(0.05, abs=0.003)
    # The median probe height sits well above it, which is what buried an
    # object resting on the real bottom inside the floor obstacle.
    median = measure_container(mesh, pose, interior_inset=0.0,
                               floor_percentile=50.0)["floor_z"]
    assert median > m["floor_z"] + 0.005
    # And the floor cuboid stays inside that bottom band.
    floor = container_obstacles(m)["box_floor"]
    assert floor["pose"][2] + floor["dims"][2] / 2 == pytest.approx(m["floor_z"], abs=1e-9)


def test_measure_rejects_a_mesh_with_no_interior():
    solid = trimesh.creation.box(extents=OUTER, transform=_pose())
    with pytest.raises(RuntimeError, match="no interior surface"):
        measure_container(solid, np.eye(4))


def test_walls_fill_the_outer_footprint_and_leave_the_cavity_free():
    m = measure_container(_open_box_mesh(), _pose(), interior_inset=0.0)
    cuboids = container_obstacles(m)
    assert set(cuboids) == {"box_wall_x_pos", "box_wall_x_neg", "box_wall_y_pos",
                            "box_wall_y_neg", "box_floor"}
    for name, cuboid in cuboids.items():
        dims = np.asarray(cuboid["dims"])
        pose = np.asarray(cuboid["pose"][:3])
        lo, hi = pose - dims / 2, pose + dims / 2
        assert lo[0] >= 0.60 - OUTER[0] / 2 - 1e-6 and hi[0] <= 0.60 + OUTER[0] / 2 + 1e-6
        assert lo[2] >= 0.04 - 1e-6 and hi[2] <= 0.04 + OUTER[2] + 1e-6
        if name != "box_floor":
            # Every wall spans the full height, so the hand cannot clip a wall
            # top while climbing out with the object.
            assert dims[2] == pytest.approx(OUTER[2], abs=1e-6)
        centre = np.array([0.60, 0.0, 0.04 + OUTER[2] / 2])
        inside = np.all(centre >= lo - 1e-9) and np.all(centre <= hi + 1e-9)
        assert not inside, f"{name} fills the box cavity"


def test_wall_thickness_matches_the_measured_mesh():
    m = measure_container(_open_box_mesh(), _pose(), interior_inset=0.0)
    cuboids = container_obstacles(m)
    # Probe sampling can only over-estimate a wall, never cut into the cavity.
    for tag, axis in (("x_pos", 0), ("x_neg", 0), ("y_pos", 1), ("y_neg", 1)):
        thickness = cuboids[f"box_wall_{tag}"]["dims"][axis]
        assert WALL - 1e-6 <= thickness <= WALL + 0.012


def test_interior_region_accepts_the_cavity_and_rejects_the_rim():
    fixture = build_fixture("box", "open_box", "unused.obj", _pose(),
                            mesh=_open_box_mesh(), interior_inset=0.015)
    region = fixture.interior_region
    assert point_in_region([0.60, 0.0, 0.10], region)
    assert not point_in_region([0.60 + OUTER[0] / 2 - 0.005, 0.0, 0.10], region)  # on the rim
    assert not point_in_region([0.60, 0.0, 0.04], region)                        # under the floor
    inside = np.eye(4); inside[:3, 3] = [0.60, 0.0, 0.10]
    outside = np.eye(4); outside[:3, 3] = [0.60, 0.35, 0.10]
    assert fixture.contains(inside) and not fixture.contains(outside)
    assert fixture.floor_z == pytest.approx(0.04 + FLOOR, abs=1e-6)
    assert fixture.rim_z == pytest.approx(0.04 + OUTER[2], abs=1e-6)


def test_fixture_json_round_trip():
    fixture = build_fixture("box", "open_box", "unused.obj", _pose(yaw=0.3),
                            mesh=_open_box_mesh(), perception={"n_views": 12})
    restored = ContainerFixture.from_json(fixture.to_json())
    np.testing.assert_allclose(restored.pose_robot, fixture.pose_robot)
    assert restored.obstacles == fixture.obstacles
    assert restored.floor_z == fixture.floor_z
    assert restored.perception == {"n_views": 12}


def test_add_box_cuboids_keeps_the_object_and_the_table():
    fixture = build_fixture("box", "open_box", "unused.obj", _pose(), mesh=_open_box_mesh())
    cfg = {"mesh": {"target": {"pose": [0.6, 0, 0.1, 1, 0, 0, 0], "file_path": "o.obj"}},
           "cuboid": {"table": {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.06, 1, 0, 0, 0]}}}
    merged = add_container_cuboids(cfg, fixture)
    assert "table" in merged["cuboid"] and "box_wall_x_pos" in merged["cuboid"]
    assert set(cfg["cuboid"]) == {"table"}          # the input is not mutated
    with pytest.raises(ValueError, match="collides"):
        add_container_cuboids(merged, fixture)
