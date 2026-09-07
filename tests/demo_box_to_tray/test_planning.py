"""Scene-hook tests: what the planner actually sees for a box pick."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

trimesh = pytest.importorskip("trimesh")

from src.demo.box_to_tray.planning import make_scene_hook, object_pose_robot  # noqa: E402
from tests.demo_box_to_tray.test_fixtures import _open_box_mesh, _pose  # noqa: E402
from src.demo.box_to_tray.fixtures import build_fixture  # noqa: E402



def _fixture(name="box", pose=None):
    return build_fixture(name, "open_box", "unused.obj", pose if pose is not None else _pose(),
                         mesh=_open_box_mesh())


def _scene_cfg(pose_xyz=(0.60, 0.0, 0.12)):
    """Minimal scene_cfg in the shape ``pose_world_to_scene_cfg`` returns."""
    return {
        "mesh": {"target": {"pose": [*pose_xyz, 1.0, 0.0, 0.0, 0.0],
                            "file_path": "/does/not/exist.obj"}},
        "cuboid": {"table": {"dims": [2, 3, 0.2], "pose": [1.1, 0, -0.06, 1, 0, 0, 0]}},
    }


def _cube_vertices(half=0.03):
    """A 6 cm cube centred on the object frame, so its bottom is -half."""
    return np.array([[sx * half, sy * half, sz * half]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)


def test_hook_adds_the_container_mesh_and_leaves_a_well_placed_pose_alone():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    # Cube bottom at 0.12 - 0.03 = 0.09 m, above the 0.055 m box floor.
    cfg = hook(_scene_cfg((0.60, 0.0, 0.12)))
    # Default model is the container's own mesh: exact for a round bowl, where
    # the fitted boxes would block the approach they are meant to guard.
    assert "box" in cfg["mesh"] and cfg["mesh"]["box"]["file_path"] == "unused.obj"
    assert cfg["mesh"]["target"]["file_path"] == "/does/not/exist.obj"
    assert "table" in cfg["cuboid"] and not any(k.startswith("box_wall") for k in cfg["cuboid"])
    np.testing.assert_allclose(object_pose_robot(cfg)[:3, 3], [0.60, 0.0, 0.12])


def test_cuboid_model_adds_the_fitted_walls_instead():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture, model="cuboid",
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    cfg = hook(_scene_cfg((0.60, 0.0, 0.12)))
    assert "box_wall_y_neg" in cfg["cuboid"] and "box" not in cfg.get("mesh", {})


def test_both_model_adds_mesh_and_walls():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture, model="both",
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    cfg = hook(_scene_cfg((0.60, 0.0, 0.12)))
    assert "box" in cfg["mesh"] and "box_wall_y_neg" in cfg["cuboid"]


def test_hook_raises_a_pose_that_sank_below_the_measured_box_floor():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    # Cube bottom 1 cm under the box floor (floor 0.055, bottom 0.045).
    cfg = hook(_scene_cfg((0.60, 0.0, 0.075)))
    assert object_pose_robot(cfg)[2, 3] == pytest.approx(fixture.floor_z + 0.03, abs=1e-6)


def test_hook_refuses_a_correction_larger_than_the_snap_limit():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture, snap_max_m=0.005,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    cfg = hook(_scene_cfg((0.60, 0.0, 0.075)))
    assert object_pose_robot(cfg)[2, 3] == pytest.approx(0.075)


def test_hook_rejects_an_object_outside_the_box_before_any_motion():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    with pytest.raises(RuntimeError, match="outside the measured box interior"):
        hook(_scene_cfg((0.60, 0.40, 0.10)))       # beside the box, on the table


def test_hook_rejects_a_pose_sitting_on_the_box_rim():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    with pytest.raises(RuntimeError, match="outside the measured box interior"):
        hook(_scene_cfg((0.60 + 0.145, 0.0, 0.10)))


def test_table_source_skips_the_interior_check_but_keeps_the_walls():
    """A step that picks off the table still has to avoid every container."""
    box = _fixture("box")
    bowl = _fixture("bowl", pose=_pose(x=0.45, y=0.30))
    hook = make_scene_hook([box, bowl], source=None,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    cfg = hook(_scene_cfg((0.55, 0.20, 0.07)))       # on the table, in neither
    assert {"box", "bowl"} <= set(cfg["mesh"])
    # The table-standing pose is left exactly as perceived: only a container
    # source applies its floor correction.
    np.testing.assert_allclose(object_pose_robot(cfg)[:3, 3], [0.55, 0.20, 0.07])


def test_target_container_is_an_obstacle_for_a_table_pick():
    bowl = _fixture("bowl", pose=_pose(x=0.45, y=0.30))
    hook = make_scene_hook([bowl], source=None,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    cfg = hook(_scene_cfg((0.65, -0.10, 0.07)))
    assert "bowl" in cfg["mesh"]


def test_source_check_can_be_disabled():
    fixture = _fixture()
    hook = make_scene_hook([fixture], source=fixture, require_in_source=False,
                           load_vertices=lambda _p: _cube_vertices(), report=None)
    cfg = hook(_scene_cfg((0.60, 0.40, 0.10)))
    assert "box" in cfg["mesh"]
