"""Offline tests for fixture and v8 axial-symmetry contracts."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.geometry import freeze_fixture_pose, validate_se3  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.symmetry import (  # noqa: E402
    load_axial_symmetry, snap_axisymmetric_tabletop_pose,
)
from precision_insertion.world import (  # noqa: E402
    add_fixed_mesh_fixtures, build_trial_scene_from_session,
)
from autodex.utils.conversion import se32cart  # noqa: E402
from autodex.utils.tabletop_geometry import table_cuboid  # noqa: E402


def _pose(*, x_mm=0.0, yaw_deg=0.0, tilt_deg=0.0):
    yaw = np.radians(yaw_deg)
    tilt = np.radians(tilt_deg)
    cy, sy, ct, st = np.cos(yaw), np.sin(yaw), np.cos(tilt), np.sin(tilt)
    result = np.eye(4)
    result[:3, :3] = (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    )
    result[0, 3] = x_mm / 1000.0
    return result


def _object(tmp_path, *, name="key", kind="Dinf", tabletop=None):
    info = tmp_path / name / "processed_data" / "info"
    poses = info / "tabletop"
    poses.mkdir(parents=True)
    data = {"type": kind, "axes": [{"axis": [0, 0, 1], "fold": "inf"}]}
    if kind == "Dinf":
        data["axes"].append({"axis": [1, 0, 0], "fold": 2})
    (info / "symmetry.json").write_text(json.dumps(data), encoding="utf-8")
    np.save(poses / "000.npy", np.eye(4) if tabletop is None else tabletop)
    return name


def test_fixture_medoid_is_observed_and_repeatable():
    poses = [_pose(x_mm=0.0), _pose(x_mm=0.4), _pose(x_mm=0.8)]
    selected, report = freeze_fixture_pose(
        poses, translation_limit_mm=1.0, angle_limit_deg=1.0)
    np.testing.assert_allclose(selected, poses[1])
    assert report["selected_index"] == 1
    assert report["accepted"] is True


def test_fixture_rejects_outlier_and_invalid_transform():
    with pytest.raises(ValueError, match="not repeatable"):
        freeze_fixture_pose(
            [_pose(), _pose(x_mm=0.2), _pose(x_mm=4.0)],
            translation_limit_mm=1.0, angle_limit_deg=1.0)
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        validate_se3(invalid)


def test_round_socket_ignores_yaw_but_rejects_axis_tilt():
    selected, report = freeze_fixture_pose(
        [_pose(yaw_deg=0), _pose(x_mm=0.2, yaw_deg=93),
         _pose(x_mm=0.4, yaw_deg=-151)],
        continuous_axis_local=[0, 0, 1],
        translation_limit_mm=1.0, angle_limit_deg=1.0)
    assert report["method"] == "observed_axisymmetric_se3_medoid"
    assert report["max_angle_residual_deg"] == pytest.approx(0.0)
    assert selected.shape == (4, 4)
    with pytest.raises(ValueError, match="not repeatable"):
        freeze_fixture_pose(
            [_pose(), _pose(tilt_deg=0.2), _pose(tilt_deg=4.0)],
            continuous_axis_local=[0, 0, 1],
            translation_limit_mm=1.0, angle_limit_deg=1.0)


def test_dinf_key_can_exchange_identical_ends(tmp_path):
    end_down = np.eye(4)
    end_down[:3, :3] = np.diag([1.0, -1.0, -1.0])
    name = _object(tmp_path, tabletop=end_down)
    result = snap_axisymmetric_tabletop_pose(
        np.eye(4), object_root=tmp_path, object_name=name)
    np.testing.assert_allclose(result[:3, 2], [0, 0, 1])
    assert load_axial_symmetry(tmp_path, name).end_exchange is True


def test_cinf_socket_cannot_flip_open_and_closed_ends(tmp_path):
    name = _object(tmp_path, kind="Cinf")
    upside_down = _pose()
    upside_down[:3, :3] = np.diag([1.0, -1.0, -1.0])
    with pytest.raises(ValueError, match="does not match"):
        snap_axisymmetric_tabletop_pose(
            upside_down, object_root=tmp_path, object_name=name)


def test_fixed_fixture_addition_is_copy_and_rejects_shadowing(tmp_path):
    mesh = tmp_path / "socket.obj"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    original = {"mesh": {"target": {"pose": [0] * 7,
                                       "file_path": "key.obj"}}}
    added = add_fixed_mesh_fixtures(original, {
        "fixture_socket": {"pose_robot": _pose(), "collision_mesh": mesh}
    })
    assert "fixture_socket" not in original["mesh"]
    assert added["mesh"]["fixture_socket"]["file_path"] == str(mesh)
    with pytest.raises(ValueError, match="invalid fixed fixture"):
        add_fixed_mesh_fixtures(original, {
            "target": {"pose_robot": _pose(), "collision_mesh": mesh}
        })


def _session(tmp_path, mode):
    mesh = (tmp_path / "object_processing" / mode.socket_object /
            "processed_data" / "mesh" / "static_collision.obj")
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    frozen = _pose(x_mm=12)
    board = {"table_surface_z_m": 0.043}
    fixed = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {"table": table_cuboid(board)}},
        {"fixture_socket": {"pose_robot": frozen, "collision_mesh": mesh}})
    record = {
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "c2r": np.eye(4).tolist(),
        "socket_pose_robot": frozen.tolist(),
        "socket_collision_mesh": str(mesh),
        "socket_collision_mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    }
    return SessionCalibration(board, frozen, {}, fixed, record), mesh


def test_trial_scene_reuses_frozen_table_and_socket_but_refreshes_key(
    tmp_path, monkeypatch,
):
    mode = select_mode("square", 1.5)
    session, _ = _session(tmp_path, mode)
    original = json.loads(json.dumps(session.collision_scene))
    seen = []

    def make_scene(pose_world, c2r, obj_name, *, obj_root, tabletop_geometry):
        seen.append((pose_world.copy(), obj_name, obj_root, tabletop_geometry))
        return {"mesh": {"target": {"pose": se32cart(pose_world).tolist(),
                                    "file_path": "fresh_key.obj"}},
                "cuboid": {"table": table_cuboid(tabletop_geometry)}}

    monkeypatch.setattr("src.execution.scene_cfg.pose_world_to_scene_cfg", make_scene)
    first = build_trial_scene_from_session(
        mode=mode, shared_root=tmp_path, calibration=session,
        key_pose_world=_pose(x_mm=100))
    second = build_trial_scene_from_session(
        mode=mode, shared_root=tmp_path, calibration=session,
        key_pose_world=_pose(x_mm=200))
    assert first["mesh"]["target"]["pose"][0] == pytest.approx(0.1)
    assert second["mesh"]["target"]["pose"][0] == pytest.approx(0.2)
    assert first["mesh"]["fixture_socket"] == second["mesh"]["fixture_socket"]
    assert first["cuboid"]["table"] == session.collision_scene["cuboid"]["table"]
    assert session.collision_scene == original
    assert all(row[1] == mode.key_object and
               row[2] == str(tmp_path / "object_processing") and
               row[3] is session.board for row in seen)


def test_trial_scene_rejects_wrong_socket_mode_or_changed_mesh(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    session, mesh = _session(tmp_path, mode)
    with pytest.raises(ValueError, match="does not match trial"):
        build_trial_scene_from_session(
            mode=select_mode("square", 1.0), shared_root=tmp_path,
            calibration=session, key_pose_world=np.eye(4))
    mesh.write_text("v 1 0 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from frozen session asset"):
        build_trial_scene_from_session(
            mode=mode, shared_root=tmp_path, calibration=session,
            key_pose_world=np.eye(4))


def test_trial_scene_rejects_inconsistent_frozen_socket_record(tmp_path):
    mode = select_mode("square", 1.5)
    session, _ = _session(tmp_path, mode)
    session.record["socket_pose_robot"] = np.eye(4).tolist()
    with pytest.raises(ValueError, match="record socket pose differs"):
        build_trial_scene_from_session(
            mode=mode, shared_root=tmp_path, calibration=session,
            key_pose_world=np.eye(4))


def test_trial_scene_uses_original_v8_scene_converter_with_measured_table(tmp_path):
    mode = select_mode("square", 1.5)
    session, _ = _session(tmp_path, mode)
    planning_mesh = (tmp_path / "object_processing" / mode.key_object /
                     "processed_data" / "mesh" / "simplified.obj")
    planning_mesh.parent.mkdir(parents=True)
    planning_mesh.write_text(
        "v 0 0 0\nv 0.01 0 0\nv 0 0.01 0\n"
        "v 0 0 0.01\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n",
        encoding="utf-8")
    key_pose = np.eye(4)
    key_pose[:3, 3] = [0.1, 0.2, 0.05]
    scene = build_trial_scene_from_session(
        mode=mode, shared_root=tmp_path, calibration=session,
        key_pose_world=key_pose)
    assert scene["mesh"]["target"]["file_path"] == str(planning_mesh)
    assert scene["mesh"]["target"]["pose"][:3] == pytest.approx([0.1, 0.2, 0.05])
    assert scene["cuboid"]["table"] == table_cuboid(session.board)
    assert scene["mesh"]["fixture_socket"] == session.collision_scene[
        "mesh"]["fixture_socket"]


def test_trial_scene_snaps_cylinder_with_local_z_v8_symmetry(tmp_path, monkeypatch):
    mode = select_mode("cylinder", 1)
    session, _ = _session(tmp_path, mode)
    info = (tmp_path / "object_processing" / mode.key_object /
            "processed_data" / "info")
    tabletop = info / "tabletop"
    tabletop.mkdir(parents=True)
    (info / "symmetry.json").write_text(json.dumps({
        "type": "Dinf", "axes": [
            {"axis": [0, 0, 1], "fold": "inf"},
            {"axis": [1, 0, 0], "fold": 2}],
    }), encoding="utf-8")
    end_down = np.diag([1.0, -1.0, -1.0, 1.0])
    np.save(tabletop / "000.npy", end_down)
    seen = []

    def make_scene(pose_world, c2r, obj_name, *, obj_root, tabletop_geometry):
        seen.append(pose_world.copy())
        return {"mesh": {"target": {"pose": se32cart(pose_world).tolist()}},
                "cuboid": {"table": table_cuboid(tabletop_geometry)}}

    monkeypatch.setattr("src.execution.scene_cfg.pose_world_to_scene_cfg", make_scene)
    source = end_down.copy()
    source[:3, 3] = [0.1, 0.2, 0.08]
    scene = build_trial_scene_from_session(
        mode=mode, shared_root=tmp_path, calibration=session,
        key_pose_world=source)
    assert scene["mesh"]["target"]["pose"] is not None
    np.testing.assert_allclose(seen[0][:3, 2], [0, 0, -1])
    assert "fixture_socket" in scene["mesh"]
