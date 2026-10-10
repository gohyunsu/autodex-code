"""Offline tests for fixture and v8 axial-symmetry contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.geometry import freeze_fixture_pose, validate_se3  # noqa: E402
from precision_insertion.symmetry import (  # noqa: E402
    load_axial_symmetry, snap_axisymmetric_tabletop_pose,
)
from precision_insertion.world import add_fixed_mesh_fixtures  # noqa: E402


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
