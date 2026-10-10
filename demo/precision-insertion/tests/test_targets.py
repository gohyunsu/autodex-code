"""Rigid, socket-frame insertion target composition without robot motion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.targets import build_rigid_insertion_targets  # noqa: E402
from precision_insertion.world import add_fixed_mesh_fixtures  # noqa: E402


def _fixture(tmp_path, mode):
    paths = AssetPaths(tmp_path, mode)
    mesh = paths.socket_collision_mesh
    mesh.parent.mkdir(parents=True)
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    socket = np.eye(4)
    # Deliberately tilt the socket axis: an axial move is not always world Z.
    socket[:3, :3] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    socket[:3, 3] = [0.4, 0.1, 0.2]
    scene = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {}},
        {"fixture_socket": {"pose_robot": socket, "collision_mesh": mesh}})
    record = {
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "socket_pose_robot": socket.tolist(),
        "socket_collision_mesh": str(mesh),
        "socket_collision_mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
    }
    session = SessionCalibration({}, socket, {}, scene, record)

    entry = np.eye(4)
    entry[:3, :3] = np.diag([1.0, -1.0, -1.0])
    entry[2, 3] = 0.135
    verification = entry.copy()
    verification[2, 3] -= 0.020
    preinsert = entry.copy()
    preinsert[2, 3] += 0.030
    geometry = {
        "units": "m", "socket_pose_object": mode.socket_object,
        "key_object": mode.key_object,
        "T_socket_pose_object": np.eye(4).tolist(),
        "T_socket_key_entry": entry.tolist(),
        "T_socket_key_verification": verification.tolist(),
        "T_socket_key_preinsert": preinsert.tolist(),
        "verification_insertion_depth_m": 0.020,
        "preinsert_clearance_m": 0.030,
        "insertion_direction_socket": [0.0, 0.0, -1.0],
        "key_frame": {"insertion_axis": [0.0, 0.0, 1.0]},
    }
    paths.task_geometry.parent.mkdir(parents=True, exist_ok=True)
    paths.task_geometry.write_text(json.dumps(geometry), encoding="utf-8")
    return session, paths, geometry


@pytest.mark.parametrize("family,gap", [("square", 1.5), ("cylinder", 20)])
def test_targets_keep_key_hand_rigid_through_tilted_socket(
    tmp_path, family, gap,
):
    mode = select_mode(family, gap)
    session, _, _ = _fixture(tmp_path, mode)
    hand_in_key = np.eye(4)
    hand_in_key[:3, 3] = [0.01, -0.02, 0.03]
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=tmp_path, calibration=session,
        T_key_hand=hand_in_key, xy_offset_socket_m=(0.001, -0.002))
    np.testing.assert_allclose(targets.insertion_axis_robot, [-1, 0, 0])
    np.testing.assert_allclose(
        targets.T_robot_key_entry[:3, 3] -
        targets.T_robot_key_preinsert[:3, 3],
        targets.insertion_axis_robot * 0.030)
    np.testing.assert_allclose(
        targets.T_robot_key_verification[:3, 3] -
        targets.T_robot_key_entry[:3, 3],
        targets.insertion_axis_robot * 0.020)
    for key, hand in (
        (targets.T_robot_key_preinsert, targets.T_robot_hand_preinsert),
        (targets.T_robot_key_entry, targets.T_robot_hand_entry),
        (targets.T_robot_key_verification, targets.T_robot_hand_verification),
    ):
        np.testing.assert_allclose(np.linalg.inv(key) @ hand, hand_in_key)
    assert targets.to_record()["robot_ready"] is False
    assert targets.to_record()["mode"]["socket_object"] == mode.socket_object
    assert len(targets.to_record()["task_geometry_sha256"]) == 64
    assert "target" not in session.collision_scene["mesh"]


def test_targets_reject_wrong_socket_geometry_and_bad_preinsert(tmp_path):
    mode = select_mode("square", 1.5)
    session, paths, geometry = _fixture(tmp_path, mode)
    geometry["T_socket_key_preinsert"][2][3] = 0.130
    paths.task_geometry.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="not above entry"):
        build_rigid_insertion_targets(
            mode=mode, shared_root=tmp_path, calibration=session,
            T_key_hand=np.eye(4))
    geometry["T_socket_key_preinsert"][2][3] = 0.165
    paths.task_geometry.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match trial"):
        build_rigid_insertion_targets(
            mode=select_mode("square", 1.0), shared_root=tmp_path,
            calibration=session, T_key_hand=np.eye(4))
    with pytest.raises(ValueError, match="two finite meters"):
        build_rigid_insertion_targets(
            mode=mode, shared_root=tmp_path, calibration=session,
            T_key_hand=np.eye(4), xy_offset_socket_m=(float("nan"), 0.0))


def test_cylinder_targets_preserve_observed_yaw_gauge(tmp_path):
    mode = select_mode("cylinder", 20)
    session, _, _ = _fixture(tmp_path, mode)
    yaw = 0.8
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=tmp_path, calibration=session,
        T_key_hand=np.eye(4), cylinder_yaw_gauge_socket_rad=yaw)
    socket_key = np.linalg.inv(session.socket_pose_robot) @ (
        targets.T_robot_key_verification)
    assert socket_key[:3, :3] @ [0., 0., 1.] == pytest.approx([0., 0., -1.])
    assert socket_key[:3, :3] @ [1., 0., 0.] == pytest.approx(
        [np.cos(yaw), np.sin(yaw), 0.])
    assert targets.to_record()["cylinder_yaw_gauge_socket_rad"] == yaw
    square = select_mode("square", 1.5)
    square_session, _, _ = _fixture(tmp_path / "square", square)
    with pytest.raises(ValueError, match="only cylinder"):
        build_rigid_insertion_targets(
            mode=square, shared_root=tmp_path / "square",
            calibration=square_session, T_key_hand=np.eye(4),
            cylinder_yaw_gauge_socket_rad=yaw)
