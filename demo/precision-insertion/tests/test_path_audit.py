"""Sampled held-key collision checks, separate from arm and contact safety."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import coal  # noqa: F401 -- load before trimesh on the AutoDex host
import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.tabletop_geometry import table_cuboid  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.path_audit import (  # noqa: E402
    PathAuditLimits, audit_held_joint_paths,
)
from precision_insertion.targets import build_rigid_insertion_targets  # noqa: E402
from precision_insertion.world import add_fixed_mesh_fixtures  # noqa: E402


class _FakePlanner:
    def fk_wrist(self, q):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        return pose


def _fixture(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    paths = AssetPaths(tmp_path, mode)
    key = paths.raw_mesh(mode.key_object)
    socket = paths.socket_collision_mesh
    key.parent.mkdir(parents=True)
    socket.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.006, 0.006, 0.006)).export(key)
    obstacle = trimesh.creation.box(extents=(0.02, 0.02, 0.06))
    obstacle.apply_translation([0.05, 0.0, 0.11])
    obstacle.export(socket)
    paths.robot_urdf.parent.mkdir(parents=True)
    paths.robot_urdf.write_text("test URDF stand-in", encoding="utf-8")
    hand = trimesh.creation.box(extents=(0.008,) * 3)
    hand.apply_translation([0, 0, 0.06])
    monkeypatch.setattr(
        "precision_insertion.path_audit._hand_link_meshes",
        lambda *_: {"hand": hand})

    board = {"table_surface_z_m": -0.1}
    socket_pose = np.eye(4)
    scene = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {"table": table_cuboid(board)}},
        {"fixture_socket": {"pose_robot": socket_pose,
                            "collision_mesh": socket}})
    record = {
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "socket_pose_robot": socket_pose.tolist(),
        "socket_collision_mesh": str(socket),
        "socket_collision_mesh_sha256": hashlib.sha256(
            socket.read_bytes()).hexdigest(),
    }
    calibration = SessionCalibration(board, socket_pose, {}, scene, record)
    entry = np.eye(4)
    entry[2, 3] = 0.105
    preinsert = entry.copy()
    preinsert[2, 3] = 0.135
    verify = entry.copy()
    verify[2, 3] = 0.085
    geometry = {
        "units": "m", "socket_pose_object": mode.socket_object,
        "key_object": mode.key_object,
        "T_socket_pose_object": np.eye(4).tolist(),
        "T_socket_key_entry": entry.tolist(),
        "T_socket_key_preinsert": preinsert.tolist(),
        "T_socket_key_verification": verify.tolist(),
        "verification_insertion_depth_m": 0.020,
        "preinsert_clearance_m": 0.030,
        "insertion_direction_socket": [0, 0, -1],
        "key_frame": {"insertion_axis": [0, 0, -1]},
    }
    paths.task_geometry.parent.mkdir(parents=True)
    paths.task_geometry.write_text(json.dumps(geometry), encoding="utf-8")
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=tmp_path, calibration=calibration,
        T_key_hand=np.eye(4))
    return calibration, targets, paths


def _paths(start_x=-0.1):
    transfer = np.zeros((11, 13))
    transfer[:, 0] = np.linspace(start_x, 0, 11)
    transfer[:, 2] = 0.135
    descent = np.repeat(transfer[-1:], 6, axis=0)
    descent[:, 2] = np.linspace(0.135, 0.085, 6)
    return transfer, descent


def _limits():
    return PathAuditLimits(
        max_joint_step_rad=0.015,
        max_wrist_step_m=0.015,
        max_wrist_rotation_deg=1.0,
        goal_position_tolerance_m=0.0005,
        goal_rotation_tolerance_deg=1.0,
        axial_lateral_tolerance_m=0.001,
        axial_rotation_tolerance_deg=1.0,
        minimum_hand_clearance_m=0.001,
    )


def _audit(tmp_path, fixture, transfer, descent):
    calibration, targets, _ = fixture
    return audit_held_joint_paths(
        shared_root=tmp_path, calibration=calibration, targets=targets,
        planner=_FakePlanner(), transfer_trajectory=transfer,
        descent_trajectory=descent, held_hand_q=np.zeros(6),
        limits=_limits())


def test_sampled_path_checks_full_held_key_and_hand_without_authorizing_robot(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    result = _audit(tmp_path, fixture, *_paths())
    assert result["sampled_clear"] is True
    assert result["robot_ready"] is False
    assert result["sample_counts"] == {"transfer": 11, "descent": 6}
    assert "key->mesh/fixture_socket" in result["minimum_surface_distances_m"]
    assert "hand/hand->cuboid/table" in result["minimum_surface_distances_m"]
    assert len(result["input_sha256"]["key_mesh"]) == 64
    assert len(result["input_sha256"]["transfer_trajectory"]) == 64
    assert len(result["input_sha256"]["session_calibration_record"]) == 64
    assert "continuous swept geometry between FK samples" in result["not_validated"]


def test_optional_lift_checks_key_socket_and_monotone_world_z(
    tmp_path, monkeypatch,
):
    calibration, targets, _ = _fixture(tmp_path, monkeypatch)
    transfer, descent = _paths()
    lift = np.repeat(transfer[:1], 11, axis=0)
    lift[:, 2] = np.linspace(0.035, 0.135, 11)
    result = audit_held_joint_paths(
        shared_root=tmp_path, calibration=calibration, targets=targets,
        planner=_FakePlanner(), lift_trajectory=lift,
        transfer_trajectory=transfer, descent_trajectory=descent,
        held_hand_q=np.zeros(6), limits=_limits())
    assert result["sampled_clear"] is True
    assert result["sample_counts"]["lift"] == 11
    assert len(result["input_sha256"]["lift_trajectory"]) == 64
    lift[5, 2] -= 0.020
    rejected = audit_held_joint_paths(
        shared_root=tmp_path, calibration=calibration, targets=targets,
        planner=_FakePlanner(), lift_trajectory=lift,
        transfer_trajectory=transfer, descent_trajectory=descent,
        held_hand_q=np.zeros(6), limits=_limits())
    assert rejected["sampled_clear"] is False
    assert any(row["reason"] == "not_monotone_world_z_lift"
               for row in rejected["failures"])


def test_socket_intersection_and_sparse_sampling_fail_closed(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    transfer, descent = _paths(start_x=0.1)
    collision = _audit(tmp_path, fixture, transfer, descent)
    assert collision["sampled_clear"] is False
    assert any(row["reason"] == "held_geometry_collision_or_clearance"
               and row["moving"] == "key"
               and row["obstacle"] == "mesh/fixture_socket"
               for row in collision["failures"])
    transfer, descent = _paths()
    sparse = _audit(tmp_path, fixture, transfer[[0, -1]], descent)
    assert sparse["sampled_clear"] is False
    assert any(row["reason"] == "path_sampling_too_sparse"
               for row in sparse["failures"])


def test_nonaxial_descent_and_changed_finger_pose_rejected(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    transfer, descent = _paths()
    descent[2, 0] = 0.003
    result = _audit(tmp_path, fixture, transfer, descent)
    assert result["sampled_clear"] is False
    assert any(row["reason"] == "not_monotone_socket_axis_stroke"
               for row in result["failures"])
    transfer, descent = _paths()
    transfer[4, 7] = 0.01
    with pytest.raises(ValueError, match="changes Inspire joints"):
        _audit(tmp_path, fixture, transfer, descent)


def test_stale_geometry_or_discontinuous_joint_segments_rejected(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    transfer, descent = _paths()
    descent[0, 1] = 0.01
    with pytest.raises(ValueError, match="discontinuous"):
        _audit(tmp_path, fixture, transfer, descent)
    geometry_path = fixture[2].task_geometry
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    geometry["preinsert_clearance_m"] = 0.031
    geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
    with pytest.raises(ValueError, match="preinsert CAD pose"):
        _audit(tmp_path, fixture, *_paths())
