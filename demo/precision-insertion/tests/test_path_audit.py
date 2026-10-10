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
    PathAuditLimits, audit_held_joint_paths, audit_held_lateral_path,
)
from precision_insertion.solid_occupancy import (  # noqa: E402
    CylinderSocketOccupancy,
)
from precision_insertion.repose_path_audit import (  # noqa: E402
    audit_repose_held_paths,
)
from precision_insertion.targets import build_rigid_insertion_targets  # noqa: E402
from precision_insertion.world import (  # noqa: E402
    add_fixed_mesh_fixtures, build_held_scene_from_trial,
)


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
    monkeypatch.setattr(
        "precision_insertion.repose_path_audit._hand_link_meshes",
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


def test_cylinder_path_audit_uses_same_validated_bore_as_endpoint_screen(
        tmp_path, monkeypatch):
    from precision_insertion.path_audit import _fixed_world_models

    mode = select_mode("cylinder", 1.0)
    paths = AssetPaths(tmp_path, mode)
    profile = np.array([
        [0., 0.], [.06, 0.], [.06, .005], [.021, .005],
        [.021, .055], [.016, .055], [.016, .005], [0., .005],
    ])
    paths.socket_collision_mesh.parent.mkdir(parents=True)
    trimesh.creation.revolve(profile, sections=256).export(
        paths.socket_collision_mesh)
    geometry = {
        "socket_bore_radius_m": .016,
        "socket_bore_bottom_z_m": .005,
        "socket_rim_z_m": .055,
    }
    paths.task_geometry.parent.mkdir(parents=True)
    paths.task_geometry.write_text(json.dumps(geometry), encoding="utf-8")
    # The task-geometry contract has its own tests. This isolates the path
    # auditor's choice of already-validated analytic cylinder occupancy.
    monkeypatch.setattr(
        "precision_insertion.path_audit.validate_task_geometry",
        lambda _geometry, _mode: np.eye(4))
    board = {"table_surface_z_m": -.1}
    scene = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {"table": table_cuboid(board)}},
        {"fixture_socket": {"pose_robot": np.eye(4),
                            "collision_mesh": paths.socket_collision_mesh}})
    calibration = SessionCalibration(board, np.eye(4), {}, scene, {})
    fixed, hashes = _fixed_world_models(
        calibration, mode=mode, shared_root=tmp_path)
    occupancy = fixed["mesh/fixture_socket"][3]
    assert isinstance(occupancy, CylinderSocketOccupancy)
    assert occupancy.classify(np.array([[0., 0., .03]])).intersects_solid is False
    assert occupancy.classify(np.array([[.018, 0., .03]])).intersects_solid is True
    assert len(hashes["task_geometry"]) == 64


def test_sampled_path_checks_full_held_key_and_hand_without_authorizing_robot(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    result = _audit(tmp_path, fixture, *_paths())
    assert result["sampled_clear"] is True
    assert result["robot_ready"] is False
    assert result["sample_counts"] == {"transfer": 11, "descent": 6}
    endpoint = result["nominal_endpoint_metrics"]
    assert endpoint["rigid_model_depth_past_entry_m"] == pytest.approx(0.020)
    assert endpoint["nominal_target_depth_m"] == pytest.approx(0.020)
    assert endpoint["lateral_from_socket_axis_m"] == pytest.approx(0.0)
    assert endpoint["hand_target_translation_error_m"] == pytest.approx(0.0)
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


def _lateral(tmp_path, fixture, path, **overrides):
    calibration, targets, _ = fixture
    kwargs = dict(
        shared_root=tmp_path, mode=targets.mode, calibration=calibration,
        planner=_FakePlanner(), trajectory=path,
        held_hand_q=np.zeros(6), T_key_hand=np.eye(4),
        increment_socket_xy_m=(0.001, 0.0), limits=_limits(),
        max_path_deviation_m=0.0001,
        max_hold_height_deviation_m=0.0001,
        max_hold_rotation_deg=1.0,
        key_surface_bound_m=0.0001, hand_surface_bound_m=0.0001)
    kwargs.update(overrides)
    return audit_held_lateral_path(**kwargs)


def _lateral_path(x0=-0.002, z=0.135):
    path = np.zeros((6, 13))
    path[:, 0] = np.linspace(x0, x0 + 0.001, len(path))
    path[:, 2] = z
    return path


def test_lateral_hold_audit_reuses_full_key_and_hand_collision(
        tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    passed = _lateral(tmp_path, fixture, _lateral_path())
    assert passed["sampled_clear"] is True
    assert passed["robot_ready"] is False
    assert passed["sample_count"] == 6
    assert "key->mesh/fixture_socket" in passed[
        "minimum_surface_distances_m"]
    assert len(passed["input_sha256"]["lateral_trajectory"]) == 64
    assert all(row["clear"] for row in passed["future_surface_margins"].values())
    rejected = _lateral(
        tmp_path, fixture, _lateral_path(), key_surface_bound_m=1.0)
    assert rejected["sampled_clear"] is False
    assert any(row["reason"] ==
               "future_surface_bound_exceeds_sampled_clearance"
               for row in rejected["failures"])


def test_lateral_hold_rejects_vertical_detour_and_changed_hand(
        tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    path = _lateral_path()
    path[3, 2] += 0.001
    rejected = _lateral(tmp_path, fixture, path)
    assert rejected["sampled_clear"] is False
    assert any(row["reason"] == "not_socket_plane_lateral_segment"
               for row in rejected["failures"])
    path = _lateral_path()
    path[2, 7] = 0.01
    with pytest.raises(ValueError, match="changes Inspire joints"):
        _lateral(tmp_path, fixture, path)
    with pytest.raises(ValueError, match="at most 1 mm"):
        _lateral(tmp_path, fixture, _lateral_path(),
                 increment_socket_xy_m=(0.002, 0.0))


def test_lateral_hold_has_no_initial_socket_collision_exemption(
        tmp_path, monkeypatch):
    fixture = _fixture(tmp_path, monkeypatch)
    collision = _lateral(tmp_path, fixture, _lateral_path(x0=0.05, z=0.11))
    assert collision["sampled_clear"] is False
    assert any(row["reason"] == "held_geometry_collision_or_clearance"
               and row["sample"] == 0 and row["moving"] == "key" and
               row["obstacle"] == "mesh/fixture_socket"
               for row in collision["failures"])


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


def _repose_paths():
    lift = np.zeros((11, 13))
    lift[:, 0] = -0.10
    lift[:, 2] = np.linspace(0.035, 0.135, 11)
    transfer = np.repeat(lift[-1:], 11, axis=0)
    transfer[:, 0] = np.linspace(-0.10, -0.05, 11)
    transfer[:, 2] = np.linspace(0.135, 0.235, 11)
    descent = np.repeat(transfer[-1:], 11, axis=0)
    descent[:, 2] = np.linspace(0.235, 0.135, 11)
    return lift, transfer, descent


def _repose_audit(tmp_path, calibration, paths):
    initial = np.eye(4)
    initial[:3, 3] = (-0.10, 0.0, 0.035)
    rest = np.eye(4)
    rest[:3, 3] = (-0.05, 0.0, 0.035)
    return audit_repose_held_paths(
        shared_root=tmp_path, mode=select_mode("square", 1.5),
        calibration=calibration, planner=_FakePlanner(),
        lift_trajectory=paths[0], transfer_trajectory=paths[1],
        descent_trajectory=paths[2], held_hand_q=np.zeros(6),
        T_key_hand=np.eye(4), T_robot_key_initial=initial,
        T_robot_key_rest=rest, release_height_m=0.10, limits=_limits())


def test_repose_keeps_frozen_socket_in_carried_world_and_samples_held_key(
    tmp_path, monkeypatch,
):
    calibration, _, _ = _fixture(tmp_path, monkeypatch)
    scene = {"mesh": dict(calibration.collision_scene["mesh"]),
             "cuboid": calibration.collision_scene["cuboid"]}
    scene["mesh"]["target"] = {"file_path": "key", "pose": [0] * 7}
    carried = build_held_scene_from_trial(
        trial_scene=scene, calibration=calibration)
    assert "target" not in carried["mesh"]
    assert carried["mesh"]["fixture_socket"] == scene["mesh"]["fixture_socket"]
    carried["mesh"].clear()
    assert "fixture_socket" in calibration.collision_scene["mesh"]
    passed = _repose_audit(tmp_path, calibration, _repose_paths())
    assert passed["sampled_clear"] is True
    assert passed["robot_ready"] is False
    assert "key->mesh/fixture_socket" in passed["minimum_surface_distances_m"]
    assert passed["sample_counts"] == {
        "lift": 11, "transfer": 11, "descent": 11}


def test_repose_rejects_changed_socket_or_held_path_collision(
    tmp_path, monkeypatch,
):
    calibration, _, _ = _fixture(tmp_path, monkeypatch)
    scene = {"mesh": dict(calibration.collision_scene["mesh"]),
             "cuboid": calibration.collision_scene["cuboid"]}
    scene["mesh"]["target"] = {"file_path": "key", "pose": [0] * 7}
    scene["mesh"]["fixture_socket"] = {"file_path": "wrong"}
    with pytest.raises(ValueError, match="differ from frozen"):
        build_held_scene_from_trial(trial_scene=scene, calibration=calibration)
    lift, transfer, descent = _repose_paths()
    transfer[:, 0] = np.linspace(-0.10, 0.05, 11)
    descent[:, 0] = 0.05
    collision = _repose_audit(tmp_path, calibration, (lift, transfer, descent))
    assert collision["sampled_clear"] is False
    assert any(row["reason"] == "held_geometry_collision_or_clearance"
               and row["obstacle"] == "mesh/fixture_socket"
               for row in collision["failures"]), (
                   collision["failures"], collision["minimum_surface_distances_m"])
