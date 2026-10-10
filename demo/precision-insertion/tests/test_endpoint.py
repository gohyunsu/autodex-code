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
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.endpoint import (  # noqa: E402
    _nominal_inspire_hold_poses,
    _validate_geometry,
    screen_grasp_endpoint,
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


def test_endpoint_uses_squeeze_not_merely_grasp_pose():
    pre = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    grasp = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    poses = _nominal_inspire_hold_poses(pre, grasp)
    assert poses["mujoco_squeeze"] == pytest.approx([0.3] * 6)
    assert poses["autodex_default_controller_hold"] == pytest.approx([0.38] * 6)
    assert not np.allclose(poses["autodex_default_controller_hold"], grasp)


def test_endpoint_hardware_nominal_hold_clips_to_inspire_limits():
    poses = _nominal_inspire_hold_poses(np.zeros(6), np.ones(6))
    assert poses["autodex_default_controller_hold"] == pytest.approx(
        [1.15, 0.55, 1.6, 1.6, 1.6, 1.6])


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
        "from precision_insertion.solid_occupancy import SolidMeshOccupancy",
        "fixed = trimesh.creation.box(extents=(1.0, 1.0, 1.0))",
        "moving = trimesh.creation.box(extents=(0.1, 0.1, 0.1))",
        "model = _coal_mesh(fixed)",
        "T = np.eye(4); T[2, 3] = 0.60",
        "clear = _mesh_pair_report(moving, model, T)",
        "T[2, 3] = 0.52",
        "overlap = _mesh_pair_report(moving, model, T)",
        "T = np.eye(4)",
        "contained_surface_only = _mesh_pair_report(moving, model, T)",
        "contained_solid = _mesh_pair_report(moving, model, T, SolidMeshOccupancy(fixed))",
        "print(json.dumps({'clear': clear, 'overlap': overlap, 'contained_surface_only': contained_surface_only, 'contained_solid': contained_solid}))",
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
    assert data["contained_surface_only"]["colliding"] is False
    assert data["contained_solid"]["colliding"] is True
    assert data["contained_solid"]["moving_vertices_inside_fixed"] == 8
    assert data["contained_solid"]["minimum_surface_distance_m"] == 0.0


def test_retry_endpoint_uses_observed_key_hand_transform(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    paths = AssetPaths(tmp_path, mode)
    for file in (paths.raw_mesh(mode.key_object), paths.socket_collision_mesh,
                 paths.robot_urdf):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("placeholder mesh", encoding="utf-8")
    paths.task_geometry.parent.mkdir(parents=True, exist_ok=True)
    paths.task_geometry.write_text(json.dumps(_geometry()), encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    np.save(candidate / "wrist_se3.npy", np.eye(4))
    np.save(candidate / "pregrasp_pose.npy", np.zeros(6))
    np.save(candidate / "grasp_pose.npy", np.ones(6) * 0.1)

    class FakeMesh:
        is_watertight = True

    monkeypatch.setattr("precision_insertion.endpoint._load_mesh",
                        lambda _path: FakeMesh())
    monkeypatch.setattr("precision_insertion.endpoint._coal_mesh",
                        lambda _mesh: object())
    monkeypatch.setattr("precision_insertion.endpoint.SolidMeshOccupancy",
                        lambda _mesh: object())
    monkeypatch.setattr("precision_insertion.endpoint._mesh_pair_report",
                        lambda *_args: {
                            "colliding": False,
                            "minimum_surface_distance_m": 0.01,
                        })
    monkeypatch.setattr("precision_insertion.endpoint._hand_link_meshes",
                        lambda _urdf, _q: {"finger": FakeMesh()})
    observed = np.eye(4)
    observed[0, 3] = 0.001
    result = screen_grasp_endpoint(
        shared_root=tmp_path, mode=mode, candidate_dir=candidate,
        minimum_hand_clearance_m=0.0002, T_key_hand_override=observed)
    assert result["endpoint_pass"] is True
    assert result["T_key_hand_source"] == "observed_postlift_override"
    assert result["T_key_hand"][0][3] == pytest.approx(0.001)
    assert result["candidate_T_key_hand"][0][3] == pytest.approx(0.0)

    achieved = screen_grasp_endpoint(
        shared_root=tmp_path, mode=mode, candidate_dir=candidate,
        minimum_hand_clearance_m=0.0002, T_key_hand_override=observed,
        hand_poses_override={"achieved_mujoco_squeeze": np.ones(6) * 0.42},
        override_source="simulated_end_squeeze")
    assert achieved["T_key_hand_source"] == "simulated_end_squeeze"
    assert list(achieved["hold_pose_screens"]) == ["achieved_mujoco_squeeze"]
    assert achieved["hold_pose_screens"]["achieved_mujoco_squeeze"]["hand_q"] == pytest.approx([0.42] * 6)
    with pytest.raises(ValueError, match="paired T_key_hand"):
        screen_grasp_endpoint(
            shared_root=tmp_path, mode=mode, candidate_dir=candidate,
            minimum_hand_clearance_m=0.0002,
            hand_poses_override={"unpaired": np.zeros(6)})


def test_cylinder_endpoint_uses_same_yaw_gauge_as_target(tmp_path, monkeypatch):
    mode = select_mode("cylinder", 20)
    paths = AssetPaths(tmp_path, mode)
    for file in (paths.raw_mesh(mode.key_object), paths.socket_collision_mesh,
                 paths.robot_urdf):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("placeholder", encoding="utf-8")
    geometry = _geometry()
    geometry["socket_pose_object"] = mode.socket_object
    geometry["key_object"] = mode.key_object
    paths.task_geometry.parent.mkdir(parents=True, exist_ok=True)
    paths.task_geometry.write_text(json.dumps(geometry), encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    np.save(candidate / "wrist_se3.npy", np.eye(4))
    np.save(candidate / "pregrasp_pose.npy", np.zeros(6))
    np.save(candidate / "grasp_pose.npy", np.ones(6) * .1)

    class FakeMesh:
        is_watertight = True

    checked = []
    monkeypatch.setattr("precision_insertion.endpoint._load_mesh",
                        lambda _path: FakeMesh())
    monkeypatch.setattr("precision_insertion.endpoint._coal_mesh",
                        lambda _mesh: object())
    monkeypatch.setattr("precision_insertion.endpoint.CylinderSocketOccupancy",
                        lambda _mesh, _geometry: object())
    monkeypatch.setattr("precision_insertion.endpoint._hand_link_meshes",
                        lambda _urdf, _q: {"finger": FakeMesh()})
    def pair(_moving, _fixed, transform, _occupancy):
        checked.append(transform.copy())
        return {"colliding": False, "minimum_surface_distance_m": .01}
    monkeypatch.setattr("precision_insertion.endpoint._mesh_pair_report", pair)
    result = screen_grasp_endpoint(
        shared_root=tmp_path, mode=mode, candidate_dir=candidate,
        minimum_hand_clearance_m=.0002,
        cylinder_yaw_gauge_socket_rad=.7)
    expected = np.diag([1., -1., -1., 1.])
    c, s = np.cos(.7), np.sin(.7)
    expected[:3, :3] = np.array([[c, -s, 0.],
                                  [s, c, 0.], [0., 0., 1.]]) @ expected[:3, :3]
    expected[2, 3] = .124
    np.testing.assert_allclose(result["T_socket_key_tested"], expected)
    np.testing.assert_allclose(checked[0], expected)
    assert result["cylinder_yaw_gauge_socket_rad"] == .7
    with pytest.raises(ValueError, match="only cylinder"):
        screen_grasp_endpoint(
            shared_root=tmp_path, mode=select_mode("square", 1.5),
            candidate_dir=candidate, minimum_hand_clearance_m=.0002,
            cylinder_yaw_gauge_socket_rad=.7)
