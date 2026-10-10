"""Session ChArUco/socket calibration contracts without camera or robot I/O."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import (  # noqa: E402
    SocketObservation, calibrate_session, load_session_calibration,
    write_session_calibration,
)
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


def _pose(x=0.0, *, yaw_deg=0.0, tilt_deg=0.0):
    yaw, tilt = np.deg2rad([yaw_deg, tilt_deg])
    cy, sy = np.cos(yaw), np.sin(yaw)
    ct, st = np.cos(tilt), np.sin(tilt)
    pose = np.eye(4)
    pose[:3, :3] = (
        np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        @ np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    )
    pose[0, 3] = x
    return pose


def _arguments(tmp_path, *, family="square"):
    mode = select_mode(family, 1.5 if family == "square" else 20)
    mesh = tmp_path / "static_collision.obj"
    mesh.write_text("v 0 0 0\n", encoding="utf-8")
    if family == "cylinder":
        info = (tmp_path / "objects" / mode.socket_object /
                "processed_data" / "info")
        info.mkdir(parents=True)
        (info / "symmetry.json").write_text(json.dumps({
            "type": "Cinf", "axes": [{"axis": [0, 0, 1], "fold": "inf"}]
        }), encoding="utf-8")
    c2r = _pose(1.0)
    return {
        "mode": mode,
        "object_root": tmp_path / "objects",
        "board_images_bgr": {
            "cam_a": np.zeros((8, 8, 3), dtype=np.uint8),
            "cam_b": np.zeros((8, 8, 3), dtype=np.uint8),
        },
        "board_timestamps_s": {"cam_a": 1.000, "cam_b": 1.005},
        "board_timestamp_source": "camera_acquisition",
        "socket_observations": [
            SocketObservation("first", "cam_a", 2.000, _pose(1.1000),
                              "camera_acquisition"),
            SocketObservation("first", "cam_b", 2.005, _pose(1.1002),
                              "camera_acquisition"),
            SocketObservation("second", "cam_a", 3.000, _pose(1.1003),
                              "camera_acquisition"),
            SocketObservation("second", "cam_b", 3.005, _pose(1.1004),
                              "camera_acquisition"),
        ],
        "intrinsics_full": {"cam_a": {}, "cam_b": {}},
        "extrinsics_full": {"cam_a": np.eye(4), "cam_b": np.eye(4)},
        "c2r": c2r,
        "base_scene": {"mesh": {}, "cuboid": {}},
        "socket_collision_mesh": mesh,
        "max_capture_skew_s": 0.020,
        "max_socket_translation_mm": 1.0,
        "max_socket_angle_deg": 1.0,
    }


@pytest.fixture
def board_measurement(monkeypatch):
    calls = []

    def measure(images, intrinsics, extrinsics, c2r):
        calls.append((tuple(sorted(images)), c2r.copy()))
        return {"board": "11", "table_surface_z_m": 0.0}

    monkeypatch.setattr(
        "src.execution.charuco_tabletop.measure_tabletop_from_images", measure)
    return calls


def test_calibration_freezes_robot_pose_and_adds_socket_without_mutation(
    tmp_path, board_measurement,
):
    args = _arguments(tmp_path)
    result = calibrate_session(**args)
    assert len(board_measurement) == 1
    assert result.board["board"] == "11"
    assert result.socket_pose_robot[0, 3] == pytest.approx(0.1002, abs=0.0002)
    assert result.socket_diagnostics["accepted"] is True
    assert "fixture_socket" not in args["base_scene"]["mesh"]
    assert "table" not in args["base_scene"]["cuboid"]
    assert result.collision_scene["mesh"]["fixture_socket"]["file_path"] == str(
        args["socket_collision_mesh"])
    table = result.collision_scene["cuboid"]["table"]
    assert table["pose"][2] + table["dims"][2] / 2 == pytest.approx(0.0)
    assert result.record["socket_pose_robot"] == result.socket_pose_robot.tolist()
    assert len(result.record["socket_observations"]) == 4
    assert result.record["board_timestamp_source"] == "camera_acquisition"
    assert all(row["timestamp_source"] == "camera_acquisition"
               for row in result.record["socket_observations"])
    assert len(result.record["socket_collision_mesh_sha256"]) == 64
    assert len(result.record["camera_calibration_sha256"]) == 64
    assert set(result.record["camera_calibration"]["extrinsics_full"]) == {
        "cam_a", "cam_b"}
    assert result.record["robot_ready"] is False
    saved = write_session_calibration(result, tmp_path / "run" / "calibration.json")
    assert json.loads(saved.read_text(encoding="utf-8")) == result.record
    with pytest.raises(FileExistsError):
        write_session_calibration(result, saved)


@pytest.mark.parametrize("change,error", [
    ({"board_timestamp_source": "publish_time"}, "acquisition timestamps"),
    ({"board_timestamps_s": {"cam_a": 1.0, "cam_b": 1.1}}, "not synchronized"),
    ({"board_timestamps_s": {"cam_a": 1.0, "missing": 1.005}}, "identical camera IDs"),
    ({"max_capture_skew_s": 0.0}, "must be positive"),
])
def test_bad_board_or_limits_stop_before_measurement(
    tmp_path, board_measurement, change, error,
):
    args = _arguments(tmp_path)
    args.update(change)
    with pytest.raises(ValueError, match=error):
        calibrate_session(**args)
    assert not board_measurement


def test_fixed_session_world_rejects_stale_key_target_before_board_measurement(
    tmp_path, board_measurement,
):
    args = _arguments(tmp_path)
    args["base_scene"]["mesh"]["target"] = {"file_path": "old_key.obj"}
    with pytest.raises(ValueError, match="must not contain a key target"):
        calibrate_session(**args)
    assert not board_measurement


def test_socket_captures_must_be_later_multiview_and_repeatable(
    tmp_path, board_measurement,
):
    args = _arguments(tmp_path)
    originals = args["socket_observations"]
    cases = [
        ([replace(originals[0], timestamp_source="foundpose_publish"),
          *originals[1:]], "acquisition timestamp"),
        ([replace(originals[0], timestamp_s=0.900),
          replace(originals[1], timestamp_s=0.905), *originals[2:]], "follow board"),
        ([replace(originals[0], camera_id="unknown"), *originals[1:]], "uncalibrated"),
        ([originals[0], originals[1],
          replace(originals[2], camera_id="cam_b"), originals[3]], "duplicate views"),
        ([replace(originals[0], timestamp_s=2.1), *originals[1:]], "not synchronized"),
        ([*originals[:3], replace(originals[3], pose_world=_pose(1.110))],
         "not repeatable"),
    ]
    for observations, error in cases:
        with pytest.raises(ValueError, match=error):
            calibrate_session(**{**args, "socket_observations": observations})


def test_cylinder_ignores_yaw_but_not_open_rim_flip(tmp_path, board_measurement):
    args = _arguments(tmp_path, family="cylinder")
    observations = args["socket_observations"]
    angles = [0, 80, -110, 170]
    args["socket_observations"] = [
        replace(o, pose_world=_pose(o.pose_world[0, 3], yaw_deg=yaw))
        for o, yaw in zip(observations, angles)
    ]
    accepted = calibrate_session(**args)
    assert accepted.socket_diagnostics["max_angle_residual_deg"] == pytest.approx(0)
    flipped = list(args["socket_observations"])
    pose = flipped[3].pose_world.copy()
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    flipped[3] = replace(flipped[3], pose_world=pose)
    with pytest.raises(ValueError, match="not repeatable"):
        calibrate_session(**{**args, "socket_observations": flipped})


def test_saved_session_replays_only_its_frozen_table_socket_and_mesh_bytes(
    tmp_path, board_measurement,
):
    args = _arguments(tmp_path)
    paths = AssetPaths(tmp_path, args["mode"])
    paths.socket_collision_mesh.parent.mkdir(parents=True)
    paths.socket_collision_mesh.write_bytes(args["socket_collision_mesh"].read_bytes())
    args["socket_collision_mesh"] = paths.socket_collision_mesh
    session = calibrate_session(**args)
    saved = write_session_calibration(session, tmp_path / "calibration.json")
    restored = load_session_calibration(
        saved, mode=args["mode"], shared_root=tmp_path)
    np.testing.assert_allclose(restored.socket_pose_robot,
                               session.socket_pose_robot)
    assert restored.collision_scene == session.collision_scene
    assert restored.record["fixed_mesh_sha256"]["fixture_socket"] == (
        restored.record["socket_collision_mesh_sha256"])
    paths.socket_collision_mesh.write_text("modified geometry", encoding="utf-8")
    with pytest.raises(ValueError, match="changed or is missing"):
        load_session_calibration(saved, mode=args["mode"], shared_root=tmp_path)


def test_saved_session_rejects_missing_or_modified_world_snapshot(
    tmp_path, board_measurement,
):
    args = _arguments(tmp_path)
    paths = AssetPaths(tmp_path, args["mode"])
    paths.socket_collision_mesh.parent.mkdir(parents=True)
    paths.socket_collision_mesh.write_bytes(args["socket_collision_mesh"].read_bytes())
    args["socket_collision_mesh"] = paths.socket_collision_mesh
    session = calibrate_session(**args)
    saved = write_session_calibration(session, tmp_path / "calibration.json")
    record = json.loads(saved.read_text(encoding="utf-8"))
    del record["collision_scene"]
    old = tmp_path / "old.json"
    old.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="no frozen collision world"):
        load_session_calibration(old, mode=args["mode"], shared_root=tmp_path)
    record = json.loads(saved.read_text(encoding="utf-8"))
    record["collision_scene"]["cuboid"]["table"]["pose"][2] += 0.01
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="descriptor hash changed"):
        load_session_calibration(tampered, mode=args["mode"], shared_root=tmp_path)
    record = json.loads(saved.read_text(encoding="utf-8"))
    record["socket_observations"][0]["timestamp_source"] = "foundpose_publish"
    bad_clock = tmp_path / "bad_clock.json"
    bad_clock.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="camera acquisition timestamps"):
        load_session_calibration(bad_clock, mode=args["mode"], shared_root=tmp_path)
