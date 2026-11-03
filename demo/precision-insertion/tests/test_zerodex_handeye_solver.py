"""Synthetic check for the project four-view hand-eye transform convention."""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.charuco import load_board
from solve_zerodex_handeye import main


def _T(rotvec=(0.0, 0.0, 0.0), translation=(0.0, 0.0, 0.0)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    value[:3, 3] = translation
    return value


def test_joint_handeye_reprojection_recovers_world_robot_and_board_mount(tmp_path):
    demo = Path(__file__).resolve().parents[1]
    board_config = demo / "config" / "zerodex_charuco_handeye_board2_60mm.json"
    _, board, _ = load_board(board_config)
    points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    ids = np.arange(len(points), dtype=np.int32)
    K = np.array([[900.0, 0.0, 1024.0],
                  [0.0, 900.0, 768.0],
                  [0.0, 0.0, 1.0]])
    cameras = {
        "cam0": np.eye(4),
        "cam1": _T((0.0, 0.05, 0.04), (-0.22, 0.01, 0.02)),
    }
    camera_dir = tmp_path / "camera"
    camera_dir.mkdir()
    (camera_dir / "intrinsics.json").write_text(json.dumps({
        camera: {
            "original_intrinsics": K.tolist(),
            "dist_params": [0.0, 0.0, 0.0, 0.0],
        } for camera in cameras
    }))
    (camera_dir / "extrinsics.json").write_text(json.dumps({
        camera: transform[:3].tolist()
        for camera, transform in cameras.items()
    }))

    world_robot = _T((0.04, -0.03, 0.08), (-0.08, 0.03, 0.78))
    ee_board = _T((0.12, -0.08, 0.06), (0.025, -0.015, 0.09))
    robot_poses = [
        _T((0.00, 0.00, 0.00), (-0.08, -0.05, 0.00)),
        _T((0.15, 0.02, -0.08), (0.08, -0.04, 0.03)),
        _T((-0.08, 0.16, 0.05), (-0.04, 0.08, 0.05)),
        _T((0.10, -0.12, 0.12), (0.07, 0.06, -0.01)),
        _T((-0.13, -0.06, -0.10), (0.00, -0.08, 0.07)),
    ]
    session = tmp_path / "session"
    for index, robot_ee in enumerate(robot_poses):
        capture = session / f"{index:04d}"
        markers = capture / "markers_2d"
        images = capture / "images"
        markers.mkdir(parents=True)
        images.mkdir()
        record = {
            "capture_index": index,
            "robot_after": {"T_robot_ee": robot_ee.tolist()},
        }
        (capture / "capture.json").write_text(json.dumps(record))
        for camera, camera_world in cameras.items():
            camera_board = camera_world @ world_robot @ robot_ee @ ee_board
            rvec = Rotation.from_matrix(camera_board[:3, :3]).as_rotvec()
            pixels, _ = cv2.projectPoints(
                points, rvec, camera_board[:3, 3], K, np.zeros(4))
            np.save(markers / f"{camera}_corner.npy", pixels.reshape(-1, 2))
            np.save(markers / f"{camera}_id.npy", ids)
            cv2.imwrite(str(images / f"{camera}.png"),
                        np.zeros((1536, 2048, 3), dtype=np.uint8))

    output = tmp_path / "solution"
    assert main([
        "--session-dir", str(session),
        "--camera-calibration-dir", str(camera_dir),
        "--board-config", str(board_config),
        "--output-dir", str(output),
    ]) == 0
    assert np.allclose(np.load(output / "C2R.npy"), world_robot, atol=2e-5)
    assert np.allclose(np.load(output / "T_ee_board.npy"), ee_board, atol=2e-5)
    report = json.loads((output / "report.json").read_text())
    assert report["reprojection"]["mean_px"] < 1e-4
    assert (output / "debug_reprojection" / "report.json").is_file()
