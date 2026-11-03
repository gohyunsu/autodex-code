#!/usr/bin/env python3
"""Solve FR3-to-four-camera hand-eye calibration by reprojection.

The four-camera extrinsics and intrinsics stay fixed. Two rigid transforms are
estimated jointly: ``T_world_robot`` (the runtime C2R transform) and the
unknown rigid board mount ``T_ee_board``. Every saved ChArUco pixel from every
camera contributes directly to the final robust reprojection objective.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibration.charuco import load_board
from calibration.reprojection_debug import render_handeye_debug


def _transform(vector: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    result[:3, 3] = vector[3:]
    return result


def _vector(transform: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
                 transform[:3, 3]]


def _se3_error(measured: np.ndarray, predicted: np.ndarray,
               translation_scale: float) -> np.ndarray:
    delta = np.linalg.inv(measured) @ predicted
    return np.r_[Rotation.from_matrix(delta[:3, :3]).as_rotvec(),
                 delta[:3, 3] / translation_scale]


def _full_transform(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape == (3, 4):
        return np.vstack((value, [0.0, 0.0, 0.0, 1.0]))
    if value.shape != (4, 4):
        raise ValueError(f"expected 3x4 or 4x4 transform, got {value.shape}")
    return value


def _intrinsic(row: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(row["original_intrinsics"], dtype=np.float64),
            np.asarray(row["dist_params"], dtype=np.float64).reshape(-1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--camera-calibration-dir", type=Path, required=True)
    parser.add_argument("--board-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    camera_dir = args.camera_calibration_dir.expanduser().resolve()
    intrinsics = json.loads(
        (camera_dir / "intrinsics.json").read_text(encoding="utf-8"))
    extrinsics_raw = json.loads(
        (camera_dir / "extrinsics.json").read_text(encoding="utf-8"))
    cameras = tuple(extrinsics_raw)
    world_to_camera = {
        camera: _full_transform(extrinsics_raw[camera]) for camera in cameras}
    _, board, _ = load_board(args.board_config)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    board_span = np.ptp(board_points, axis=0)
    board_scale = float(np.linalg.norm(board_span))

    observations = []
    world_board_measurements: dict[str, list[np.ndarray]] = {}
    robot_ee = {}
    capture_dirs = sorted(
        path for path in args.session_dir.expanduser().resolve().iterdir()
        if path.is_dir() and path.name.isdigit())
    for capture_dir in capture_dirs:
        record = json.loads(
            (capture_dir / "capture.json").read_text(encoding="utf-8"))
        robot_ee[capture_dir.name] = _full_transform(
            record["robot_after"]["T_robot_ee"])
        for camera in cameras:
            marker_dir = capture_dir / "markers_2d"
            corner_path = marker_dir / f"{camera}_corner.npy"
            id_path = marker_dir / f"{camera}_id.npy"
            if not corner_path.is_file() or not id_path.is_file():
                continue
            pixels = np.load(corner_path).reshape(-1, 2).astype(np.float64)
            ids = np.load(id_path).reshape(-1).astype(np.int32)
            if len(ids) < 4:
                continue
            points = board_points[ids]
            K, dist = _intrinsic(intrinsics[camera])
            ok, rvec, tvec = cv2.solvePnP(
                points, pixels, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            camera_board = np.eye(4)
            camera_board[:3, :3] = Rotation.from_rotvec(
                rvec.reshape(3)).as_matrix()
            camera_board[:3, 3] = tvec.reshape(3)
            world_board = np.linalg.inv(world_to_camera[camera]) @ camera_board
            world_board_measurements.setdefault(capture_dir.name, []).append(
                world_board)
            observations.append(
                (camera, capture_dir.name, ids, pixels, world_board))

    captures = sorted(world_board_measurements)
    if len(captures) < 3:
        raise RuntimeError(
            "hand-eye needs at least three captured robot/board poses with "
            "four or more ChArUco corners")

    first = captures[0]
    first_world_board = world_board_measurements[first][0]
    initial_ee_board = np.eye(4)
    initial_world_robot = first_world_board @ np.linalg.inv(robot_ee[first])
    initial = np.r_[_vector(initial_world_robot), _vector(initial_ee_board)]

    def unpack(vector):
        return _transform(vector[:6]), _transform(vector[6:12])

    def rigid_residual(vector):
        world_robot, ee_board = unpack(vector)
        chunks = []
        for _, capture, _, _, measured in observations:
            predicted = world_robot @ robot_ee[capture] @ ee_board
            chunks.append(_se3_error(measured, predicted, board_scale))
        return np.concatenate(chunks)

    rigid = least_squares(
        rigid_residual, initial, loss="soft_l1", f_scale=0.01,
        max_nfev=1000)

    def pixel_residual(vector):
        world_robot, ee_board = unpack(vector)
        chunks = []
        for camera, capture, ids, pixels, _ in observations:
            camera_board = (
                world_to_camera[camera] @ world_robot @
                robot_ee[capture] @ ee_board)
            rvec = Rotation.from_matrix(
                camera_board[:3, :3]).as_rotvec()
            K, dist = _intrinsic(intrinsics[camera])
            projected, _ = cv2.projectPoints(
                board_points[ids], rvec, camera_board[:3, 3], K, dist)
            chunks.append((projected.reshape(-1, 2) - pixels).ravel())
        return np.concatenate(chunks)

    refined = least_squares(
        pixel_residual, rigid.x, loss="soft_l1", f_scale=1.0,
        max_nfev=1000, verbose=1)
    world_robot, ee_board = unpack(refined.x)

    per_camera = {}
    for camera in cameras:
        chunks = []
        for cam, capture, ids, pixels, _ in observations:
            if cam != camera:
                continue
            camera_board = (
                world_to_camera[cam] @ world_robot @
                robot_ee[capture] @ ee_board)
            rvec = Rotation.from_matrix(camera_board[:3, :3]).as_rotvec()
            K, dist = _intrinsic(intrinsics[cam])
            projected, _ = cv2.projectPoints(
                board_points[ids], rvec, camera_board[:3, 3], K, dist)
            chunks.extend(np.linalg.norm(
                projected.reshape(-1, 2) - pixels, axis=1).tolist())
        values = np.asarray(chunks, dtype=np.float64)
        per_camera[camera] = {
            "observations": int(len(values)),
            "mean_px": float(np.mean(values)),
            "median_px": float(np.median(values)),
            "max_px": float(np.max(values)),
        }
    errors = np.linalg.norm(pixel_residual(refined.x).reshape(-1, 2), axis=1)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "C2R.npy", world_robot)
    np.save(output / "T_ee_board.npy", ee_board)
    debug = render_handeye_debug(
        session_dir=args.session_dir.expanduser().resolve(),
        board_config=args.board_config.expanduser().resolve(),
        camera_calibration_dir=camera_dir,
        c2r_path=output / "C2R.npy",
        ee_board_path=output / "T_ee_board.npy",
        output_dir=output / "debug_reprojection")
    report = {
        "schema_version": 1,
        "transform_convention": {
            "C2R.npy": "T_world_robot",
            "T_ee_board.npy": "T_end_effector_board",
            "camera_extrinsics": "T_camera_world",
        },
        "camera_calibration_dir": str(camera_dir),
        "capture_count": len(captures),
        "view_observation_count": len(observations),
        "optimizer": {
            "rigid_success": bool(rigid.success),
            "rigid_message": rigid.message,
            "reprojection_success": bool(refined.success),
            "reprojection_message": refined.message,
            "reprojection_cost": float(refined.cost),
            "reprojection_nfev": int(refined.nfev),
        },
        "reprojection": {
            "mean_px": float(np.mean(errors)),
            "median_px": float(np.median(errors)),
            "max_px": float(np.max(errors)),
            "per_camera": per_camera,
        },
        "debug_reprojection": {
            "directory": str(output / "debug_reprojection"),
            "mean_px": debug["mean_px"],
            "median_px": debug["median_px"],
            "max_px": debug["max_px"],
        },
        "T_world_robot": world_robot.tolist(),
        "T_ee_board": ee_board.tolist(),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
