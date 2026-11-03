#!/usr/bin/env python3
"""Solve four-camera metric extrinsics from saved ChArUco observations.

Intrinsics remain fixed. Per-view PnP initializes camera/board transforms and
a joint robust least-squares fit minimizes distorted-pixel reprojection error.
The first camera is the arbitrary world frame; hand-eye calibration later
establishes the robot-to-world transform.
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


def _T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    result[:3, 3] = tvec
    return result


def _x(T: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]]


def _mean_transform(values: list[np.ndarray]) -> np.ndarray:
    rotations = Rotation.from_matrix(np.stack([T[:3, :3] for T in values]))
    result = np.eye(4)
    result[:3, :3] = rotations.mean().as_matrix()
    result[:3, 3] = np.median(
        np.stack([T[:3, 3] for T in values]), axis=0)
    return result


def _intrinsic(row: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(row["original_intrinsics"], dtype=np.float64),
            np.asarray(row["dist_params"], dtype=np.float64).reshape(-1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--board-config", type=Path, required=True)
    parser.add_argument("--camera-ids", default=(
        "26053248,25322639,25305462,25322642"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    cameras = tuple(x.strip() for x in args.camera_ids.split(",") if x.strip())
    intrinsics_all = json.loads(args.intrinsics.read_text(encoding="utf-8"))
    intrinsics = {camera: intrinsics_all[camera] for camera in cameras}
    _, board, _ = load_board(args.board_config)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)

    observations = []
    pnp: dict[tuple[str, str], np.ndarray] = {}
    pose_dirs = sorted(path for path in args.session_dir.iterdir()
                       if path.is_dir() and path.name.isdigit())
    for pose_dir in pose_dirs:
        for camera in cameras:
            marker_dir = pose_dir / "markers_2d"
            corner_file = marker_dir / f"{camera}_corner.npy"
            id_file = marker_dir / f"{camera}_id.npy"
            if not corner_file.is_file() or not id_file.is_file():
                continue
            pixels = np.load(corner_file).reshape(-1, 2).astype(np.float64)
            ids = np.load(id_file).reshape(-1).astype(np.int32)
            if len(ids) < 4:
                continue
            points = board_points[ids]
            K, dist = _intrinsic(intrinsics[camera])
            ok, rvec, tvec = cv2.solvePnP(
                points, pixels, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            pnp[(camera, pose_dir.name)] = _T(rvec.reshape(3), tvec.reshape(3))
            observations.append((camera, pose_dir.name, ids, pixels))
    poses = sorted({pose for _, pose in pnp})
    if not poses:
        raise RuntimeError("no ChArUco pose has four or more detected corners")
    reference = max(cameras, key=lambda camera: sum(
        (camera, pose) in pnp for pose in poses))

    camera_T = {reference: np.eye(4)}
    unresolved = set(cameras) - {reference}
    while unresolved:
        changed = False
        for camera in tuple(unresolved):
            proposals = []
            for known, T_known_world in camera_T.items():
                for pose in poses:
                    if ((camera, pose) in pnp and (known, pose) in pnp):
                        proposals.append(
                            pnp[(camera, pose)] @
                            np.linalg.inv(pnp[(known, pose)]) @ T_known_world)
            if proposals:
                camera_T[camera] = _mean_transform(proposals)
                unresolved.remove(camera)
                changed = True
        if not changed:
            raise RuntimeError(
                f"camera overlap graph is disconnected: {sorted(unresolved)}")

    board_T = {}
    for pose in poses:
        proposals = [np.linalg.inv(camera_T[camera]) @ pnp[(camera, pose)]
                     for camera in cameras if (camera, pose) in pnp]
        board_T[pose] = _mean_transform(proposals)

    variable_cameras = [camera for camera in cameras if camera != reference]
    x0 = np.concatenate([*(_x(camera_T[c]) for c in variable_cameras),
                         *(_x(board_T[p]) for p in poses)])
    cam_slice = {camera: slice(6 * i, 6 * (i + 1))
                 for i, camera in enumerate(variable_cameras)}
    board_offset = 6 * len(variable_cameras)
    pose_slice = {pose: slice(board_offset + 6 * i,
                              board_offset + 6 * (i + 1))
                  for i, pose in enumerate(poses)}

    def unpack(x):
        cams = {reference: np.eye(4)}
        cams.update({camera: _T(x[s][:3], x[s][3:])
                     for camera, s in cam_slice.items()})
        boards = {pose: _T(x[s][:3], x[s][3:])
                  for pose, s in pose_slice.items()}
        return cams, boards

    def residual(x):
        cams, boards = unpack(x)
        chunks = []
        for camera, pose, ids, pixels in observations:
            T_cam_board = cams[camera] @ boards[pose]
            rvec = Rotation.from_matrix(T_cam_board[:3, :3]).as_rotvec()
            K, dist = _intrinsic(intrinsics[camera])
            projected, _ = cv2.projectPoints(
                board_points[ids], rvec, T_cam_board[:3, 3], K, dist)
            chunks.append((projected.reshape(-1, 2) - pixels).ravel())
        return np.concatenate(chunks)

    result = least_squares(residual, x0, loss="soft_l1", f_scale=1.0,
                           max_nfev=400, verbose=1)
    camera_T, board_T = unpack(result.x)
    all_residuals = residual(result.x).reshape(-1, 2)
    per_camera = {}
    for camera in cameras:
        values = []
        for cam, pose, ids, pixels in observations:
            if cam != camera:
                continue
            T_cam_board = camera_T[cam] @ board_T[pose]
            rvec = Rotation.from_matrix(T_cam_board[:3, :3]).as_rotvec()
            K, dist = _intrinsic(intrinsics[cam])
            projected, _ = cv2.projectPoints(
                board_points[ids], rvec, T_cam_board[:3, 3], K, dist)
            values.extend(np.linalg.norm(
                projected.reshape(-1, 2) - pixels, axis=1).tolist())
        per_camera[camera] = {
            "observations": len(values),
            "mean_px": float(np.mean(values)),
            "median_px": float(np.median(values)),
            "max_px": float(np.max(values)),
        }
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "extrinsics.json").write_text(json.dumps(
        {camera: camera_T[camera][:3].tolist() for camera in cameras},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "intrinsics.json").write_text(json.dumps(
        intrinsics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "world_frame": f"camera_{reference}",
        "reference_camera": reference,
        "camera_ids": list(cameras),
        "capture_count": len(poses),
        "view_observation_count": len(observations),
        "optimizer": {
            "success": bool(result.success), "message": result.message,
            "cost": float(result.cost), "nfev": int(result.nfev),
        },
        "reprojection": {
            "mean_px": float(np.mean(np.linalg.norm(all_residuals, axis=1))),
            "median_px": float(np.median(np.linalg.norm(all_residuals, axis=1))),
            "max_px": float(np.max(np.linalg.norm(all_residuals, axis=1))),
            "per_camera": per_camera,
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
