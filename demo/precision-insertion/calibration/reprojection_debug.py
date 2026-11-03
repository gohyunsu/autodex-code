"""Render observed-versus-predicted calibration pixels for visual review."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .charuco import load_board


_CAMERA_ORDER = ("26053248", "25322639", "25305462", "25322642")
_OBSERVED = (0, 255, 0)
_PREDICTED = (255, 0, 255)
_RESIDUAL = (0, 255, 255)


def _full_transform(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    return (np.vstack((value, [0.0, 0.0, 0.0, 1.0]))
            if value.shape == (3, 4) else value)


def _T(vector: np.ndarray) -> np.ndarray:
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(vector[:3]).as_matrix()
    value[:3, 3] = vector[3:]
    return value


def _x(transform: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(transform[:3, :3]).as_rotvec(),
                 transform[:3, 3]]


def _mean_transform(values: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in values])).mean().as_matrix()
    result[:3, 3] = np.median(
        np.stack([value[:3, 3] for value in values]), axis=0)
    return result


def _intrinsic(row: dict) -> tuple[np.ndarray, np.ndarray]:
    return (np.asarray(row["original_intrinsics"], dtype=np.float64),
            np.asarray(row["dist_params"], dtype=np.float64).reshape(-1))


def _project(points: np.ndarray, camera_board: np.ndarray,
             intrinsic: dict) -> np.ndarray:
    K, dist = _intrinsic(intrinsic)
    rvec = Rotation.from_matrix(camera_board[:3, :3]).as_rotvec()
    pixels, _ = cv2.projectPoints(
        points, rvec, camera_board[:3, 3], K, dist)
    return pixels.reshape(-1, 2)


def _observation(capture_dir: Path, camera: str):
    marker_dir = capture_dir / "markers_2d"
    corner_path = marker_dir / f"{camera}_corner.npy"
    id_path = marker_dir / f"{camera}_id.npy"
    if not corner_path.is_file() or not id_path.is_file():
        return np.empty((0, 2)), np.empty((0,), dtype=np.int32)
    return (np.load(corner_path).reshape(-1, 2).astype(np.float64),
            np.load(id_path).reshape(-1).astype(np.int32))


def _draw(image: np.ndarray, observed: np.ndarray, ids: np.ndarray,
          predicted_all: np.ndarray, camera: str) -> tuple[np.ndarray, list[float]]:
    overlay = image.copy()
    height, width = overlay.shape[:2]
    for x, y in predicted_all:
        point = (int(round(x)), int(round(y)))
        if 0 <= point[0] < width and 0 <= point[1] < height:
            cv2.drawMarker(overlay, point, _PREDICTED,
                           cv2.MARKER_TILTED_CROSS, 12, 2, cv2.LINE_AA)
    errors = []
    for pixel, identifier in zip(observed, ids):
        predicted = predicted_all[int(identifier)]
        p_observed = tuple(np.rint(pixel).astype(int))
        p_predicted = tuple(np.rint(predicted).astype(int))
        cv2.line(overlay, p_observed, p_predicted, _RESIDUAL, 2, cv2.LINE_AA)
        cv2.circle(overlay, p_observed, 5, _OBSERVED, -1, cv2.LINE_AA)
        errors.append(float(np.linalg.norm(pixel - predicted)))
    mean = float(np.mean(errors)) if errors else None
    label = (f"{camera} | n={len(errors)} | mean={mean:.2f}px"
             if mean is not None else f"{camera} | no observed corner")
    cv2.rectangle(overlay, (0, 0), (760, 58), (15, 15, 15), -1)
    cv2.putText(overlay, label, (18, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay, errors


def _save_debug(
        output_dir: Path, session_dir: Path, board_points: np.ndarray,
        intrinsics: dict, cameras: tuple[str, ...], camera_board_for_capture,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    ordered = tuple(camera for camera in _CAMERA_ORDER if camera in cameras)
    ordered += tuple(camera for camera in cameras if camera not in ordered)
    capture_dirs = sorted(
        path for path in session_dir.iterdir()
        if path.is_dir() and path.name.isdigit())
    rows = []
    grids = []
    all_errors = []
    per_camera = {camera: [] for camera in ordered}
    for capture_dir in capture_dirs:
        capture_output = output_dir / capture_dir.name
        capture_output.mkdir()
        tiles = []
        capture_errors = []
        transforms = camera_board_for_capture(capture_dir, ordered)
        for camera in ordered:
            image = cv2.imread(str(capture_dir / "images" / f"{camera}.png"))
            if image is None:
                continue
            observed, ids = _observation(capture_dir, camera)
            predicted = _project(
                board_points, transforms[camera], intrinsics[camera])
            overlay, errors = _draw(
                image, observed, ids, predicted, camera)
            cv2.imwrite(str(capture_output / f"{camera}.png"), overlay)
            all_errors.extend(errors)
            capture_errors.extend(errors)
            per_camera[camera].extend(errors)
            tile_width = 640
            tile_height = int(round(tile_width * image.shape[0] / image.shape[1]))
            tiles.append(cv2.resize(
                overlay, (tile_width, tile_height), interpolation=cv2.INTER_AREA))
        if len(tiles) == 4:
            grid = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
            grid_path = capture_output / "grid.png"
            cv2.imwrite(str(grid_path), grid)
            grids.append(grid_path)
        rows.append({
            "capture": capture_dir.name,
            "observations": len(capture_errors),
            "mean_px": (None if not capture_errors else
                        float(np.mean(capture_errors))),
            "max_px": (None if not capture_errors else
                       float(np.max(capture_errors))),
        })
    if grids:
        frames = [Image.open(path).convert("RGB") for path in grids]
        frames[0].save(
            output_dir / "reprojection_4view.gif", save_all=True,
            append_images=frames[1:], duration=900, loop=0,
            optimize=False)
        for frame in frames:
            frame.close()
    report = {
        "legend_bgr": {
            "observed": list(_OBSERVED),
            "predicted": list(_PREDICTED),
            "residual_line": list(_RESIDUAL),
        },
        "capture_count": len(rows),
        "observations": len(all_errors),
        "mean_px": float(np.mean(all_errors)),
        "median_px": float(np.median(all_errors)),
        "max_px": float(np.max(all_errors)),
        "per_camera": {
            camera: {
                "observations": len(values),
                "mean_px": (None if not values else float(np.mean(values))),
                "median_px": (None if not values else float(np.median(values))),
                "max_px": (None if not values else float(np.max(values))),
            } for camera, values in per_camera.items()
        },
        "captures": rows,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return report


def render_extrinsic_debug(
        *, session_dir: Path, board_config: Path, intrinsics_path: Path,
        extrinsics_path: Path, output_dir: Path) -> dict:
    """Estimate one board pose per capture with fixed solved cameras and render."""
    intrinsics = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    extrinsics_raw = json.loads(extrinsics_path.read_text(encoding="utf-8"))
    cameras = tuple(extrinsics_raw)
    camera_world = {
        camera: _full_transform(extrinsics_raw[camera]) for camera in cameras}
    _, board, _ = load_board(board_config)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    board_poses = {}
    for capture_dir in sorted(
            path for path in session_dir.iterdir()
            if path.is_dir() and path.name.isdigit()):
        proposals = []
        observations = []
        for camera in cameras:
            pixels, ids = _observation(capture_dir, camera)
            if len(ids) < 4:
                continue
            K, dist = _intrinsic(intrinsics[camera])
            ok, rvec, tvec = cv2.solvePnP(
                board_points[ids], pixels, K, dist,
                flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            camera_board = _T(np.r_[rvec.reshape(3), tvec.reshape(3)])
            proposals.append(np.linalg.inv(camera_world[camera]) @ camera_board)
            observations.append((camera, pixels, ids))
        if not proposals:
            continue
        initial = _x(_mean_transform(proposals))

        def residual(vector):
            world_board = _T(vector)
            chunks = []
            for camera, pixels, ids in observations:
                predicted = _project(
                    board_points[ids], camera_world[camera] @ world_board,
                    intrinsics[camera])
                chunks.append((predicted - pixels).ravel())
            return np.concatenate(chunks)

        result = least_squares(
            residual, initial, loss="soft_l1", f_scale=1.0,
            max_nfev=200)
        board_poses[capture_dir.name] = _T(result.x)

    def transforms(capture_dir, ordered):
        world_board = board_poses[capture_dir.name]
        return {camera: camera_world[camera] @ world_board
                for camera in ordered}

    return _save_debug(
        output_dir, session_dir, board_points, intrinsics, cameras, transforms)


def render_handeye_debug(
        *, session_dir: Path, board_config: Path,
        camera_calibration_dir: Path, c2r_path: Path,
        ee_board_path: Path, output_dir: Path) -> dict:
    """Render fixed-camera and fixed-board-mount hand-eye reprojections."""
    intrinsics = json.loads(
        (camera_calibration_dir / "intrinsics.json").read_text(encoding="utf-8"))
    extrinsics_raw = json.loads(
        (camera_calibration_dir / "extrinsics.json").read_text(encoding="utf-8"))
    cameras = tuple(extrinsics_raw)
    camera_world = {
        camera: _full_transform(extrinsics_raw[camera]) for camera in cameras}
    world_robot = np.load(c2r_path)
    ee_board = np.load(ee_board_path)
    _, board, _ = load_board(board_config)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)

    def transforms(capture_dir, ordered):
        record = json.loads(
            (capture_dir / "capture.json").read_text(encoding="utf-8"))
        robot_ee = _full_transform(record["robot_after"]["T_robot_ee"])
        world_board = world_robot @ robot_ee @ ee_board
        return {camera: camera_world[camera] @ world_board
                for camera in ordered}

    return _save_debug(
        output_dir, session_dir, board_points, intrinsics, cameras, transforms)
