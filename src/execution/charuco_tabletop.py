"""Empty-board Charuco preflight for the integrated execution pipeline.

This is deliberately a *board-in-robot-frame* measurement.  It uses the
already calibrated camera-to-robot transform (``inv(C2R)``); it never tries to
overwrite hand-eye calibration from a board that an operator may have moved.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


BOARD_ID = "11"
EXPECTED_CORNERS = 54
# Paradex's Charuco configuration uses checkerLength=1.0 as a grid unit, while
# the physical boards are printed with 5 cm squares.  Keep this conversion in
# one explicit place; triangulated camera geometry is already in metres.
CHARUCO_GRID_UNIT_M = 0.05
MAX_REPROJECTION_PX = 3.0
MAX_PLANE_RMS_M = 0.002
MAX_PLANE_ERROR_M = 0.005
MAX_FIT_RMS_M = 0.002
MAX_FIT_ERROR_M = 0.005
MIN_UPWARD_NORMAL_Z = 0.95


def _triangulate_dlt(observations: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Linear multi-view triangulation from (world->camera P, undistorted uv)."""
    rows = []
    for projection, uv in observations:
        u, v = np.asarray(uv, dtype=np.float64).reshape(2)
        P = np.asarray(projection, dtype=np.float64).reshape(3, 4)
        rows.extend((u * P[2] - P[0], v * P[2] - P[1]))
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64), full_matrices=False)
    h = vt[-1]
    if abs(float(h[3])) <= 1e-12:
        raise ValueError("degenerate Charuco triangulation")
    return h[:3] / h[3]


def _reprojection_error(point_world: np.ndarray, projection: np.ndarray,
                        uv: np.ndarray) -> float:
    h = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    p = np.asarray(projection, dtype=np.float64).reshape(3, 4) @ h
    if abs(float(p[2])) <= 1e-12:
        return float("inf")
    return float(np.linalg.norm(p[:2] / p[2] - np.asarray(uv, dtype=np.float64)))


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t with ``target ~= R @ source + t`` (no scale fit)."""
    sc = source.mean(axis=0)
    tc = target.mean(axis=0)
    H = (source - sc).T @ (target - tc)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1] *= -1.0
        R = Vt.T @ U.T
    return R, tc - R @ sc


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    if normal[2] < 0.0:
        normal = -normal
    error = np.abs((points - centroid) @ normal)
    return centroid, normal, error


def _board_corner_model() -> np.ndarray:
    """Read board-11's physical internal-corner coordinates in metres.

    ``getChessboardCorners()`` returns the ``checkerLength`` grid units from
    the Paradex board config (board 11 uses ``1.0``), not metres.  Comparing
    those values directly to meter-scale triangulated points turns this
    45 cm x 30 cm board into a fictitious 9 m x 6 m board, which a rigid fit
    cannot correct because it deliberately has no scale parameter.
    """
    from paradex.image.aruco import _charuco_board_cache

    board = _charuco_board_cache.get(BOARD_ID)
    if board is None:
        raise RuntimeError(f"Charuco board {BOARD_ID} is not configured")
    corners = (np.asarray(board.getChessboardCorners(), dtype=np.float64)
               .reshape(-1, 3) * CHARUCO_GRID_UNIT_M)
    if len(corners) != EXPECTED_CORNERS:
        raise RuntimeError(
            f"board {BOARD_ID}: expected {EXPECTED_CORNERS} internal corners, "
            f"but config contains {len(corners)}")
    return corners


def measure_tabletop_from_images(
    images_bgr: Mapping[str, np.ndarray],
    intrinsics_full: Mapping[str, Mapping[str, np.ndarray]],
    extrinsics_full: Mapping[str, np.ndarray],
    c2r: np.ndarray,
) -> dict:
    """Measure board 11 from one empty-table multi-camera snapshot.

    Snapshot JPEGs are treated as original/distorted camera images and each
    detected pixel is converted to the calibration's undistorted pixel space
    before triangulation against its world->camera extrinsic.

    Raises ``RuntimeError`` when the empty-board quality contract is not met.
    No historical centre/height comparison is performed.
    """
    from paradex.image.aruco import detect_charuco

    observations: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    camera_corner_counts: dict[str, int] = {}
    cameras_used: set[str] = set()
    for serial, image in images_bgr.items():
        if image is None or serial not in intrinsics_full or serial not in extrinsics_full:
            continue
        detected = detect_charuco(image)
        info = detected.get(BOARD_ID)
        if info is None:
            camera_corner_counts[str(serial)] = 0
            continue
        ids = np.asarray(info.get("checkerIDs", []), dtype=np.int64).reshape(-1)
        pixels = np.asarray(info.get("checkerCorner", []), dtype=np.float64).reshape(-1, 2)
        if len(ids) != len(pixels):
            raise RuntimeError(f"board {BOARD_ID}: malformed detection from {serial}")
        calib = intrinsics_full[serial]
        K_orig = np.asarray(calib["K_orig"], dtype=np.float64).reshape(3, 3)
        K_undist = np.asarray(calib["K_undist"], dtype=np.float64).reshape(3, 3)
        dist = np.asarray(calib["dist_params"], dtype=np.float64).reshape(-1)
        undistorted = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), K_orig, dist, P=K_undist,
        ).reshape(-1, 2)
        ext = np.asarray(extrinsics_full[serial], dtype=np.float64)
        if ext.shape == (4, 4):
            ext = ext[:3, :]
        projection = K_undist @ ext.reshape(3, 4)
        valid = 0
        for corner_id, uv in zip(ids, undistorted):
            if not 0 <= int(corner_id) < EXPECTED_CORNERS:
                continue
            observations.setdefault(int(corner_id), []).append((projection, uv))
            valid += 1
        camera_corner_counts[str(serial)] = valid
        if valid:
            cameras_used.add(str(serial))

    model = _board_corner_model()
    missing = sorted(set(range(EXPECTED_CORNERS)) - set(observations))
    if missing:
        raise RuntimeError(
            f"board {BOARD_ID}: empty-board preflight requires all "
            f"{EXPECTED_CORNERS}/{EXPECTED_CORNERS} corners; missing {missing}")

    points_world = np.empty((EXPECTED_CORNERS, 3), dtype=np.float64)
    reprojection_error_px: dict[int, float] = {}
    views_per_corner: dict[int, int] = {}
    for corner_id in range(EXPECTED_CORNERS):
        obs = observations[corner_id]
        if len(obs) < 2:
            raise RuntimeError(
                f"board {BOARD_ID}: corner {corner_id} was seen by only {len(obs)} camera(s)")
        point = _triangulate_dlt(obs)
        errors = np.asarray([_reprojection_error(point, P, uv) for P, uv in obs])
        # A bad corner in one camera must not pull a 20-view DLT estimate away
        # from the physical board.  Refit only if at least two inliers remain.
        keep = errors <= MAX_REPROJECTION_PX
        if int(keep.sum()) >= 2 and not bool(keep.all()):
            point = _triangulate_dlt([o for o, ok in zip(obs, keep) if ok])
            errors = np.asarray([
                _reprojection_error(point, P, uv) for P, uv in obs
            ])
        if float(np.median(errors)) > MAX_REPROJECTION_PX:
            raise RuntimeError(
                f"board {BOARD_ID}: corner {corner_id} reprojection median "
                f"{float(np.median(errors)):.2f}px exceeds {MAX_REPROJECTION_PX:.1f}px")
        points_world[corner_id] = point
        reprojection_error_px[corner_id] = float(np.median(errors))
        views_per_corner[corner_id] = len(obs)

    c2r = np.asarray(c2r, dtype=np.float64).reshape(4, 4)
    world_h = np.concatenate([points_world, np.ones((len(points_world), 1))], axis=1)
    points_robot = (np.linalg.inv(c2r) @ world_h.T).T[:, :3]
    R_rb, t_rb = _rigid_fit(model, points_robot)
    fitted_robot = (R_rb @ model.T).T + t_rb
    fit_error = np.linalg.norm(points_robot - fitted_robot, axis=1)
    plane_point, normal, plane_error = _fit_plane(points_robot)
    if normal[2] < MIN_UPWARD_NORMAL_Z:
        raise RuntimeError(
            f"board {BOARD_ID}: normal z={normal[2]:.4f} is not a plausible tabletop")
    if float(np.sqrt(np.mean(plane_error ** 2))) > MAX_PLANE_RMS_M or float(plane_error.max()) > MAX_PLANE_ERROR_M:
        raise RuntimeError(
            f"board {BOARD_ID}: plane residual rms/max="
            f"{np.sqrt(np.mean(plane_error ** 2))*1000:.2f}/"
            f"{plane_error.max()*1000:.2f}mm exceeds "
            f"{MAX_PLANE_RMS_M*1000:.1f}/{MAX_PLANE_ERROR_M*1000:.1f}mm")
    if float(np.sqrt(np.mean(fit_error ** 2))) > MAX_FIT_RMS_M or float(fit_error.max()) > MAX_FIT_ERROR_M:
        raise RuntimeError(
            f"board {BOARD_ID}: rigid-fit residual rms/max="
            f"{np.sqrt(np.mean(fit_error ** 2))*1000:.2f}/"
            f"{fit_error.max()*1000:.2f}mm exceeds "
            f"{MAX_FIT_RMS_M*1000:.1f}/{MAX_FIT_ERROR_M*1000:.1f}mm")

    board_center_local = model.mean(axis=0)
    board_center = R_rb @ board_center_local + t_rb
    T_robot_board = np.eye(4)
    T_robot_board[:3, :3] = R_rb
    T_robot_board[:3, 3] = t_rb
    return {
        "schema_version": 1,
        "source": "charuco_empty_board_preflight",
        "board": BOARD_ID,
        "center_robot_m": board_center.tolist(),
        # Runtime execution assumes a level board after the normal-quality
        # check.  Use the fitted plane's centroid height as the one shared
        # table/release reference; retain the full normal for diagnostics.
        "table_surface_z_m": float(plane_point[2]),
        "plane_point_robot_m": plane_point.tolist(),
        "normal_robot": normal.tolist(),
        "T_robot_board": T_robot_board.tolist(),
        "corners_robot_m": points_robot.tolist(),
        "corners_world_m": points_world.tolist(),
        "metrics": {
            "corners_expected": EXPECTED_CORNERS,
            "corners_triangulated": EXPECTED_CORNERS,
            "cameras_used": len(cameras_used),
            "per_camera_corners": camera_corner_counts,
            "views_per_corner": views_per_corner,
            "reprojection_median_px": float(np.median(list(reprojection_error_px.values()))),
            "reprojection_max_px": float(max(reprojection_error_px.values())),
            "plane_rms_mm": float(np.sqrt(np.mean(plane_error ** 2)) * 1000.0),
            "plane_max_mm": float(plane_error.max() * 1000.0),
            "rigid_fit_rms_mm": float(np.sqrt(np.mean(fit_error ** 2)) * 1000.0),
            "rigid_fit_max_mm": float(fit_error.max() * 1000.0),
        },
    }


def save_tabletop_measurement(measurement: Mapping, out_dir: str | Path) -> Path:
    """Persist a human-readable preflight result without changing calibration."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "charuco_tabletop.json"
    with open(path, "w") as f:
        json.dump(measurement, f, indent=2, sort_keys=True)
    return path
