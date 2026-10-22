"""Fit an upright fixed socket from its acquired SAM masks and board plane.

The socket's wide flange is not consistently included in SAM masks.  Fit the
visible cylindrical wall instead of applying a whole-mesh silhouette loss to
different object parts. This is a socket-specific geometric estimate built
from stock SAM masks and frame provenance; this mode needs no socket FoundPose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .geometry import validate_se3
from .timing_profile import timed


def _barrel_dimensions(raw_mesh: Path) -> tuple[float, float, float]:
    vertices = []
    with Path(raw_mesh).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    xyz = np.asarray(vertices, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("socket raw mesh has no finite vertices")
    z_top = float(xyz[:, 2].max())
    z_bottom = float(xyz[:, 2].min())
    top = xyz[np.isclose(xyz[:, 2], z_top, atol=1e-6)]
    radius = float(np.linalg.norm(top[:, :2], axis=1).max())
    if not (z_top > z_bottom and radius > 0):
        raise ValueError("socket mesh lacks an upright cylindrical wall")
    wall = xyz[np.isclose(np.linalg.norm(xyz[:, :2], axis=1), radius,
                             atol=1e-6)]
    wall_bottom = float(wall[:, 2].min())
    if wall_bottom >= z_top:
        raise ValueError("socket mesh lacks a finite cylindrical wall")
    return radius, wall_bottom, z_top


def fit_board_socket_body(
    *, masks: Mapping[str, np.ndarray],
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_camera_world: Mapping[str, np.ndarray],
    c2r: np.ndarray, board_pose_robot: np.ndarray, raw_mesh: Path,
) -> tuple[np.ndarray, dict]:
    """Return world socket pose and per-view wall IoU from saved frame masks.

    The rigid fixture sits on the measured board and has its open axis along
    the board normal.  Each mask centroid supplies a board-plane intersection
    at the wall's mid-height.  A coordinate-wise median rejects views whose
    SAM mask contains the flange instead of the wall.  Only the two board XY
    coordinates are inferred; axial yaw is unobservable for this cylinder.
    """
    if (not masks or set(masks) != set(intrinsics_undist) or
            set(masks) != set(extrinsics_camera_world)):
        raise ValueError("socket wall fit requires matching calibrated views")
    board = validate_se3(board_pose_robot, name="measured board pose")
    world_robot = validate_se3(c2r, name="C2R")
    up = -board[:3, 2]
    x_axis = board[:3, 0]
    y_axis = np.cross(up, x_axis)
    rotation = np.column_stack((x_axis, y_axis, up))
    with timed("socket.body_mesh_dimensions", phase="socket", kind="geometry"):
        radius, bottom, top = _barrel_dimensions(raw_mesh)
    mid_height = (bottom + top) / 2.0
    board_points: list[np.ndarray] = []
    views = []
    for camera in sorted(masks):
        with timed("socket.body_centroid_ray", phase="socket",
                   kind="geometry", camera_id=camera):
            mask = np.asarray(masks[camera], dtype=bool)
            K = np.asarray(intrinsics_undist[camera], dtype=np.float64)
            E = validate_se3(extrinsics_camera_world[camera],
                             name=f"{camera} camera/world") @ world_robot
            if (mask.ndim != 2 or not mask.any() or K.shape != (3, 3) or
                    not np.all(np.isfinite(K)) or K[0, 0] <= 0 or K[1, 1] <= 0 or
                    not np.allclose(K[2], [0, 0, 1], atol=1e-8)):
                raise ValueError(f"invalid socket wall image or intrinsics: {camera}")
            yy, xx = np.nonzero(mask)
            pixel = np.array([float(xx.mean()), float(yy.mean()), 1.0])
            robot_camera = np.linalg.inv(E)
            origin = robot_camera[:3, 3]
            ray = robot_camera[:3, :3] @ np.linalg.solve(K, pixel)
            denominator = float(np.dot(ray, up))
            if abs(denominator) < 1e-8:
                raise ValueError(f"socket ray is parallel to the board: {camera}")
            distance = float(np.dot(board[:3, 3] + mid_height * up - origin, up)
                             / denominator)
            if distance <= 0:
                raise ValueError(f"socket mask ray points behind camera: {camera}")
            board_points.append(origin + distance * ray - mid_height * up)
            views.append((camera, mask, K, E))

    with timed("socket.body_multiview_median", phase="socket",
               kind="geometry", camera_count=len(views)):
        points = np.asarray(board_points)
        xy = np.column_stack(((points - board[:3, 3]) @ x_axis,
                              (points - board[:3, 3]) @ y_axis))
        median_xy = np.median(xy, axis=0)
        pose_robot = np.eye(4)
        pose_robot[:3, :3] = rotation
        pose_robot[:3, 3] = (board[:3, 3] + median_xy[0] * x_axis
                             + median_xy[1] * y_axis)

    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    rings = np.asarray([
        [radius * np.cos(angle), radius * np.sin(angle), z, 1.0]
        for z in (bottom, top) for angle in angles], dtype=np.float64)
    vertices_robot = pose_robot @ rings.T
    per_view = {}
    for camera, mask, K, E in views:
        with timed("socket.body_wall_iou", phase="socket", kind="geometry",
                   camera_id=camera):
            vertices_camera = (E @ vertices_robot)[:3].T
            if np.any(vertices_camera[:, 2] <= 0):
                raise ValueError(f"socket wall is behind camera: {camera}")
            projected = (K @ vertices_camera.T).T
            pixels = np.rint(projected[:, :2] /
                             projected[:, 2, None]).astype(np.int32)
            silhouette = np.zeros(mask.shape, dtype=np.uint8)
            cv2.fillConvexPoly(silhouette, cv2.convexHull(pixels), 1)
            intersection = np.count_nonzero(silhouette & mask)
            union = np.count_nonzero(silhouette | mask)
            per_view[camera] = float(intersection / union) if union else 0.0
    diagnostics = {
        "method": "board_normal_wall_mask_centroid_median",
        "body_radius_m": radius, "body_bottom_z_m": bottom,
        "body_top_z_m": top, "per_view_iou": per_view,
        "mean_body_iou": float(np.mean(list(per_view.values()))),
        "per_view_board_xy_m": {
            camera: xy[index].tolist() for index, camera in enumerate(sorted(masks))},
        "board_xy_m": median_xy.tolist(),
    }
    return validate_se3(world_robot @ pose_robot,
                        name="board-constrained socket pose"), diagnostics
