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


def _mesh_vertices(raw_mesh: Path) -> np.ndarray:
    vertices = []
    with Path(raw_mesh).open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
    xyz = np.asarray(vertices, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.all(np.isfinite(xyz)):
        raise ValueError("socket raw mesh has no finite vertices")
    return xyz


def _barrel_dimensions_from_vertices(xyz: np.ndarray) -> tuple[float, float, float]:
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


def _barrel_dimensions(raw_mesh: Path) -> tuple[float, float, float]:
    return _barrel_dimensions_from_vertices(_mesh_vertices(raw_mesh))


def _region_rings(radius: float, bottom: float, top: float) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    return np.asarray([
        [radius * np.cos(angle), radius * np.sin(angle), z, 1.0]
        for z in (bottom, top) for angle in angles], dtype=np.float64)


def _project_hull(points: np.ndarray, pose_robot: np.ndarray,
                  robot_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    camera = (robot_camera @ pose_robot @ points.T)[:3].T
    if np.any(camera[:, 2] <= 0):
        raise ValueError("socket region is behind camera")
    projected = (intrinsic @ camera.T).T
    pixels = (projected[:, :2] / projected[:, 2, None]).astype(np.float32)
    return cv2.convexHull(pixels)


def _hull_center(hull: np.ndarray) -> np.ndarray:
    moments = cv2.moments(hull)
    if moments["m00"] <= 0:
        raise ValueError("socket region has no projected area")
    return np.asarray([moments["m10"], moments["m01"]]) / moments["m00"]


def _region_iou(hull: np.ndarray, mask: np.ndarray,
                mask_bounds: tuple[int, int, int, int]) -> float:
    """Compare only the union ROI, not a full-resolution frame per hypothesis."""
    pixels = np.rint(hull).astype(np.int32)
    x, y, width, height = cv2.boundingRect(pixels)
    left = max(0, min(x, mask_bounds[0]))
    top = max(0, min(y, mask_bounds[1]))
    right = min(mask.shape[1], max(x + width, mask_bounds[2]))
    bottom = min(mask.shape[0], max(y + height, mask_bounds[3]))
    silhouette = np.zeros((bottom - top, right - left), dtype=np.uint8)
    cv2.fillConvexPoly(
        silhouette, pixels - np.array([[[left, top]]], dtype=np.int32), 1)
    observed = mask[top:bottom, left:right]
    intersection = np.count_nonzero(silhouette & observed)
    union = np.count_nonzero(silhouette | observed)
    return float(intersection / union) if union else 0.0


def fit_board_socket_body(
    *, masks: Mapping[str, np.ndarray],
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_camera_world: Mapping[str, np.ndarray],
    c2r: np.ndarray, board_pose_robot: np.ndarray, raw_mesh: Path,
) -> tuple[np.ndarray, dict]:
    """Return world socket pose and per-view wall IoU from saved frame masks.

    The rigid fixture sits on the measured board and has its open axis along
    the board normal.  Each mask centroid supplies a board-plane intersection
    at the height of the mask region (wall, flange or their union). Region
    choice uses the known mesh silhouette and the acquired mask, not another
    learned model or a new pass/fail threshold. The median combines the
    resulting board XY points; axial yaw is unobservable for this cylinder.
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
        xyz = _mesh_vertices(raw_mesh)
        radius, bottom, top = _barrel_dimensions_from_vertices(xyz)
        radii = np.linalg.norm(xyz[:, :2], axis=1)
        flange_radius = float(radii.max())
        flange_vertices = xyz[np.isclose(radii, flange_radius, atol=1e-6)]
        flange_bottom = float(flange_vertices[:, 2].min())
        flange_top = float(flange_vertices[:, 2].max())
        models = {"wall": (_region_rings(radius, bottom, top),
                           (bottom + top) / 2.0)}
        if flange_radius > radius and flange_top > flange_bottom:
            flange = _region_rings(flange_radius, flange_bottom, flange_top)
            models["flange"] = (flange, (flange_bottom + flange_top) / 2.0)
            models["whole_body"] = (
                np.vstack((models["wall"][0], flange)),
                (flange_bottom + top) / 2.0)
    board_points: list[np.ndarray] = []
    views = []
    selected_regions: dict[str, str] = {}
    hypothesis_iou: dict[str, dict[str, float]] = {}
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
            mask_bounds = (int(xx.min()), int(yy.min()),
                           int(xx.max()) + 1, int(yy.max()) + 1)
            robot_camera = np.linalg.inv(E)
            origin = robot_camera[:3, 3]
            ray = robot_camera[:3, :3] @ np.linalg.solve(K, pixel)
            denominator = float(np.dot(ray, up))
            if abs(denominator) < 1e-8:
                raise ValueError(f"socket ray is parallel to the board: {camera}")
            scores = {}
            estimates = {}
            for region, (rings, height) in models.items():
                distance = float(np.dot(
                    board[:3, 3] + height * up - origin, up) / denominator)
                if distance <= 0:
                    raise ValueError(f"socket mask ray points behind camera: {camera}")
                guess = origin + distance * ray - height * up
                xy_guess = np.asarray([
                    (guess - board[:3, 3]) @ x_axis,
                    (guess - board[:3, 3]) @ y_axis])

                def hull_at(xy: np.ndarray) -> np.ndarray:
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = (board[:3, 3] + xy[0] * x_axis
                                   + xy[1] * y_axis)
                    return _project_hull(rings, pose, E, K)

                center = _hull_center(hull_at(xy_guess))
                jacobian = np.column_stack([
                    (_hull_center(hull_at(xy_guess + direction)) - center)
                    / 0.001
                    for direction in (np.array([0.001, 0.0]),
                                      np.array([0.0, 0.001]))])
                aligned_xy = xy_guess + np.linalg.solve(
                    jacobian, pixel[:2] - center)
                estimates[region] = aligned_xy
                scores[region] = _region_iou(
                    hull_at(aligned_xy), mask, mask_bounds)
            selected = max(scores, key=scores.__getitem__)
            chosen_xy = estimates[selected]
            board_points.append(board[:3, 3] + chosen_xy[0] * x_axis
                                + chosen_xy[1] * y_axis)
            views.append((camera, mask, K, E, mask_bounds))
            selected_regions[camera] = selected
            hypothesis_iou[camera] = scores

    with timed("socket.body_multiview_median", phase="socket",
               kind="geometry", camera_count=len(views)):
        points = np.asarray(board_points)
        xy = np.column_stack(((points - board[:3, 3]) @ x_axis,
                              (points - board[:3, 3]) @ y_axis))
        initial_median_xy = np.median(xy, axis=0)
        initial_residual_mm = np.linalg.norm(
            xy - initial_median_xy, axis=1) * 1000.0
        # Select the socket pose from this capture's internally consistent
        # centre cluster. Camera IDs are never blacklisted: a new socket-only
        # capture recomputes this set. The relative scale floor avoids
        # dropping equivalent views merely because the MAD is numerically
        # tiny, while retaining at least two views for the fixture pose.
        if len(xy) >= 3:
            residual_median = float(np.median(initial_residual_mm))
            residual_mad = float(np.median(np.abs(
                initial_residual_mm-residual_median)))
            residual_scale = max(
                1.4826*residual_mad, .1*max(residual_median, 1e-9))
            selection_cutoff_mm = residual_median + 3.*residual_scale
            inlier_indices = np.flatnonzero(
                initial_residual_mm <= selection_cutoff_mm)
            if len(inlier_indices) < 2:
                inlier_indices = np.argsort(initial_residual_mm)[:2]
        else:
            residual_median = float(np.median(initial_residual_mm))
            residual_mad = 0.
            selection_cutoff_mm = None
            inlier_indices = np.arange(len(xy))
        median_xy = np.median(xy[inlier_indices], axis=0)
        residual_mm = np.linalg.norm(xy - median_xy, axis=1) * 1000.0
        pairwise_mm = np.linalg.norm(
            xy[:, None, :] - xy[None, :, :], axis=2) * 1000.0
        pose_robot = np.eye(4)
        pose_robot[:3, :3] = rotation
        pose_robot[:3, 3] = (board[:3, 3] + median_xy[0] * x_axis
                             + median_xy[1] * y_axis)

    per_view = {}
    for camera, mask, K, E, mask_bounds in views:
        with timed("socket.body_wall_iou", phase="socket", kind="geometry",
                   camera_id=camera):
            rings = models[selected_regions[camera]][0]
            per_view[camera] = _region_iou(
                _project_hull(rings, pose_robot, E, K), mask, mask_bounds)
    diagnostics = {
        "method": "board_normal_mesh_region_mask_robust_inlier_median",
        "body_radius_m": radius, "body_bottom_z_m": bottom,
        "body_top_z_m": top, "per_view_iou": per_view,
        "flange_radius_m": flange_radius,
        "flange_bottom_z_m": flange_bottom,
        "flange_top_z_m": flange_top,
        "per_view_selected_region": selected_regions,
        "per_view_region_iou": hypothesis_iou,
        "mean_body_iou": float(np.mean(list(per_view.values()))),
        "per_view_board_xy_m": {
            camera: xy[index].tolist() for index, camera in enumerate(sorted(masks))},
        "per_view_center_residual_mm": {
            camera: float(residual_mm[index])
            for index, camera in enumerate(sorted(masks))},
        "per_view_initial_center_residual_mm": {
            camera: float(initial_residual_mm[index])
            for index, camera in enumerate(sorted(masks))},
        "socket_pose_inlier_camera_ids": [
            sorted(masks)[index] for index in inlier_indices],
        "socket_pose_excluded_camera_ids": [
            camera for index, camera in enumerate(sorted(masks))
            if index not in set(inlier_indices.tolist())],
        "socket_pose_selection": {
            "method": "median_mad_relative_cluster",
            "median_initial_residual_mm": residual_median,
            "mad_initial_residual_mm": residual_mad,
            "cutoff_mm": selection_cutoff_mm,
            "minimum_views_retained": min(2, len(xy)),
            "scope": "per_capture_pose_estimation_not_camera_blacklist",
        },
        "maximum_pairwise_center_disagreement_mm": float(np.max(pairwise_mm)),
        "board_xy_m": median_xy.tolist(),
    }
    return validate_se3(world_robot @ pose_robot,
                        name="board-constrained socket pose"), diagnostics
