"""Fit an upright fixed socket from its acquired SAM masks and support plane.

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


def fit_support_surface_socket_body(
    *, masks: Mapping[str, np.ndarray],
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_camera_world: Mapping[str, np.ndarray],
    c2r: np.ndarray, support_pose_robot: np.ndarray, raw_mesh: Path,
    support_source: str = "measured_support_surface",
) -> tuple[np.ndarray, dict]:
    """Return world socket pose and per-view wall IoU from saved frame masks.

    The rigid fixture sits on the measured support surface and has its open
    axis along the support normal. Each mask centroid supplies a plane intersection
    at the height of the mask region (wall, flange or their union). Region
    choice uses the known mesh silhouette and the acquired mask, not another
    learned model or a new pass/fail threshold. One median over every
    available view combines the resulting board XY points; there is no
    camera-centre consistency ranking or per-session view selection. Axial
    yaw is unobservable for this cylinder.
    """
    if (not masks or set(masks) != set(intrinsics_undist) or
            set(masks) != set(extrinsics_camera_world)):
        raise ValueError("socket wall fit requires matching calibrated views")
    support = validate_se3(support_pose_robot, name="measured support pose")
    world_robot = validate_se3(c2r, name="C2R")
    up = support[:3, 2]
    x_axis = support[:3, 0]
    y_axis = support[:3, 1]
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
    support_points: list[np.ndarray] = []
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
                    support[:3, 3] + height * up - origin, up) / denominator)
                if distance <= 0:
                    raise ValueError(f"socket mask ray points behind camera: {camera}")
                guess = origin + distance * ray - height * up
                xy_guess = np.asarray([
                    (guess - support[:3, 3]) @ x_axis,
                    (guess - support[:3, 3]) @ y_axis])

                def hull_at(xy: np.ndarray) -> np.ndarray:
                    pose = np.eye(4)
                    pose[:3, :3] = rotation
                    pose[:3, 3] = (support[:3, 3] + xy[0] * x_axis
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
            support_points.append(support[:3, 3] + chosen_xy[0] * x_axis
                                + chosen_xy[1] * y_axis)
            views.append((camera, mask, K, E, mask_bounds))
            selected_regions[camera] = selected
            hypothesis_iou[camera] = scores

    with timed("socket.body_multiview_median", phase="socket",
               kind="geometry", camera_count=len(views)):
        points = np.asarray(support_points)
        xy = np.column_stack(((points - support[:3, 3]) @ x_axis,
                              (points - support[:3, 3]) @ y_axis))
        median_xy = np.median(xy, axis=0)
        pose_robot = np.eye(4)
        pose_robot[:3, :3] = rotation
        pose_robot[:3, 3] = (support[:3, 3] + median_xy[0] * x_axis
                             + median_xy[1] * y_axis)

    per_view = {}
    for camera, mask, K, E, mask_bounds in views:
        with timed("socket.body_wall_iou", phase="socket", kind="geometry",
                   camera_id=camera):
            rings = models[selected_regions[camera]][0]
            per_view[camera] = _region_iou(
                _project_hull(rings, pose_robot, E, K), mask, mask_bounds)
    diagnostics = {
        "method": "support_normal_mesh_region_mask_all_view_median",
        "support_source": support_source,
        "body_radius_m": radius, "body_bottom_z_m": bottom,
        "body_top_z_m": top, "per_view_iou": per_view,
        "flange_radius_m": flange_radius,
        "flange_bottom_z_m": flange_bottom,
        "flange_top_z_m": flange_top,
        "per_view_selected_region": selected_regions,
        "per_view_region_iou": hypothesis_iou,
        "mean_body_iou": float(np.mean(list(per_view.values()))),
        "per_view_support_xy_m": {
            camera: xy[index].tolist() for index, camera in enumerate(sorted(masks))},
        "socket_pose_camera_ids": sorted(masks),
        "socket_pose_aggregation": {
            "method": "median_all_available_views",
            "camera_count": len(masks),
            "scope": "no_camera_center_consistency_comparison_or_selection",
        },
        "support_xy_m": median_xy.tolist(),
    }
    return validate_se3(world_robot @ pose_robot,
                        name="support-constrained socket pose"), diagnostics


def fit_support_surface_key_body(
    *, masks: Mapping[str, np.ndarray],
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_camera_world: Mapping[str, np.ndarray],
    c2r: np.ndarray, support_pose_robot: np.ndarray, raw_mesh: Path,
    orientation_robot: np.ndarray, center_local: np.ndarray,
    support_source: str = "measured_support_surface",
) -> tuple[np.ndarray, dict]:
    """Fit a known tabletop key at a fixed support-relative orientation.

    FoundPose can explain a single silhouette at the wrong depth along that
    camera's ray.  The tabletop class supplies the cylinder orientation and
    the frozen support plane supplies its missing depth.  Each admitted mask
    then estimates the two tangential coordinates by aligning the projected
    mesh hull to the mask centroid; the median combines all available views.

    This is a geometric pose refinement, not a new acceptance threshold.  It
    works for both an end-down and a side-down cylinder because contact with
    the plane is computed from the complete metric mesh after applying the
    requested orientation.
    """
    if (not masks or set(masks) != set(intrinsics_undist) or
            set(masks) != set(extrinsics_camera_world)):
        raise ValueError("key support fit requires matching calibrated views")
    support = validate_se3(support_pose_robot, name="measured support pose")
    world_robot = validate_se3(c2r, name="C2R")
    rotation = np.asarray(orientation_robot, dtype=np.float64)
    center = np.asarray(center_local, dtype=np.float64)
    if (rotation.shape != (3, 3) or center.shape != (3,) or
            not np.all(np.isfinite(rotation)) or
            not np.all(np.isfinite(center)) or
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or
            np.linalg.det(rotation) < 0.999):
        raise ValueError("key support fit needs a proper orientation and center")
    xyz = _mesh_vertices(raw_mesh)
    xyz1 = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
    up = support[:3, 2]
    x_axis = support[:3, 0]
    y_axis = support[:3, 1]
    rotated = xyz @ rotation.T
    support_clearance = -float(np.min(rotated @ up))
    center_height = float(np.dot(rotation @ center, up) + support_clearance)

    def pose_at(xy: np.ndarray) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = (support[:3, 3] + float(xy[0]) * x_axis
                       + float(xy[1]) * y_axis + support_clearance * up)
        return pose

    estimates: dict[str, np.ndarray] = {}
    views = []
    for camera in sorted(masks):
        with timed("key.body_centroid_ray", phase="key", kind="geometry",
                   camera_id=camera):
            mask = np.asarray(masks[camera], dtype=bool)
            K = np.asarray(intrinsics_undist[camera], dtype=np.float64)
            E = validate_se3(extrinsics_camera_world[camera],
                             name=f"{camera} camera/world") @ world_robot
            if (mask.ndim != 2 or not mask.any() or K.shape != (3, 3) or
                    not np.all(np.isfinite(K)) or K[0, 0] <= 0 or
                    K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1], atol=1e-8)):
                raise ValueError(f"invalid key image or intrinsics: {camera}")
            yy, xx = np.nonzero(mask)
            pixel = np.array([float(xx.mean()), float(yy.mean()), 1.0])
            bounds = (int(xx.min()), int(yy.min()), int(xx.max()) + 1,
                      int(yy.max()) + 1)
            robot_camera = np.linalg.inv(E)
            origin = robot_camera[:3, 3]
            ray = robot_camera[:3, :3] @ np.linalg.solve(K, pixel)
            denominator = float(np.dot(ray, up))
            if abs(denominator) < 1e-8:
                raise ValueError(f"key ray is parallel to support plane: {camera}")
            distance = float(np.dot(
                support[:3, 3] + center_height * up - origin, up) / denominator)
            if distance <= 0:
                raise ValueError(f"key mask ray points behind camera: {camera}")
            center_guess = origin + distance * ray
            origin_guess = center_guess - rotation @ center
            xy = np.asarray([
                np.dot(origin_guess - support[:3, 3], x_axis),
                np.dot(origin_guess - support[:3, 3], y_axis),
            ])

            def hull_at(value: np.ndarray) -> np.ndarray:
                return _project_hull(xyz1, pose_at(value), E, K)

            # Two local image-Jacobian steps are enough for the millimetre-
            # scale tangential correction while retaining the exact support
            # constraint at every iterate.
            for _ in range(2):
                projected_center = _hull_center(hull_at(xy))
                jacobian = np.column_stack([
                    (_hull_center(hull_at(xy + direction)) - projected_center)
                    / 0.001
                    for direction in (np.array([0.001, 0.0]),
                                      np.array([0.0, 0.001]))
                ])
                delta, *_ = np.linalg.lstsq(
                    jacobian, pixel[:2] - projected_center, rcond=None)
                xy = xy + delta
            estimates[camera] = xy
            views.append((camera, mask, K, E, bounds))

    with timed("key.body_multiview_median", phase="key", kind="geometry",
               camera_count=len(views)):
        per_camera_xy = np.asarray([estimates[camera]
                                    for camera in sorted(estimates)])
        median_xy = np.median(per_camera_xy, axis=0)
        pose_robot = pose_at(median_xy)

    per_view = {}
    for camera, mask, K, E, bounds in views:
        with timed("key.body_hull_iou", phase="key", kind="geometry",
                   camera_id=camera):
            per_view[camera] = _region_iou(
                _project_hull(xyz1, pose_robot, E, K), mask, bounds)
    diagnostics = {
        "method": "support_normal_full_mesh_mask_all_view_median",
        "support_source": support_source,
        "support_clearance_m": support_clearance,
        "center_height_m": center_height,
        "per_view_iou": per_view,
        "mean_body_iou": float(np.mean(list(per_view.values()))),
        "per_view_support_xy_m": {
            camera: estimates[camera].tolist() for camera in sorted(estimates)},
        "support_xy_m": median_xy.tolist(),
        "camera_ids": sorted(masks),
        "aggregation": {
            "method": "median_all_available_views",
            "camera_count": len(masks),
            "scope": "tabletop_support_geometry_not_an_acceptance_gate",
        },
    }
    return validate_se3(world_robot @ pose_robot,
                        name="support-constrained key pose"), diagnostics


def fit_board_socket_body(
    *, masks: Mapping[str, np.ndarray],
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_camera_world: Mapping[str, np.ndarray],
    c2r: np.ndarray, board_pose_robot: np.ndarray, raw_mesh: Path,
) -> tuple[np.ndarray, dict]:
    """Compatibility wrapper for the ChArUco-supported cylinder route."""
    board = validate_se3(board_pose_robot, name="measured board pose")
    up = -board[:3, 2]
    support = np.eye(4)
    support[:3, 0] = board[:3, 0]
    support[:3, 1] = np.cross(up, support[:3, 0])
    support[:3, 2] = up
    support[:3, 3] = board[:3, 3]
    pose, diagnostics = fit_support_surface_socket_body(
        masks=masks, intrinsics_undist=intrinsics_undist,
        extrinsics_camera_world=extrinsics_camera_world, c2r=c2r,
        support_pose_robot=support, raw_mesh=raw_mesh,
        support_source="measured_charuco_board")
    # Preserve legacy diagnostic keys for saved AutoDex/board evidence.
    diagnostics["method"] = "board_normal_mesh_region_mask_all_view_median"
    diagnostics["per_view_board_xy_m"] = diagnostics.pop(
        "per_view_support_xy_m")
    diagnostics["board_xy_m"] = diagnostics.pop("support_xy_m")
    return pose, diagnostics
