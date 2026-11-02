"""Session-local camera selection shared by metric and visual checkpoints.

Camera serials are never blacklisted here.  A session's socket-only fit ranks
the currently available views, so moving the socket or recalibrating a camera
changes the next session's selection without editing a machine profile.
"""

from __future__ import annotations

import math

import numpy as np


def socket_consistent_alignment_cameras(
    calibration_record: dict, camera_ids,
) -> dict:
    """Select this session's geometrically consistent socket views.

    Prefer the inliers recorded by the latest board-constrained socket fit.
    Older records fall back to a robust cluster over their centre residuals.
    This ranks visual evidence; it does not add a session-failure condition.
    """
    available = list(dict.fromkeys(map(str, camera_ids)))
    refinements = calibration_record.get("socket_capture_refinements", [])
    if refinements:
        latest_body_fit = refinements[-1].get(
            "stock_selection", {}).get("body_fit", {})
        pose_inliers = latest_body_fit.get("socket_pose_inlier_camera_ids")
        if isinstance(pose_inliers, list):
            selected = [camera for camera in available
                        if camera in set(map(str, pose_inliers))]
            if len(selected) >= 2:
                scores = latest_body_fit.get(
                    "per_view_center_residual_mm", {})
                return {
                    "method": "session_socket_pose_inliers",
                    "selected_camera_ids": selected,
                    "excluded_camera_ids": [
                        camera for camera in available
                        if camera not in selected],
                    "per_view_socket_center_residual_mm": {
                        camera: float(value)
                        for camera, value in scores.items()
                        if camera in available},
                    "scope": "endpoint_view_selection_from_socket_pose_fit",
                }
    by_camera: dict[str, list[float]] = {
        camera: [] for camera in available}
    for refinement in refinements:
        body_fit = refinement.get("stock_selection", {}).get("body_fit", {})
        for camera, value in body_fit.get(
                "per_view_center_residual_mm", {}).items():
            if camera in by_camera:
                try:
                    residual = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(residual) and residual >= 0:
                    by_camera[camera].append(residual)
    scores = {camera: float(np.median(values))
              for camera, values in by_camera.items() if values}
    if len(scores) < 3:
        return {
            "method": "all_views_socket_residual_unavailable",
            "selected_camera_ids": available,
            "excluded_camera_ids": [],
            "per_view_socket_center_residual_mm": scores,
        }
    values = np.asarray(list(scores.values()), dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_scale = max(1.4826 * mad, .1 * max(median, 1e-9))
    cutoff = median + 3. * robust_scale
    selected = [camera for camera in available
                if camera not in scores or scores[camera] <= cutoff]
    if len(selected) < 2:
        selected = sorted(available, key=lambda camera: scores.get(
            camera, float("inf")))[:2]
    excluded = [camera for camera in available if camera not in selected]
    return {
        "method": "session_socket_center_median_mad_cluster",
        "selected_camera_ids": selected,
        "excluded_camera_ids": excluded,
        "per_view_socket_center_residual_mm": scores,
        "median_residual_mm": median,
        "mad_residual_mm": mad,
        "selection_cutoff_mm": cutoff,
        "scope": "endpoint_view_selection_not_camera_blacklist",
    }


def limit_ranked_camera_selection(selection: dict, max_views: int) -> dict:
    """Keep the lowest-residual session views without inventing a blacklist."""
    if type(max_views) is not int or not 2 <= max_views <= 5:
        raise ValueError("camera view limit must be an integer from two to five")
    selected = list(selection.get("selected_camera_ids", []))
    scores = selection.get("per_view_socket_center_residual_mm", {})
    ranked = sorted(
        selected,
        key=lambda camera: (float(scores.get(camera, float("inf"))), camera),
    )
    result = dict(selection)
    result["candidate_camera_ids"] = selected
    result["selected_camera_ids"] = ranked[:max_views]
    result["view_limit"] = max_views
    result["ranking"] = "ascending_session_socket_center_residual_mm"
    return result
