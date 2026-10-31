"""Session-local ChArUco camera selection for the capture4 rig.

The stock tabletop fitter is the source of the geometric quality checks.  A
single miscalibrated view can, however, bias its initial all-view DLT before
the per-corner residual refit has an inlier.  This wrapper retries the same
unchanged fit on the largest camera subsets and records which subset actually
agreed.  Camera serials are never blacklisted across sessions.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

import numpy as np


def _score(measurement: dict) -> tuple[float, float, float]:
    metrics = measurement.get("metrics", {})
    return (
        float(metrics.get("reprojection_median_px", float("inf"))),
        float(metrics.get("rigid_fit_rms_mm", float("inf"))),
        float(metrics.get("plane_rms_mm", float("inf"))),
    )


def measure_tabletop_with_view_selection(
    images_bgr: Mapping[str, np.ndarray],
    intrinsics_full: Mapping,
    extrinsics_full: Mapping,
    c2r: np.ndarray,
    *,
    board_id: str,
    minimum_board_corners: int,
) -> dict:
    """Run the stock fit on the largest geometrically consistent view set.

    The all-view fit remains the fast path.  Only when it fails do we enumerate
    same-size subsets, stopping at the first subset size with a valid stock
    measurement.  Selection therefore removes a session outlier without
    weakening any stock reprojection, plane, rigid-fit or corner-coverage
    condition.
    """
    from src.execution.charuco_tabletop import measure_tabletop_from_images

    cameras = sorted(images_bgr)
    if len(cameras) < 2:
        return measure_tabletop_from_images(
            images_bgr, intrinsics_full, extrinsics_full, c2r,
            board_id=board_id,
            minimum_board_corners=minimum_board_corners,
        )

    attempts: list[dict] = []
    first_error: RuntimeError | None = None
    for subset_size in range(len(cameras), 1, -1):
        accepted: list[tuple[tuple[float, float, float], tuple[str, ...], dict]] = []
        for subset in combinations(cameras, subset_size):
            selected_images = {camera: images_bgr[camera] for camera in subset}
            try:
                measurement = measure_tabletop_from_images(
                    selected_images, intrinsics_full, extrinsics_full, c2r,
                    board_id=board_id,
                    minimum_board_corners=minimum_board_corners,
                )
            except RuntimeError as exc:
                if first_error is None:
                    first_error = exc
                attempts.append({
                    "camera_ids": list(subset),
                    "accepted": False,
                    "error": str(exc),
                })
                continue
            score = _score(measurement)
            attempts.append({
                "camera_ids": list(subset),
                "accepted": True,
                "score": list(score),
            })
            accepted.append((score, subset, measurement))
        if not accepted:
            continue
        score, selected, measurement = min(
            accepted, key=lambda row: (row[0], row[1]))
        selected_set = set(selected)
        selection = {
            "method": (
                "all_views_stock_fit" if len(selected) == len(cameras)
                else "largest_stock_consistent_camera_subset"
            ),
            "available_camera_ids": cameras,
            "selected_camera_ids": list(selected),
            "excluded_camera_ids": [
                camera for camera in cameras if camera not in selected_set
            ],
            "selected_score": list(score),
            "attempts": attempts,
            "scope": "session_board_fit_not_persistent_camera_blacklist",
        }
        measurement = dict(measurement)
        measurement["camera_view_selection"] = selection
        metrics = dict(measurement.get("metrics", {}))
        metrics["camera_view_selection"] = selection
        measurement["metrics"] = metrics
        return measurement

    if first_error is not None:
        raise first_error
    raise RuntimeError("ChArUco board fit has no two-camera subset")
