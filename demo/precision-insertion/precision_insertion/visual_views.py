"""Choose the operator's fixed camera views for visual assessments."""

from __future__ import annotations

import os
from collections.abc import Iterable


def assessment_cameras(available: Iterable[str]) -> list[str]:
    """Use all views offline, or the configured live camera selection."""
    cameras = sorted(set(available))
    configured = os.environ.get("PRECISION_VLM_CAMERA_IDS", "").strip()
    if not configured:
        return cameras
    selected = [item.strip() for item in configured.split(",")]
    if (len(selected) not in {3, 4, 5} or len(set(selected)) != len(selected) or
            any(not item.isdecimal() for item in selected)):
        raise ValueError("PRECISION_VLM_CAMERA_IDS needs three to five distinct camera IDs")
    missing = set(selected) - set(cameras)
    if missing:
        raise ValueError(f"selected VLM cameras lack paired evidence: {sorted(missing)}")
    return sorted(selected)


def final_assessment_cameras(
    available: Iterable[str], *, selected: Iterable[str] | None = None,
) -> list[str]:
    """Allow the released-socket label to use a reviewed camera subset.

    The XY estimator and earlier VLM stages keep their own view selection.
    Without a dedicated final subset, retain the existing assessment views.
    """
    configured = os.environ.get("PRECISION_FINAL_VLM_CAMERA_IDS", "").strip()
    if selected is None and not configured:
        return assessment_cameras(available)
    cameras = set(available)
    chosen = (list(selected) if selected is not None else
              [item.strip() for item in configured.split(",")])
    if (len(chosen) < 2 or len(chosen) > 5 or
            len(set(chosen)) != len(chosen) or
            any(not isinstance(item, str) or not item.isdecimal()
                for item in chosen)):
        raise ValueError(
            "PRECISION_FINAL_VLM_CAMERA_IDS needs two to five distinct camera IDs")
    missing = set(chosen) - cameras
    if missing:
        raise ValueError(
            f"selected final VLM cameras lack paired evidence: {sorted(missing)}")
    return sorted(chosen)


def final_cameras_with_visible_socket(
    preferred: Iterable[str], available: Iterable[str],
    socket_projections: dict,
) -> tuple[list[str], dict[str, bool]]:
    """Replace a view with no projected socket by an available visible view.

    The reviewed camera order remains the default. This only handles a socket
    moved outside one camera's frame; projection area does not rank views.
    If no replacement is visible, keep the original camera and its existing
    full-frame VLM fallback.
    """
    chosen = sorted(set(preferred))
    cameras = sorted(set(available))
    visible = {}
    for camera in cameras:
        row = socket_projections.get(camera) or {}
        bbox, size = row.get("projected_hull_bbox_px"), row.get("image_size_wh")
        visible[camera] = bool(
            isinstance(bbox, list) and len(bbox) == 4 and
            isinstance(size, list) and len(size) == 2 and
            all(type(v) is int for v in bbox + size) and
            size[0] > 0 and size[1] > 0 and
            max(0, bbox[0]) < min(size[0], bbox[2]) and
            max(0, bbox[1]) < min(size[1], bbox[3]))
    replacements = [camera for camera in cameras
                    if camera not in chosen and visible[camera]]
    for camera in list(chosen):
        if not visible.get(camera, False) and replacements:
            chosen.remove(camera)
            chosen.append(replacements.pop(0))
    return sorted(chosen), visible
