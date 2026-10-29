"""Compare the same calibrated socket pixels before and after a trial.

The initial tabletop snapshot shows an empty socket. Both captures come from
fixed, undistorted cameras in the same session. This measures a visible change
in the socket mouth, not penetration depth or the identity of that change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .observer import LabeledFrame


# A pixel change is recorded as a diagnostic only. It is not a success rule:
# motion, illumination and occlusion can also change these pixels.
CHANGED_PIXEL_DELTA = 15.


def socket_temporal_change(
    initial_frames: Sequence[LabeledFrame],
    final_frames: Sequence[LabeledFrame],
    socket_projections: Mapping[str, dict],
) -> dict:
    """Measure change in each calibrated socket mouth without labeling it."""
    before = {frame.camera_id: frame for frame in initial_frames}
    after = {frame.camera_id: frame for frame in final_frames}
    if set(before) != set(after):
        raise ValueError("socket temporal comparison needs paired cameras")
    rows = []
    for camera in sorted(before):
        first, last = before[camera], after[camera]
        if first.image.size != last.image.size:
            raise ValueError("socket temporal camera geometry changed")
        projection = socket_projections.get(camera) or {}
        bbox = projection.get("projected_hull_bbox_px")
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(value) is not int for value in bbox)):
            continue
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0, y1 - y0
        center_x = (x0 + x1) / 2
        roi = (
            max(0, int(center_x - .22 * width)),
            max(0, int(y0 - .40 * height)),
            min(first.image.width, int(center_x + .22 * width)),
            min(first.image.height, int(y0 + .45 * height)),
        )
        if roi[0] >= roi[2] or roi[1] >= roi[3]:
            continue
        a = np.asarray(first.image.crop(roi), dtype=np.int16)
        b = np.asarray(last.image.crop(roi), dtype=np.int16)
        delta = np.abs(a - b).mean(axis=2)
        fraction = float(np.mean(delta > CHANGED_PIXEL_DELTA))
        rows.append({
            "camera_id": camera, "roi_xyxy_full_px": list(roi),
            "changed_fraction": fraction,
            "changed_pixel_delta": CHANGED_PIXEL_DELTA,
        })
    return {
        "schema": "precision_insertion_socket_temporal_change_v1",
        "per_view": rows,
        "median_changed_fraction": (
            float(np.median([row["changed_fraction"] for row in rows]))
            if rows else None),
        "scope": "visual_change_not_key_identity_or_20mm_depth",
    }
