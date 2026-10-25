"""Track a white cylindrical key's visible end from real camera frames.

The initial AutoDex key mask supplies this camera's appearance. A projected
key shaft restricts association, while the visible end is measured from the
segmented component itself. Consecutive frames associate the same component;
an occluded end stays unobserved. No VLM pixel response enters this estimator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .grounded_alignment import (
    GroundedLineView, NEXT_TRIAL_XY_MAX_STEP_M,
    resolve_xy_only_cylinder_alignment,
)
from .xy_overlay import CalibratedXYFrame


@dataclass(frozen=True)
class KeyAppearance:
    lab_b_median: float
    luma_low: float
    reference_area_px: int

    @classmethod
    def from_reference(cls, image_bgr: np.ndarray,
                       mask: np.ndarray) -> "KeyAppearance":
        if (image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or
                mask.shape[:2] != image_bgr.shape[:2]):
            raise ValueError("key appearance needs aligned RGB image and mask")
        selected = mask[..., 0] > 0 if mask.ndim == 3 else mask > 0
        if int(np.count_nonzero(selected)) < 100:
            raise ValueError("initial key mask has too few pixels")
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[selected]
        return cls(float(np.median(lab[:, 2])),
                   float(np.percentile(lab[:, 0], 5)),
                   int(lab.shape[0]))

    def to_record(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TipObservation:
    tip_px: tuple[float, float]
    component_area_px: int
    component_bbox_xywh: tuple[int, int, int, int]
    key_color_fraction: float

    def to_record(self) -> dict:
        return {"tip_px": list(self.tip_px),
                "component_area_px": self.component_area_px,
                "component_bbox_xywh": list(self.component_bbox_xywh),
                "key_color_fraction": self.key_color_fraction}


def detect_visible_tip(
    image_bgr: np.ndarray, appearance: KeyAppearance, *,
    projected_rear_px: Sequence[float], projected_tip_px: Sequence[float],
    prior_tip_px: Sequence[float] | None = None,
) -> TipObservation | None:
    """Measure the centre of the key's visible end, or return no observation.

    The projected shaft defines a search corridor, not the returned tip. Lab
    colour is learned separately for each camera from its saved initial key
    mask. A component must extend back along the shaft so a white socket below
    the key is not mistaken for the key.
    """
    if (not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or
            image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8 or
            not isinstance(appearance, KeyAppearance)):
        raise ValueError("tip tracking needs a BGR uint8 frame and key appearance")
    rear = np.asarray(projected_rear_px, dtype=float)
    tip = np.asarray(projected_tip_px, dtype=float)
    if (rear.shape != (2,) or tip.shape != (2,) or
            not np.all(np.isfinite(rear)) or not np.all(np.isfinite(tip))):
        raise ValueError("projected shaft endpoints must be finite pixels")
    span = float(np.linalg.norm(tip - rear))
    if span < 20:
        return None
    axis = (tip - rear) / span
    lateral = np.array([-axis[1], axis[0]])
    anchor = (tip if prior_tip_px is None else
              np.asarray(prior_tip_px, dtype=float))
    if anchor.shape != (2,) or not np.all(np.isfinite(anchor)):
        raise ValueError("prior tip must be finite pixels")
    height, width = image_bgr.shape[:2]
    pad = max(120., 0.55 * span)
    bounds = np.stack((rear, tip, anchor))
    x0 = max(0, int(np.floor(bounds[:, 0].min() - pad)))
    y0 = max(0, int(np.floor(bounds[:, 1].min() - pad)))
    x1 = min(width, int(np.ceil(bounds[:, 0].max() + pad)))
    y1 = min(height, int(np.ceil(bounds[:, 1].max() + pad)))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    lab = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2LAB)
    colour = ((lab[:, :, 2] >= appearance.lab_b_median - 2) &
              (lab[:, :, 0] >= max(75., appearance.luma_low - 20)))
    mask = cv2.morphologyEx(
        colour.astype(np.uint8), cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for component in range(1, count):
        bx, by, bw, bh, area = (int(v) for v in stats[component])
        if not (max(400, .15 * appearance.reference_area_px) <= area <=
                max(6000, 6 * appearance.reference_area_px)):
            continue
        ys, xs = np.nonzero(labels[by:by + bh, bx:bx + bw] == component)
        xy = np.column_stack((xs + bx + x0, ys + by + y0)).astype(float)
        axial = (xy - tip) @ axis
        cross = (xy - tip) @ lateral
        t05, t95 = np.percentile(axial, [5, 95])
        c05, c95 = np.percentile(cross, [5, 95])
        if (t05 >= -.15 * span or t95 <= -.6 * span or
                t95 >= .45 * span or
                c95 - c05 >= max(180., 1.3 * span) or
                abs(float(np.median(cross))) >= max(160., .7 * span)):
            continue
        end = axial >= t95
        observed = tip + axis * float(np.median(axial[end])) + \
            lateral * float(np.median(cross[end]))
        score = float(np.linalg.norm(observed - anchor))
        candidates.append((score, -area, TipObservation(
            (float(observed[0]), float(observed[1])), area,
            (bx + x0, by + y0, bw, bh),
            float(np.mean(colour[by:by + bh, bx:bx + bw]
                          [labels[by:by + bh, bx:bx + bw] == component])))))
    return min(candidates, key=lambda row: (row[0], row[1]))[2] if candidates else None


def track_tip_frames(
    frames: Sequence[tuple[int, int, np.ndarray]],
    appearance: KeyAppearance, *, target_frame_id: int,
    projected_rear_px: Sequence[float], projected_tip_px: Sequence[float],
) -> dict:
    """Associate the key across a short video window around an exact frame ID."""
    if not frames or len({row[0] for row in frames}) != len(frames):
        raise ValueError("tracking needs unique camera frame IDs")
    ordered = sorted(frames, key=lambda row: row[0])
    anchor_index = min(range(len(ordered)),
                       key=lambda i: abs(ordered[i][0] - target_frame_id))
    anchor_frame = ordered[anchor_index]
    if abs(anchor_frame[0] - target_frame_id) > 2:
        return {"status": "target_frame_missing", "target_frame_id": target_frame_id,
                "observations": []}
    observations = {}
    anchor = detect_visible_tip(
        anchor_frame[2], appearance,
        projected_rear_px=projected_rear_px,
        projected_tip_px=projected_tip_px)
    observations[anchor_index] = anchor
    for direction in (-1, 1):
        prior = None if anchor is None else anchor.tip_px
        for index in range(anchor_index + direction,
                           -1 if direction < 0 else len(ordered), direction):
            found = detect_visible_tip(
                ordered[index][2], appearance,
                projected_rear_px=projected_rear_px,
                projected_tip_px=projected_tip_px, prior_tip_px=prior)
            observations[index] = found
            if found is not None:
                prior = found.tip_px
    trace = [{"source_frame_id": fid, "read_unix_ns": timestamp_ns,
              "observation": None if observations[i] is None else
              observations[i].to_record()}
             for i, (fid, timestamp_ns, _image) in enumerate(ordered)]
    return {"status": "observed" if anchor is not None else "occluded",
            "target_frame_id": target_frame_id,
            "observed_frame_id": anchor_frame[0],
            "tip_px": None if anchor is None else list(anchor.tip_px),
            "observations": trace}


def tracked_tip_xy_alignment(
    views: Sequence[GroundedLineView], *, nominal_tip_z_socket_m: float,
    search_index: int, max_reprojection_px: float, min_parallax_deg: float,
    max_nominal_tip_height_error_m: float,
) -> dict:
    """Convert corresponding visible key tips into bounded next-trial XY.

    With three or more views, use one robust multi-view fit. This avoids a
    misleading two-view hypothesis winning solely because both points have a
    small pixel residual. Sparse views use the existing AutoDex XY fallback.
    """
    fallback = lambda selected: resolve_xy_only_cylinder_alignment(
        selected, {"status": "abstain"},
        nominal_tip_z_socket_m=nominal_tip_z_socket_m,
        search_index=search_index, max_reprojection_px=max_reprojection_px,
        min_parallax_deg=min_parallax_deg,
        max_nominal_tip_height_error_m=max_nominal_tip_height_error_m)
    if len(views) < 3:
        return dict(fallback(views), estimator="tracked_visible_key_tip")

    projections = [np.asarray(view.frame.intrinsics, dtype=float) @
                   np.asarray(view.frame.T_camera_socket, dtype=float)[:3]
                   for view in views]
    pixels = [np.asarray(view.tip_uv_px, dtype=float) for view in views]
    rows = []
    for projection, pixel in zip(projections, pixels):
        rows.extend((pixel[0] * projection[2] - projection[0],
                     pixel[1] * projection[2] - projection[1]))
    _u, _s, vh = np.linalg.svd(np.asarray(rows))
    seed = vh[-1]
    if abs(seed[3]) < 1e-9:
        return dict(fallback([]), estimator="tracked_visible_key_tip",
                    tracker_reason="triangulation_at_infinity")
    seed = seed[:3] / seed[3]

    def residual(point):
        values = []
        for projection, pixel in zip(projections, pixels):
            h = projection @ np.r_[point, 1.]
            values.extend(h[:2] / h[2] - pixel)
        return np.asarray(values)

    fit = least_squares(residual, seed, loss="soft_l1", f_scale=7.)
    errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    # A segmented cylinder end spans several pixels. The camera-consistency
    # tolerance is an observation precision, not a robot motion gate.
    pixel_tolerance = max(15., 4 * max_reprojection_px)
    inlier_indices = np.flatnonzero(errors <= pixel_tolerance)
    if fit.success and 3 <= len(inlier_indices) < len(views):
        # The robust all-view fit identifies bad camera observations. Refit
        # the reported point using only the accepted views; otherwise a large
        # outlier still shifts the command correction by several millimetres.
        def accepted_residual(point):
            return residual(point).reshape(-1, 2)[inlier_indices].ravel()

        refined = least_squares(accepted_residual, fit.x)
        if refined.success and np.all(np.isfinite(refined.x)):
            refined_errors = np.linalg.norm(
                residual(refined.x).reshape(-1, 2), axis=1)
            if np.all(refined_errors[inlier_indices] <= pixel_tolerance):
                fit = refined
                errors = refined_errors
    inliers = [views[index].frame.camera_id for index in inlier_indices]
    if (not fit.success or not np.all(np.isfinite(fit.x)) or
            len(inliers) < 3 or
            abs(fit.x[2] - nominal_tip_z_socket_m) >
            max_nominal_tip_height_error_m):
        return dict(fallback([]), estimator="tracked_visible_key_tip",
                    tracker_reason="no_consistent_multiview_tip",
                    tracker_reprojection_px={
                        view.frame.camera_id: float(error)
                        for view, error in zip(views, errors)})
    correction = -fit.x[:2]
    magnitude = float(np.linalg.norm(correction))
    increment = correction * min(
        1., NEXT_TRIAL_XY_MAX_STEP_M / max(magnitude, 1e-12))
    return {
        "status": "diagnostic_metric_xy_correction",
        "reason": "multiview_tip_xy_vertical_axis_assumption",
        "xy_source": "multiview_tip_xy_vertical_axis_assumption",
        "estimator": "tracked_visible_key_tip",
        "inlier_cameras": inliers,
        "nominal_tip_z_socket_m": float(nominal_tip_z_socket_m),
        "tip_socket_m": fit.x.tolist(),
        "xy_correction_socket_m": correction.tolist(),
        "bounded_xy_increment_socket_m": increment.tolist(),
        "xy_correction_norm_m": magnitude,
        "increment_norm_m": float(np.linalg.norm(increment)),
        "tracker_reprojection_px": {
            view.frame.camera_id: float(error)
            for view, error in zip(views, errors)},
    }


def tracked_view(
    *, camera_id: str, timestamp_s: float, frame_id: int,
    preinsert_bgr: np.ndarray, reference_bgr: np.ndarray,
    reference_mask: np.ndarray, T_camera_socket: np.ndarray,
    intrinsics: np.ndarray, projected_rear_px: Sequence[float],
    projected_tip_px: Sequence[float],
) -> tuple[GroundedLineView | None, dict]:
    """Measure one saved camera view with its own initial key appearance."""
    appearance = KeyAppearance.from_reference(reference_bgr, reference_mask)
    observation = detect_visible_tip(
        preinsert_bgr, appearance, projected_rear_px=projected_rear_px,
        projected_tip_px=projected_tip_px)
    shaft_px = float(np.linalg.norm(
        np.asarray(projected_tip_px) - np.asarray(projected_rear_px)))
    visible = usable_tip_observation(observation, appearance, shaft_px)
    record = {
        "camera_id": camera_id, "frame_id": int(frame_id),
        "timestamp_s": float(timestamp_s),
        "projected_rear_px": list(map(float, projected_rear_px)),
        "projected_tip_px": list(map(float, projected_tip_px)),
        "reference_appearance": appearance.to_record(),
        "observation": None if observation is None else observation.to_record(),
        "admitted": bool(visible),
    }
    if not visible:
        return None, record
    frame = CalibratedXYFrame(
        camera_id, timestamp_s,
        Image.fromarray(cv2.cvtColor(preinsert_bgr, cv2.COLOR_BGR2RGB)),
        np.asarray(T_camera_socket, dtype=float),
        np.asarray(intrinsics, dtype=float))
    return GroundedLineView(frame, observation.tip_px, None), record


def usable_tip_observation(
    observation: TipObservation | None, appearance: KeyAppearance,
    shaft_span_px: float,
) -> bool:
    """Reject weak components that match the white socket or Inspire hand."""
    return (observation is not None and shaft_span_px >= 100 and
            observation.component_area_px >= max(
                3000, .25 * appearance.reference_area_px) and
            observation.key_color_fraction >= .90)
