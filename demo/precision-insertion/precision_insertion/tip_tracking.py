"""Track a white cylindrical key's visible end from real camera frames.

The initial AutoDex key mask supplies this camera's appearance. A projected
key shaft restricts association, while the visible end is measured from the
segmented component itself. Consecutive frames associate the same component;
an occluded end stays unobserved. No VLM pixel response enters this estimator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from .grounded_alignment import (
    GroundedLineView, NEXT_TRIAL_XY_MAX_STEP_M,
    _parallax, _project, resolve_xy_only_cylinder_alignment,
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

    With three or more views, first use one robust multi-view fit. This avoids
    a misleading two-view hypothesis winning solely because both points have a
    small pixel residual. If the visible partial-face centres are not the same
    triangulable 3D point, intersect every valid camera ray with the commanded
    tip-height plane and robustly fuse those XY hypotheses. This fallback uses
    all available cameras and never blacklists a camera serial. One or two
    usable views retain their explicit planar or triangulated hypotheses; no
    visible tip means no XY advice.
    """
    def fallback(selected):
        proposal = resolve_xy_only_cylinder_alignment(
            selected, {"status": "abstain"},
            nominal_tip_z_socket_m=nominal_tip_z_socket_m,
            search_index=search_index, max_reprojection_px=max_reprojection_px,
            min_parallax_deg=min_parallax_deg,
            max_nominal_tip_height_error_m=max_nominal_tip_height_error_m)
        if proposal["status"] != "diagnostic_exploratory_xy_step":
            return proposal
        # The VLM route may explicitly explore after an unobservable image.
        # A tracker/SAM result without a consistent visible tip has no point
        # from which to derive a wrist correction.
        return {
            "status": "abstain", "reason": "no_consistent_visible_key_tip",
            "xy_source": "none", "inlier_cameras": [],
            "nominal_tip_z_socket_m": float(nominal_tip_z_socket_m),
            "tip_socket_m": None,
        }

    def robust_planar_fallback(selected, *, tracker_reason: str):
        """Fuse calibrated ray/known-height intersections without depth claims.

        A partly occluded circular face can yield a slightly different proxy
        centre in each camera, so forcing those pixels to triangulate as one
        3D point is sometimes inappropriate. The pre-insertion controller does
        know the commanded tip height. Equal-weight geometric median fusion on
        that plane is invariant to socket-frame XY rotation and has the usual
        50% breakdown point; unlike the former fallback, contour smoothness in
        one image cannot by itself select the wrist command.
        """
        hypotheses = []
        for view in selected:
            one = fallback([view])
            point = one.get("tip_socket_m")
            if (one.get("status") != "diagnostic_planar_xy_hypothesis" or
                    not isinstance(point, list) or len(point) != 3):
                continue
            xy = np.asarray(point[:2], dtype=float)
            if np.all(np.isfinite(xy)):
                hypotheses.append((view.frame.camera_id, xy))
        if len(hypotheses) < 2:
            return None

        points = np.stack([xy for _camera, xy in hypotheses])
        # Weiszfeld's algorithm for the geometric median. The tolerance is
        # numerical only; it neither accepts nor rejects a camera observation.
        centre = np.mean(points, axis=0)
        for _ in range(64):
            distances = np.linalg.norm(points - centre, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= 1e-12:
                centre = points[nearest].copy()
                break
            weights = 1. / distances
            updated = np.sum(points * weights[:, None], axis=0) / np.sum(weights)
            if np.linalg.norm(updated - centre) <= 1e-12:
                centre = updated
                break
            centre = updated

        correction = -centre
        magnitude = float(np.linalg.norm(correction))
        increment = correction * min(
            1., NEXT_TRIAL_XY_MAX_STEP_M / max(magnitude, 1e-12))
        return {
            "status": "diagnostic_planar_xy_hypothesis",
            "reason": "robust_multiview_nominal_tip_height_plane",
            "xy_source": "robust_multiview_nominal_tip_height_plane",
            "estimator": "tracked_visible_key_tip",
            "inlier_cameras": [camera for camera, _xy in hypotheses],
            "nominal_tip_z_socket_m": float(nominal_tip_z_socket_m),
            "tip_socket_m": [float(centre[0]), float(centre[1]),
                             float(nominal_tip_z_socket_m)],
            "xy_correction_socket_m": correction.tolist(),
            "bounded_xy_increment_socket_m": increment.tolist(),
            "xy_correction_norm_m": magnitude,
            "increment_norm_m": float(np.linalg.norm(increment)),
            "tracker_reason": tracker_reason,
            "planar_hypotheses_socket_xy_m": {
                camera: xy.tolist() for camera, xy in hypotheses},
        }
    if len(views) < 3:
        return dict(fallback(views), estimator="tracked_visible_key_tip")
    if _parallax(views) < min_parallax_deg:
        planar = robust_planar_fallback(
            views, tracker_reason="insufficient_camera_parallax")
        if planar is not None:
            return planar
        return dict(fallback(views), estimator="tracked_visible_key_tip",
                    tracker_reason="insufficient_camera_parallax")

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
    seed = None if abs(seed[3]) < 1e-9 else seed[:3] / seed[3]

    def residual(point):
        values = []
        for view, pixel in zip(views, pixels):
            projected = _project(view, point)
            values.extend((1e6, 1e6) if projected is None else
                          projected - pixel)
        return np.asarray(values)

    # A segmented cylinder end spans several pixels. The camera-consistency
    # tolerance is an observation precision, not a robot motion gate.
    pixel_tolerance = max(15., 4 * max_reprojection_px)
    fit = None if seed is None else least_squares(
        residual, seed, loss="soft_l1", f_scale=7.)

    def assess(candidate):
        if candidate is None or not candidate.success or not np.all(
                np.isfinite(candidate.x)):
            return None
        errors = np.linalg.norm(residual(candidate.x).reshape(-1, 2), axis=1)
        inliers = np.flatnonzero(errors <= pixel_tolerance)
        if len(inliers) < 3:
            return None
        if len(inliers) < len(views):
            # Fit the reported point to observations we actually accepted.
            def selected(point):
                return residual(point).reshape(-1, 2)[inliers].ravel()

            candidate = least_squares(selected, candidate.x)
            if not candidate.success or not np.all(np.isfinite(candidate.x)):
                return None
            errors = np.linalg.norm(residual(candidate.x).reshape(-1, 2), axis=1)
            if np.any(errors[inliers] > pixel_tolerance):
                return None
        if (abs(candidate.x[2] - nominal_tip_z_socket_m) >
                max_nominal_tip_height_error_m):
            return None
        if _parallax([views[index] for index in inliers]) < min_parallax_deg:
            return None
        return candidate, errors, inliers

    accepted = assess(fit)
    if accepted is None:
        # A single grossly wrong camera point can corrupt the all-view SVD
        # seed before the robust loss has a chance to reject it. Retry only
        # that case with deterministic two-camera seeds, ranking by how many
        # independently projected views agree. This adds no new threshold.
        candidates = []
        for first, second in combinations(range(len(views)), 2):
            pair_rows = []
            for index in (first, second):
                projection, pixel = projections[index], pixels[index]
                pair_rows.extend((pixel[0] * projection[2] - projection[0],
                                  pixel[1] * projection[2] - projection[1]))
            _u, _s, pair_vh = np.linalg.svd(np.asarray(pair_rows))
            homogeneous = pair_vh[-1]
            if abs(homogeneous[3]) < 1e-9:
                continue
            point = homogeneous[:3] / homogeneous[3]
            if not np.all(np.isfinite(point)):
                continue
            pair_errors = np.linalg.norm(residual(point).reshape(-1, 2), axis=1)
            pair_inliers = np.flatnonzero(pair_errors <= pixel_tolerance)
            if len(pair_inliers) < 3:
                continue
            def selected(xyz):
                return residual(xyz).reshape(-1, 2)[pair_inliers].ravel()

            result = assess(least_squares(selected, point))
            if result is not None:
                refined, refined_errors, refined_inliers = result
                candidates.append((
                    -len(refined_inliers),
                    float(np.sum(refined_errors[refined_inliers] ** 2)),
                    first, second, result))
        if candidates:
            accepted = min(candidates, key=lambda row: row[:4])[4]
    if accepted is None:
        reason = ("triangulation_at_infinity" if seed is None else
                  "no_consistent_multiview_tip")
        planar = robust_planar_fallback(views, tracker_reason=reason)
        if planar is not None:
            return planar
        report = dict(fallback([]), estimator="tracked_visible_key_tip",
                      tracker_reason=reason)
        if fit is not None and fit.success and np.all(np.isfinite(fit.x)):
            primary_errors = np.linalg.norm(
                residual(fit.x).reshape(-1, 2), axis=1)
            report["tracker_reprojection_px"] = {
                view.frame.camera_id: float(error)
                for view, error in zip(views, primary_errors)}
            primary_inliers = np.flatnonzero(primary_errors <= pixel_tolerance)
            if (len(primary_inliers) >= 3 and _parallax(
                    [views[index] for index in primary_inliers]) <
                    min_parallax_deg):
                report["tracker_reason"] = "insufficient_inlier_camera_parallax"
        return report
    fit, errors, inlier_indices = accepted
    inliers = [views[index].frame.camera_id for index in inlier_indices]
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
