"""Read-only, calibrated multi-view key-axis alignment diagnostic.

The VLM supplies *image observations*, never a metric displacement. The
cylinder-preferred route uses an insertion-tip centre and the visible image
axis line in each view; its endpoints need no physical cross-view point
correspondence. A marked-object route uses two corresponding CAD landmarks.
If the shaft is hidden, one triangulated tip does not determine key tilt.

All pixels refer to original, undistorted AutoDex images. T_camera_socket is
composed from the frozen session camera/robot/socket transforms. The returned
1 mm step is diagnostic; it does not bypass grasp, collision, path or guarded
contact gates and cannot command the Franka.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
import math
import time
from typing import Sequence

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.stats import chi2

from .geometry import validate_se3
from .observer import ImageVLM, VLMObservation, _backend_model
from .xy_overlay import CalibratedXYFrame
from .xy_voting import VLM_XY_STEP_M


@dataclass(frozen=True)
class GroundedView:
    frame: CalibratedXYFrame
    tip_uv_px: tuple[float, float] | None
    axis_ref_uv_px: tuple[float, float] | None
    evidence: str = ""


@dataclass(frozen=True)
class GroundedLineView:
    """Tip pixel and two pixels on one visible projected cylinder axis line.

    The line endpoints do not need to identify the same 3D cross-sections in
    other views; only the *axis line* is corresponding geometry.
    """

    frame: CalibratedXYFrame
    tip_uv_px: tuple[float, float] | None
    axis_line_uv_px: tuple[tuple[float, float], tuple[float, float]] | None
    evidence: str = ""


@dataclass(frozen=True)
class AlignmentLimits:
    # Values must come from held-out point-label and rig-calibration trials.
    pixel_sigma_px: float
    max_reprojection_px: float
    min_parallax_deg: float
    max_axis_tilt_deg: float
    max_20mm_axis_sweep_m: float
    max_lateral_uncertainty_95_m: float
    # Empirical floor for calibration, socket pose and correlated point bias.
    systematic_lateral_sigma_m: float
    cad_spacing_sigma_m: float
    minimum_views: int = 3

    def validate(self) -> None:
        positive = (self.pixel_sigma_px, self.max_reprojection_px,
                    self.min_parallax_deg, self.max_axis_tilt_deg,
                    self.max_20mm_axis_sweep_m,
                    self.max_lateral_uncertainty_95_m,
                    self.cad_spacing_sigma_m)
        if (not all(math.isfinite(float(v)) and v > 0 for v in positive) or
                not math.isfinite(self.systematic_lateral_sigma_m) or
                self.systematic_lateral_sigma_m < 0 or
                self.min_parallax_deg >= 90 or self.max_axis_tilt_deg >= 90 or
                type(self.minimum_views) is not int or self.minimum_views < 2):
            raise ValueError("alignment limits need commissioned positive values")


def _K(value: np.ndarray) -> np.ndarray:
    K = np.asarray(value, dtype=float)
    if (K.shape != (3, 3) or not np.all(np.isfinite(K)) or
            K[0, 0] <= 0 or K[1, 1] <= 0 or
            not np.allclose(K[2], [0, 0, 1], atol=1e-9)):
        raise ValueError("invalid full-resolution undistorted intrinsics")
    return K


def _projection(view: GroundedView) -> np.ndarray:
    return _K(view.frame.intrinsics) @ validate_se3(
        view.frame.T_camera_socket, name="T_camera_socket")[:3]


def _project(view: GroundedView, xyz: np.ndarray) -> np.ndarray | None:
    E = view.frame.T_camera_socket
    camera_xyz = E[:3, :3] @ xyz + E[:3, 3]
    if camera_xyz[2] <= 1e-6:
        return None
    pixel = view.frame.intrinsics @ camera_xyz
    return pixel[:2] / pixel[2]


def _dlt(pair: Sequence[GroundedView], field: str) -> np.ndarray | None:
    rows = []
    for view in pair:
        u, v = getattr(view, field)
        P = _projection(view)
        rows.extend((u * P[2] - P[0], v * P[2] - P[1]))
    _u, _s, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vh[-1]
    if abs(h[3]) < 1e-10:
        return None
    xyz = h[:3] / h[3]
    return xyz if np.all(np.isfinite(xyz)) else None


def _ray(view: GroundedView, uv: tuple[float, float]) -> np.ndarray:
    camera_ray = np.linalg.solve(view.frame.intrinsics,
                                 np.array([*uv, 1.0]))
    socket_ray = view.frame.T_camera_socket[:3, :3].T @ camera_ray
    return socket_ray / np.linalg.norm(socket_ray)


def _parallax(views: Sequence[GroundedView]) -> float:
    return max(math.degrees(math.acos(float(np.clip(np.dot(
        _ray(a, a.tip_uv_px), _ray(b, b.tip_uv_px)), -1, 1))))
               for a, b in combinations(views, 2))


def _axis_residuals(x: np.ndarray, views: Sequence[GroundedView],
                    length_m: float, limits: AlignmentLimits) -> np.ndarray:
    tip, ref = x[:3], x[3:]
    rows = []
    for view in views:
        for point, field in ((tip, "tip_uv_px"), (ref, "axis_ref_uv_px")):
            pixel = _project(view, point)
            if pixel is None:
                rows.extend((1e6, 1e6))
            else:
                rows.extend((pixel - np.asarray(getattr(view, field))) /
                            limits.pixel_sigma_px)
    rows.append((np.linalg.norm(tip - ref) - length_m) /
                limits.cad_spacing_sigma_m)
    return np.asarray(rows, dtype=float)


def _mean_axis_error(x: np.ndarray, rim_z_m: float,
                     depth_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tip, ref = x[:3], x[3:]
    axis = (tip - ref) / np.linalg.norm(tip - ref)
    if abs(axis[2]) < 1e-8:
        raise ValueError("key axis is parallel to socket entry plane")
    entry = tip[:2] + (rim_z_m - tip[2]) * axis[:2] / axis[2]
    depth = tip[:2] + (rim_z_m - depth_m - tip[2]) * axis[:2] / axis[2]
    return (entry + depth) / 2, entry, depth


def _abstain(reason: str, **extra) -> dict:
    return {
        "schema": "precision_insertion_grounded_alignment_v1",
        "status": "abstain", "reason": reason, "step_socket_m": None,
        "scope": "read_only_metric_diagnostic_not_robot_motion",
        "robot_ready": False, **extra,
    }


def estimate_grounded_alignment(
    views: Sequence[GroundedView], *, landmark_spacing_m: float,
    socket_rim_z_m: float, verification_depth_m: float,
    limits: AlignmentLimits, yaw_relevant: bool = False,
) -> dict:
    """Robust two-landmark triangulation, CAD check, uncertainty, 1 mm step.

    A single VLM-selected pixel cannot determine scale or axis by itself.
    The 1 mm cardinal step minimizes mean lateral error at rim and 20 mm,
    subject to a 95% lower confidence bound on squared-error improvement.
    It remains advisory even when all checks pass.
    """
    limits.validate()
    if (not all(math.isfinite(v) and v > 0 for v in
                (landmark_spacing_m, verification_depth_m)) or
            not math.isfinite(socket_rim_z_m)):
        raise ValueError("CAD landmark spacing, rim and depth are required")
    if yaw_relevant:
        return _abstain("two_collinear_landmarks_cannot_estimate_square_key_yaw")
    if len({v.frame.camera_id for v in views}) != len(views):
        raise ValueError("grounding needs distinct camera IDs")
    admitted = []
    for view in views:
        frame = view.frame
        _K(frame.intrinsics)
        validate_se3(frame.T_camera_socket, name="T_camera_socket")
        if (not isinstance(frame.raw, Image.Image) or
                not math.isfinite(float(frame.timestamp_s))):
            raise ValueError("grounding requires timestamped full images")
        for point in (view.tip_uv_px, view.axis_ref_uv_px):
            if point is None:
                continue
            if (len(point) != 2 or not all(math.isfinite(float(x)) for x in point)
                    or not (0 <= point[0] < frame.raw.width and
                            0 <= point[1] < frame.raw.height)):
                raise ValueError("grounded pixel lies outside the original image")
        if view.tip_uv_px is not None and view.axis_ref_uv_px is not None:
            admitted.append(view)
    if len(admitted) < limits.minimum_views:
        return _abstain("insufficient_visible_corresponding_axis_landmarks",
                        visible_cameras=[v.frame.camera_id for v in admitted])
    if _parallax(admitted) < limits.min_parallax_deg:
        return _abstain("insufficient_multiview_parallax")

    # Minimal-pair hypotheses; a camera counts as an inlier only when *both*
    # independently observed landmark reprojections agree. This rejects a
    # different physical point accidentally labelled in one view.
    best = None
    for pair in combinations(admitted, 2):
        tip = _dlt(pair, "tip_uv_px")
        ref = _dlt(pair, "axis_ref_uv_px")
        if tip is None or ref is None or np.linalg.norm(tip - ref) < 1e-8:
            continue
        x = np.r_[tip, ref]
        inliers, total = [], 0.0
        for view in admitted:
            projections = (_project(view, tip), _project(view, ref))
            if any(p is None for p in projections):
                continue
            errors = [float(np.linalg.norm(p - getattr(view, name)))
                      for p, name in zip(projections,
                                         ("tip_uv_px", "axis_ref_uv_px"))]
            if max(errors) <= limits.max_reprojection_px:
                inliers.append(view)
                total += sum(e * e for e in errors)
        score = (len(inliers), -total)
        if best is None or score > best[0]:
            best = (score, x, inliers)
    if best is None or len(best[2]) < limits.minimum_views:
        return _abstain("no_consistent_multiview_landmark_hypothesis")
    _, seed, inliers = best
    if _parallax(inliers) < limits.min_parallax_deg:
        return _abstain("inlier_views_have_insufficient_parallax")
    fit = least_squares(
        _axis_residuals, seed, args=(inliers, landmark_spacing_m, limits),
        max_nfev=200, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return _abstain("nonlinear_triangulation_failed")
    tip, ref = fit.x[:3], fit.x[3:]
    if any(_project(view, point) is None for view in inliers
           for point in (tip, ref)):
        return _abstain("landmark_behind_camera")
    reprojection = [float(np.linalg.norm(_project(view, point) -
                                         getattr(view, field)))
                    for view in inliers
                    for point, field in ((tip, "tip_uv_px"),
                                         (ref, "axis_ref_uv_px"))]
    if max(reprojection) > limits.max_reprojection_px:
        return _abstain("reprojection_error_exceeds_commissioned_limit")
    spacing_error = abs(np.linalg.norm(tip - ref) - landmark_spacing_m)
    if spacing_error > 3 * limits.cad_spacing_sigma_m:
        return _abstain("observed_landmark_spacing_disagrees_with_CAD")
    axis = (tip - ref) / np.linalg.norm(tip - ref)
    tilt_deg = math.degrees(math.acos(float(np.clip(-axis[2], -1, 1))))
    if tilt_deg > limits.max_axis_tilt_deg:
        return _abstain("axis_tilt_requires_reorientation_not_xy",
                        axis_tilt_deg=tilt_deg)
    mean, entry, depth = _mean_axis_error(
        fit.x, socket_rim_z_m, verification_depth_m)
    sweep = float(np.linalg.norm(depth - entry))
    if sweep > limits.max_20mm_axis_sweep_m:
        return _abstain("axis_sweep_cannot_be_fixed_by_xy",
                        axis_sweep_m=sweep)
    # J is with whitened pixel and CAD-spacing residuals. Its inverse normal
    # matrix estimates 3D landmark covariance conditional on stated noise.
    normal = fit.jac.T @ fit.jac
    if np.linalg.cond(normal) > 1e10:
        return _abstain("ill_conditioned_axis_estimate")
    cov_6d = np.linalg.inv(normal)
    J = np.empty((2, 6))
    for j in range(6):
        h = 1e-6
        x_plus, x_minus = fit.x.copy(), fit.x.copy()
        x_plus[j] += h
        x_minus[j] -= h
        J[:, j] = (_mean_axis_error(x_plus, socket_rim_z_m,
                                    verification_depth_m)[0] -
                   _mean_axis_error(x_minus, socket_rim_z_m,
                                    verification_depth_m)[0]) / (2 * h)
    cov_xy = J @ cov_6d @ J.T + np.eye(2) * (
        limits.systematic_lateral_sigma_m ** 2)
    if (not np.all(np.isfinite(cov_xy)) or
            np.linalg.eigvalsh(cov_xy).min() <= 0):
        return _abstain("invalid_metric_uncertainty")
    # 2D radial 95% confidence ellipse, not the 1D 1.96-sigma interval.
    uncertainty_95 = math.sqrt(float(chi2.ppf(0.95, df=2))) * math.sqrt(
        float(np.linalg.eigvalsh(cov_xy).max()))
    if uncertainty_95 > limits.max_lateral_uncertainty_95_m:
        return _abstain("lateral_uncertainty_exceeds_budget",
                        lateral_uncertainty_95_m=uncertainty_95)
    step = VLM_XY_STEP_M
    steps = ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step))
    choices = []
    for move in steps:
        delta = np.asarray(move)
        # Equivalent to improvement in the sum of squared rim/depth errors.
        improvement = -2 * float(np.dot(mean, delta)) - step * step
        improvement_sigma = 2 * math.sqrt(float(delta @ cov_xy @ delta))
        choices.append((improvement - 1.96 * improvement_sigma,
                        improvement, move))
    lower, improvement, move = max(choices, key=lambda row: row[0])
    details = {
        "inlier_cameras": [v.frame.camera_id for v in inliers],
        "rejected_cameras": sorted(set(v.frame.camera_id for v in admitted) -
                                   set(v.frame.camera_id for v in inliers)),
        "tip_socket_m": tip.tolist(), "axis_ref_socket_m": ref.tolist(),
        "insertion_axis_socket": axis.tolist(),
        "axis_tilt_deg": tilt_deg, "axis_sweep_20mm_m": sweep,
        "rim_error_xy_m": entry.tolist(),
        "depth_error_xy_m": depth.tolist(),
        "mean_error_xy_m": mean.tolist(),
        "reprojection_max_px": max(reprojection),
        "landmark_spacing_residual_m": float(spacing_error),
        "parallax_deg": _parallax(inliers),
        "lateral_covariance_m2": cov_xy.tolist(),
        "lateral_uncertainty_95_m": uncertainty_95,
        "best_step_squared_error_improvement_lower_95_m2": lower,
    }
    if lower <= 0:
        return _abstain("no_1mm_cardinal_step_has_confident_improvement",
                        **details)
    return {
        "schema": "precision_insertion_grounded_alignment_v1",
        "status": "diagnostic_1mm_step", "reason": "confident_lateral_reduction",
        "step_socket_m": list(move), **details,
        "scope": "read_only_metric_diagnostic_not_robot_motion",
        "robot_ready": False,
    }


def observe_grounded_key_axis(
    backend: ImageVLM, frames: Sequence[CalibratedXYFrame],
) -> tuple[list[GroundedView], list[VLMObservation]]:
    """Ground per-view original pixels. Never show a predicted key overlay here.

    The axis-reference must be the same CAD-defined cross-section in every
    view, at the spacing supplied to the estimator. An arbitrary point along
    a smooth cylinder is not a valid correspondence.
    """
    views, records = [], []
    if len({frame.camera_id for frame in frames}) != len(frames):
        raise ValueError("grounding needs unique camera IDs")
    for frame in frames:
        if not isinstance(frame.raw, Image.Image) or frame.raw.mode != "RGB":
            raise ValueError("grounding needs original undistorted RGB frames")
        width, height = frame.raw.size
        prompt = (
            f"Camera {frame.camera_id}; ORIGINAL UNDISTORTED image "
            f"{width}x{height} pixels. Use only visible raw pixels, no robot "
            "command or CAD overlay as evidence. Mark the centre of the "
            "key's insertion tip and the centre of the specifically marked "
            "axis-reference cross-section behind that tip. The caller has "
            "measured their CAD spacing; do not choose an arbitrary point "
            "along a smooth sidewall. If either centre is occluded, not "
            "physically identifiable, or ambiguous, set it to null. "
            "Return JSON only: "
            '{"tip_px":[x,y]|null,"axis_ref_px":[x,y]|null,'
            '"evidence":"visible image features only"}. '
            "Coordinates are original-image pixel centres, origin top-left."
        )
        started = time.perf_counter()
        answer = backend.infer([frame.raw], prompt)
        latency = time.perf_counter() - started
        if not isinstance(answer, str):
            raise TypeError("VLM backend must return text")
        try:
            parsed = json.loads(answer)
            if not isinstance(parsed, dict) or set(parsed) != {
                    "tip_px", "axis_ref_px", "evidence"} or not isinstance(
                        parsed["evidence"], str):
                raise ValueError("invalid grounded landmark JSON")
            for field in ("tip_px", "axis_ref_px"):
                point = parsed[field]
                if point is not None and (
                        not isinstance(point, list) or len(point) != 2 or
                        not all(type(p) in (int, float) and math.isfinite(p)
                                for p in point) or
                        not (0 <= point[0] < width and 0 <= point[1] < height)):
                    raise ValueError("landmark must be an original-image pixel")
            error = None
        except (json.JSONDecodeError, ValueError) as exc:
            parsed = {"tip_px": None, "axis_ref_px": None, "evidence": ""}
            error = str(exc)
        views.append(GroundedView(
            frame, None if parsed["tip_px"] is None else tuple(parsed["tip_px"]),
            None if parsed["axis_ref_px"] is None else
            tuple(parsed["axis_ref_px"]), parsed["evidence"]))
        records.append(VLMObservation(
            "key_axis_grounding", parsed, answer, prompt,
            (f"raw_preinsert_hold/{frame.camera_id}@{frame.timestamp_s:.6f}",),
            error, _backend_model(backend), latency))
    return views, records


def _line_geometry(view: GroundedLineView) -> tuple[np.ndarray, np.ndarray]:
    p, q = (np.asarray(v, dtype=float) for v in view.axis_line_uv_px)
    line = np.cross(np.r_[p, 1.0], np.r_[q, 1.0])
    line /= np.linalg.norm(line[:2])  # signed pixel distance to line
    E = view.frame.T_camera_socket
    normal = E[:3, :3].T @ (view.frame.intrinsics.T @ line)
    normal /= np.linalg.norm(normal)
    return line, normal


def _line_axis(views: Sequence[GroundedLineView]) -> tuple[np.ndarray, float]:
    normals = np.vstack([_line_geometry(view)[1] for view in views])
    plane_angle = max(math.degrees(math.acos(float(np.clip(
        abs(np.dot(a, b)), -1, 1)))) for a, b in combinations(normals, 2))
    _u, singular, vh = np.linalg.svd(normals)
    if len(singular) < 2 or singular[1] < 1e-5:
        raise ValueError("axis planes do not determine a 3D line direction")
    axis = vh[-1]
    if axis[2] > 0:
        axis = -axis
    return axis, plane_angle


def _tip_fit(views: Sequence[GroundedLineView]) -> tuple[np.ndarray, np.ndarray]:
    # _dlt only needs frame and the named tip coordinate.
    seed_pair = max(combinations(views, 2), key=lambda pair: math.degrees(
        math.acos(float(np.clip(np.dot(
            _ray(pair[0], pair[0].tip_uv_px),
            _ray(pair[1], pair[1].tip_uv_px)), -1, 1)))))
    seed = _dlt(seed_pair, "tip_uv_px")
    if seed is None:
        raise ValueError("tip rays cannot be triangulated")
    def residual(x):
        values = []
        for view in views:
            pixel = _project(view, x)
            values.extend((pixel - view.tip_uv_px) if pixel is not None
                          else (1e6, 1e6))
        return np.asarray(values)
    fit = least_squares(residual, seed, max_nfev=100)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("tip triangulation did not converge")
    return fit.x, fit.fun


def _line_errors(view: GroundedLineView, tip: np.ndarray,
                 axis: np.ndarray, span_m: float) -> tuple[float, float, float]:
    tip_pixel = _project(view, tip)
    next_pixel = _project(view, tip + span_m * axis)
    if tip_pixel is None or next_pixel is None:
        return float("inf"), float("inf"), float("inf")
    line, _normal = _line_geometry(view)
    tip_error = float(np.linalg.norm(tip_pixel - view.tip_uv_px))
    line_tip_error = abs(float(line @ np.r_[tip_pixel, 1.0]))
    line_next_error = abs(float(line @ np.r_[next_pixel, 1.0]))
    return tip_error, line_tip_error, line_next_error


def _line_solution(views: Sequence[GroundedLineView], span_m: float
                   ) -> tuple[np.ndarray, np.ndarray, float]:
    tip, _errors = _tip_fit(views)
    axis, plane_angle = _line_axis(views)
    return tip, axis, plane_angle


def estimate_grounded_line_alignment(
    views: Sequence[GroundedLineView], *, socket_rim_z_m: float,
    verification_depth_m: float, limits: AlignmentLimits,
) -> dict:
    """Triangulate a tip and back-project 2D axis lines to a 3D direction.

    This is the cylinder-preferred option: the visible shaft *line* can be
    matched across views without inventing a second axial point on a smooth
    surface. A segmentation/CAD silhouette fit should eventually replace
    VLM-drawn lines; until commissioned, output is diagnostic only.
    """
    limits.validate()
    if (not math.isfinite(socket_rim_z_m) or
            not math.isfinite(verification_depth_m) or
            verification_depth_m <= 0):
        raise ValueError("socket rim and verification depth are required")
    if len({v.frame.camera_id for v in views}) != len(views):
        raise ValueError("line grounding needs distinct camera IDs")
    admitted = []
    for view in views:
        frame = view.frame
        _K(frame.intrinsics)
        validate_se3(frame.T_camera_socket, name="T_camera_socket")
        if (not isinstance(frame.raw, Image.Image) or
                not math.isfinite(float(frame.timestamp_s))):
            raise ValueError("line grounding requires timestamped full images")
        points = ([view.tip_uv_px] if view.tip_uv_px is not None else [])
        if view.axis_line_uv_px is not None:
            if len(view.axis_line_uv_px) != 2:
                raise ValueError("axis line needs two image points")
            points.extend(view.axis_line_uv_px)
        for point in points:
            if (len(point) != 2 or not all(math.isfinite(float(x)) for x in point)
                    or not (0 <= point[0] < frame.raw.width and
                            0 <= point[1] < frame.raw.height)):
                raise ValueError("grounded line pixel lies outside original image")
        if view.tip_uv_px is not None and view.axis_line_uv_px is not None:
            p, q = view.axis_line_uv_px
            if math.dist(p, q) < 8.0:
                continue
            admitted.append(view)
    if len(admitted) < limits.minimum_views:
        return _abstain("insufficient_visible_tip_and_shaft_axis_views")
    if _parallax(admitted) < limits.min_parallax_deg:
        return _abstain("insufficient_tip_ray_parallax")
    span = verification_depth_m
    best = None
    for pair in combinations(admitted, 2):
        try:
            tip, axis, plane_angle = _line_solution(pair, span)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if plane_angle < limits.min_parallax_deg:
            continue
        inliers, total = [], 0.0
        for view in admitted:
            errors = _line_errors(view, tip, axis, span)
            if max(errors) <= limits.max_reprojection_px:
                inliers.append(view)
                total += sum(e * e for e in errors)
        score = (len(inliers), -total)
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or len(best[1]) < limits.minimum_views:
        return _abstain("no_consistent_multiview_tip_and_axis_line")
    inliers = best[1]
    try:
        tip, axis, plane_angle = _line_solution(inliers, span)
    except (ValueError, np.linalg.LinAlgError):
        return _abstain("line_axis_refinement_failed")
    if (plane_angle < limits.min_parallax_deg or
            _parallax(inliers) < limits.min_parallax_deg):
        return _abstain("inlier_axis_planes_or_tip_rays_degenerate")
    errors = [e for view in inliers for e in _line_errors(view, tip, axis, span)]
    if max(errors) > limits.max_reprojection_px:
        return _abstain("tip_or_axis_line_reprojection_exceeds_limit")
    tilt_deg = math.degrees(math.acos(float(np.clip(-axis[2], -1, 1))))
    if tilt_deg > limits.max_axis_tilt_deg:
        return _abstain("axis_tilt_requires_reorientation_not_xy",
                        axis_tilt_deg=tilt_deg)
    # The line direction needs no physical second landmark. A virtual 20 mm
    # point is sufficient to calculate the axis/rim intersections.
    x = np.r_[tip, tip - span * axis]
    mean, entry, depth = _mean_axis_error(
        x, socket_rim_z_m, verification_depth_m)
    sweep = float(np.linalg.norm(entry - depth))
    if sweep > limits.max_20mm_axis_sweep_m:
        return _abstain("axis_sweep_cannot_be_fixed_by_xy",
                        axis_sweep_m=sweep)
    # Numerical propagation of commissioned pixel noise through tip DLT/LS
    # and the multi-view plane-intersection axis. This is conditional on the
    # *same* inlier set; systematic model/calibration error is a separate floor.
    numerical_jacobian = []
    for index, view in enumerate(inliers):
        coordinates = [*view.tip_uv_px, *view.axis_line_uv_px[0],
                       *view.axis_line_uv_px[1]]
        for coordinate_index in range(6):
            perturbed = []
            for sign in (1, -1):
                modified = coordinates.copy()
                modified[coordinate_index] += sign * 0.01
                replacement = GroundedLineView(
                    view.frame, tuple(modified[:2]),
                    (tuple(modified[2:4]), tuple(modified[4:6])),
                    view.evidence)
                varied = list(inliers)
                varied[index] = replacement
                try:
                    tip_v, axis_v, _ = _line_solution(varied, span)
                    x_v = np.r_[tip_v, tip_v - span * axis_v]
                    perturbed.append(_mean_axis_error(
                        x_v, socket_rim_z_m, verification_depth_m)[0])
                except (ValueError, np.linalg.LinAlgError):
                    return _abstain("uncertainty_propagation_failed")
            numerical_jacobian.append((perturbed[0] - perturbed[1]) / 0.02)
    J = np.stack(numerical_jacobian, axis=1)
    cov_xy = (limits.pixel_sigma_px ** 2) * (J @ J.T) + np.eye(2) * (
        limits.systematic_lateral_sigma_m ** 2)
    if (not np.all(np.isfinite(cov_xy)) or
            np.linalg.eigvalsh(cov_xy).min() <= 0):
        return _abstain("invalid_metric_uncertainty")
    uncertainty_95 = math.sqrt(float(chi2.ppf(0.95, df=2))) * math.sqrt(
        float(np.linalg.eigvalsh(cov_xy).max()))
    if uncertainty_95 > limits.max_lateral_uncertainty_95_m:
        return _abstain("lateral_uncertainty_exceeds_budget",
                        lateral_uncertainty_95_m=uncertainty_95)
    step = VLM_XY_STEP_M
    options = ((step, 0.), (-step, 0.), (0., step), (0., -step))
    candidates = []
    for move in options:
        delta = np.asarray(move)
        improvement = -2 * float(np.dot(mean, delta)) - step * step
        lower = improvement - 1.96 * 2 * math.sqrt(
            float(delta @ cov_xy @ delta))
        candidates.append((lower, move))
    lower, move = max(candidates, key=lambda row: row[0])
    details = {
        "estimator": "multiview_tip_plus_projected_axis_planes",
        "inlier_cameras": [v.frame.camera_id for v in inliers],
        "rejected_cameras": sorted(set(v.frame.camera_id for v in admitted) -
                                   set(v.frame.camera_id for v in inliers)),
        "tip_socket_m": tip.tolist(),
        "insertion_axis_socket": axis.tolist(),
        "axis_tilt_deg": tilt_deg, "axis_sweep_20mm_m": sweep,
        "rim_error_xy_m": entry.tolist(),
        "depth_error_xy_m": depth.tolist(),
        "mean_error_xy_m": mean.tolist(),
        "reprojection_max_px": max(errors),
        "tip_parallax_deg": _parallax(inliers),
        "axis_plane_angle_deg": plane_angle,
        "lateral_covariance_m2": cov_xy.tolist(),
        "lateral_uncertainty_95_m": uncertainty_95,
        "best_step_squared_error_improvement_lower_95_m2": lower,
    }
    if lower <= 0:
        return _abstain("no_1mm_cardinal_step_has_confident_improvement",
                        **details)
    return {
        "schema": "precision_insertion_grounded_alignment_v1",
        "status": "diagnostic_1mm_step", "reason": "confident_lateral_reduction",
        "step_socket_m": list(move), **details,
        "scope": "read_only_metric_diagnostic_not_robot_motion",
        "robot_ready": False,
    }


def observe_grounded_cylinder_axis(
    backend: ImageVLM, frames: Sequence[CalibratedXYFrame],
) -> tuple[list[GroundedLineView], list[VLMObservation]]:
    """Ask for a tip centre and a visible shaft centreline, raw images only."""
    views, records = [], []
    if len({frame.camera_id for frame in frames}) != len(frames):
        raise ValueError("line grounding needs unique camera IDs")
    for frame in frames:
        if not isinstance(frame.raw, Image.Image) or frame.raw.mode != "RGB":
            raise ValueError("line grounding needs original undistorted RGB")
        width, height = frame.raw.size
        prompt = (
            f"Camera {frame.camera_id}. ORIGINAL UNDISTORTED image "
            f"{width}x{height} pixels. From visible raw key pixels only, "
            "mark the insertion-tip centre and TWO separated points along "
            "the image-projected centreline of the visible straight cylinder "
            "shaft. The line points need not mark the same physical shaft "
            "locations in other cameras. If the tip or straight shaft is "
            "occluded, ambiguous, or indistinguishable from the hand/socket, "
            "return null for it. Never extrapolate a hidden line from a "
            "robot/CAD overlay. Return JSON only: "
            '{"tip_px":[x,y]|null,"axis_line_px":[[x1,y1],[x2,y2]]|null,'
            '"evidence":"visible image features only"}. '
            "Coordinates are original-image pixels, origin top-left."
        )
        started = time.perf_counter()
        answer = backend.infer([frame.raw], prompt)
        latency = time.perf_counter() - started
        if not isinstance(answer, str):
            raise TypeError("VLM backend must return text")
        try:
            parsed = json.loads(answer)
            if not isinstance(parsed, dict) or set(parsed) != {
                    "tip_px", "axis_line_px", "evidence"} or not isinstance(
                        parsed["evidence"], str):
                raise ValueError("invalid grounded line JSON")
            points = []
            if parsed["tip_px"] is not None:
                points.append(parsed["tip_px"])
            if parsed["axis_line_px"] is not None:
                if (not isinstance(parsed["axis_line_px"], list) or
                        len(parsed["axis_line_px"]) != 2):
                    raise ValueError("axis line needs two pixels")
                points.extend(parsed["axis_line_px"])
            for point in points:
                if (not isinstance(point, list) or len(point) != 2 or
                        not all(type(p) in (int, float) and math.isfinite(p)
                                for p in point) or
                        not (0 <= point[0] < width and 0 <= point[1] < height)):
                    raise ValueError("line landmark outside original pixels")
            error = None
        except (json.JSONDecodeError, ValueError) as exc:
            parsed = {"tip_px": None, "axis_line_px": None, "evidence": ""}
            error = str(exc)
        segment = parsed["axis_line_px"]
        views.append(GroundedLineView(
            frame, None if parsed["tip_px"] is None else tuple(parsed["tip_px"]),
            None if segment is None else (tuple(segment[0]), tuple(segment[1])),
            parsed["evidence"]))
        records.append(VLMObservation(
            "cylinder_tip_axis_line_grounding", parsed, answer, prompt,
            (f"raw_preinsert_hold/{frame.camera_id}@{frame.timestamp_s:.6f}",),
            error, _backend_model(backend), latency))
    return views, records
