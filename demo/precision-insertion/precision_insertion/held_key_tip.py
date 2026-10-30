"""Read-only key-end image estimate from a VLM box and a key silhouette.

The frozen socket pose and planned held-key transform select the original
undistorted crop. Qwen identifies a coarse key box; stock AutoDex SAM3 can
segment the key within that box. The distal silhouette edge midpoint is an
image observation, not a 3D pose, insertion-depth measurement, or robot target.
"""

from __future__ import annotations

import json
import math
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations


KEY_BOX_PROMPT = (
    "Find the WHITE CYLINDRICAL KEY directly held by the BLACK ROBOT HAND. "
    "Ignore any white socket, board, or fingers. Return the tight bounding "
    'box of the white cylindrical KEY ONLY as JSON {"bbox":[x1,y1,x2,y2]}. '
    "Coordinates are normalized 0-1000 for the whole displayed image."
)
KEY_MASK_PROMPT = "white cylinder"


def key_box_socket_bbox_overlap_fraction(
    key_box_crop_px: Sequence[float], crop_xyxy_full_px: Sequence[int],
    socket_bbox_full_px: Sequence[int],
) -> float:
    """Measure how much a proposed key box occupies the fixed socket image.

    The socket projection is a broad bounding box, so this is an identity
    check for a box almost entirely on the socket, not a collision test.
    """
    if (len(key_box_crop_px) != 4 or len(crop_xyxy_full_px) != 4 or
            len(socket_bbox_full_px) != 4):
        raise ValueError("key/socket overlap needs four coordinates per box")
    x0, y0 = crop_xyxy_full_px[:2]
    a = [float(key_box_crop_px[0]) + x0,
         float(key_box_crop_px[1]) + y0,
         float(key_box_crop_px[2]) + x0,
         float(key_box_crop_px[3]) + y0]
    b = [float(value) for value in socket_bbox_full_px]
    if (not all(math.isfinite(value) for value in a + b) or
            a[0] >= a[2] or a[1] >= a[3] or
            b[0] >= b[2] or b[1] >= b[3]):
        raise ValueError("key/socket boxes need finite positive area")
    intersection = (max(0., min(a[2], b[2]) - max(a[0], b[0])) *
                    max(0., min(a[3], b[3]) - max(a[1], b[1])))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]))


def projected_key_axis_crop(
    *, image_size_wh: tuple[int, int], intrinsics: np.ndarray,
    T_camera_robot: np.ndarray, T_robot_key: np.ndarray,
    tip_z_m: float, side_px: int = 512,
) -> tuple[tuple[int, int, int, int], tuple[float, float], tuple[float, float]]:
    """Frame the planned key axis without resampling source camera pixels.

    ``T_robot_key`` comes from the frozen socket pose and planned held relation.
    Its projections locate a crop only; they are not measured key landmarks.
    """
    width, height = image_size_wh
    if (type(width) is not int or type(height) is not int or
            type(side_px) is not int or side_px <= 0 or
            min(width, height) < side_px or
            not math.isfinite(tip_z_m) or tip_z_m <= 0):
        raise ValueError("key-axis crop needs finite geometry and source pixels")
    K = np.asarray(intrinsics, dtype=float)
    T = np.asarray(T_camera_robot, dtype=float) @ np.asarray(T_robot_key, dtype=float)
    if (K.shape != (3, 3) or T.shape != (4, 4) or
            not np.all(np.isfinite(K)) or not np.all(np.isfinite(T))):
        raise ValueError("key-axis crop needs finite calibrated matrices")

    def project(z: float) -> np.ndarray:
        point = T @ np.array([0., 0., z, 1.])
        if point[2] <= 0:
            raise ValueError("planned key axis lies behind camera")
        homogeneous = K @ point[:3]
        return homogeneous[:2] / homogeneous[2]

    rear, tip = project(0.), project(float(tip_z_m))
    center = (rear + tip) / 2
    x0 = int(round(np.clip(center[0] - side_px/2, 0, width-side_px)))
    y0 = int(round(np.clip(center[1] - side_px/2, 0, height-side_px)))
    return ((x0, y0, x0+side_px, y0+side_px),
            tuple(map(float, rear)), tuple(map(float, tip)))


def parse_normalized_key_box(
    answer: str, *, canvas_size_wh: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Map Qwen's documented 0-1000 box convention to crop pixels.

    An absent or malformed box is an unobserved result, never a fabricated
    key location. Extra JSON explanation fields are ignored.
    """
    if not isinstance(answer, str):
        raise TypeError("Qwen answer must be text")
    width, height = canvas_size_wh
    if width <= 0 or height <= 0:
        raise ValueError("positive canvas dimensions are required")
    try:
        value = json.loads(answer[answer.index("{"):answer.rindex("}")+1])["bbox"]
    except (ValueError, KeyError, TypeError):
        return None
    if (not isinstance(value, list) or len(value) != 4 or
            not all(type(v) in (int, float) and math.isfinite(v) and
                    0 <= v <= 1000 for v in value) or
            not value[0] < value[2] or not value[1] < value[3]):
        return None
    return (value[0]*width/1000., value[1]*height/1000.,
            value[2]*width/1000., value[3]*height/1000.)


def padded_key_box_roi(
    box_xyxy: Sequence[float], *, canvas_size_wh: tuple[int, int],
    padding_px: int = 25,
) -> tuple[int, int, int, int]:
    """Crop a Qwen key box with a small visual margin for SAM3."""
    width, height = canvas_size_wh
    if (len(box_xyxy) != 4 or type(padding_px) is not int or padding_px < 0 or
            not all(math.isfinite(float(v)) for v in box_xyxy)):
        raise ValueError("key box ROI needs finite bounds and nonnegative padding")
    x1, y1, x2, y2 = box_xyxy
    if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
        raise ValueError("key box lies outside displayed crop")
    return (max(0, int(x1)-padding_px),
            max(0, int(y1)-padding_px),
            min(width, int(x2)+padding_px),
            min(height, int(y2)+padding_px))


def distal_key_mask_edge_midpoint(
    mask: np.ndarray, *, planned_rear_px: Sequence[float],
    planned_tip_px: Sequence[float], mask_origin_px: Sequence[float] = (0., 0.),
) -> tuple[float, float] | None:
    """Find the far end of a segmented held cylinder in original pixels.

    The planned axis chooses which short edge of an oriented silhouette box is
    distal. It never supplies the observed endpoint. Hand occlusion can bias
    a rectangle, so the result remains a visual estimate for multi-view QA.
    """
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("held-key mask must be 2D")
    ys, xs = np.nonzero(binary)
    if len(xs) < 50:
        return None
    rear = np.asarray(planned_rear_px, dtype=float)
    tip = np.asarray(planned_tip_px, dtype=float)
    origin = np.asarray(mask_origin_px, dtype=float)
    if (rear.shape != (2,) or tip.shape != (2,) or origin.shape != (2,) or
            not np.all(np.isfinite(rear)) or not np.all(np.isfinite(tip)) or
            not np.all(np.isfinite(origin))):
        raise ValueError("distal edge needs finite projected key axis")
    direction = tip - rear
    length = np.linalg.norm(direction)
    if length <= 1e-6:
        raise ValueError("projected planned key axis is degenerate")
    direction /= length
    points = np.column_stack((xs, ys)).astype(np.float32)
    corners = cv2.boxPoints(cv2.minAreaRect(points)).astype(float) + origin
    distal = corners[np.argsort(corners @ direction)[-2:]]
    center = distal.mean(axis=0)
    return float(center[0]), float(center[1])


def distal_key_mask_boundary_fit(
    mask: np.ndarray, *, planned_rear_px: Sequence[float],
    planned_tip_px: Sequence[float], mask_origin_px: Sequence[float] = (0., 0.),
) -> dict | None:
    """Estimate the distal face center from the mask's actual outer boundary.

    Preserve the whole-mask oriented box's lateral centerline, then gather
    actual end-boundary pixels in central lateral strips and fit their axial
    location. This corrects the box end along the key axis without allowing
    occlusion on one side to move the centerline sideways. The mask cannot
    reveal an occluded face's true center, so this remains a visual proxy.
    """
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("held-key mask must be 2D")
    ys, xs = np.nonzero(binary)
    if len(xs) < 50:
        return None
    rear = np.asarray(planned_rear_px, dtype=float)
    tip = np.asarray(planned_tip_px, dtype=float)
    origin = np.asarray(mask_origin_px, dtype=float)
    if (rear.shape != (2,) or tip.shape != (2,) or origin.shape != (2,) or
            not np.all(np.isfinite(rear)) or not np.all(np.isfinite(tip)) or
            not np.all(np.isfinite(origin))):
        raise ValueError("distal boundary needs finite projected key axis")
    direction = tip - rear
    span = float(np.linalg.norm(direction))
    if span <= 1e-6:
        raise ValueError("projected planned key axis is degenerate")
    direction /= span
    lateral = np.array([-direction[1], direction[0]])
    points = np.column_stack((xs, ys)).astype(float) + origin
    axial = points @ direction
    cross = points @ lateral
    low, high = np.percentile(axial, (55., 85.))
    body = (axial >= low) & (axial <= high)
    left, right = np.percentile(cross[body], (7., 93.))
    if right-left < 4.:
        return None
    rectangle_end = distal_key_mask_edge_midpoint(
        mask, planned_rear_px=rear, planned_tip_px=tip,
        mask_origin_px=origin)
    if rectangle_end is None:
        return None
    center_cross = float(np.asarray(rectangle_end) @ lateral)
    half_width = .35 * (right-left)
    bins = np.linspace(center_cross-half_width,
                       center_cross+half_width, 9)
    edge_cross, edge_axial = [], []
    for start, end in zip(bins[:-1], bins[1:]):
        selected = (cross >= start) & (cross < end)
        if np.count_nonzero(selected) < 8:
            continue
        edge_cross.append((start+end)/2. - center_cross)
        edge_axial.append(float(np.percentile(axial[selected], 99.)))
    if len(edge_cross) < 5:
        return None
    x = np.asarray(edge_cross)
    y = np.asarray(edge_axial)
    fit = least_squares(lambda coeff: coeff[0] + coeff[1]*x-y,
                        (float(np.median(y)), 0.), loss="soft_l1", f_scale=3.)
    if not fit.success:
        return None
    center = direction*fit.x[0] + lateral*center_cross
    width = half_width
    ends = [direction*(fit.x[0]+fit.x[1]*cross_offset) +
            lateral*(center_cross+cross_offset)
            for cross_offset in (-width, width)]
    samples = [direction*axial_value +
               lateral*(center_cross+cross_offset)
               for cross_offset, axial_value in zip(edge_cross, edge_axial)]
    return {
        "center_full_px": center.tolist(),
        "fitted_boundary_full_px": [point.tolist() for point in ends],
        "boundary_samples_full_px": [point.tolist() for point in samples],
        "body_lateral_width_px": float(right-left),
    }


def distal_key_mask_boundary_center(
    mask: np.ndarray, *, planned_rear_px: Sequence[float],
    planned_tip_px: Sequence[float], mask_origin_px: Sequence[float] = (0., 0.),
) -> tuple[float, float] | None:
    """Return the mask-boundary point without the visualization diagnostics."""
    fit = distal_key_mask_boundary_fit(
        mask, planned_rear_px=planned_rear_px,
        planned_tip_px=planned_tip_px, mask_origin_px=mask_origin_px)
    return None if fit is None else tuple(fit["center_full_px"])


def distal_key_mask_half_ellipse(
    mask: np.ndarray, *, planned_rear_px: Sequence[float],
    planned_tip_px: Sequence[float], mask_origin_px: Sequence[float] = (0., 0.),
) -> dict | None:
    """Fit a distal circular-face projection from the SAM3 mask alone.

    The largest connected mask component avoids disconnected hand/key flecks.
    Its long rectangle provides the observed cylinder direction and
    diameter in pixels. For each lateral bin, the farthest mask pixels trace
    the exposed half of a projected circle. Fit that half ellipse with its
    major radius fixed by the observed shaft width and its minor radius free.
    The planned axis chooses only *which* end is distal. No planned circle or
    socket pose enters the fit. A partial/occluded cap is still ambiguous.
    """
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError("held-key mask must be 2D")
    count, labels, stats, _centers = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None
    key_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == key_component)
    if len(xs) < 50:
        return None
    rear = np.asarray(planned_rear_px, dtype=float)
    tip = np.asarray(planned_tip_px, dtype=float)
    origin = np.asarray(mask_origin_px, dtype=float)
    if (rear.shape != (2,) or tip.shape != (2,) or origin.shape != (2,) or
            not np.all(np.isfinite(rear)) or not np.all(np.isfinite(tip)) or
            not np.all(np.isfinite(origin))):
        raise ValueError("SAM3 ellipse needs a finite end direction and origin")
    planned = tip-rear
    if np.linalg.norm(planned) <= 1e-6:
        # A valid end-on view can project both ends to the same pixel. Its
        # distal mask edge is unobservable; other views can still be fitted.
        return None
    local = np.column_stack((xs, ys)).astype(np.float32)
    corners = cv2.boxPoints(cv2.minAreaRect(local)).astype(float)
    edge_a, edge_b = corners[1]-corners[0], corners[2]-corners[1]
    axis = edge_a if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b) else edge_b
    axis /= np.linalg.norm(axis)
    if axis @ planned < 0:
        axis *= -1
    lateral = np.array([-axis[1], axis[0]])
    points = local.astype(float)+origin
    axial, cross = points @ axis, points @ lateral
    body = (axial >= np.percentile(axial, 50.)) & (
        axial <= np.percentile(axial, 80.))
    left, right = np.percentile(cross[body], (7., 93.))
    radius = float((right-left)/2.)
    if radius < 3.:
        return None
    rectangle_end = distal_key_mask_edge_midpoint(
        labels == key_component, planned_rear_px=rear, planned_tip_px=tip,
        mask_origin_px=origin)
    if rectangle_end is None:
        return None
    center_cross = float(np.asarray(rectangle_end) @ lateral)
    bins = np.linspace(center_cross-.9*radius,
                       center_cross+.9*radius, 15)
    offsets, boundary = [], []
    for start, end in zip(bins[:-1], bins[1:]):
        selected = (cross >= start) & (cross < end)
        if np.count_nonzero(selected) < 8:
            continue
        offsets.append((start+end)/2. - center_cross)
        boundary.append(float(np.percentile(axial[selected], 99.)))
    if len(offsets) < 5:
        return None
    offsets = np.asarray(offsets)
    boundary = np.asarray(boundary)
    shape = np.sqrt(np.clip(1.-(offsets/radius)**2, 0., 1.))
    coefficients = np.linalg.lstsq(
        np.column_stack((np.ones_like(shape), shape)), boundary,
        rcond=None)[0]
    initial = (float(coefficients[0]), float(np.clip(coefficients[1], 0., radius)))
    fit = least_squares(
        lambda values: values[0]+values[1]*shape-boundary,
        initial, bounds=([-np.inf, 0.], [np.inf, radius]),
        loss="soft_l1", f_scale=3.)
    if not fit.success:
        return None
    center = axis*fit.x[0] + lateral*center_cross
    angles = np.linspace(0., 2.*np.pi, 96, endpoint=False)
    ellipse = [center + lateral*radius*np.cos(angle) +
               axis*fit.x[1]*np.sin(angle) for angle in angles]
    samples = [axis*value + lateral*(center_cross+offset)
               for offset, value in zip(offsets, boundary)]
    residual = fit.x[0]+fit.x[1]*shape-boundary
    return {
        "center_full_px": center.tolist(),
        # Unit image direction from the rear of the cylinder toward its
        # distal face.  Together with the fitted face centre this is a 2D
        # projection of the cylinder shaft; calibrated views can intersect
        # the corresponding interpretation planes to recover the 3D key
        # axis without asking the VLM for another landmark.
        "axis_unit_full_px": axis.tolist(),
        "ellipse_full_px": [point.tolist() for point in ellipse],
        "distal_boundary_samples_full_px": [point.tolist() for point in samples],
        "major_radius_px": radius,
        "minor_radius_px": float(fit.x[1]),
        "boundary_median_abs_error_px": float(np.median(np.abs(residual))),
        "source": "sam3_mask_only_observed_diameter_partial_ellipse",
    }


def triangulate_tip_proxy(
    observations: Sequence[tuple[str, np.ndarray, Sequence[float]]],
    *, robust_pixel_scale: float = 12.,
) -> dict:
    """Triangulate silhouette-end proxies with the saved undistorted cameras.

    The image points are only approximate correspondences to the same lower
    key face. Pair-seeded robust least squares limits one poor view's influence
    on this *read-only* spatial proxy; its per-view residuals stay in the
    result for review. No residual is used as a motion approval threshold.
    """
    if len(observations) < 2 or not math.isfinite(robust_pixel_scale) or robust_pixel_scale <= 0:
        raise ValueError("proxy triangulation needs at least two cameras")
    camera_ids = [item[0] for item in observations]
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError("proxy triangulation needs unique camera IDs")
    matrices, pixels = [], []
    for camera, projection, pixel in observations:
        P = np.asarray(projection, dtype=float)
        uv = np.asarray(pixel, dtype=float)
        if (not camera or P.shape != (3, 4) or uv.shape != (2,) or
                not np.all(np.isfinite(P)) or not np.all(np.isfinite(uv))):
            raise ValueError("proxy triangulation needs finite calibrated projections")
        matrices.append(P)
        pixels.append(uv)

    def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
        h = P @ np.r_[xyz, 1.]
        if h[2] <= 0:
            return np.array([1e6, 1e6])
        return h[:2] / h[2]

    def residual(xyz: np.ndarray) -> np.ndarray:
        return np.concatenate([project(P, xyz)-uv
                               for P, uv in zip(matrices, pixels)])

    seeds = []
    for first, second in combinations(range(len(observations)), 2):
        rows = []
        for index in (first, second):
            u, v = pixels[index]
            P = matrices[index]
            rows.extend((u*P[2]-P[0], v*P[2]-P[1]))
        _u, _s, vh = np.linalg.svd(np.asarray(rows))
        homogeneous = vh[-1]
        if abs(homogeneous[3]) < 1e-10:
            continue
        xyz = homogeneous[:3] / homogeneous[3]
        if (np.all(np.isfinite(xyz)) and
                all((P @ np.r_[xyz, 1.])[2] > 0 for P in matrices)):
            values = residual(xyz)
            scale = robust_pixel_scale
            score = float(np.sum(2*scale**2*(np.sqrt(1+(values/scale)**2)-1)))
            seeds.append((score, first, second, xyz))
    if not seeds:
        raise ValueError("saved view rays have no finite point in front of cameras")
    _score, first, second, seed = min(seeds, key=lambda row: row[0])
    fit = least_squares(residual, seed, loss="soft_l1", f_scale=robust_pixel_scale)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("saved key-tip proxy fit did not converge")
    errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    return {
        "schema": "precision_insertion_key_tip_spatial_proxy_v1",
        "point_socket_m": fit.x.tolist(),
        "reprojection_error_px": {
            camera: float(error) for camera, error in zip(camera_ids, errors)},
        "pair_seed_cameras": [camera_ids[first], camera_ids[second]],
        "robust_pixel_scale": float(robust_pixel_scale),
        "scope": "read_only_approximate_visual_proxy_not_robot_target",
    }
