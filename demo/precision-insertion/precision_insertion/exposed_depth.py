"""Read-only key-depth interval from a visible rear centre and shaft axis.

The insertion tip may be hidden by the socket. Given a *visible* rear-centre
point and projected shaft centreline in multiple calibrated cameras, reuse
the demo's line triangulator and extrapolate the CAD rear-to-tip length.
The interval is only conservative if separately commissioned error bounds
cover VLM landmark, triangulation, camera/socket and CAD systematic errors.
It is not an insertion-success label or a guarded-execution producer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import validate_task_geometry
from .geometry import validate_se3
from .grounded_alignment import (
    GroundedLineView, _K, _line_errors, _line_solution, _parallax,
)


@dataclass(frozen=True)
class ExposedDepthLimits:
    minimum_views: int
    max_capture_skew_s: float
    min_rear_parallax_deg: float
    max_reprojection_px: float
    max_axis_line_error_px: float
    min_axis_plane_angle_deg: float
    max_axis_tilt_deg: float
    # Deterministic worst-case, not covariance-derived confidence intervals.
    rear_position_error_bound_m: float
    axis_angle_error_bound_deg: float
    rim_height_error_bound_m: float
    cad_tip_projection_error_bound_m: float

    def validate(self) -> None:
        if type(self.minimum_views) is not int or self.minimum_views < 2:
            raise ValueError("depth needs at least two calibrated camera views")
        for name in (
            "max_capture_skew_s", "min_rear_parallax_deg",
            "max_reprojection_px", "max_axis_line_error_px",
            "min_axis_plane_angle_deg", "max_axis_tilt_deg",
            "rear_position_error_bound_m",
            "axis_angle_error_bound_deg", "rim_height_error_bound_m",
            "cad_tip_projection_error_bound_m",
        ):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value) or
                    value <= 0):
                raise ValueError(f"{name} needs a commissioned positive bound")
        if (self.min_rear_parallax_deg >= 90 or
                self.min_axis_plane_angle_deg >= 90 or
                self.max_axis_tilt_deg >= 90 or
                self.axis_angle_error_bound_deg >= 90):
            raise ValueError("depth geometry angle limits must be below 90 deg")


def _abstain(reason: str, **details) -> dict:
    return {
        "schema": "precision_insertion_exposed_depth_v1",
        "status": "abstain", "reason": reason,
        "key_depth_source": None, "key_depth_interval_m": None,
        "scope": "read_only_visible_landmark_diagnostic_not_task_label",
        "robot_ready": False, **details,
    }


def estimate_exposed_depth(
    views: Sequence[GroundedLineView], *,
    socket_rim_z_m: float, key_rear_to_tip_m: float,
    limits: ExposedDepthLimits,
) -> dict:
    """Bound rim-to-tip axial penetration in the frozen socket frame.

    ``view.tip_uv_px`` is the *rear-centre* pixel in this call; it is not the
    hidden insertion tip. ``axis_line_uv_px`` is the visible projected key
    centreline. The returned bound requires independent held-out worst-case
    landmark and calibration errors; reprojection alone cannot supply them.
    """
    limits.validate()
    if (not math.isfinite(socket_rim_z_m) or
            not math.isfinite(key_rear_to_tip_m) or
            key_rear_to_tip_m <= 0):
        raise ValueError("depth needs finite CAD rim and positive key length")
    if len({view.frame.camera_id for view in views}) != len(views):
        raise ValueError("depth camera IDs must be unique")
    admitted = []
    for view in views:
        frame = view.frame
        _K(frame.intrinsics)
        validate_se3(frame.T_camera_socket, name="T_camera_socket")
        if (not isinstance(frame.raw, Image.Image) or
                not math.isfinite(float(frame.timestamp_s))):
            raise ValueError("depth needs timestamped camera frames")
        if view.tip_uv_px is None or view.axis_line_uv_px is None:
            continue
        if len(view.axis_line_uv_px) != 2:
            raise ValueError("visible axis line needs two pixels")
        points = (view.tip_uv_px, *view.axis_line_uv_px)
        for point in points:
            if (len(point) != 2 or
                    not all(type(value) in (int, float) and math.isfinite(value)
                            for value in point) or
                    not (0 <= point[0] < frame.raw.width and
                         0 <= point[1] < frame.raw.height)):
                raise ValueError("depth landmark leaves original image")
        if np.linalg.norm(np.subtract(*view.axis_line_uv_px)) < 4:
            return _abstain("projected_axis_line_too_short")
        admitted.append(view)
    if len(admitted) < limits.minimum_views:
        return _abstain("rear_center_or_axis_not_visible_in_enough_views",
                        visible_cameras=[v.frame.camera_id for v in admitted])
    skew = max(v.frame.timestamp_s for v in admitted) - min(
        v.frame.timestamp_s for v in admitted)
    if skew > limits.max_capture_skew_s:
        return _abstain("multiview_exposures_not_synchronized",
                        capture_skew_s=skew)
    parallax = _parallax(admitted)
    if parallax < limits.min_rear_parallax_deg:
        return _abstain("rear_center_parallax_too_small", parallax_deg=parallax)
    try:
        rear, axis, plane_angle = _line_solution(admitted, key_rear_to_tip_m)
    except (ValueError, np.linalg.LinAlgError):
        return _abstain("rear_center_or_axis_triangulation_failed")
    if plane_angle < limits.min_axis_plane_angle_deg:
        return _abstain("axis_planes_have_insufficient_separation",
                        axis_plane_angle_deg=plane_angle)
    tilt = math.degrees(math.acos(float(np.clip(-axis[2], -1, 1))))
    errors = [_line_errors(view, rear, axis, key_rear_to_tip_m)
              for view in admitted]
    reprojection = max(row[0] for row in errors)
    line_error = max(max(row[1:]) for row in errors)
    if (not np.all(np.isfinite(rear)) or
            reprojection > limits.max_reprojection_px or
            line_error > limits.max_axis_line_error_px):
        return _abstain("rear_or_axis_reprojection_exceeds_limit",
                        rear_reprojection_px=reprojection,
                        axis_line_error_px=line_error)
    if tilt > limits.max_axis_tilt_deg:
        return _abstain("key_axis_tilt_exceeds_limit", axis_tilt_deg=tilt)
    tip = rear + key_rear_to_tip_m * axis
    nominal_depth = float(socket_rim_z_m - tip[2])
    # ||u - u_hat|| <= 2 sin(theta/2) for unit directions within theta.
    axis_bound = key_rear_to_tip_m * 2 * math.sin(
        math.radians(limits.axis_angle_error_bound_deg) / 2)
    total_bound = (limits.rear_position_error_bound_m + axis_bound +
                   limits.rim_height_error_bound_m +
                   limits.cad_tip_projection_error_bound_m)
    return {
        "schema": "precision_insertion_exposed_depth_v1",
        "status": "bounded_visual_depth",
        "key_depth_source": "exposed_length_cad",
        "key_depth_interval_m": [nominal_depth - total_bound,
                                 nominal_depth + total_bound],
        "rear_center_socket_m": rear.tolist(),
        "insertion_axis_socket": axis.tolist(),
        "extrapolated_tip_socket_m": tip.tolist(),
        "socket_rim_z_m": float(socket_rim_z_m),
        "key_rear_to_tip_m": float(key_rear_to_tip_m),
        "nominal_depth_m": nominal_depth,
        "worst_case_depth_error_bound_m": total_bound,
        "limits": asdict(limits),
        "inlier_cameras": [view.frame.camera_id for view in admitted],
        "capture_skew_s": skew,
        "rear_parallax_deg": parallax,
        "axis_plane_angle_deg": plane_angle,
        "axis_tilt_deg": tilt,
        "rear_reprojection_max_px": reprojection,
        "axis_line_error_max_px": line_error,
        "scope": "read_only_visible_landmark_diagnostic_not_task_label",
        "robot_ready": False,
    }


def estimate_exposed_depth_for_mode(
    views: Sequence[GroundedLineView], *, mode: TaskMode,
    shared_root: Path, limits: ExposedDepthLimits,
) -> dict:
    """Use the same v8 task geometry file as endpoint/target planning."""
    path = AssetPaths(Path(shared_root).expanduser().resolve(), mode).task_geometry
    raw = path.read_bytes()
    geometry = json.loads(raw)
    validate_task_geometry(geometry, mode)
    rear = geometry["key_frame"].get(
        "grasp_rear_z_m", geometry["key_frame"].get("handle_rear_z_m"))
    tip = geometry["key_frame"]["tip_z_m"]
    rim = geometry["socket_entry_plane_z_m"]
    if rear is None:
        raise ValueError("task geometry lacks the visible rear key reference")
    if ("socket_rim_z_m" in geometry and
            not math.isclose(float(geometry["socket_rim_z_m"]), float(rim),
                             rel_tol=0, abs_tol=1e-8)):
        raise ValueError("socket rim differs from the CAD entry plane")
    result = estimate_exposed_depth(
        views, socket_rim_z_m=float(rim),
        key_rear_to_tip_m=float(tip) - float(rear), limits=limits)
    result["mode"] = {"family": mode.family, "gap_mm": mode.gap_mm,
                      "key_object": mode.key_object,
                      "socket_object": mode.socket_object}
    result["task_geometry_path"] = str(path.resolve())
    result["task_geometry_sha256"] = hashlib.sha256(raw).hexdigest()
    return result
