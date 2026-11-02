"""Fit a partially visible cylindrical key directly in calibrated images.

The endpoint estimator represents every camera by one distal-face centre.
That is useful when the face is visible, but a partly occluded face can move
that centre appreciably.  This module keeps the same observed centre while
also fitting the visible shaft direction to the known CAD length.  It solves
the symmetry-reduced pre-insertion pose ``(x, y, tilt_x, tilt_y)`` jointly in
all available views.  Cylinder yaw is intentionally absent.

The fit is image based: it does not triangulate each face centre first.  A
robust image loss lets an occluded or incorrectly segmented view contribute
less without turning its residual into a new pass/fail gate.  The caller gets
all per-view residuals and observability information for replay and review.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import validate_se3
from .grounded_alignment import GroundedLineView


@dataclass(frozen=True)
class PartialCylinderView:
    """One calibrated, visible portion of the held cylindrical key."""

    camera_id: str
    intrinsics: np.ndarray
    T_camera_socket: np.ndarray
    distal_center_uv_px: tuple[float, float]
    shaft_axis_uv: tuple[float, float]
    distal_center_weight: float = 1.
    shaft_axis_weight: float = 1.

    def validated(self) -> "PartialCylinderView":
        K = np.asarray(self.intrinsics, dtype=float)
        T = validate_se3(self.T_camera_socket, name="T_camera_socket")
        center = np.asarray(self.distal_center_uv_px, dtype=float)
        axis = np.asarray(self.shaft_axis_uv, dtype=float)
        center_weight = float(self.distal_center_weight)
        axis_weight = float(self.shaft_axis_weight)
        if (not self.camera_id or K.shape != (3, 3) or
                center.shape != (2,) or axis.shape != (2,) or
                not np.all(np.isfinite(K)) or
                not np.all(np.isfinite(center)) or
                not np.all(np.isfinite(axis)) or
                not math.isfinite(center_weight) or center_weight < 0 or
                not math.isfinite(axis_weight) or axis_weight < 0 or
                center_weight + axis_weight <= 0):
            raise ValueError("partial cylinder view needs finite calibration")
        length = float(np.linalg.norm(axis))
        if length <= 1e-9:
            raise ValueError("visible shaft direction is degenerate")
        return PartialCylinderView(
            camera_id=self.camera_id,
            intrinsics=K,
            T_camera_socket=T,
            distal_center_uv_px=tuple(center.tolist()),
            shaft_axis_uv=tuple((axis / length).tolist()),
            distal_center_weight=center_weight,
            shaft_axis_weight=axis_weight,
        )


def partial_shape_measurement_weights(half_ellipse: dict,
                                      shape_profile: dict) -> dict:
    """Convert visible geometry into complementary centre/axis information.

    A front-facing circular cap makes its centre observable but its projected
    shaft direction ambiguous.  A long side silhouette has the opposite
    behavior.  The ellipse aspect and silhouette eccentricity provide those
    two normalized information fractions; mask solidity reduces both when a
    hand removes part of the observed key.  No threshold rejects a view.
    """
    major = float(half_ellipse["major_radius_px"])
    minor = float(half_ellipse["minor_radius_px"])
    axial = float(shape_profile["visible_axial_span_px"])
    lateral = float(shape_profile["visible_lateral_span_px"])
    solidity = float(shape_profile["solidity"])
    if (not all(math.isfinite(value)
                for value in (major, minor, axial, lateral, solidity)) or
            major <= 0 or minor < 0 or axial <= 0 or lateral <= 0 or
            not 0 <= solidity <= 1.05):
        raise ValueError("partial shape weights need finite observed geometry")
    solidity = float(np.clip(solidity, 0., 1.))
    face_visibility = float(np.clip(minor / major, 0., 1.))
    shaft_eccentricity_sq = float(np.clip(
        1. - (lateral / axial) ** 2, 0., 1.))
    return {
        "distal_center_weight": solidity * face_visibility ** 2,
        "shaft_axis_weight": solidity * shaft_eccentricity_sq,
        "distal_face_visibility": face_visibility,
        "shaft_silhouette_eccentricity": math.sqrt(shaft_eccentricity_sq),
        "mask_solidity": solidity,
        "source": "visible_geometry_information_weights_no_view_rejection",
    }


def partial_view_from_grounded(view: GroundedLineView) -> PartialCylinderView:
    """Convert the existing SAM centre/shaft evidence without resampling it."""
    if view.tip_uv_px is None or view.axis_line_uv_px is None:
        raise ValueError("partial shape fit needs a visible centre and shaft")
    first = np.asarray(view.axis_line_uv_px[0], dtype=float)
    second = np.asarray(view.axis_line_uv_px[1], dtype=float)
    return PartialCylinderView(
        camera_id=view.frame.camera_id,
        intrinsics=np.asarray(view.frame.intrinsics, dtype=float),
        T_camera_socket=np.asarray(view.frame.T_camera_socket, dtype=float),
        distal_center_uv_px=tuple(view.tip_uv_px),
        shaft_axis_uv=tuple((second - first).tolist()),
    ).validated()


def _project(view: PartialCylinderView, point_socket: np.ndarray) -> np.ndarray:
    camera = view.T_camera_socket[:3, :3] @ point_socket + \
        view.T_camera_socket[:3, 3]
    if camera[2] <= 1e-9:
        return np.array([1e6, 1e6], dtype=float)
    homogeneous = view.intrinsics @ camera
    return homogeneous[:2] / homogeneous[2]


def _pose(parameters: np.ndarray, tip_z_socket_m: float,
          key_length_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tip = np.array([parameters[0], parameters[1], tip_z_socket_m], dtype=float)
    # The key insertion axis points from its rear toward the distal face.  In
    # the aligned socket frame that direction is -Z.  Axial yaw is irrelevant
    # for a circular key, hence the two-component rotation vector.
    axis = Rotation.from_rotvec(
        [parameters[2], parameters[3], 0.]).apply([0., 0., -1.])
    axis /= np.linalg.norm(axis)
    rear = tip - key_length_m * axis
    return tip, rear, axis


def _view_residual(view: PartialCylinderView, tip: np.ndarray,
                   rear: np.ndarray) -> tuple[np.ndarray, dict]:
    predicted_tip = _project(view, tip)
    predicted_rear = _project(view, rear)
    vector = predicted_tip - predicted_rear
    span = float(np.linalg.norm(vector))
    if span <= 1e-9:
        predicted_axis = np.zeros(2)
        orientation_px = 1e6
        orientation_rad = math.pi / 2.
    else:
        predicted_axis = vector / span
        observed_axis = np.asarray(view.shaft_axis_uv, dtype=float)
        # The SAM direction is already oriented rear-to-tip, but retain the
        # undirected cylinder-line equivalence for saved evidence that predates
        # that convention.
        if predicted_axis @ observed_axis < 0:
            observed_axis = -observed_axis
        cross = float(predicted_axis[0] * observed_axis[1] -
                      predicted_axis[1] * observed_axis[0])
        dot = float(np.clip(predicted_axis @ observed_axis, -1., 1.))
        orientation_rad = math.atan2(abs(cross), dot)
        orientation_px = span * cross
    center = predicted_tip - np.asarray(view.distal_center_uv_px, dtype=float)
    weighted = np.r_[center * math.sqrt(view.distal_center_weight),
                     orientation_px * math.sqrt(view.shaft_axis_weight)]
    return weighted, {
        "camera_id": view.camera_id,
        "observed_distal_center_uv_px": list(view.distal_center_uv_px),
        "predicted_distal_center_uv_px": predicted_tip.tolist(),
        "predicted_rear_uv_px": predicted_rear.tolist(),
        "center_residual_px": center.tolist(),
        "center_residual_norm_px": float(np.linalg.norm(center)),
        "shaft_orientation_residual_deg": math.degrees(orientation_rad),
        "shaft_orientation_residual_px": float(orientation_px),
        "predicted_shaft_span_px": span,
        "distal_center_weight": view.distal_center_weight,
        "shaft_axis_weight": view.shaft_axis_weight,
    }


def estimate_partial_cylinder_alignment(
    views: Sequence[PartialCylinderView | GroundedLineView], *,
    nominal_tip_z_socket_m: float, key_length_m: float,
    initial_xy_socket_m: Sequence[float] = (0., 0.),
    robust_pixel_scale: float = 8., max_nfev: int = 120,
) -> dict:
    """Fit key XY and tilt to visible centre/shaft evidence in image space.

    Two calibrated views provide six scalar image constraints for four pose
    variables.  A single view is therefore returned as unobservable rather
    than being promoted to a metric 3D correction.  This is a rank condition,
    not a task-success or robot-safety threshold.
    """
    converted = [
        partial_view_from_grounded(view)
        if isinstance(view, GroundedLineView) else view.validated()
        for view in views
    ]
    if len({view.camera_id for view in converted}) != len(converted):
        raise ValueError("partial shape views need distinct camera IDs")
    if (not math.isfinite(nominal_tip_z_socket_m) or
            not math.isfinite(key_length_m) or key_length_m <= 0 or
            not math.isfinite(robust_pixel_scale) or robust_pixel_scale <= 0 or
            type(max_nfev) is not int or max_nfev <= 0):
        raise ValueError("partial shape fit needs finite geometry and loss")
    initial_xy = np.asarray(initial_xy_socket_m, dtype=float)
    if initial_xy.shape != (2,) or not np.all(np.isfinite(initial_xy)):
        raise ValueError("initial partial shape XY needs two finite metres")
    if len(converted) < 2:
        return {
            "status": "insufficient_multiview_shape_geometry",
            "camera_ids": [view.camera_id for view in converted],
            "required_pose_parameters": ["x", "y", "tilt_x", "tilt_y"],
            "scope": "diagnostic_partial_shape_fit_not_robot_target",
        }

    def residual(parameters: np.ndarray) -> np.ndarray:
        tip, rear, _axis = _pose(
            parameters, nominal_tip_z_socket_m, key_length_m)
        return np.concatenate([
            _view_residual(view, tip, rear)[0] for view in converted])

    seed = np.r_[initial_xy, 0., 0.]
    fit = least_squares(
        residual, seed, loss="soft_l1", f_scale=robust_pixel_scale,
        max_nfev=max_nfev)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return {
            "status": "partial_shape_optimization_failed",
            "camera_ids": [view.camera_id for view in converted],
            "optimizer_message": fit.message,
            "scope": "diagnostic_partial_shape_fit_not_robot_target",
        }
    tip, rear, axis = _pose(fit.x, nominal_tip_z_socket_m, key_length_m)
    per_view = []
    for view in converted:
        _residual_values, record = _view_residual(view, tip, rear)
        per_view.append(record)
    singular_values = np.linalg.svd(fit.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(fit.jac))
    condition = (None if singular_values[-1] <= 1e-12 else
                 float(singular_values[0] / singular_values[-1]))
    tilt = math.degrees(math.acos(float(np.clip(-axis[2], -1., 1.))))
    return {
        "status": "diagnostic_partial_shape_pose",
        "camera_ids": [view.camera_id for view in converted],
        "tip_socket_m": tip.tolist(),
        "rear_socket_m": rear.tolist(),
        "insertion_axis_socket": axis.tolist(),
        "xy_correction_socket_m": (-tip[:2]).tolist(),
        "axis_tilt_deg": tilt,
        "pose_parameters": {
            "x_m": float(fit.x[0]), "y_m": float(fit.x[1]),
            "tilt_x_rad": float(fit.x[2]),
            "tilt_y_rad": float(fit.x[3]),
        },
        "per_view": per_view,
        "robust_pixel_scale": robust_pixel_scale,
        "cost": float(fit.cost),
        "optimality": float(fit.optimality),
        "jacobian_rank": rank,
        "jacobian_singular_values": singular_values.tolist(),
        "jacobian_condition": condition,
        "scope": "diagnostic_partial_shape_fit_not_robot_target",
    }
