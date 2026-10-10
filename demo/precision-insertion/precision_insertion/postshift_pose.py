"""Gauge-fixed held-key hypothesis from a post-shift cylinder tip and axis.

The round key does not expose axial yaw. The measured wrist and the physically
calibrated grasp medoid supply that gauge; a multi-view observation supplies
only the insertion-tip centre and directed shaft axis. This is a read-only
geometric hypothesis, not a measured 6-DOF key pose or motion authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import validate_se3


def _unit(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if (vector.shape != (3,) or not np.all(np.isfinite(vector)) or
            np.linalg.norm(vector) < 1e-10):
        raise ValueError(f"{name} must be a finite nonzero 3-vector")
    return vector / np.linalg.norm(vector)


def _minimal_axis_rotation(source: np.ndarray,
                           destination: np.ndarray) -> np.ndarray:
    """Smallest SO(3) rotation taking one *directed* unit axis to another.

    The opposite-axis case has no unique minimum-rotation plane and would
    select a yaw gauge arbitrarily; it is deliberately rejected.
    """
    cosine = float(np.clip(np.dot(source, destination), -1.0, 1.0))
    if cosine < -1.0 + 1e-8:
        raise ValueError("observed key axis is antiparallel to grasp prior")
    cross = np.cross(source, destination)
    skew = np.array([[0., -cross[2], cross[1]],
                     [cross[2], 0., -cross[0]],
                     [-cross[1], cross[0], 0.]], dtype=np.float64)
    return np.eye(3) + skew + (skew @ skew) / (1.0 + cosine)


@dataclass(frozen=True)
class AxisymmetricHeldHypothesis:
    T_socket_key: np.ndarray
    T_robot_key: np.ndarray
    T_key_hand: np.ndarray
    predicted_tip_socket_m: np.ndarray
    observed_tip_socket_m: np.ndarray
    predicted_axis_socket: np.ndarray
    observed_axis_socket: np.ndarray
    tip_prior_residual_m: float
    axis_prior_residual_deg: float

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_axisymmetric_held_hypothesis_v1",
            "T_socket_key": self.T_socket_key.tolist(),
            "T_robot_key": self.T_robot_key.tolist(),
            "T_key_hand": self.T_key_hand.tolist(),
            "predicted_tip_socket_m": self.predicted_tip_socket_m.tolist(),
            "observed_tip_socket_m": self.observed_tip_socket_m.tolist(),
            "predicted_axis_socket": self.predicted_axis_socket.tolist(),
            "observed_axis_socket": self.observed_axis_socket.tolist(),
            "tip_prior_residual_m": self.tip_prior_residual_m,
            "axis_prior_residual_deg": self.axis_prior_residual_deg,
            "unobservable_dof": "rotation_about_cylinder_axis",
            "yaw_gauge": "minimum_axis_rotation_from_physical_grasp_medoid",
            "scope": "read_only_observed_tip_axis_plus_prior_yaw_hypothesis",
            "robot_ready": False,
        }


def reconstruct_axisymmetric_held_hypothesis(
    *, T_robot_socket: np.ndarray, T_robot_hand_measured: np.ndarray,
    T_key_hand_prior: np.ndarray, tip_key_m: np.ndarray,
    insertion_axis_key: np.ndarray, tip_socket_m: np.ndarray,
    insertion_axis_socket: np.ndarray,
    max_tip_prior_residual_m: float, max_axis_prior_residual_deg: float,
) -> AxisymmetricHeldHypothesis:
    """Preserve the prior yaw gauge while fitting observed tip and axis.

    Let H be the measured wrist and G the prior key-to-hand transform. The
    prior key pose is H G^-1. In socket coordinates, a minimum rotation maps
    its directed insertion axis to the triangulated axis; translation then
    places the local insertion tip at the triangulated 3-D tip. The resulting
    G' = K'^-1 H is a *hypothesis* whose uncertainty must be included in a
    separately commissioned whole-surface bound before any endpoint audit.
    """
    socket = validate_se3(T_robot_socket, name="frozen T_robot_socket")
    hand = validate_se3(T_robot_hand_measured, name="measured T_robot_hand")
    prior = validate_se3(T_key_hand_prior, name="physical medoid T_key_hand")
    tip_local = np.asarray(tip_key_m, dtype=np.float64)
    tip_observed = np.asarray(tip_socket_m, dtype=np.float64)
    if (tip_local.shape != (3,) or tip_observed.shape != (3,) or
            not np.all(np.isfinite(tip_local)) or
            not np.all(np.isfinite(tip_observed))):
        raise ValueError("key-local and observed insertion tips must be finite xyz")
    key_axis = _unit(insertion_axis_key, "key insertion axis")
    observed_axis = _unit(insertion_axis_socket, "observed insertion axis")
    tip_limit = float(max_tip_prior_residual_m)
    axis_limit = float(max_axis_prior_residual_deg)
    if (not math.isfinite(tip_limit) or tip_limit <= 0 or
            not math.isfinite(axis_limit) or not 0 < axis_limit < 90):
        raise ValueError("tip/axis prior residual limits must be commissioned")

    predicted = validate_se3(
        np.linalg.inv(socket) @ hand @ np.linalg.inv(prior),
        name="prior T_socket_key")
    predicted_tip = predicted[:3, :3] @ tip_local + predicted[:3, 3]
    predicted_axis = _unit(predicted[:3, :3] @ key_axis,
                           "predicted insertion axis")
    tip_error = float(np.linalg.norm(tip_observed - predicted_tip))
    axis_error = math.degrees(math.acos(float(np.clip(
        np.dot(predicted_axis, observed_axis), -1.0, 1.0))))
    if tip_error > tip_limit or axis_error > axis_limit:
        raise ValueError("observed key tip/axis disagrees with physical grasp prior")

    rotation = (_minimal_axis_rotation(predicted_axis, observed_axis) @
                predicted[:3, :3])
    observed = np.eye(4, dtype=np.float64)
    observed[:3, :3] = rotation
    observed[:3, 3] = tip_observed - rotation @ tip_local
    observed = validate_se3(observed, name="gauge-fixed T_socket_key")
    robot_key = validate_se3(socket @ observed, name="gauge-fixed T_robot_key")
    held = validate_se3(np.linalg.inv(robot_key) @ hand,
                        name="gauge-fixed T_key_hand")
    return AxisymmetricHeldHypothesis(
        observed, robot_key, held, predicted_tip, tip_observed,
        predicted_axis, observed_axis, tip_error, axis_error)


def tip_axis_visual_surface_bound(
    *, key_vertices_m: np.ndarray, tip_key_m: np.ndarray,
    max_tip_error_m: float, max_axis_error_deg: float,
) -> float:
    """Conservative extra key-surface error for the observed tip/axis fit.

    A tip translation error e and a rotation error a about that tip move any
    key vertex by at most e + 2 L sin(a/2), where L is its maximum distance
    from the chosen tip. This does *not* bound axial-yaw slip; the separate
    physical grasp-medoid bound and cylinder symmetry must cover that.
    """
    vertices = np.asarray(key_vertices_m, dtype=np.float64)
    tip = np.asarray(tip_key_m, dtype=np.float64)
    translation = float(max_tip_error_m)
    angle = float(max_axis_error_deg)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0 or
            tip.shape != (3,) or not np.all(np.isfinite(vertices)) or
            not np.all(np.isfinite(tip)) or
            not math.isfinite(translation) or translation <= 0 or
            not math.isfinite(angle) or not 0 < angle < 90):
        raise ValueError("visual surface bound needs metric CAD and error limits")
    lever = float(np.linalg.norm(vertices - tip, axis=1).max())
    return translation + 2.0 * lever * math.sin(math.radians(angle) / 2.0)
