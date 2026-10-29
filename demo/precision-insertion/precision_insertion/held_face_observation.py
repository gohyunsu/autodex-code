"""Geometry for an optional held-key face observation before insertion.

The nominal key-in-hand pose is used only to plan and project an observation
pose. A measured face centre from calibrated views can then be mapped back to
the vertical insertion pose under the current translation-only error model.
This module does not command the robot or claim that the face is unoccluded.
"""

from __future__ import annotations

from itertools import combinations
import math
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from .geometry import validate_se3


def _turn(socket: np.ndarray, tilt_deg: float,
          axis_azimuth_deg: float) -> np.ndarray:
    if not all(math.isfinite(value) for value in
               (tilt_deg, axis_azimuth_deg)):
        raise ValueError("tilt and azimuth must be finite degrees")
    azimuth = math.radians(axis_azimuth_deg)
    axis = (math.cos(azimuth) * socket[:3, 0] +
            math.sin(azimuth) * socket[:3, 1])
    rotation, _ = cv2.Rodrigues(axis * math.radians(tilt_deg))
    return rotation


def tilted_observation_goals(
    *, T_robot_key_preinsert: np.ndarray,
    T_robot_wrist_current: np.ndarray,
    T_robot_socket: np.ndarray,
    raise_m: float, tilt_deg: float, axis_azimuth_deg: float,
) -> dict[str, np.ndarray]:
    """Raise the held key, then rotate the wrist about its nominal key origin.

    The returned wrist poses preserve the nominal rigid key-to-wrist relation.
    A planner must still find and audit the raise, tilt, untilt and return
    trajectories from the *measured* joints before any motion is commanded.
    """
    key = validate_se3(T_robot_key_preinsert, name="preinsert key")
    wrist = validate_se3(T_robot_wrist_current, name="measured wrist")
    socket = validate_se3(T_robot_socket, name="measured socket")
    if not math.isfinite(raise_m) or raise_m < 0:
        raise ValueError("observation raise must be nonnegative metres")
    turn = _turn(socket, tilt_deg, axis_azimuth_deg)
    held_relation = np.linalg.inv(key) @ wrist
    raised_key = key.copy()
    raised_key[:3, 3] += socket[:3, 2] * raise_m
    observed_key = raised_key.copy()
    observed_key[:3, :3] = turn @ raised_key[:3, :3]
    return {
        "T_robot_key_raised": raised_key,
        "T_robot_key_observation": observed_key,
        "T_robot_wrist_raised": raised_key @ held_relation,
        "T_robot_wrist_observation": observed_key @ held_relation,
        "T_robot_wrist_return": wrist.copy(),
        "R_robot_tilt": turn,
    }


def rank_face_view_tilts(
    *, T_robot_key_preinsert: np.ndarray,
    T_robot_socket: np.ndarray,
    camera_centers_robot: Mapping[str, Sequence[float]],
    tip_z_m: float, raise_m: float,
    tilt_degrees: Sequence[float], azimuth_degrees: Sequence[float],
) -> list[dict]:
    """Rank face-facing camera pairs by their weaker view and parallax.

    A positive face dot only means the camera is in front of the CAD face.
    Hand/socket occlusion, image pixels and FR3 reachability are not tested.
    No threshold on the ranking score rejects a grasp or an insertion.
    """
    key = validate_se3(T_robot_key_preinsert, name="preinsert key")
    socket = validate_se3(T_robot_socket, name="measured socket")
    if not math.isfinite(tip_z_m) or tip_z_m <= 0:
        raise ValueError("key tip z must be positive metres")
    if not math.isfinite(raise_m) or raise_m < 0:
        raise ValueError("observation raise must be nonnegative metres")
    if len(tilt_degrees) == 0 or len(azimuth_degrees) == 0:
        raise ValueError("observation ranking needs sampled rotations")
    cameras = {}
    for name, value in camera_centers_robot.items():
        center = np.asarray(value, dtype=float)
        if (not isinstance(name, str) or not name or
                center.shape != (3,) or not np.isfinite(center).all()):
            raise ValueError("camera centres need names and finite robot XYZ")
        cameras[name] = center
    if len(cameras) < 2:
        raise ValueError("face observation needs at least two cameras")

    rows = []
    tip_local = np.array([0., 0., tip_z_m, 1.])
    for tilt_deg in tilt_degrees:
        for azimuth_deg in azimuth_degrees:
            turn = _turn(socket, float(tilt_deg), float(azimuth_deg))
            observed_key = key.copy()
            observed_key[:3, :3] = turn @ key[:3, :3]
            observed_key[:3, 3] += socket[:3, 2] * raise_m
            tip = (observed_key @ tip_local)[:3]
            normal = observed_key[:3, 2]
            rays = {}
            for camera, center in cameras.items():
                direction = center - tip
                length = float(np.linalg.norm(direction))
                if length > 0:
                    rays[camera] = direction / length
            dots = {camera: float(normal @ ray)
                    for camera, ray in rays.items()}
            pairs = []
            for first, second in combinations(sorted(rays), 2):
                if min(dots[first], dots[second]) <= 0:
                    continue
                parallax = float(np.linalg.norm(
                    np.cross(rays[first], rays[second])))
                pairs.append((min(dots[first], dots[second]) * parallax,
                              first, second, parallax))
            if not pairs:
                continue
            score, first, second, parallax = max(pairs)
            rows.append({
                "tilt_deg": float(tilt_deg),
                "axis_azimuth_socket_deg": float(azimuth_deg),
                "best_pair": [first, second],
                "best_pair_score": score,
                "best_pair_parallax_sin": parallax,
                "face_dot_by_camera": dots,
                "predicted_tip_robot_m": tip.tolist(),
            })
    return sorted(rows, key=lambda row: row["best_pair_score"], reverse=True)


def xy_correction_from_tilted_tip(
    *, observed_tip_robot_m: Sequence[float],
    T_robot_key_observation: np.ndarray,
    R_robot_tilt: np.ndarray,
    T_robot_socket: np.ndarray, tip_z_m: float,
) -> dict:
    """Undo a known tilt and convert a measured tip translation to socket XY.

    The face centre must be an actual 3D image measurement. The inverse
    rotation is valid when the key stays rigid in the hand and its departure
    from the nominal pose is translation only; an orientation error needs a
    separate 6D pose estimate.
    """
    key = validate_se3(T_robot_key_observation, name="observed key hypothesis")
    socket = validate_se3(T_robot_socket, name="measured socket")
    turn = np.asarray(R_robot_tilt, dtype=float)
    observed = np.asarray(observed_tip_robot_m, dtype=float)
    if (turn.shape != (3, 3) or not np.isfinite(turn).all() or
            not np.allclose(turn.T @ turn, np.eye(3), atol=1e-3) or
            observed.shape != (3,) or not np.isfinite(observed).all() or
            not math.isfinite(tip_z_m) or tip_z_m <= 0):
        raise ValueError("tilted tip conversion needs finite rigid geometry")
    nominal_tip = (key @ np.array([0., 0., tip_z_m, 1.]))[:3]
    translation_robot = turn.T @ (observed - nominal_tip)
    translation_socket = socket[:3, :3].T @ translation_robot
    return {
        "measured_tip_robot_m": observed.tolist(),
        "nominal_tilted_tip_robot_m": nominal_tip.tolist(),
        "key_translation_socket_m": translation_socket.tolist(),
        "xy_correction_socket_m": (-translation_socket[:2]).tolist(),
    }
