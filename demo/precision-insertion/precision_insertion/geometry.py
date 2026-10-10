"""Validate and freeze session fixture poses without camera/robot side effects."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def validate_se3(value, *, name: str = "pose") -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError(f"{name} rotation determinant is not +1")
    return pose.copy()


def _unit_axis(value) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("local_axis must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("local_axis must be non-zero")
    return axis / norm


def pose_angle_deg(a: np.ndarray, b: np.ndarray, *, local_axis=None) -> float:
    """Full SO(3) distance, or oriented-axis distance for a C∞ socket.

    Axial yaw is unobservable for the round socket, but an upside-down open
    rim is not equivalent to an upright one, so the axis sign is preserved.
    """
    ra = validate_se3(a, name="pose_a")[:3, :3]
    rb = validate_se3(b, name="pose_b")[:3, :3]
    if local_axis is None:
        cosine = (np.trace(ra.T @ rb) - 1.0) / 2.0
    else:
        axis = _unit_axis(local_axis)
        cosine = np.dot(ra @ axis, rb @ axis)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def freeze_fixture_pose(
    poses: Iterable[np.ndarray],
    *,
    translation_limit_mm: float,
    angle_limit_deg: float,
    continuous_axis_local=None,
) -> tuple[np.ndarray, dict]:
    """Choose an observed medoid and reject non-repeatable measurements.

    The chosen transform is always one input sample. This avoids averaging
    rotation matrices and keeps the collision pose tied to saved evidence.
    """
    limits = (translation_limit_mm, angle_limit_deg)
    if not all(np.isfinite(v) and v > 0 for v in limits):
        raise ValueError("fixture repeatability limits must be finite and positive")
    axis = (None if continuous_axis_local is None
            else _unit_axis(continuous_axis_local))
    samples = [validate_se3(value, name=f"pose[{i}]")
               for i, value in enumerate(poses)]
    if len(samples) < 2:
        raise ValueError("at least two fixture measurements are required")

    count = len(samples)
    translation_mm = np.zeros((count, count), dtype=np.float64)
    angle_deg = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.linalg.norm(
                samples[i][:3, 3] - samples[j][:3, 3]) * 1000.0)
            angle = pose_angle_deg(samples[i], samples[j], local_axis=axis)
            translation_mm[i, j] = translation_mm[j, i] = distance
            angle_deg[i, j] = angle_deg[j, i] = angle

    costs = (translation_mm / translation_limit_mm
             + angle_deg / angle_limit_deg).sum(axis=1)
    selected_index = int(np.argmin(costs))
    translation_residuals = translation_mm[selected_index]
    angle_residuals = angle_deg[selected_index]
    max_translation = float(translation_residuals.max())
    max_angle = float(angle_residuals.max())
    diagnostics = {
        "method": ("observed_axisymmetric_se3_medoid" if axis is not None
                   else "observed_se3_medoid"),
        "sample_count": count,
        "selected_index": selected_index,
        "translation_limit_mm": float(translation_limit_mm),
        "angle_limit_deg": float(angle_limit_deg),
        "translation_residuals_mm": translation_residuals.tolist(),
        "angle_residuals_deg": angle_residuals.tolist(),
        "max_translation_residual_mm": max_translation,
        "max_angle_residual_deg": max_angle,
        "continuous_axis_local": None if axis is None else axis.tolist(),
        "accepted": (max_translation <= translation_limit_mm
                     and max_angle <= angle_limit_deg),
    }
    if not diagnostics["accepted"]:
        raise ValueError(
            "fixture pose measurements are not repeatable: "
            f"translation {max_translation:.3f} mm "
            f"(limit {translation_limit_mm:.3f}), angle {max_angle:.3f} deg "
            f"(limit {angle_limit_deg:.3f})"
        )
    return samples[selected_index], diagnostics
