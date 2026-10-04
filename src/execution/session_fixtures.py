"""Session-scoped fixed-fixture pose validation and freezing.

The precision-insertion socket is measured at session startup, then treated as
immutable for every trial in that process.  These helpers deliberately contain
no camera or robot dependencies so the repeatability gate can be unit tested.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def validate_se3(value, *, name: str = "pose") -> np.ndarray:
    """Return ``value`` as a validated homogeneous rigid transform."""
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
    return pose


def rotation_distance_deg(a, b) -> float:
    """Geodesic SO(3) distance in degrees."""
    ra = validate_se3(a, name="pose_a")[:3, :3]
    rb = validate_se3(b, name="pose_b")[:3, :3]
    cosine = np.clip((np.trace(ra.T @ rb) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def freeze_pose_medoid(
    poses: Iterable[np.ndarray],
    *,
    translation_limit_mm: float,
    rotation_limit_deg: float,
) -> tuple[np.ndarray, dict]:
    """Select an observed SE(3) medoid and enforce session repeatability.

    Rotations must not be averaged elementwise.  Instead, the returned pose is
    one of the actual measurements: the sample with the smallest normalized
    pairwise translation-plus-rotation cost.  Limits apply to every sample's
    residual from that medoid and therefore reject a single unstable estimate.
    """
    if translation_limit_mm <= 0 or rotation_limit_deg <= 0:
        raise ValueError("fixture repeatability limits must be positive")
    samples = [validate_se3(p, name=f"pose[{i}]").copy()
               for i, p in enumerate(poses)]
    if len(samples) < 2:
        raise ValueError("at least two fixture measurements are required")

    n_samples = len(samples)
    translation_mm = np.zeros((n_samples, n_samples), dtype=np.float64)
    rotation_deg = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dt = float(np.linalg.norm(
                samples[i][:3, 3] - samples[j][:3, 3]) * 1000.0)
            dr = rotation_distance_deg(samples[i], samples[j])
            translation_mm[i, j] = translation_mm[j, i] = dt
            rotation_deg[i, j] = rotation_deg[j, i] = dr

    normalized_cost = (
        translation_mm / float(translation_limit_mm)
        + rotation_deg / float(rotation_limit_deg)
    ).sum(axis=1)
    selected_index = int(np.argmin(normalized_cost))
    selected_translation = translation_mm[selected_index]
    selected_rotation = rotation_deg[selected_index]
    max_translation = float(selected_translation.max())
    max_rotation = float(selected_rotation.max())
    accepted = (
        max_translation <= translation_limit_mm
        and max_rotation <= rotation_limit_deg
    )
    diagnostics = {
        "method": "observed_se3_medoid",
        "sample_count": n_samples,
        "selected_index": selected_index,
        "translation_limit_mm": float(translation_limit_mm),
        "rotation_limit_deg": float(rotation_limit_deg),
        "translation_residuals_mm": selected_translation.tolist(),
        "rotation_residuals_deg": selected_rotation.tolist(),
        "max_translation_residual_mm": max_translation,
        "max_rotation_residual_deg": max_rotation,
        "accepted": accepted,
    }
    if not accepted:
        raise ValueError(
            "fixture pose measurements are not repeatable: "
            f"translation {max_translation:.3f} mm "
            f"(limit {translation_limit_mm:.3f}), rotation "
            f"{max_rotation:.3f} deg (limit {rotation_limit_deg:.3f})"
        )
    return samples[selected_index], diagnostics
