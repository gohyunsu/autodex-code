"""v8 object_processing axis symmetry for precision-key tabletop poses."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .geometry import validate_se3


@dataclass(frozen=True)
class AxialSymmetry:
    axis_local: np.ndarray
    end_exchange_axis_local: np.ndarray | None

    @property
    def end_exchange(self) -> bool:
        return self.end_exchange_axis_local is not None


def _axis(value, *, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return axis / norm


def load_axial_symmetry(object_root: Path, object_name: str) -> AxialSymmetry:
    """Read the cylinder's local axis from its own v8 asset, not legacy IDs."""
    root = Path(object_root) / object_name / "processed_data" / "info"
    path = root / "symmetry.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("type") not in ("Cinf", "Dinf"):
        raise ValueError(f"{object_name}: expected Cinf or Dinf symmetry")
    axes = data.get("axes")
    if not isinstance(axes, list):
        raise ValueError(f"{object_name}: symmetry axes missing")
    continuous = [_axis(item.get("axis"), name="continuous axis")
                  for item in axes if item.get("fold") in ("inf", "Cinf")]
    if len(continuous) != 1:
        raise ValueError(f"{object_name}: expected one continuous symmetry axis")
    axis = continuous[0]
    flips = [_axis(item.get("axis"), name="end-exchange axis")
             for item in axes if item.get("fold") == 2]
    perpendicular = [value for value in flips
                     if abs(float(np.dot(value, axis))) < 1e-4]
    if data["type"] == "Dinf" and not perpendicular:
        raise ValueError(f"{object_name}: Dinf end-exchange axis missing")
    if data["type"] == "Cinf" and flips:
        raise ValueError(f"{object_name}: Cinf must not exchange open and closed ends")
    return AxialSymmetry(axis, perpendicular[0] if perpendicular else None)


def snap_axisymmetric_tabletop_pose(
    pose_robot: np.ndarray,
    *,
    object_root: Path,
    object_name: str,
    max_axis_error_deg: float = 20.0,
) -> np.ndarray:
    """Snap cylinder rotation to a v8 tabletop class, preserving translation.

    Axial yaw is canonical when upright. A D∞ key can exchange identical ends;
    a C∞ socket cannot. Large axis disagreement is rejected rather than
    silently converted into a different tabletop pose.
    """
    snapped, _, _ = match_axisymmetric_tabletop_pose(
        pose_robot, object_root=object_root, object_name=object_name,
        max_axis_error_deg=max_axis_error_deg)
    return snapped


def match_axisymmetric_tabletop_pose(
    pose_robot: np.ndarray,
    *,
    object_root: Path,
    object_name: str,
    max_axis_error_deg: float = 20.0,
) -> tuple[np.ndarray, str, float]:
    """Return snapped D∞ key pose, selected v8 stem and axis residual."""
    pose = validate_se3(pose_robot)
    if not np.isfinite(max_axis_error_deg) or not 0 < max_axis_error_deg < 180:
        raise ValueError("max_axis_error_deg must be in (0, 180)")
    symmetry = load_axial_symmetry(object_root, object_name)
    tabletop_dir = (Path(object_root) / object_name / "processed_data" /
                    "info" / "tabletop")
    tabletop_files = sorted(tabletop_dir.glob("*.npy"))
    if not tabletop_files:
        raise FileNotFoundError(f"{object_name}: v8 tabletop poses missing")

    estimated_axis = pose[:3, :3] @ symmetry.axis_local
    best_rotation = None
    best_stem = None
    best_error = float("inf")
    for path in tabletop_files:
        tabletop = validate_se3(np.load(path), name=f"tabletop {path.name}")
        rotation = tabletop[:3, :3].copy()
        table_axis = rotation @ symmetry.axis_local
        if symmetry.end_exchange and np.dot(estimated_axis, table_axis) < 0:
            flip_axis = symmetry.end_exchange_axis_local
            flip_rotation = 2.0 * np.outer(flip_axis, flip_axis) - np.eye(3)
            rotation = rotation @ flip_rotation
            table_axis = rotation @ symmetry.axis_local

        estimated_xy = np.linalg.norm(estimated_axis[:2])
        table_xy = np.linalg.norm(table_axis[:2])
        yaw = (0.0 if min(estimated_xy, table_xy) < 1e-8 else
               np.arctan2(estimated_axis[1], estimated_axis[0])
               - np.arctan2(table_axis[1], table_axis[0]))
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ rotation
        snapped_axis = rotation @ symmetry.axis_local
        error = float(np.degrees(np.arccos(np.clip(
            np.dot(estimated_axis, snapped_axis), -1.0, 1.0))))
        if error < best_error:
            best_error = error
            best_rotation = rotation
            best_stem = path.stem

    if best_rotation is None or best_error > max_axis_error_deg:
        raise ValueError(
            f"{object_name}: axis does not match a v8 tabletop pose "
            f"(error {best_error:.2f} deg)")
    result = pose.copy()
    result[:3, :3] = best_rotation
    return result, best_stem, best_error
