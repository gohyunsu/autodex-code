"""Explicit v8 object identities for the two precision-insertion modes."""

from __future__ import annotations

from dataclasses import dataclass


SQUARE_GAPS_MM = (0.3, 0.5, 1.0, 1.5)
CYLINDER_RADIAL_GAPS_MM = (1, 3, 5, 10, 15, 20)


@dataclass(frozen=True)
class TaskMode:
    family: str
    gap_mm: float
    key_object: str
    socket_object: str
    yaw_relevant: bool
    target_depth_m: float = 0.020


def select_mode(family: str, gap_mm: float) -> TaskMode:
    """Resolve exact runtime IDs; a proxy mesh is never a physical target."""
    gap = float(gap_mm)
    if family == "square":
        if gap not in SQUARE_GAPS_MM:
            raise ValueError(f"square gap must be one of {SQUARE_GAPS_MM} mm")
        stem = str(gap).replace(".", "p")
        return TaskMode(
            family, gap, f"precision_key_{stem}mm",
            "precision_socket_unified", True,
        )
    if family == "cylinder":
        if gap not in CYLINDER_RADIAL_GAPS_MM:
            raise ValueError(
                f"cylinder radial gap must be one of {CYLINDER_RADIAL_GAPS_MM} mm"
            )
        return TaskMode(
            family, gap, "precision_key_cylinder_r15_h80",
            f"precision_socket_cylinder_gap_{int(gap):02d}mm", False,
        )
    raise ValueError("family must be 'square' or 'cylinder'")
