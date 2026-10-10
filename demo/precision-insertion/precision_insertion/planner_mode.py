"""Declare and verify AutoDex's read-only Cartesian planner experiment mode."""

from __future__ import annotations

import os


def require_declared_cartesian_mode(mode: str) -> bool:
    """Return the native-mode flag or reject an ambiguous environment.

    The native locked-hand route is explicitly experimental in AutoDex. This
    check makes offline planner artifacts reproducible; it does not approve
    the planner for physical motion.
    """
    if mode not in {"default", "native-locked-experimental"}:
        raise ValueError("unknown AutoDex Cartesian planner mode")
    enabled = os.environ.get("AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS") == "1"
    if enabled != (mode == "native-locked-experimental"):
        raise ValueError(
            "--planner-mode must match AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS=1 "
            "(native-locked-experimental) or the unset/default planner")
    return enabled


def cartesian_mode_from_planner(planner) -> str:
    """Capture the actual initialized AutoDex mode in the trial record."""
    return ("native-locked-experimental" if getattr(
        planner, "_native_pose_constraints_enabled", False) else "default")
