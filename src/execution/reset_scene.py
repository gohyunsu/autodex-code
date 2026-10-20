"""Scene pose for a carried release or an observed missed grasp."""

from __future__ import annotations

import numpy as np

from autodex.utils.conversion import cart2se3, se32cart


def scene_for_reset(
    scene_cfg: dict, *, key_was_carried: bool,
    grasp_wrist: np.ndarray | None = None,
    current_wrist: np.ndarray | None = None,
) -> dict:
    """Keep a missed key on the table; project only a carried key from wrist.

    Copy the target row because a reset preflight must not rewrite the scene
    used by the original pickup plan.
    """
    scene = dict(scene_cfg)
    scene["mesh"] = dict(scene_cfg["mesh"])
    scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
    scene["mesh"]["target"]["pose"] = list(
        scene_cfg["mesh"]["target"]["pose"])
    if key_was_carried:
        if grasp_wrist is None or current_wrist is None:
            raise ValueError("carried-key reset needs grasp and current wrists")
        target_at_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        target_in_wrist = np.linalg.inv(grasp_wrist) @ target_at_grasp
        scene["mesh"]["target"]["pose"] = se32cart(
            current_wrist @ target_in_wrist).tolist()
    return scene


def vertical_clearance_poses(
    planner, start_full_q: np.ndarray, lift_height_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Request a vertical lift with the actual planner wrist orientation."""
    start = np.asarray(planner.fk_wrist(start_full_q), dtype=np.float64)
    end = start.copy()
    end[2, 3] += lift_height_m
    return start, end
