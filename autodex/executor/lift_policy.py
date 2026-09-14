"""Arm-neutral policy for replaying a grasp-time lift preflight.

Robot executors own hardware IO, but deciding whether a trajectory planned
from an expected joint state may be replayed is the same safety rule for every
arm.  Keeping this small module free of robot SDK imports makes the contract
usable by xArm, FR3, and future executors.
"""
from dataclasses import dataclass

import numpy as np


# Joint tracking error after the approach/grasp sequence.  Both bounds are
# deliberate: max error catches one slipped joint, while L2 catches many small
# errors.  Finger joints are excluded because a squeeze intentionally changes
# them from the planner's nominal grasp configuration.
LIFT_START_MAX_ABS_RAD = 0.12
LIFT_START_L2_RAD = 0.20
# Planner-space hand joints are not universally angular (Franka fingers are
# prismatic, and controller APIs vary), so this is an explicit per-joint-value
# bound rather than a value labelled "rad".
LIFT_START_HAND_MAX_ABS = 0.08
LIFT_START_HAND_L2 = 0.16


class LiftExecutionError(RuntimeError):
    """Lift cannot be safely executed; the object must remain held.

    ``run_auto`` treats this differently from an approach/contact failure:
    opening the hand or issuing an unverified lateral reset would turn a
    planning failure into a drop/collision hazard.
    """

    object_held = True


@dataclass(frozen=True)
class LiftStartCheck:
    """Measured-vs-planned arm-state comparison for execution logging."""
    accepted: bool
    max_abs_rad: float
    l2_rad: float
    expected_arm_qpos: np.ndarray
    live_arm_qpos: np.ndarray
    hand_checked: bool = False
    hand_accepted: bool = True
    hand_max_abs: float = 0.0
    hand_l2: float = 0.0
    expected_hand_qpos: np.ndarray | None = None
    live_hand_qpos: np.ndarray | None = None


def check_lift_start(live_arm_qpos: np.ndarray, expected_full_qpos: np.ndarray,
                     *, arm_dof: int,
                     max_abs_tol_rad: float = LIFT_START_MAX_ABS_RAD,
                     l2_tol_rad: float = LIFT_START_L2_RAD,
                     live_hand_qpos: np.ndarray | None = None,
                     max_hand_abs_tol: float = LIFT_START_HAND_MAX_ABS,
                     hand_l2_tol: float = LIFT_START_HAND_L2) -> LiftStartCheck:
    """Return whether a planner preflight can be replayed from ``live`` q.

    Hand state is checked when an adapter can provide planner-space measured
    qpos.  A missing reading is represented explicitly rather than silently
    accepting a preflight whose collision model used a different hand shape.
    """
    live = np.asarray(live_arm_qpos, dtype=np.float64).reshape(-1)[:arm_dof]
    expected = np.asarray(expected_full_qpos, dtype=np.float64).reshape(-1)[:arm_dof]
    if len(live) != arm_dof or len(expected) != arm_dof:
        raise ValueError(
            f"lift-start comparison requires {arm_dof} arm joints, got "
            f"live={len(live)}, expected={len(expected)}")
    delta = live - expected
    max_abs = float(np.max(np.abs(delta)))
    l2 = float(np.linalg.norm(delta))
    expected_hand = np.asarray(expected_full_qpos, dtype=np.float64).reshape(-1)[arm_dof:]
    hand_checked = live_hand_qpos is not None
    live_hand = None
    hand_max_abs = 0.0
    hand_l2 = 0.0
    hand_accepted = True
    if hand_checked:
        live_hand = np.asarray(live_hand_qpos, dtype=np.float64).reshape(-1)
        if live_hand.shape != expected_hand.shape or not np.isfinite(live_hand).all():
            hand_accepted = False
            hand_max_abs = float("inf")
            hand_l2 = float("inf")
        else:
            hand_delta = live_hand - expected_hand
            hand_max_abs = float(np.max(np.abs(hand_delta))) if len(hand_delta) else 0.0
            hand_l2 = float(np.linalg.norm(hand_delta))
            hand_accepted = (hand_max_abs <= max_hand_abs_tol
                             and hand_l2 <= hand_l2_tol)
    return LiftStartCheck(
        accepted=(max_abs <= max_abs_tol_rad and l2 <= l2_tol_rad and hand_accepted),
        max_abs_rad=max_abs,
        l2_rad=l2,
        expected_arm_qpos=expected.copy(),
        live_arm_qpos=live.copy(),
        hand_checked=hand_checked,
        hand_accepted=hand_accepted,
        hand_max_abs=hand_max_abs,
        hand_l2=hand_l2,
        expected_hand_qpos=expected_hand.copy(),
        live_hand_qpos=None if live_hand is None else live_hand.copy(),
    )
