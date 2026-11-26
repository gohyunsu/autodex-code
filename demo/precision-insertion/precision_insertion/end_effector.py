"""End-effector contracts shared by precision-insertion runtime modules.

The ZeroDex FR3 can be fitted with either the historical Inspire hand or the
stock Franka parallel gripper.  Keep those assets and joint conventions
separate: an Inspire BODex candidate is not a valid Franka-gripper candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from autodex.utils.robot_config import INSPIRE_INIT


@dataclass(frozen=True)
class EndEffectorSpec:
    name: str
    planner_hand: str
    asset_namespace: str
    hand_dof: int
    open_q: np.ndarray
    feedback_units: str

    @property
    def full_dof(self) -> int:
        return 7 + self.hand_dof


_SPECS = {
    "inspire": EndEffectorSpec(
        name="inspire", planner_hand="fr3_inspire",
        asset_namespace="inspire", hand_dof=6,
        open_q=np.asarray(INSPIRE_INIT, dtype=np.float64),
        feedback_units="inspire_raw_0_1000"),
    "franka_gripper": EndEffectorSpec(
        name="franka_gripper", planner_hand="fr3_gripper",
        asset_namespace="franka_gripper", hand_dof=2,
        # Each URDF prismatic joint is one half of the reported total width.
        open_q=np.asarray([0.04, 0.04], dtype=np.float64),
        feedback_units="finger_displacement_m"),
}


def normalize_end_effector(name: str | None) -> str:
    value = (name or "inspire").strip().lower().replace("-", "_")
    aliases = {
        "inspire_left": "inspire",
        "gripper": "franka_gripper",
        "stock_gripper": "franka_gripper",
        "parallel_gripper": "franka_gripper",
        "franka_hand": "franka_gripper",
    }
    value = aliases.get(value, value)
    if value not in _SPECS:
        raise ValueError(
            "PRECISION_END_EFFECTOR must be inspire or franka_gripper")
    return value


def end_effector_spec(name: str | None = None) -> EndEffectorSpec:
    selected = normalize_end_effector(
        os.environ.get("PRECISION_END_EFFECTOR") if name is None else name)
    spec = _SPECS[selected]
    # Return independent arrays even though the dataclass itself is frozen.
    return EndEffectorSpec(
        name=spec.name, planner_hand=spec.planner_hand,
        asset_namespace=spec.asset_namespace, hand_dof=spec.hand_dof,
        open_q=spec.open_q.copy(), feedback_units=spec.feedback_units)


def planner_end_effector_spec(planner) -> EndEffectorSpec:
    """Resolve the end effector from the planner, independent of the shell env."""
    planner_hand = getattr(planner, "_hand", None)
    name = {
        "fr3_inspire": "inspire",
        "fr3_gripper": "franka_gripper",
    }.get(planner_hand)
    if name is None:
        # Older planner test doubles and archived adapters predate the explicit
        # _hand field.  Their historical contract is the selected environment
        # mode (Inspire unless explicitly overridden).  A live planner always
        # carries _hand and still takes the strict branch below.
        if planner_hand is None:
            return end_effector_spec()
        raise ValueError(
            f"unsupported precision planner hand: {planner_hand!r}")
    spec = end_effector_spec(name)
    expected_link = "base_link" if spec.name == "inspire" else "fr3_hand"
    ee_link = getattr(planner, "_robot_cfg", {}).get(
        "kinematics", {}).get("ee_link")
    if getattr(planner, "_n_arm", None) != 7 or ee_link != expected_link:
        raise ValueError(
            f"precision planner needs the {expected_link} planner contract "
            f"for FR3/{spec.name}")
    return spec


def full_dof_end_effector_spec(full_dof: int) -> EndEffectorSpec:
    """Infer the saved end-effector contract from a full joint-vector width."""
    matches = [spec for spec in _SPECS.values()
               if spec.full_dof == int(full_dof)]
    if len(matches) != 1:
        raise ValueError(f"unsupported precision full DoF: {full_dof}")
    return end_effector_spec(matches[0].name)


def hand_q_from_gripper_width(width_m: float) -> np.ndarray:
    """Map Franka's total jaw width to the two URDF prismatic joints."""
    width = float(width_m)
    if not np.isfinite(width) or width < 0.0 or width > 0.08 + 1e-6:
        raise ValueError("Franka gripper width must be finite and in [0, 0.08] m")
    finger = np.clip(width * 0.5, 0.0, 0.04)
    return np.asarray([finger, finger], dtype=np.float64)


def gripper_width_from_hand_q(hand_q: np.ndarray) -> float:
    q = np.asarray(hand_q, dtype=np.float64).reshape(-1)
    if (q.shape != (2,) or not np.all(np.isfinite(q)) or
            np.any(q < -1e-6) or np.any(q > 0.04 + 1e-6)):
        raise ValueError("Franka gripper planner q must contain two finger joints")
    return float(np.clip(q.sum(), 0.0, 0.08))
