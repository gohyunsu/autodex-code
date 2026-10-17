"""Explicit experimental v8 key-in-hand hypothesis for an occluded lift."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .live_robot_state import LiveRobotState
from .session_policy import SessionDecision


_PARAMETERS = frozenset({
    "max_arm_hand_skew_s", "max_hand_command_error_raw",
    "max_arm_velocity_rad_s",
    "axial_waypoint_step_m",
})


@dataclass
class LiveNominalRelationHandlers:
    """Replan a held transfer without claiming a measured held-key pose."""

    live: object
    planner: object
    read_state: Callable[[], LiveRobotState]
    limits: object
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if not callable(self.read_state) or not isinstance(self.parameters, Mapping):
            raise TypeError("nominal relation needs measured joints and limits")
        missing = _PARAMETERS - self.parameters.keys()
        extra = self.parameters.keys() - _PARAMETERS
        if missing or extra:
            raise ValueError(
                f"nominal relation limits missing {sorted(missing)} or "
                f"unknown {sorted(extra)}")

    def plan_nominal_transfer(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "held_relation_evidence_required" or
                not decision.candidate_id):
            raise ValueError("nominal relation belongs to another held attempt")
        state = self.read_state()
        if not isinstance(state, LiveRobotState):
            raise TypeError("nominal relation needs measured Franka/Inspire feedback")
        result = runner.prepare_nominal_postlift_transfer(
            planner=self.planner, joint_sample=state,
            limits=self.limits, **dict(self.parameters))
        if result.status != "sampled_postlift_preflight_pass":
            raise ValueError(f"nominal held transfer preflight rejected: {result.status}")

    def handlers(self) -> dict[str, ActionHandler]:
        return {"held_relation_evidence_required": self.plan_nominal_transfer}
