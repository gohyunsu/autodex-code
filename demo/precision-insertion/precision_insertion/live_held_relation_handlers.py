"""Resolve an occluded held key from grasp-specific physical calibration.

Raw Qwen lift success alone cannot determine the key-to-hand transform.
This handler uses the existing bounded post-lift planner; a MuJoCo squeeze
or nominal BODex pose is not promoted to an observed physical relation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .live_robot_state import LiveRobotState
from .session_policy import SessionDecision


_PARAMETERS = frozenset({
    "max_state_age_s", "max_arm_hand_skew_s",
    "max_hand_command_error_raw", "max_arm_velocity_rad_s",
    "max_postlift_arm_drift_rad", "max_postlift_hand_drift_raw",
    "max_calibration_hand_excess_rad", "axial_waypoint_step_m",
})


@dataclass
class LiveHeldRelationHandlers:
    """Plan held transfer from calibrated key/hand uncertainty, without motion."""

    live: object
    planner: object
    read_state: Callable[[], LiveRobotState]
    calibration_for_candidate: Callable[[str], Path]
    bounds: object
    path_limits: object
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        if (not callable(self.read_state) or
                not callable(self.calibration_for_candidate) or
                not isinstance(self.parameters, Mapping)):
            raise TypeError("held relation needs measured state and physical calibration")
        missing = _PARAMETERS - self.parameters.keys()
        extra = self.parameters.keys() - _PARAMETERS
        if missing or extra:
            raise ValueError(
                f"held relation limits missing {sorted(missing)} or "
                f"unknown {sorted(extra)}")

    def plan_bounded_transfer(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "held_relation_evidence_required" or
                not decision.candidate_id):
            raise ValueError("held relation belongs to another session or decision")
        source = Path(self.calibration_for_candidate(
            decision.candidate_id)).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"selected grasp lacks physical key/hand calibration: {source}")
        state = self.read_state()
        if not isinstance(state, LiveRobotState):
            raise TypeError("held relation needs measured Franka/Inspire feedback")
        result = runner.prepare_bounded_postlift_transfer(
            planner=self.planner, physical_calibration_path=source,
            joint_sample=state, bounds=self.bounds, limits=self.path_limits,
            **dict(self.parameters))
        if result.status != "sampled_postlift_preflight_pass":
            raise ValueError(f"held transfer preflight rejected: {result.status}")

    def handlers(self) -> dict[str, ActionHandler]:
        return {"held_relation_evidence_required": self.plan_bounded_transfer}
