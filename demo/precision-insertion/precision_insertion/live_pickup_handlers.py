"""Begin a selected AutoDex v8 attempt and plan lift from measured squeeze.

This is one live action-loop boundary. Candidate selection and pickup/lift
planning stay in SessionRunner and the existing demo execution primitives.
The lift is *not* commanded here and no grasp-success label is inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .live_robot_state import LiveRobotState
from .pickup_execution import PickupExecutionLimits, execute_bound_pickup
from .session_policy import SessionDecision


_MEASURED_LIFT_KEYS = frozenset({
    "max_state_age_s", "max_post_squeeze_arm_drift_rad",
    "max_post_squeeze_hand_drift_raw", "max_arm_hand_skew_s",
    "max_hand_command_error_raw", "max_arm_velocity_rad_s",
    "axial_waypoint_step_m",
})


@dataclass
class LivePickupHandlers:
    """Compose the existing selected-attempt, pickup and lift-plan steps.

    ``executor`` is the independently commissioned pickup executor required
    by ``execute_bound_pickup``. This handler does not replace its interlock
    or make the unchanged stock follower suitable for physical execution.
    """

    live: object
    planner: object
    executor: object
    read_state: Callable[[], LiveRobotState]
    motion_interlock: Callable[[], bool]
    pickup_limits: PickupExecutionLimits
    lift_bounds: object
    path_limits: object
    measured_lift_parameters: Mapping[str, float]
    enable_robot_motion: bool = False
    attempt_id_factory: Callable[[], str] = lambda: f"attempt_{uuid4().hex}"
    started_at_factory: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if (not callable(self.read_state) or
                not callable(self.motion_interlock) or
                not callable(self.attempt_id_factory) or
                not callable(self.started_at_factory) or
                not isinstance(self.pickup_limits, PickupExecutionLimits) or
                not isinstance(self.measured_lift_parameters, Mapping)):
            raise TypeError("live pickup needs feedback, limits and an interlock")
        missing = _MEASURED_LIFT_KEYS - self.measured_lift_parameters.keys()
        extra = self.measured_lift_parameters.keys() - _MEASURED_LIFT_KEYS
        if missing or extra:
            raise ValueError(
                f"measured lift limits missing {sorted(missing)} or unknown "
                f"{sorted(extra)}")

    def begin_and_squeeze(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "execution_gate_required"):
            raise ValueError("pickup belongs to another session or decision")
        if self.enable_robot_motion is not True:
            raise PermissionError("live pickup motion is not enabled")
        attempt = runner.begin_selected_attempt(
            attempt_id=self.attempt_id_factory(),
            started_at_s=self.started_at_factory())
        if attempt.candidate_id != decision.candidate_id:
            raise ValueError("started attempt differs from selected v8 grasp")
        before = self.read_state()
        after: LiveRobotState | None = None

        def read_post_state() -> LiveRobotState:
            nonlocal after
            after = self.read_state()
            return after

        execute_bound_pickup(
            runner=runner, executor=self.executor, planner=self.planner,
            pre_state=before, read_post_state=read_post_state,
            limits=self.pickup_limits,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion)
        if not isinstance(after, LiveRobotState):
            raise RuntimeError("pickup completed without measured squeeze feedback")
        pickup_log = (Path(runner.output_dir) / "attempts" /
                      attempt.attempt_id / "pickup_execution.json")
        chain = runner.prepare_measured_lift_chain(
            planner=self.planner, pickup_execution_log=pickup_log,
            joint_sample=after, bounds=self.lift_bounds,
            limits=self.path_limits,
            **dict(self.measured_lift_parameters))
        if chain.status != "sampled_measured_chain_pass":
            raise ValueError(f"measured squeeze has no passing lift chain: {chain.status}")

    def handlers(self) -> dict[str, ActionHandler]:
        return {"execution_gate_required": self.begin_and_squeeze}
