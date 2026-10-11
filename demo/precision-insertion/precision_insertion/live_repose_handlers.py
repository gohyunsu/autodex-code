"""Plan directed AutoDex v8 reorientation when a tabletop pose is exhausted.

The same just-captured key pose and measured joints are reused. A nominal
reset plan is not motion or evidence that the key landed in the target pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .live_key_trial import PreparedLiveKeyTrial
from .session_policy import SessionDecision


_ACTIONS = (
    "preflight_repose", "continue_repose_seed_preflight",
    "preflight_repose_release", "preflight_alternative_repose_cell",
)


@dataclass
class LiveReposePlanningHandlers:
    """Call the existing directed v8 reset planner, without new perception."""

    live: object
    key_planning: object
    planner: object
    path_limits: object
    select_target: Callable
    parameters: Mapping[str, object]

    def plan(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action not in _ACTIONS):
            raise ValueError("repose planning belongs to another session or decision")
        latest = self.key_planning.latest
        if not isinstance(latest, PreparedLiveKeyTrial):
            raise ValueError("repose needs the saved exhausted tabletop capture")
        target = self.select_target(runner, decision, latest)
        if (not isinstance(target, Mapping) or
                set(target) != {"to_pose_stem", "height_cm",
                                "release_xy_robot_m"}):
            raise ValueError("repose selector needs target pose, height and XY")
        runner.preflight_repose(
            planner=self.planner,
            key_observation=latest.observation,
            live_start_q=latest.state.full_q,
            start_q_acquisition_timestamp_s=latest.state.sample_timestamp_s,
            limits=self.path_limits, **dict(target),
            **dict(self.parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {action: self.plan for action in _ACTIONS}
