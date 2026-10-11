"""Plan directed AutoDex v8 reorientation when a tabletop pose is exhausted.

The same just-captured key pose and measured joints are reused. A nominal
reset plan is not motion or evidence that the key landed in the target pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from autodex.utils.robot_config import FR3_CLEAR_VIEW

from .action_loop import ActionHandler
from .live_key_trial import PreparedLiveKeyTrial
from .repose_target_selector import (
    DirectedReposeTargetSelector, ReposeTargetsExhausted,
)
from .reset_candidates import load_v8_reset_seeds
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
    parameters: Mapping[str, object]
    select_target: Callable = field(default_factory=DirectedReposeTargetSelector)

    def plan(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action not in _ACTIONS):
            raise ValueError("repose planning belongs to another session or decision")
        latest = self.key_planning.latest
        if not isinstance(latest, PreparedLiveKeyTrial):
            raise ValueError("repose needs the saved exhausted tabletop capture")
        parameters = dict(self.parameters)
        try:
            if isinstance(self.select_target, DirectedReposeTargetSelector):
                drift = parameters.get("max_center_in_hand_drift_m")
                tilt = parameters.get("max_symmetry_axis_tilt_deg")
                source = latest.preflight.pose_class["stem"]

                def seed_available(stem: str, height: int) -> bool:
                    # The loader performs the same source/MuJoCo checks as
                    # later preflight; fidelity filtering is opt-in. Identity avoids
                    # inventing a second key pose for this availability query.
                    return load_v8_reset_seeds(
                        shared_root=runner.shared_root, mode=runner.mode,
                        height_cm=height, from_pose_stem=source,
                        to_pose_stem=stem, T_robot_key=np.eye(4),
                        max_center_in_hand_drift_m=drift,
                        max_symmetry_axis_tilt_deg=tilt,
                        candidate_root=parameters.get(
                            "reset_candidate_root")) is not None

                target = self.select_target(
                    runner, decision, latest,
                    seed_available=seed_available)
            else:
                target = self.select_target(runner, decision, latest)
        except ReposeTargetsExhausted:
            attempted = getattr(self.select_target, "attempted_choices", None)
            runner.exhaust_repose_targets(
                attempted() if callable(attempted) else ())
            return
        if (not isinstance(target, Mapping) or
                set(target) != {"to_pose_stem", "height_cm",
                                "release_xy_robot_m"}):
            raise ValueError("repose selector needs target pose, height and XY")
        # The live action loop needs release/open/exit in the *same* preflight.
        # Keep AutoDex's clear-view retract as the default, while callers may
        # configure a larger key clearance for their printed object.
        parameters.setdefault("retreat_goal_arm_q", FR3_CLEAR_VIEW)
        parameters.setdefault("minimum_release_key_clearance_m", .005)
        runner.preflight_repose(
            planner=self.planner,
            key_observation=latest.observation,
            live_start_q=latest.state.full_q,
            start_q_acquisition_timestamp_s=latest.state.sample_timestamp_s,
            limits=self.path_limits, **dict(target),
            **parameters)

    def handlers(self) -> dict[str, ActionHandler]:
        return {action: self.plan for action in _ACTIONS}
