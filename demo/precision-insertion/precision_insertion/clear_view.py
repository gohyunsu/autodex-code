"""Keep the FR3 and open Inspire out of AutoDex's perception views.

The stock FrankaExecutor owns the initial free-space ``home(clear_view=True)``
motion. After any trial, a release handler must instead execute its planned
vertical exit and socket-aware retract before this gate admits a new image.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np

from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT

from .live_robot_state import LiveRobotState
from .repose_release import ReposeReleasePreflight


@dataclass
class ClearViewForPerception:
    """Command only the *initial* clear view; gate every later key exposure.

    ``read_state`` must return current measured FR3/Inspire feedback, not the
    executor's remembered command. A release/reset handler can use
    ``finish_planned_release_exit`` to return to clear view. No key capture is
    admitted until that handler has completed and feedback agrees.
    """

    executor: object
    read_state: Callable[[], LiveRobotState]
    max_arm_hand_skew_s: float
    max_hand_command_error_raw: float
    max_arm_velocity_rad_s: float
    max_sample_age_s: float
    max_arm_error_rad: float = 0.05
    max_hand_error_rad: float = 0.05

    def __post_init__(self) -> None:
        if not callable(self.read_state) or not callable(
                getattr(self.executor, "home", None)):
            raise TypeError("clear view needs Franka home and measured feedback")
        for value in (
                self.max_arm_hand_skew_s, self.max_hand_command_error_raw,
                self.max_arm_velocity_rad_s, self.max_sample_age_s,
                self.max_arm_error_rad, self.max_hand_error_rad):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError("clear-view feedback limits must be positive")

    def verify_state(self, state: LiveRobotState, *, phase: str,
                     require_fresh: bool = True) -> LiveRobotState:
        self._validate_feedback(state, phase=phase, require_fresh=require_fresh)
        arm_error = float(np.max(np.abs(state.full_q[:7] - FR3_CLEAR_VIEW)))
        hand_error = float(np.max(np.abs(state.full_q[7:] - INSPIRE_INIT)))
        if (arm_error > self.max_arm_error_rad or
                hand_error > self.max_hand_error_rad):
            raise ValueError(
                f"{phase}: arm is not at clear view or Inspire is not open "
                f"(arm={arm_error:.4f}, hand={hand_error:.4f} rad)")
        return state

    def _validate_feedback(self, state: LiveRobotState, *, phase: str,
                           require_fresh: bool = True) -> None:
        if not isinstance(state, LiveRobotState):
            raise TypeError(f"{phase} needs measured LiveRobotState")
        state.validate(
            max_arm_hand_skew_s=self.max_arm_hand_skew_s,
            max_hand_command_error_raw=self.max_hand_command_error_raw,
            max_arm_velocity_rad_s=self.max_arm_velocity_rad_s)
        age = time.time() - state.sample_timestamp_s
        if require_fresh and (age < -self.max_sample_age_s or
                              age > self.max_sample_age_s):
            raise ValueError(f"{phase}: clear-view feedback is stale")

    def _verify_release_waypoint(self, expected: np.ndarray, *, phase: str,
                                 require_open: bool) -> LiveRobotState:
        state = self.read_state()
        self._validate_feedback(state, phase=phase)
        if np.max(np.abs(state.full_q[:7] - expected[:7])) > self.max_arm_error_rad:
            raise ValueError(f"{phase}: arm missed the release-exit waypoint")
        if (require_open and np.max(np.abs(state.full_q[7:] - INSPIRE_INIT)) >
                self.max_hand_error_rad):
            raise ValueError(f"{phase}: Inspire is not fully open")
        return state

    def prepare_initial(self) -> LiveRobotState:
        """Open hand and park arm *before* the first board/socket capture.

        Only call this before a trial, with no held key and a known free path.
        It is deliberately not the post-release recovery operation.
        """
        self.executor.home(clear_view=True)
        return self.verify_state(self.read_state(), phase="session_start")

    def before_key_capture(self) -> LiveRobotState:
        """Every trial/retrial needs a clear, open, stationary measured state."""
        return self.verify_state(self.read_state(), phase="before_key_capture")

    def finish_planned_release_exit(
        self, *, release_plan: ReposeReleasePreflight,
        open_hand: Callable[[np.ndarray], None],
        follow_joint_trajectory: Callable[[np.ndarray], None],
    ) -> LiveRobotState:
        """Open at the return site, ascend, then follow the planned retract.

        The caller must first execute the separately checked held-key return
        to the release site. These commissioned callbacks own motor control;
        this method enforces sequence and endpoint, then reads real feedback.
        It never calls free-space ``home`` after a trial.
        """
        if (not isinstance(release_plan, ReposeReleasePreflight) or
                release_plan.status !=
                "nominal_release_exit_path_pass_drop_unobserved" or
                release_plan.post_release_lift_trajectory is None or
                release_plan.retract_trajectory is None):
            raise ValueError("clear-view return needs a passed release/exit plan")
        if (not isinstance(release_plan.release_geometry_audit, dict) or
                release_plan.release_geometry_audit.get("sampled_clear") is not True):
            raise ValueError("clear-view return needs a passed release geometry audit")
        if (not callable(open_hand) or
                not callable(follow_joint_trajectory)):
            raise TypeError("release/exit needs commissioned motion callbacks")
        if (not np.allclose(release_plan.release_hand_q, INSPIRE_INIT,
                            atol=1e-8, rtol=0) or
                not np.allclose(release_plan.retreat_goal_arm_q, FR3_CLEAR_VIEW,
                                atol=1e-8, rtol=0)):
            raise ValueError("release plan must fully open and end at clear view")
        up = np.asarray(release_plan.post_release_lift_trajectory, dtype=float)
        retract = np.asarray(release_plan.retract_trajectory, dtype=float)
        if (up.ndim != 2 or retract.ndim != 2 or up.shape[1] != 13 or
                retract.shape[1] != 13 or len(up) < 2 or len(retract) < 2 or
                not np.all(np.isfinite(up)) or
                not np.all(np.isfinite(retract)) or
                not np.allclose(up[-1], retract[0], atol=1e-4, rtol=0) or
                not np.allclose(retract[-1, :7], FR3_CLEAR_VIEW,
                                atol=1e-3, rtol=0) or
                not np.allclose(up[:, 7:], INSPIRE_INIT, atol=1e-8, rtol=0) or
                not np.allclose(retract[:, 7:], INSPIRE_INIT,
                                atol=1e-8, rtol=0)):
            raise ValueError("release exit is not a continuous open-hand clear-view path")
        self._verify_release_waypoint(up[0], phase="before_release_open",
                                      require_open=False)
        open_hand(np.asarray(INSPIRE_INIT, dtype=float).copy())
        self._verify_release_waypoint(up[0], phase="after_release_open",
                                      require_open=True)
        follow_joint_trajectory(up.copy())
        self._verify_release_waypoint(up[-1], phase="after_release_lift",
                                      require_open=True)
        follow_joint_trajectory(retract.copy())
        return self.verify_state(self.read_state(), phase="after_release_exit")
