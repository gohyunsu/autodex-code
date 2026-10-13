"""Keep the FR3 and open Inspire out of AutoDex's perception views.

AutoDex's ``src.execution.franka_executor.FrankaExecutor`` owns the initial
free-space ``home(clear_view=True)``; ``autodex.executor.real.RealExecutor``
is the xArm executor and must never be used for this demo. The Franka executor
motion. After any trial, a release handler must instead execute its planned
vertical exit and socket-aware retract before this gate admits a new image.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np

from autodex.utils.robot_config import FR3_CLEAR_VIEW, INSPIRE_INIT

from .live_robot_state import LiveRobotState
from .repose_preflight import ReposeHeldPreflight
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
        if (getattr(self.executor, "arm_dof", None) != 7 or
                getattr(self.executor, "hand_name", None) not in
                {"inspire", "inspire_left"}):
            raise TypeError(
                "precision insertion needs the 7-DOF FrankaExecutor with Inspire; "
                "AutoDex RealExecutor controls xArm")
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

    def prepare_failed_grasp_for_capture(
        self, *, pickup_plan, planner, trial_scene: dict,
    ) -> LiveRobotState:
        """Reuse AutoDex release/reset after an observed *missed* grasp.

        AutoDex's reset plans a vertical clearance then a retract to
        ``FR3_CLEAR_VIEW``. Its stock scene moves ``target`` to the predicted
        release site; on a miss the real key may instead remain at its original
        tabletop pose. Preserve that original target as a second obstacle in
        the copied world, while retaining the frozen socket. Never execute a
        free-space home here. A slip/unknown held-key state is not a miss and
        must not call this path.
        """
        state = self.read_state()
        self._validate_feedback(state, phase="failed_grasp_recovery")
        if (np.max(np.abs(state.full_q[:7] - FR3_CLEAR_VIEW)) <=
                self.max_arm_error_rad and
                np.max(np.abs(state.full_q[7:] - INSPIRE_INIT)) <=
                self.max_hand_error_rad):
            return state
        if (not getattr(pickup_plan, "success", False) or planner is None or
                not callable(getattr(self.executor, "release", None)) or
                not callable(getattr(self.executor, "reset", None)) or
                not isinstance(trial_scene, dict) or
                not isinstance(trial_scene.get("mesh"), dict) or
                not isinstance(trial_scene["mesh"].get("target"), dict) or
                "fixture_socket" not in trial_scene["mesh"]):
            raise ValueError("failed-grasp clear view needs the selected AutoDex plan and frozen world")
        scene = copy.deepcopy(trial_scene)
        if "failed_grasp_original_key" in scene["mesh"]:
            raise ValueError("failed-grasp scene already has a reserved key obstacle")
        scene["mesh"]["failed_grasp_original_key"] = copy.deepcopy(
            scene["mesh"]["target"])
        # Stock AutoDex release uses the selected grasp/pregrasp ramp. Full
        # opening here is appropriate only after a judged miss, not a slip.
        self.executor.release(pickup_plan, open_to_init=True)
        self.executor.reset(pickup_plan, planner, scene)
        return self.before_key_capture()

    def finish_planned_release_exit(
        self, *, release_plan: ReposeReleasePreflight,
        open_hand: Callable[[np.ndarray], None],
        follow_joint_trajectory: Callable[[np.ndarray], None],
        evidence_sink: list[dict] | None = None,
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
        opened = self._verify_release_waypoint(
            up[0], phase="after_release_open", require_open=True)
        if evidence_sink is not None:
            evidence_sink.append({"name": "release_open",
                                  "state": opened.to_record()})
        follow_joint_trajectory(up.copy())
        lifted = self._verify_release_waypoint(
            up[-1], phase="after_release_lift", require_open=True)
        if evidence_sink is not None:
            evidence_sink.append({"name": "post_release_lift",
                                  "state": lifted.to_record()})
        follow_joint_trajectory(retract.copy())
        final = self.verify_state(self.read_state(), phase="after_release_exit")
        if evidence_sink is not None:
            evidence_sink.append({"name": "post_release_retract",
                                  "state": final.to_record()})
        return final

    def finish_planned_release_exit_with_autodex(
        self, *, release_plan: ReposeReleasePreflight,
        evidence_sink: list[dict] | None = None,
    ) -> LiveRobotState:
        """Execute the checked exit through the existing FrankaExecutor.

        The stock public trajectory method accepts arm joints plus a separate
        Inspire controller-unit trajectory. Reuse its own conversion and
        release ramp so the full 13-DOF plan is actually followed; passing
        only the 13-column array would silently leave the hand untouched.
        """
        if (not callable(getattr(self.executor, "release", None)) or
                not callable(getattr(self.executor, "follow_joint_trajectory", None)) or
                not callable(getattr(self.executor, "_convert", None))):
            raise TypeError("planned release needs the stock FrankaExecutor")

        def open_hand(_open_q: np.ndarray) -> None:
            self.executor.release(None, open_to_init=True)

        return self.finish_planned_release_exit(
            release_plan=release_plan, open_hand=open_hand,
            follow_joint_trajectory=self._follow_autodex,
            evidence_sink=evidence_sink)

    def _follow_autodex(self, path: np.ndarray) -> None:
        hand_actions = np.asarray([
            self.executor._convert(row[7:]) for row in path],
            dtype=np.float64)
        self.executor.follow_joint_trajectory(
            path, hand_traj=hand_actions)

    def complete_failed_insertion_return_with_autodex(
        self, *, held_plan: ReposeHeldPreflight,
        release_plan: ReposeReleasePreflight,
        evidence_sink: list[dict] | None = None,
    ) -> LiveRobotState:
        """From withdrawn hold, return key to table and park for recapture.

        This does not perform the preceding guarded axial withdrawal or infer
        where the key actually landed. It follows only the already-audited
        held transfer/descent; release and empty-hand exit use the same stock
        FrankaExecutor as the v8 pipeline.
        """
        if (not isinstance(held_plan, ReposeHeldPreflight) or
                held_plan.status != "sampled_held_path_pass_release_unplanned" or
                not isinstance(held_plan.sampled_held_path_audit, dict) or
                held_plan.sampled_held_path_audit.get("sampled_clear") is not True or
                held_plan.transfer_trajectory is None or
                held_plan.descent_trajectory is None or
                not callable(getattr(self.executor, "follow_joint_trajectory", None)) or
                not callable(getattr(self.executor, "_convert", None))):
            raise ValueError("failed insertion return needs passed held paths and AutoDex executor")
        transfer = np.asarray(held_plan.transfer_trajectory, dtype=float)
        descent = np.asarray(held_plan.descent_trajectory, dtype=float)
        if (transfer.ndim != 2 or descent.ndim != 2 or
                transfer.shape[1] != 13 or descent.shape[1] != 13 or
                len(transfer) < 2 or len(descent) < 2 or
                not np.all(np.isfinite(transfer)) or
                not np.all(np.isfinite(descent)) or
                not np.allclose(transfer[-1], descent[0], atol=1e-4, rtol=0) or
                not np.allclose(transfer[:, 7:], held_plan.held_hand_q,
                                atol=1e-8, rtol=0) or
                not np.allclose(descent[:, 7:], held_plan.held_hand_q,
                                atol=1e-8, rtol=0)):
            raise ValueError("failed insertion held paths are discontinuous")
        if (not isinstance(release_plan, ReposeReleasePreflight) or
                release_plan.status !=
                "nominal_release_exit_path_pass_drop_unobserved" or
                not isinstance(release_plan.release_geometry_audit, dict) or
                release_plan.release_geometry_audit.get("sampled_clear") is not True or
                release_plan.post_release_lift_trajectory is None or
                release_plan.retract_trajectory is None or
                not np.allclose(
                    np.asarray(release_plan.post_release_lift_trajectory)[0, :7],
                    descent[-1, :7], atol=1e-4, rtol=0)):
            raise ValueError("failed insertion return lacks a continuous release/exit plan")
        started = self._verify_return_endpoint(
            transfer[0], phase="withdrawn_held_start")
        if evidence_sink is not None:
            evidence_sink.append({"name": "withdrawn_hold",
                                  "state": started.to_record()})
        self._follow_autodex(transfer.copy())
        transferred = self._verify_return_endpoint(
            transfer[-1], phase="held_transfer_end")
        if evidence_sink is not None:
            evidence_sink.append({"name": "held_transfer",
                                  "state": transferred.to_record()})
        self._follow_autodex(descent.copy())
        descended = self._verify_return_endpoint(
            descent[-1], phase="held_descent_end")
        if evidence_sink is not None:
            evidence_sink.append({"name": "held_descent",
                                  "state": descended.to_record()})
        return self.finish_planned_release_exit_with_autodex(
            release_plan=release_plan, evidence_sink=evidence_sink)

    def _verify_return_endpoint(self, expected: np.ndarray, *, phase: str):
        state = self.read_state()
        self._validate_feedback(state, phase=phase)
        if (np.max(np.abs(state.full_q[:7] - expected[:7])) >
                self.max_arm_error_rad or
                np.max(np.abs(state.full_q[7:] - expected[7:])) >
                self.max_hand_error_rad):
            raise ValueError(f"{phase}: measured robot missed the held-return waypoint")
        return state
