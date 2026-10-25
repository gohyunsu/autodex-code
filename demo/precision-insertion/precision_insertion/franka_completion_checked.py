"""Demo-local completion check around the AutoDex-derived FR3 follower.

The stock follower may exit on a stall/timeout, ramp the hand to its final
configuration, and make a blocking endpoint ``arm.move``. For precision
pickup that combination can close the fingers or cross an obstacle after the
planned stream was not followed. This subclass reuses the stock streaming
algorithm but defers its final hand ramp and checks measured joint completion
before allowing the ramp. The stock blocking landing is retained for an
empty-hand clear-view return, where exact arrival matters after release.

This is *not* a robot-side watchdog or a force servo. The installed ParaDex
daemon has its own command-expiry watchdog; this mixin checks completion and
acknowledges a stopped stream through that daemon. It does not claim that a
key physically reached the socket depth.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache

import numpy as np

from autodex.utils.robot_config import FR3_CLEAR_VIEW

_COMPOSED_TYPE = None
STOCK_COMPLETION_TOLERANCE_RAD = 0.04


class CompletionCheckedFrankaMixin:
    """Preserve stock path following but fail closed at its terminal boundary.

    The stock class is imported only when the FR3 runtime opens, after its
    planner initializes CUDA. Importing this module for CPU-only tests or
    static configuration does not initialize cuRobo or connect to hardware.
    """

    # Match the stock follower's L2 endpoint tolerance for streamed paths.
    completion_max_path_error_rad = STOCK_COMPLETION_TOLERANCE_RAD
    stop_max_joint_velocity_rad_s = 0.05
    stop_wait_s = 1.0

    def _follow(
        self, arm_traj, hand_traj=None, speed=None,
        abort_on_contact=False, stop_wrench_z=None,
        slowdown_wrist_target=None, slowdown_wrist_references=None,
        proximity_hand_traj=None,
    ):
        path = np.atleast_2d(np.asarray(arm_traj, dtype=np.float64))
        if (path.ndim != 2 or path.shape[1] < 7 or len(path) == 0 or
                not np.all(np.isfinite(path))):
            raise ValueError("completion-checked FR3 path needs finite arm joints")
        if getattr(self, "_precision_follow_active", False):
            raise RuntimeError("concurrent FR3 follow is not supported")
        original_land_tol = self.land_tol
        original_follow_tol = self.follow_tol
        had_own_ramp = "_ramp_hand" in self.__dict__
        original_ramp = self._ramp_hand
        saved_own_ramp = self.__dict__.get("_ramp_hand")
        deferred = []

        def defer_ramp(target):
            deferred.append(np.asarray(target, dtype=np.float64).copy())

        self._precision_follow_active = True
        clear_return = (not bool(getattr(self, "_holding_object", False)) and
                        np.max(np.abs(path[-1, :7] - FR3_CLEAR_VIEW)) <= 1e-3)
        # Stock landing is useful after the object has been released: the
        # last clear-view residual must not strand an otherwise completed
        # insertion. Keep it disabled while carrying a key or near contact.
        self.land_tol = original_land_tol if clear_return else math.inf
        # Use the stock L2 convergence tolerance for the streaming segment.
        self.follow_tol = min(original_follow_tol,
                              self.completion_max_path_error_rad)
        self._ramp_hand = defer_ramp
        try:
            result = super()._follow(
                path, hand_traj, speed=speed,
                abort_on_contact=(abort_on_contact or
                                  bool(getattr(self, "_holding_object", False))),
                stop_wrench_z=stop_wrench_z,
                slowdown_wrist_target=slowdown_wrist_target,
                slowdown_wrist_references=slowdown_wrist_references,
                proximity_hand_traj=proximity_hand_traj)
        finally:
            self.land_tol = original_land_tol
            self.follow_tol = original_follow_tol
            if had_own_ramp:
                self._ramp_hand = saved_own_ramp
            else:
                del self._ramp_hand
            self._precision_follow_active = False

        stop = self.stop_and_acknowledge()
        if stop.get("acknowledged") is not True:
            raise RuntimeError("FR3 stream stop was not acknowledged")
        # A deliberate force-stop is *not* endpoint completion. Its caller
        # receives the stock contact flag and must handle the partial stroke.
        if stop_wrench_z is not None and self._last_stop_on_contact:
            return result
        state = self.arm.get_data()
        if (not isinstance(state, dict) or
                np.asarray(state.get("qpos", [])).shape != (7,) or
                np.asarray(state.get("qvel", [])).shape != (7,)):
            raise RuntimeError("FR3 has no terminal joint feedback")
        measured = np.asarray(state["qpos"], dtype=np.float64)
        velocity = np.asarray(state["qvel"], dtype=np.float64)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(velocity)):
            raise RuntimeError("FR3 terminal joint feedback is not finite")
        endpoint_error = float(np.linalg.norm(measured - path[-1, :7]))
        max_velocity = float(np.max(np.abs(velocity)))
        arm_error = bool(self.arm.is_error())
        if (arm_error or
                endpoint_error > self.completion_max_path_error_rad or
                max_velocity > self.stop_max_joint_velocity_rad_s):
            raise RuntimeError(
                "FR3 stream did not finish at a measured stationary endpoint "
                f"(joint_error_l2_rad={endpoint_error:.6f}, "
                f"limit_l2_rad={self.completion_max_path_error_rad:.6f}, "
                f"max_joint_velocity_rad_s={max_velocity:.6f}, "
                f"velocity_limit_rad_s={self.stop_max_joint_velocity_rad_s:.6f}, "
                f"arm_error={arm_error})")
        if hand_traj is not None:
            if len(deferred) != 1:
                raise RuntimeError("stock follower did not defer one final hand ramp")
            original_ramp(deferred[0])
        elif deferred:
            raise RuntimeError("stock follower unexpectedly ramped the hand")
        return result

    def stop_and_acknowledge(self) -> dict:
        """Stop the stream and acknowledge only a measured stationary arm."""
        response = self.arm.stop_streaming()
        if not isinstance(response, dict) or response.get("type") != "success":
            return {"acknowledged": False, "reason": "stream_stop_not_accepted"}
        deadline = time.monotonic() + self.stop_wait_s
        while True:
            state = self.arm.get_data()
            velocity = (None if not isinstance(state, dict) else
                        np.asarray(state.get("qvel", []), dtype=np.float64))
            if (velocity is not None and velocity.shape == (7,) and
                    np.all(np.isfinite(velocity)) and
                    np.max(np.abs(velocity)) <=
                    self.stop_max_joint_velocity_rad_s):
                return {"acknowledged": True,
                        "max_joint_velocity_rad_s":
                        float(np.max(np.abs(velocity)))}
            if time.monotonic() >= deadline:
                return {"acknowledged": False,
                        "reason": "stationary_feedback_not_observed"}
            time.sleep(0.01)


@lru_cache(maxsize=1)
def completion_checked_franka_class():
    """Construct the demo subclass only after planner/CUDA initialization."""
    global _COMPOSED_TYPE
    from src.execution.franka_executor import FrankaExecutor

    class CompletionCheckedFrankaExecutor(
            CompletionCheckedFrankaMixin, FrankaExecutor):
        pass

    _COMPOSED_TYPE = CompletionCheckedFrankaExecutor
    return CompletionCheckedFrankaExecutor


def is_demo_completion_checked_executor(executor) -> bool:
    """Recognize only the factory's exact class, without importing cuRobo."""
    return _COMPOSED_TYPE is not None and type(executor) is _COMPOSED_TYPE
