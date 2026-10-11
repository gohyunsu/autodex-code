"""Construct the existing AutoDex FR3/Inspire runtime for the demo.

The robot-PC integration still supplies camera timing, contact controllers,
and per-stage callbacks. This module owns only the stock Franka executor,
FR3 cuRobo planner, measured feedback reader and clear-view gate. Importing
it does not connect to or move the robot; ``open_franka_core`` does connect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .clear_view import ClearViewForPerception
from .live_robot_state import LiveRobotState, read_live_franka_inspire_state


@dataclass(frozen=True)
class FrankaFeedbackSettings:
    max_arm_hand_skew_s: float
    max_sample_age_s: float
    max_hand_command_error_raw: float
    max_arm_update_wait_s: float
    max_arm_velocity_rad_s: float


@dataclass
class FrankaRuntimeCore:
    executor: object
    planner: object
    read_state: Callable[[], LiveRobotState]
    clear_view: ClearViewForPerception
    _closed: bool = False

    def close(self) -> None:
        """Release the stock Franka daemon/controllers once, even on failure."""
        if not self._closed:
            self._closed = True
            self.executor.shutdown()


def open_franka_core(
    feedback: FrankaFeedbackSettings, *,
    executor_factory: Callable[[], object] | None = None,
    planner_factory: Callable[[], object] | None = None,
) -> FrankaRuntimeCore:
    """Connect to FR3/Inspire without commanding home or a grasp.

    The two optional factories allow tests to exercise resource ownership
    without a robot. Real callers should use the AutoDex constructors below.
    ``run_live.py`` invokes its integration factory only after the explicit
    robot-motion opt-in; that factory can call this function and return the
    resulting ``clear_view`` and ``close`` in ``LiveRuntimeIntegration``.
    """
    if not isinstance(feedback, FrankaFeedbackSettings):
        raise TypeError("FR3 runtime needs explicit feedback settings")
    if executor_factory is None:
        from src.execution.franka_executor import FrankaExecutor

        executor_factory = lambda: FrankaExecutor(hand_name="inspire")
    if planner_factory is None:
        from autodex.planner import GraspPlanner

        planner_factory = lambda: GraspPlanner(hand="fr3_inspire")
    executor = executor_factory()
    try:
        if (getattr(executor, "arm_dof", None) != 7 or
                getattr(executor, "hand_name", None) not in
                {"inspire", "inspire_left"} or
                not callable(getattr(executor, "shutdown", None))):
            raise TypeError("FR3 runtime needs the stock Franka/Inspire executor")
        planner = planner_factory()
        if (getattr(planner, "_n_arm", None) != 7 or
                getattr(planner, "_hand", None) != "fr3_inspire"):
            raise TypeError("FR3 runtime needs GraspPlanner(hand='fr3_inspire')")

        def read_state() -> LiveRobotState:
            return read_live_franka_inspire_state(
                arm=executor.arm, hand=executor.hand,
                max_arm_hand_skew_s=feedback.max_arm_hand_skew_s,
                max_sample_age_s=feedback.max_sample_age_s,
                max_hand_command_error_raw=feedback.max_hand_command_error_raw,
                max_arm_update_wait_s=feedback.max_arm_update_wait_s,
                max_arm_velocity_rad_s=feedback.max_arm_velocity_rad_s)

        clear_view = ClearViewForPerception(
            executor=executor, read_state=read_state,
            max_arm_hand_skew_s=feedback.max_arm_hand_skew_s,
            max_hand_command_error_raw=feedback.max_hand_command_error_raw,
            max_arm_velocity_rad_s=feedback.max_arm_velocity_rad_s,
            max_sample_age_s=feedback.max_sample_age_s)
        return FrankaRuntimeCore(executor, planner, read_state, clear_view)
    except BaseException:
        # FrankaExecutor owns a daemon connection and a proximity worker even
        # if planner construction fails. Never leak them on a partial startup.
        if callable(getattr(executor, "shutdown", None)):
            executor.shutdown()
        raise
