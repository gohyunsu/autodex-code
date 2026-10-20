"""Construct the existing AutoDex FR3/Inspire runtime for the demo.

The robot-PC integration still supplies camera timing, contact controllers,
and per-stage callbacks. This module owns the demo's completion-checked
subclass of the stock Franka executor,
FR3 cuRobo planner, measured feedback reader and clear-view gate. Importing
it does not connect to or move the robot. ``open_franka_core`` constructs a
command-capable executor: its constructors may send FR3 setup commands and
Inspire angle writes, so it belongs only to the motion-authorized stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

from .clear_view import ClearViewForPerception
from .feedback_buffer import ExposureStateBuffer, RobotFeedbackSampler
from .live_robot_state import LiveRobotState, read_live_franka_inspire_state
from .mount_geometry import link7_to_hand


# Match stock AutoDex FR3's near-object and held-object speed scale.
PRECISION_OBJECT_SAFETY_SPEED_SCALE = 0.4


@dataclass(frozen=True)
class FrankaFeedbackSettings:
    max_arm_hand_skew_s: float
    max_sample_age_s: float
    max_hand_command_error_raw: float
    max_arm_velocity_rad_s: float


@dataclass(frozen=True)
class FrankaSamplingSettings:
    max_samples: int
    max_bracket_span_s: float
    max_key_state_skew_s: float
    max_arm_hold_drift_rad: float
    max_hand_hold_drift_raw: float
    max_sampling_hand_command_error_raw: float
    max_sampling_arm_velocity_rad_s: float
    period_s: float
    warmup_timeout_s: float


@dataclass
class FrankaRuntimeCore:
    executor: object
    planner: object
    read_state: Callable[[], LiveRobotState]
    clear_view: ClearViewForPerception
    feedback: FrankaFeedbackSettings
    sampler: RobotFeedbackSampler | None = None
    _closed: bool = False

    def start_feedback_sampler(
        self, settings: FrankaSamplingSettings,
    ) -> RobotFeedbackSampler:
        """Keep exposure-time feedback alive through intervening arm motion.

        The sampling limits admit moving states into the buffer. Lookup for a
        key or VLM image still uses the tighter stationary limits and rejects
        any moving sample that overlaps that image's exposure interval.
        """
        if (self._closed or self.sampler is not None or
                not isinstance(settings, FrankaSamplingSettings)):
            raise ValueError("FR3 feedback sampler is closed or already running")
        buffer = ExposureStateBuffer(
            max_samples=settings.max_samples,
            max_bracket_span_s=settings.max_bracket_span_s,
            max_key_state_skew_s=settings.max_key_state_skew_s,
            max_arm_hold_drift_rad=settings.max_arm_hold_drift_rad,
            max_hand_hold_drift_raw=settings.max_hand_hold_drift_raw,
            max_arm_hand_skew_s=self.feedback.max_arm_hand_skew_s,
            max_hand_command_error_raw=self.feedback.max_hand_command_error_raw,
            max_arm_velocity_rad_s=self.feedback.max_arm_velocity_rad_s,
            max_sampling_hand_command_error_raw=(
                settings.max_sampling_hand_command_error_raw),
            max_sampling_arm_velocity_rad_s=(
                settings.max_sampling_arm_velocity_rad_s))

        def sample_state() -> LiveRobotState:
            return read_live_franka_inspire_state(
                arm=self.executor.arm, hand=self.executor.hand,
                max_arm_hand_skew_s=self.feedback.max_arm_hand_skew_s,
                max_sample_age_s=self.feedback.max_sample_age_s,
                max_hand_command_error_raw=(
                    settings.max_sampling_hand_command_error_raw),
                max_arm_velocity_rad_s=(
                    settings.max_sampling_arm_velocity_rad_s))

        sampler = RobotFeedbackSampler(
            buffer, sample_state, period_s=settings.period_s)
        sampler.start()
        try:
            sampler.wait_until_ready(timeout_s=settings.warmup_timeout_s)
        except BaseException:
            sampler.close()
            raise
        self.sampler = sampler
        return sampler

    def close(self) -> None:
        """Release the stock Franka daemon/controllers once, even on failure."""
        if not self._closed:
            self._closed = True
            try:
                if self.sampler is not None:
                    self.sampler.close()
            finally:
                self.executor.shutdown()


def open_franka_core(
    feedback: FrankaFeedbackSettings, *,
    executor_factory: Callable[[], object] | None = None,
    planner_factory: Callable[[], object] | None = None,
) -> FrankaRuntimeCore:
    """Construct the motion-capable FR3/Inspire runtime after motion opt-in.

    The two optional factories allow tests to exercise resource ownership
    without a robot. Real callers should use the AutoDex constructors below.
    The stock executor constructor calls FR3 stop/recovery/collision setup;
    the Inspire constructor writes angle/speed/force registers. This is not a
    connection-only or feedback-only probe. ``run_live.py`` invokes its
    integration factory only after the explicit robot-motion opt-in; that
    factory can return ``clear_view`` and ``close`` from this core.
    """
    if not isinstance(feedback, FrankaFeedbackSettings):
        raise TypeError("FR3 runtime needs explicit feedback settings")
    mount_path = os.environ.get("PRECISION_FR3_INSPIRE_URDF")
    mount_urdf = (Path(mount_path).expanduser().resolve()
                  if mount_path else None)
    if planner_factory is None:
        from autodex.planner import GraspPlanner

        planner_factory = lambda: GraspPlanner(
            hand="fr3_inspire", robot_urdf_path=(
                None if mount_urdf is None else str(mount_urdf)))
    # cuRobo loads CUDA during planner construction. Do that before opening a
    # robot connection, so a missing GPU never touches the Franka daemon.
    planner = planner_factory()
    if (getattr(planner, "_n_arm", None) != 7 or
            getattr(planner, "_hand", None) != "fr3_inspire"):
        raise TypeError("FR3 runtime needs GraspPlanner(hand='fr3_inspire')")
    if executor_factory is None:
        from .franka_completion_checked import completion_checked_franka_class

        executor_factory = lambda: completion_checked_franka_class()(
            hand_name="inspire", link7_to_hand=(
                None if mount_urdf is None else link7_to_hand(mount_urdf)),
            object_safety_speed_scale=PRECISION_OBJECT_SAFETY_SPEED_SCALE)
    executor = executor_factory()
    try:
        if (getattr(executor, "arm_dof", None) != 7 or
                getattr(executor, "hand_name", None) not in
                {"inspire", "inspire_left"} or
                not callable(getattr(executor, "shutdown", None)) or
                not callable(getattr(executor, "set_speed_profile_planner", None))):
            raise TypeError("FR3 runtime needs the stock Franka/Inspire executor")
        # Match run_auto.py: generic return/recovery trajectories use this
        # planner for the stock live object-proximity speed profile.
        executor.set_speed_profile_planner(planner)
        def read_state() -> LiveRobotState:
            return read_live_franka_inspire_state(
                arm=executor.arm, hand=executor.hand,
                max_arm_hand_skew_s=feedback.max_arm_hand_skew_s,
                max_sample_age_s=feedback.max_sample_age_s,
                max_hand_command_error_raw=feedback.max_hand_command_error_raw,
                max_arm_velocity_rad_s=feedback.max_arm_velocity_rad_s)

        clear_view = ClearViewForPerception(
            executor=executor, read_state=read_state,
            max_arm_hand_skew_s=feedback.max_arm_hand_skew_s,
            max_hand_command_error_raw=feedback.max_hand_command_error_raw,
            max_arm_velocity_rad_s=feedback.max_arm_velocity_rad_s,
            max_sample_age_s=feedback.max_sample_age_s)
        return FrankaRuntimeCore(
            executor, planner, read_state, clear_view, feedback)
    except BaseException:
        # FrankaExecutor owns a daemon connection and a proximity worker even
        # if planner construction fails. Never leak them on a partial startup.
        if callable(getattr(executor, "shutdown", None)):
            executor.shutdown()
        raise
