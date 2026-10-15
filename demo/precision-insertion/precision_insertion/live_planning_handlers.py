"""Concrete non-motion handlers for a frozen session's key-planning phase.

The AutoDex camera session, FoundPose capture and v8 planner already have
owners. This adapter only passes the *same* live session and exposure-time
robot state between them; it does not implement another grasp selector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .checkpoint_camera import capture_snapshot_checkpoint_frames
from .live_key_trial import PreparedLiveKeyTrial, prepare_next_live_key
from .session_policy import SessionDecision
from .prompt_contract import reject_incorrect_object_color
from .timing_profile import timed


def _new_capture_id() -> str:
    return f"key_{uuid4().hex}"


_CONTINUATION_PARAMETERS = frozenset({
    "max_arm_hand_skew_s", "max_hand_command_error_raw",
    "max_arm_velocity_rad_s", "max_key_state_skew_s",
    "max_pose_error_deg", "axial_waypoint_step_m",
})
_OWNED_CAPTURE_PARAMETERS = frozenset({
    "runner", "init_orchestrator", "acquisition_metadata_for_request",
    "state_for_observation", "capture_root", "key_evidence_dir",
    "capture_id", "calibrated_camera_ids", "intrinsics_full",
    "extrinsics_full", "image_hw", "key_prompt", "view_limits",
    "planner", "limits",
})


@dataclass
class LiveKeyPlanningHandlers:
    """Bind fresh/same-capture v8 planning to one open camera session.

    ``key_parameters`` are the keyword-only thresholds accepted by
    ``prepare_next_live_key`` other than arguments this adapter supplies.
    The caller must keep its measured-feedback sampler running across each
    camera exposure and supply that sampler's ``state_for_observation``.
    """

    live: object
    planner: object
    state_for_observation: Callable
    limits: object
    key_prompt: str
    key_parameters: Mapping[str, object]
    capture_id_factory: Callable[[], str] = _new_capture_id
    latest: PreparedLiveKeyTrial | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if (not callable(self.state_for_observation) or
                not callable(self.capture_id_factory) or
                not isinstance(self.key_prompt, str) or
                not self.key_prompt.strip() or
                not isinstance(self.key_parameters, Mapping)):
            raise TypeError("key planning needs feedback, IDs, prompt and parameters")
        reject_incorrect_object_color(self.key_prompt)
        if not callable(getattr(getattr(self.live, "clear_view", None),
                                "before_key_capture", None)) or not callable(
                getattr(self.live.clear_view, "verify_state", None)):
            raise TypeError("key planning needs a measured clear-view gate")
        parameters = dict(self.key_parameters)
        missing = _CONTINUATION_PARAMETERS - parameters.keys()
        reserved = _OWNED_CAPTURE_PARAMETERS & parameters.keys()
        if missing or reserved:
            raise ValueError(
                f"key parameters missing {sorted(missing)} or override "
                f"session-owned {sorted(reserved)}")
        self.key_parameters = parameters

    def _require_decision(self, runner, decision: SessionDecision,
                          actions: frozenset[str]) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action not in actions or
                runner.current_decision() != decision):
            raise ValueError("key planning handler has another session or decision")

    def capture_and_preflight(self, runner, decision: SessionDecision) -> None:
        """Acquire a *new* key frame and plan against the frozen socket."""
        self._require_decision(
            runner, decision,
            frozenset({"capture_fresh_key", "reobserve_key_and_preflight"}))
        with timed("key.clear_view", phase="key", kind="motion"):
            self.live.clear_view.before_key_capture()
        capture_id = self.capture_id_factory()
        launch = self.live.cameras.launch

        def measured_clear_exposure(observation):
            state = self.state_for_observation(observation)
            return self.live.clear_view.verify_state(
                state, phase="key_exposure", require_fresh=False)

        prepared = prepare_next_live_key(
            runner=runner,
            init_orchestrator=self.live.cameras.init_orchestrator,
            acquisition_metadata_for_request=(
                self.live.cameras.acquisition_metadata_for_request),
            state_for_observation=measured_clear_exposure,
            capture_root=launch.capture_root,
            key_evidence_dir=(runner.output_dir / "key_captures" / capture_id),
            capture_id=capture_id,
            calibrated_camera_ids=set(launch.camera_ids),
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            image_hw=launch.image_hw,
            key_prompt=self.key_prompt, view_limits=launch.view_limits,
            planner=self.planner, limits=self.limits,
            capture_initial_visual=lambda: capture_snapshot_checkpoint_frames(
                self.live, phase="initial_tabletop",
                timeout_s=float(self.key_parameters["capture_timeout_s"])),
            **dict(self.key_parameters))
        self.latest = prepared

    def continue_same_capture(self, runner, decision: SessionDecision) -> None:
        """Exhaust a pilot prefix without incorrectly reacquiring the key."""
        self._require_decision(
            runner, decision, frozenset({"continue_candidate_preflight"}))
        if self.latest is None:
            raise ValueError("same-capture continuation has no saved key/state")
        previous = self.latest
        parameters = self.key_parameters
        state = previous.state
        options = parameters.get("planning_options")
        with timed("key.candidate_preflight", phase="planning", kind="planning",
                   continued=True):
            result = runner.preflight_next_key(
                planner=self.planner,
                key_observation=previous.observation,
                key_evidence_dir=previous.key_evidence_dir,
                live_start_q=state.full_q.copy(),
                start_q_acquisition_timestamp_s=state.sample_timestamp_s,
                measured_start_state=state,
                measured_state_limits={
                    key: parameters[key] for key in (
                        "max_arm_hand_skew_s", "max_hand_command_error_raw",
                        "max_arm_velocity_rad_s")},
                max_key_state_skew_s=parameters["max_key_state_skew_s"],
                limits=self.limits,
                max_pose_error_deg=parameters["max_pose_error_deg"],
                axial_waypoint_step_m=parameters["axial_waypoint_step_m"],
                max_candidate_attempts=None,
                **({} if options is None else dict(options)))
        self.latest = PreparedLiveKeyTrial(
            previous.observation, state, result, previous.key_evidence_dir)

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "capture_fresh_key": self.capture_and_preflight,
            "reobserve_key_and_preflight": self.capture_and_preflight,
            "continue_candidate_preflight": self.continue_same_capture,
        }
