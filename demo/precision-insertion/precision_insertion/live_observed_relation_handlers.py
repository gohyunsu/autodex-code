"""Use a visible post-lift key to replace the nominal BODex held relation.

This is an optional AutoDex FoundPose route. It preserves the raw Qwen lift
label and measures a *new* key pose only when the key is sufficiently visible.
An occluded or inconsistent key fails the observation; it is not promoted to
a nominal key-in-hand relation or a claimed transfer success.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .key_perception import (
    admit_postlift_key_capture, verify_key_capture_artifacts,
    write_key_capture_artifacts,
)
from .live_capture import collect_key_capture
from .live_robot_state import LiveRobotState
from .session_policy import SessionDecision


_CAPTURE_KEYS = frozenset({
    "maximum_multiview_center_error_mm",
    "maximum_multiview_angle_error_deg",
    "maximum_candidate_prior_center_error_mm",
    "maximum_candidate_prior_angle_error_deg",
    "maximum_prior_time_skew_s", "minimum_refinement_iou",
    "capture_timeout_s", "image_write_timeout_s",
})
_CAPTURE_OPTIONAL = frozenset({
    "silhouette_iterations", "silhouette_loss_threshold",
})
_PLAN_KEYS = frozenset({
    "max_state_skew_s", "max_arm_hand_skew_s",
    "max_hand_command_error_raw", "max_arm_velocity_rad_s",
    "max_grasp_translation_drift_m", "max_grasp_rotation_drift_deg",
    "axial_waypoint_step_m",
})


@dataclass
class LiveObservedHeldRelationHandlers:
    """One held-relation action from new image capture through v8 replan."""

    live: object
    planner: object
    read_state: Callable[[], LiveRobotState]
    state_for_observation: Callable
    key_prompt: str
    path_limits: object
    capture_parameters: Mapping[str, object]
    parameters: Mapping[str, object]
    capture_id_factory: Callable[[], str] = lambda: f"held_{uuid4().hex}"

    def __post_init__(self) -> None:
        if (not callable(self.read_state) or
                not callable(self.state_for_observation) or
                not callable(self.capture_id_factory) or
                not isinstance(self.key_prompt, str) or
                not self.key_prompt.strip()):
            raise TypeError("observed held relation needs cameras, feedback and key prompt")
        for label, values, required, optional in (
                ("capture", self.capture_parameters, _CAPTURE_KEYS,
                 _CAPTURE_OPTIONAL),
                ("planning", self.parameters, _PLAN_KEYS, frozenset())):
            if not isinstance(values, Mapping):
                raise TypeError(f"held-relation {label} limits must be a mapping")
            missing = required - values.keys()
            unknown = values.keys() - required - optional
            if missing or unknown:
                raise ValueError(
                    f"held-relation {label} missing {sorted(missing)} or "
                    f"unknown {sorted(unknown)}")

    def observe_and_plan(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "held_relation_evidence_required" or
                not decision.candidate_id):
            raise ValueError("observed held relation belongs to another trial")
        launch = self.live.cameras.launch
        cameras = self.live.cameras
        initial = self.read_state()
        if not isinstance(initial, LiveRobotState):
            raise TypeError("held-key prior needs measured robot feedback")
        prior = runner.postlift_candidate_pose_prior(
            planner=self.planner, joint_sample=initial,
            max_arm_hand_skew_s=self.parameters["max_arm_hand_skew_s"],
            max_hand_command_error_raw=(
                self.parameters["max_hand_command_error_raw"]),
            max_arm_velocity_rad_s=self.parameters["max_arm_velocity_rad_s"])
        capture_id = self.capture_id_factory()
        options = dict(self.capture_parameters)
        timeout = options.pop("capture_timeout_s")
        write_timeout = options.pop("image_write_timeout_s")
        capture = collect_key_capture(
            init_orchestrator=cameras.init_orchestrator,
            key_object=runner.mode.key_object, capture_id=capture_id,
            key_prompt=self.key_prompt, capture_root=launch.capture_root,
            calibrated_camera_ids=set(launch.camera_ids),
            acquisition_metadata_for_request=(
                cameras.acquisition_metadata_for_request),
            timeout_s=timeout, image_write_timeout_s=write_timeout)
        observation = admit_postlift_key_capture(
            capture=capture,
            init_orchestrator=cameras.init_orchestrator,
            mode=runner.mode, shared_root=runner.shared_root,
            calibration=runner.calibration,
            calibrated_camera_ids=set(launch.camera_ids),
            view_limits=launch.view_limits,
            candidate_pose_prior_world=prior,
            measured_wrist_timestamp_s=initial.sample_timestamp_s,
            **options)
        evidence_dir = (Path(runner.output_dir) / "held_key_captures" /
                        capture_id)
        write_key_capture_artifacts(capture, observation, evidence_dir)
        verify_key_capture_artifacts(evidence_dir)
        measured = self.state_for_observation(observation)
        if not isinstance(measured, LiveRobotState):
            raise TypeError("held key exposure needs measured robot feedback")
        result = runner.prepare_postlift_transfer(
            planner=self.planner, key_observation=observation,
            key_evidence_dir=evidence_dir, joint_sample=measured,
            limits=self.path_limits, **dict(self.parameters))
        if result.status != "sampled_postlift_preflight_pass":
            raise ValueError(
                f"observed held relation has no passing transfer: {result.status}")

    def handlers(self) -> dict[str, ActionHandler]:
        return {"held_relation_evidence_required": self.observe_and_plan}
