"""Execute a measured held lift, then obtain an observed grasp label.

The action loop supplies one commissioned lift adapter and one camera-side
raw-frame producer. Planning, execution evidence and the paired Qwen lift
checkpoint are the existing demo primitives; no label is inferred from the
planned path or a motor command alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .lift_execution import LiftExecutionLimits, execute_bound_measured_lift
from .labeling import request_annotation
from .live_robot_state import LiveRobotState
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from .recoverable_failure import acknowledged_failure
from .session_policy import SessionDecision


_ASSESS_KEYS = frozenset({
    "max_phase_skew_s", "max_lift_observation_gap_s",
    "max_arm_hand_skew_s", "max_hand_command_error_raw",
    "max_arm_velocity_rad_s",
})
_ASSESS_OPTIONAL = frozenset({"minimum_visual_views"})


@dataclass
class LiveLiftHandlers:
    """Join a bound lift execution to raw multi-view visual assessment.

    ``capture_after_lift`` must acquire the already-open AutoDex camera rig's
    *undistorted* frames and camera-side exposure metadata. It does not
    initialize another camera system or synthesize frame timestamps.
    ``state_for_capture`` reads feedback buffered at that exposure, not a
    later joint sample following Qwen inference.
    """

    live: object
    adapter: object
    read_state: Callable[[], LiveRobotState]
    state_for_capture: Callable[[RawCameraCapture], LiveRobotState]
    capture_after_lift: Callable
    backend: object
    motion_interlock: Callable[[], bool]
    limits: LiftExecutionLimits
    commissioning_record_path: Path
    assessment_parameters: Mapping[str, float]
    label_mode: str = "evidence_fused"
    manual_label_provider: Callable | None = None
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time

    def __post_init__(self) -> None:
        if (not callable(self.read_state) or
                not callable(self.state_for_capture) or
                not callable(self.capture_after_lift) or
                not callable(self.motion_interlock) or
                not callable(self.decision_time) or
                not callable(getattr(self.backend, "infer", None)) or
                not isinstance(self.limits, LiftExecutionLimits) or
                not isinstance(self.assessment_parameters, Mapping)):
            raise TypeError("live lift needs controller, cameras, Qwen and limits")
        missing = _ASSESS_KEYS - self.assessment_parameters.keys()
        extra = (self.assessment_parameters.keys() - _ASSESS_KEYS -
                 _ASSESS_OPTIONAL)
        if missing or extra:
            raise ValueError(
                f"lift checkpoint limits missing {sorted(missing)} or "
                f"unknown {sorted(extra)}")

    def lift_and_observe(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "await_lift_observation"):
            raise ValueError("held lift belongs to another session or decision")
        if self.enable_robot_motion is not True:
            raise PermissionError("live held-lift motion is not enabled")
        attempt = runner.active_attempt
        if attempt is None or attempt.candidate_id != decision.candidate_id:
            raise ValueError("held lift has no selected v8 attempt")
        before = self.read_state()
        failure_path = (Path(runner.output_dir) / "attempts" /
                        attempt.attempt_id / "lift_failure.json")
        try:
            execute_bound_measured_lift(
                runner=runner, adapter=self.adapter, pre_state=before,
                read_post_state=self.read_state, limits=self.limits,
                commissioning_record_path=self.commissioning_record_path,
                motion_interlock=self.motion_interlock,
                enable_robot_motion=self.enable_robot_motion)
        except Exception:
            if not acknowledged_failure(
                    failure_path,
                    schema="precision_insertion_lift_execution_v1",
                    attempt_id=attempt.attempt_id,
                    candidate_id=attempt.candidate_id):
                raise
            stamp = self.decision_time()
            refs = {"lift_failure": str(failure_path)}
            runner.observe_stage("grasp_success", None,
                                 timestamp_s=stamp, evidence_refs=refs)
            runner.record_failure("lift_execution_failed",
                                  timestamp_s=stamp, evidence_refs=refs)
            return
        lift_log = (Path(runner.output_dir) / "attempts" /
                    attempt.attempt_id / "lift_execution.json")
        capture = self.capture_after_lift(self.live, runner, decision)
        if not isinstance(capture, RawCameraCapture):
            raise TypeError("post-lift AutoDex cameras returned no raw capture")
        camera_ids = set(self.live.cameras.launch.camera_ids)
        if (not 2 <= len(capture.images_bgr) or
                not set(capture.images_bgr) <= camera_ids):
            raise ValueError("post-lift frames differ from session cameras")
        evidence_dir = (Path(runner.output_dir) / "attempts" /
                        attempt.attempt_id / "raw_after_lift" /
                        capture.capture_id)
        write_raw_camera_capture(capture, evidence_dir, phase="after_lift")
        verify_raw_camera_capture(evidence_dir, phase="after_lift")
        exposure_state = self.state_for_capture(capture)
        if not isinstance(exposure_state, LiveRobotState):
            raise TypeError("post-lift camera exposure lacks measured robot state")
        assessment_index = getattr(runner, "_lift_assessment_index", None)
        checkpoint = runner.prepare_raw_lift_label(
            after_raw_evidence_dir=evidence_dir,
            joint_sample=exposure_state, backend=self.backend,
            lift_execution_log_path=lift_log,
            decision_timestamp_s=self.decision_time(),
            label_mode=self.label_mode,
            manual_annotation=request_annotation(
                mode=self.label_mode, provider=self.manual_label_provider,
                stage="grasp_success",
                evidence_dirs=(
                    (Path(runner._key_evidence_dir), evidence_dir)
                    if self.label_mode == "manual" else (evidence_dir,)),
                attempt_id=attempt.attempt_id,
                candidate_id=attempt.candidate_id),
            **dict(self.assessment_parameters))
        if checkpoint.grasp_success is None:
            if type(assessment_index) is not int:
                raise ValueError("unknown lift verdict lacks a saved assessment index")
            report = (Path(runner.output_dir) / "attempts" /
                      attempt.attempt_id / "lift_assessments" /
                      f"{assessment_index:03d}" / "report.json")
            if not report.is_file():
                raise FileNotFoundError("unknown lift verdict lacks its VLM report")
            runner.observe_stage(
                "grasp_success", None, timestamp_s=self.decision_time(),
                evidence_refs={"vlm_observation": str(report),
                               "lift_execution": str(lift_log)})

    def handlers(self) -> dict[str, ActionHandler]:
        return {"await_lift_observation": self.lift_and_observe}
