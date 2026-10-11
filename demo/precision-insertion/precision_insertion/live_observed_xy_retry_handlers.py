"""Live visible-key square retry, using the existing AutoDex camera session.

The held-key capture producer supplies one admitted same-request FoundPose
observation and its raw frames. Planning, transfer, arrival, guarded contact
and final task labels are delegated to the existing demo boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Mapping
from uuid import uuid4

import cv2
import numpy as np
from PIL import Image

from .action_loop import ActionHandler
from .bounded_postlift import BoundedPostLiftPreflight
from .guarded_insertion_execution import execute_bound_guarded_insertion
from .geometry import validate_se3
from .key_perception import (
    KeyPoseObservation, admit_held_key_capture,
    verify_key_capture_artifacts, write_key_capture_artifacts,
)
from .live_capture import collect_key_capture
from .live_depth import prepare_live_depth_evidence
from .labeling import request_annotation
from .live_robot_state import LiveRobotState
from .observed_retry_transfer_execution import (
    execute_bound_observed_retry_transfer,
)
from .observer import LabeledFrame
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from .session_policy import SessionDecision


@dataclass(frozen=True)
class HeldKeyRetryCapture:
    observation: KeyPoseObservation
    evidence_dir: Path
    raw: RawCameraCapture
    state: LiveRobotState


class HeldKeyRetryUnobservable(ValueError):
    """A captured withdrawn key could not yield an admitted multiview pose."""

    def __init__(self, reason: str, raw_dir: Path | None):
        super().__init__(reason)
        self.raw_dir = raw_dir


@dataclass
class LiveObservedXYRetryHandlers:
    """Complete the square retry from guarded withdrawal through new label.

    Robot-side watchdog, contact servo and camera acquisition remain external
    producers. `capture_withdrawn_key` must return a same-request FoundPose
    capture and exposure-aligned measured state; the runner checks this again.
    """

    live: object
    planner: object
    backend: object
    read_state: Callable[[], LiveRobotState]
    state_for_capture: Callable[[RawCameraCapture], LiveRobotState]
    withdraw_guarded: Callable
    capture_arrival: Callable
    capture_final: Callable
    transfer_adapter: object
    insertion_adapter: object
    motion_interlock: Callable[[], bool]
    retry_limits: object
    transfer_limits: object
    insertion_limits: object
    contact_limits: object
    transfer_commissioning_path: Path
    insertion_watchdog_path: Path
    insertion_contact_path: Path
    arrival_parameters: Mapping[str, object]
    handoff_parameters: Mapping[str, object]
    insertion_parameters: Mapping[str, object]
    final_parameters: Mapping[str, object]
    capture_withdrawn_key: Callable | None = None
    key_prompt: str | None = None
    key_capture_parameters: Mapping[str, object] = field(default_factory=dict)
    depth_limits: object | None = None
    depth_capture_parameters: Mapping[str, float] = field(default_factory=dict)
    depth_commissioning_path: Path | None = None
    label_mode: str = "evidence_fused"
    manual_label_provider: Callable | None = None
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time
    _pending: dict | None = field(default=None, init=False)
    _withdrawn: tuple[str, LiveRobotState, Path] | None = field(
        default=None, init=False)

    def withdrawn_for_return(self):
        return self._withdrawn

    def _capture_held_key(self, runner) -> HeldKeyRetryCapture:
        """Reuse AutoDex SAM/FoundPose on one stationary withdrawn exposure."""
        if (not isinstance(self.key_prompt, str) or
                not self.key_prompt.strip() or
                not isinstance(self.key_capture_parameters, Mapping) or
                runner._postlift_preflight is None or
                runner._postlift_preflight.status !=
                "sampled_postlift_preflight_pass"):
            raise ValueError("square retry needs a key prompt and observed held pose")
        launch = self.live.cameras.launch
        cameras = self.live.cameras
        options = dict(self.key_capture_parameters)
        timeout = options.pop("capture_timeout_s")
        write_timeout = options.pop("image_write_timeout_s")
        capture_id = f"retry_held_{uuid4().hex}"
        try:
            capture = collect_key_capture(
                init_orchestrator=cameras.init_orchestrator,
                key_object=runner.mode.key_object, capture_id=capture_id,
                key_prompt=self.key_prompt, capture_root=launch.capture_root,
                calibrated_camera_ids=set(launch.camera_ids),
                acquisition_metadata_for_request=(
                    cameras.acquisition_metadata_for_request),
                timeout_s=timeout, image_write_timeout_s=write_timeout)
        except ValueError as exc:
            # This one producer verdict means no camera yielded a SAM/pose
            # payload. Treat it as a visual abstention, not permission to use
            # the old key/hand pose. Other malformed or stale captures remain
            # hard errors requiring operator review.
            if str(exc) != "FoundPose has no calibrated camera payloads":
                raise
            raise HeldKeyRetryUnobservable(str(exc), None) from exc
        raw = RawCameraCapture(
            capture_id=capture.capture_id,
            request_id=capture.request_id,
            images_bgr=capture.images_bgr,
            frame_ids={camera: row["frame_id"]
                       for camera, row in capture.frame_evidence.items()},
            acquisition_metadata={
                "request_id": capture.request_id,
                "source": "camera_acquisition",
                "frames": capture.frame_evidence})
        raw_dir = (Path(runner._attempt_dir) / "raw_captures" /
                   "retry_held" / capture_id)
        write_raw_camera_capture(raw, raw_dir, phase="preinsert")
        verify_raw_camera_capture(raw_dir, phase="preinsert")
        measured = self.state_for_capture(raw)
        if not isinstance(measured, LiveRobotState):
            raise TypeError("retry FoundPose exposure lacks measured feedback")
        relation = validate_se3(
            runner._postlift_preflight.relation.T_key_hand,
            name="observed post-lift key-to-hand relation")
        wrist = validate_se3(
            self.planner.fk_wrist(measured.full_q),
            name="measured withdrawn wrist FK")
        c2r = validate_se3(runner.calibration.record["c2r"],
                           name="session C2R")
        prior_source = (
            "measured_wrist_plus_bounded_held_relation"
            if isinstance(runner._postlift_preflight, BoundedPostLiftPreflight)
            else "measured_wrist_plus_observed_held_relation")
        prior = validate_se3(c2r @ wrist @ np.linalg.inv(relation),
                             name="measured withdrawn key prior")
        try:
            observation = admit_held_key_capture(
                capture=capture, init_orchestrator=cameras.init_orchestrator,
                mode=runner.mode, shared_root=runner.shared_root,
                calibration=runner.calibration,
                calibrated_camera_ids=set(launch.camera_ids),
                view_limits=launch.view_limits,
                held_pose_prior_world=prior,
                held_pose_prior_timestamp_s=measured.sample_timestamp_s,
                held_pose_prior_source=prior_source,
                **options)
        except ValueError as exc:
            raise HeldKeyRetryUnobservable(str(exc), raw_dir) from exc
        evidence = (Path(runner._attempt_dir) /
                    "observed_retry_held_keys" / capture_id)
        write_key_capture_artifacts(capture, observation, evidence)
        verify_key_capture_artifacts(evidence)
        return HeldKeyRetryCapture(observation, evidence, raw, measured)

    def _require(self, runner, decision: SessionDecision, action: str) -> None:
        if (runner is not self.live.opened.runner or
                runner.mode.family != "square" or
                not isinstance(decision, SessionDecision) or
                decision.action != action or
                runner.current_decision() != decision or
                runner.active_attempt is None or
                runner.active_attempt.candidate_id != decision.candidate_id or
                self.enable_robot_motion is not True):
            raise ValueError(f"square retry needs this live held {action}")

    def _capture(self, runner, decision, callback, phase: str):
        capture = callback(self.live, runner, decision)
        if not isinstance(capture, RawCameraCapture):
            raise TypeError(f"{phase} needs same-session raw AutoDex frames")
        if (len(capture.images_bgr) < 2 or
                not set(capture.images_bgr) <=
                set(self.live.cameras.launch.camera_ids)):
            raise ValueError(f"{phase} frames differ from frozen cameras")
        output = (Path(runner._attempt_dir) / "raw_captures" / phase /
                  capture.capture_id)
        write_raw_camera_capture(capture, output, phase=phase)
        verify_raw_camera_capture(output, phase=phase)
        state = self.state_for_capture(capture)
        if not isinstance(state, LiveRobotState):
            raise TypeError(f"{phase} exposure lacks measured robot state")
        return output, state

    def assess_and_shift(self, runner, decision: SessionDecision) -> None:
        self._require(runner, decision,
                      "guarded_withdrawal_then_xy_assessment")
        if self._pending is not None or runner._postlift_report_path is None:
            raise ValueError("square retry needs one observed post-lift relation")
        withdrawal = self.withdraw_guarded(runner, decision)
        if (not isinstance(withdrawal, tuple) or len(withdrawal) != 2 or
                not isinstance(withdrawal[0], LiveRobotState)):
            raise TypeError("guarded withdrawal needs measured state and log")
        withdrawal_path = Path(withdrawal[1]).expanduser().resolve()
        completed = float(json.loads(withdrawal_path.read_text(
            encoding="utf-8"))["completed_at_s"])
        self._withdrawn = (runner.active_attempt.attempt_id,
                           withdrawal[0], withdrawal_path)
        try:
            held = (self.capture_withdrawn_key(self.live, runner, decision)
                    if self.capture_withdrawn_key is not None else
                    self._capture_held_key(runner))
        except HeldKeyRetryUnobservable as exc:
            assessment_path = (
                Path(runner._attempt_dir) / "xy_retry_assessments" /
                f"{runner._retry_assessment_index:03d}" / "report.json")
            assessment_path.parent.mkdir(parents=True, exist_ok=False)
            assessment_path.write_text(json.dumps({
                "schema": "precision_insertion_held_retry_unobservable_v1",
                "attempt_id": runner.active_attempt.attempt_id,
                "candidate_id": runner.active_attempt.candidate_id,
                "raw_capture_dir": (
                    None if exc.raw_dir is None else str(exc.raw_dir)),
                "withdrawal_evidence_path": str(withdrawal_path),
                "reason": str(exc),
                "scope": "no_observed_key_pose_no_xy_motion",
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            runner.record_xy_retry_unavailable(
                reason="held_key_pose_unavailable",
                retry_assessment_path=assessment_path,
                timestamp_s=self.decision_time())
            return
        if (not isinstance(held, HeldKeyRetryCapture) or
                not isinstance(held.raw, RawCameraCapture) or
                not isinstance(held.state, LiveRobotState) or
                not isinstance(held.observation, KeyPoseObservation) or
                len(held.raw.images_bgr) < 2 or
                not set(held.raw.images_bgr) <=
                set(self.live.cameras.launch.camera_ids)):
            raise TypeError("square retry needs same-frame admitted held key")
        frames = [LabeledFrame(
            camera, "preinsert_hold",
            held.raw.acquisition_metadata["frames"][camera]["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            for camera, image in sorted(held.raw.images_bgr.items())]
        index = runner._retry_assessment_index
        launch = self.live.cameras.launch
        result = runner.prepare_observed_xy_retry(
            planner=self.planner, held_key_observation=held.observation,
            held_key_evidence_dir=held.evidence_dir,
            joint_sample=held.state, frames=frames,
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            frame_request_id=held.raw.request_id,
            frame_ids=held.raw.frame_ids,
            acquisition_metadata=held.raw.acquisition_metadata,
            backend=self.backend, withdrawal_completed_at_s=completed,
            withdrawal_evidence_path=withdrawal_path,
            postlift_preflight_report_path=runner._postlift_report_path,
            decision_timestamp_s=self.decision_time(),
            limits=self.retry_limits)
        source = (Path(runner._attempt_dir) / "xy_retry_assessments" /
                  f"{index:03d}")
        if result.status != "ready_to_record_pending_retry":
            runner.record_xy_retry_unavailable(
                reason=("vlm_abstain" if result.assessment is None or
                        result.assessment.decision is None else
                        "lateral_preflight_failed"),
                retry_assessment_path=source / "report.json",
                timestamp_s=self.decision_time())
            return
        preflight = result.preflight
        assert preflight is not None
        plan_path = source / "preflight" / "report.json"
        self._withdrawn = None
        execute_bound_observed_retry_transfer(
            runner=runner, preflight=preflight,
            preflight_report_path=plan_path,
            adapter=self.transfer_adapter, pre_state=self.read_state(),
            read_post_state=self.read_state, limits=self.transfer_limits,
            commissioning_record_path=self.transfer_commissioning_path,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion)
        transfer_path = (Path(runner._attempt_dir) /
                         "observed_retry_transfer_execution.json")
        arrival_raw, arrival_state = self._capture(
            runner, decision, self.capture_arrival, "preinsert")
        arrival_index = len(list((Path(runner._attempt_dir) /
                                  "observed_retry_arrivals").glob("*")))
        arrival = runner.assess_observed_xy_retry_arrival(
            preflight=preflight, preflight_report_path=plan_path,
            transfer_execution_path=transfer_path,
            raw_bundle=arrival_raw, joint_sample=arrival_state,
            backend=self.backend, **dict(self.arrival_parameters))
        arrival_path = (Path(runner._attempt_dir) /
                        "observed_retry_arrivals" /
                        f"{arrival_index:03d}" / "report.json")
        if arrival.preinsert_reached is not True:
            runner.record_failure(
                "perception_unreliable", timestamp_s=self.decision_time(),
                evidence_refs={"retry_arrival": str(arrival_path)})
            return
        self._pending = {
            "attempt_id": runner.active_attempt.attempt_id,
            "preflight": preflight, "preflight_path": plan_path,
            "arrival_path": arrival_path, "arrival_raw": arrival_raw,
        }

    def insert_and_observe(self, runner, decision: SessionDecision) -> None:
        self._require(runner, decision,
                      "await_retry_execution_and_observation")
        pending = self._pending
        if (pending is None or
                pending["attempt_id"] != runner.active_attempt.attempt_id):
            raise ValueError("square retry insertion lacks its new arrival")
        handoff = runner.prepare_observed_xy_retry_axial_handoff(
            preflight=pending["preflight"],
            preflight_report_path=pending["preflight_path"],
            arrival_report_path=pending["arrival_path"],
            measured_start=self.read_state(),
            decision_timestamp_s=self.decision_time(),
            **dict(self.handoff_parameters))
        executed = execute_bound_guarded_insertion(
            runner=runner, adapter=self.insertion_adapter,
            handoff_report_path=handoff,
            pre_state=self.read_state(), read_post_state=self.read_state,
            limits=self.insertion_limits, contact_limits=self.contact_limits,
            watchdog_commissioning_path=self.insertion_watchdog_path,
            contact_commissioning_path=self.insertion_contact_path,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion,
            observed_xy_retry=True, **dict(self.insertion_parameters))
        metric_path = Path(executed["metric_record"]["path"])
        final, _ = self._capture(
            runner, decision, self.capture_final, "final_or_abort")
        decision_at = self.decision_time()
        depth_path = None
        if self.depth_limits is not None:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            saved, visible = prepare_live_depth_evidence(
                runner=runner, final_bundle=final, backend=self.backend,
                completed_at_s=metric["completed_at_s"],
                decision_timestamp_s=decision_at,
                limits=self.depth_limits,
                capture_parameters=self.depth_capture_parameters,
                slot="retry")
            if visible and self.depth_commissioning_path is not None:
                depth_path = saved
        runner.prepare_observed_insertion_label(
            preinsert_bundle=pending["arrival_raw"], final_bundle=final,
            metric_record_path=metric_path, backend=self.backend,
            decision_timestamp_s=decision_at,
            retry_arrival_report_path=pending["arrival_path"],
            depth_report_path=depth_path,
            depth_commissioning_path=(
                self.depth_commissioning_path if depth_path is not None else None),
            label_mode=self.label_mode,
            manual_annotation=request_annotation(
                mode=self.label_mode, provider=self.manual_label_provider,
                stage="insertion_success",
                evidence_dirs=(pending["arrival_raw"], final),
                attempt_id=runner.active_attempt.attempt_id,
                candidate_id=runner.active_attempt.candidate_id),
            **dict(self.final_parameters))
        self._pending = None

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "guarded_withdrawal_then_xy_assessment": self.assess_and_shift,
            "await_retry_execution_and_observation": self.insert_and_observe,
        }
