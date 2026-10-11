"""Join the existing grounded 1 mm retry stages in the live action loop.

This module supplies no new planner, VLM policy, or robot controller. The
injected producers must return real camera frames and measured robot state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Mapping

import cv2
from PIL import Image

from .action_loop import ActionHandler
from .frame_provenance import verify_frame_provenance
from .lateral_execution import execute_bound_lateral_shift
from .live_depth import prepare_live_depth_evidence
from .live_robot_state import LiveRobotState
from .observer import LabeledFrame
from .postshift_transfer_execution import execute_bound_postshift_transfer
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from .retry_guarded_execution import execute_bound_retry_guarded_insertion
from .session_policy import SessionDecision


@dataclass
class LiveGroundedRetryHandlers:
    """One failed-insertion correction and its fresh guarded reattempt.

    Every ``*_parameters`` mapping contains the corresponding SessionRunner
    or execution function's unowned keyword arguments. This deliberately
    reuses those contracts instead of defining another retry policy.
    """

    live: object
    planner: object
    backend: object
    read_state: Callable[[], LiveRobotState]
    state_for_capture: Callable[[RawCameraCapture], LiveRobotState]
    withdraw_guarded: Callable
    capture_withdrawn: Callable
    capture_postshift: Callable
    capture_arrival: Callable
    capture_preinsert: Callable
    capture_final: Callable
    lateral_adapter: object
    transfer_adapter: object
    insertion_adapter: object
    motion_interlock: Callable[[], bool]
    retry_limits: object
    alignment_limits: object
    deviation_bounds: object
    lateral_limits: object
    transfer_limits: object
    insertion_limits: object
    contact_limits: object
    lateral_commissioning_path: Path
    transfer_commissioning_path: Path
    insertion_watchdog_path: Path
    insertion_contact_path: Path
    diagnostic_parameters: Mapping[str, object]
    lateral_parameters: Mapping[str, object]
    lateral_execution_parameters: Mapping[str, object]
    checkpoint_parameters: Mapping[str, object]
    insertion_preflight_parameters: Mapping[str, object]
    path_handoff_parameters: Mapping[str, object]
    transfer_execution_parameters: Mapping[str, object]
    arrival_parameters: Mapping[str, object]
    axial_replan_parameters: Mapping[str, object]
    retry_handoff_parameters: Mapping[str, object]
    retry_execution_parameters: Mapping[str, object]
    final_parameters: Mapping[str, object]
    depth_limits: object | None = None
    depth_capture_parameters: Mapping[str, float] = field(default_factory=dict)
    depth_commissioning_path: Path | None = None
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time
    _pending: dict | None = field(default=None, init=False)

    def _require(self, runner, decision: SessionDecision, action: str) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action != action or
                runner.current_decision() != decision or
                runner.active_attempt is None or
                runner.active_attempt.candidate_id != decision.candidate_id):
            raise ValueError(f"{action} is not this session's held attempt")
        if self.enable_robot_motion is not True:
            raise PermissionError("retry robot motion is not enabled")

    def _capture(self, runner, decision, callback, phase: str):
        capture = callback(self.live, runner, decision)
        if not isinstance(capture, RawCameraCapture):
            raise TypeError(f"{phase} needs same-session raw AutoDex frames")
        cameras = set(self.live.cameras.launch.camera_ids)
        if len(capture.images_bgr) < 2 or not set(capture.images_bgr) <= cameras:
            raise ValueError(f"{phase} frames differ from frozen cameras")
        root = (Path(runner._attempt_dir) / "raw_captures" / phase /
                capture.capture_id)
        write_raw_camera_capture(capture, root, phase=phase)
        verify_raw_camera_capture(root, phase=phase)
        state = self.state_for_capture(capture)
        if not isinstance(state, LiveRobotState):
            raise TypeError(f"{phase} exposure lacks measured robot state")
        return root, capture, state

    @staticmethod
    def _report(runner, folder: str, index: int) -> Path:
        return Path(runner._attempt_dir) / folder / f"{index:03d}" / "report.json"

    def _unavailable(self, runner, reason: str, report: Path) -> None:
        runner.record_xy_retry_unavailable(
            reason=reason, retry_assessment_path=report,
            timestamp_s=self.decision_time())
        self._pending = None

    def _unreliable_held_relation(self, runner, report: Path) -> None:
        """Do not execute a held return from a visibly inconsistent grasp."""
        runner.record_failure(
            "perception_unreliable", timestamp_s=self.decision_time(),
            evidence_refs={"postshift_checkpoint": str(report)})
        self._pending = None

    def assess_and_shift(self, runner, decision: SessionDecision) -> None:
        self._require(runner, decision,
                      "guarded_withdrawal_then_xy_assessment")
        if self._pending is not None:
            raise ValueError("a grounded retry is already pending")
        if runner._postlift_report_path is None:
            raise ValueError("retry needs the measured post-lift held relation")
        withdrawal = self.withdraw_guarded(runner, decision)
        if (not isinstance(withdrawal, tuple) or len(withdrawal) != 2 or
                not isinstance(withdrawal[0], LiveRobotState)):
            raise TypeError("guarded withdrawal must return measured state and log")
        withdrawal_path = Path(withdrawal[1]).expanduser().resolve()
        completed = float(json.loads(withdrawal_path.read_text(
            encoding="utf-8"))["completed_at_s"])
        capture = self.capture_withdrawn(self.live, runner, decision)
        if not isinstance(capture, RawCameraCapture):
            raise TypeError("withdrawn hold needs same-session camera frames")
        if (len(capture.images_bgr) < 2 or
                not set(capture.images_bgr) <=
                set(self.live.cameras.launch.camera_ids)):
            raise ValueError("withdrawn frames differ from session cameras")
        frame_rows = verify_frame_provenance(
            capture.acquisition_metadata, request_id=capture.request_id,
            images_bgr=capture.images_bgr, frame_ids=capture.frame_ids)
        frames = [LabeledFrame(
            camera, "preinsert_hold", frame_rows[camera]["timestamp_s"],
            Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            for camera, image in sorted(capture.images_bgr.items())]
        state = self.state_for_capture(capture)
        if not isinstance(state, LiveRobotState):
            raise TypeError("withdrawn exposure lacks measured robot state")
        launch = self.live.cameras.launch
        diagnostic_index = runner._retry_assessment_index
        diagnostic = runner.prepare_grounded_xy_diagnostic(
            joint_sample=state, frames=frames,
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            frame_request_id=capture.request_id, frame_ids=capture.frame_ids,
            acquisition_metadata=capture.acquisition_metadata,
            backend=self.backend, withdrawal_completed_at_s=completed,
            withdrawal_evidence_path=withdrawal_path,
            postlift_preflight_report_path=runner._postlift_report_path,
            decision_timestamp_s=self.decision_time(),
            limits=self.retry_limits, alignment_limits=self.alignment_limits,
            **dict(self.diagnostic_parameters))
        diagnostic_path = self._report(
            runner, "xy_retry_assessments", diagnostic_index)
        if diagnostic.status != "diagnostic_metric_xy_correction":
            self._unavailable(runner, "vlm_abstain", diagnostic_path)
            return
        lateral_index = runner._lateral_preflight_index
        plan = runner.prepare_grounded_lateral_hold_preflight(
            planner=self.planner, diagnostic=diagnostic,
            diagnostic_report_path=diagnostic_path,
            joint_sample=self.read_state(),
            decision_timestamp_s=self.decision_time(),
            limits=self.retry_limits,
            observed_bounds=self.deviation_bounds,
            **dict(self.lateral_parameters))
        plan_path = self._report(runner, "lateral_hold_preflights", lateral_index)
        if plan.lateral.status != "sampled_lateral_hold_shift_pass":
            self._unavailable(runner, "lateral_preflight_failed", plan_path)
            return
        execute_bound_lateral_shift(
            runner=runner, plan=plan, plan_report_path=plan_path,
            adapter=self.lateral_adapter, pre_state=self.read_state(),
            read_post_state=self.read_state, limits=self.lateral_limits,
            commissioning_record_path=self.lateral_commissioning_path,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion,
            **dict(self.lateral_execution_parameters))
        shift_log = (Path(runner._attempt_dir) / "lateral_executions" /
                     plan_path.parent.name / "execution.json")
        postshift_dir, _, postshift_state = self._capture(
            runner, decision, self.capture_postshift, "post_lateral_hold")
        checkpoint_index = runner._postshift_checkpoint_index
        checkpoint = runner.assess_postshift_lateral_alignment(
            plan=plan, plan_report_path=plan_path,
            execution_log_path=shift_log, capture_dir=postshift_dir,
            joint_sample=postshift_state,
            decision_timestamp_s=self.decision_time(),
            planner=self.planner,
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            backend=self.backend, alignment_limits=self.alignment_limits,
            **dict(self.checkpoint_parameters))
        checkpoint_path = self._report(
            runner, "postshift_checkpoints", checkpoint_index)
        if checkpoint.status in {"held_relation_inconsistent", "visual_abstain"}:
            self._unreliable_held_relation(runner, checkpoint_path)
            return
        if checkpoint.status != "visual_alignment_within_budget":
            self._unavailable(runner, "postshift_alignment_failed",
                              checkpoint_path)
            return
        preflight_index = runner._postshift_20mm_index
        previous = runner.prepare_postshift_insertion_preflight(
            planner=self.planner, shift_plan=plan,
            checkpoint=checkpoint, checkpoint_report_path=checkpoint_path,
            bounds=self.deviation_bounds,
            **dict(self.insertion_preflight_parameters))
        previous_path = self._report(
            runner, "postshift_20mm_preflights", preflight_index)
        if previous.status != "sampled_postshift_20mm_preflight_pass":
            self._unavailable(runner, "insertion_preflight_failed",
                              previous_path)
            return
        handoff = runner.prepare_postshift_path_handoff(
            preflight=previous, preflight_report_path=previous_path,
            checkpoint=checkpoint, shift_plan=plan,
            measured_start=self.read_state(),
            decision_timestamp_s=self.decision_time(),
            **dict(self.path_handoff_parameters))
        if not json.loads(handoff.read_text(
                encoding="utf-8"))["transfer_required"]:
            self._unavailable(runner, "arrival_preflight_failed", handoff)
            return
        execute_bound_postshift_transfer(
            runner=runner, expected=previous,
            checkpoint=checkpoint, shift_plan=plan,
            handoff_report_path=handoff, adapter=self.transfer_adapter,
            pre_state=self.read_state(), read_post_state=self.read_state,
            limits=self.transfer_limits,
            commissioning_record_path=self.transfer_commissioning_path,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion,
            **dict(self.transfer_execution_parameters))
        transfer_log = (Path(runner._attempt_dir) /
                        "postshift_transfer_executions" /
                        handoff.parent.name / "execution.json")
        arrival_dir, _, arrival_state = self._capture(
            runner, decision, self.capture_arrival, "preinsert")
        arrival_index = runner._postshift_arrival_index
        arrival = runner.assess_postshift_transfer_arrival(
            preflight=previous, checkpoint=checkpoint, shift_plan=plan,
            handoff_report_path=handoff,
            transfer_execution_path=transfer_log,
            capture_dir=arrival_dir, joint_sample=arrival_state,
            decision_timestamp_s=self.decision_time(),
            planner=self.planner,
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            backend=self.backend, alignment_limits=self.alignment_limits,
            **dict(self.arrival_parameters))
        arrival_path = self._report(
            runner, "postshift_arrival_checkpoints", arrival_index)
        if arrival.status in {"held_relation_inconsistent", "visual_abstain"}:
            self._unreliable_held_relation(runner, arrival_path)
            return
        if arrival.status != "visual_alignment_within_budget":
            self._unavailable(runner, "arrival_preflight_failed", arrival_path)
            return
        replan_index = runner._postshift_arrival_replan_index
        replan = runner.prepare_postshift_arrival_axial_replan(
            planner=self.planner, previous=previous,
            previous_report_path=previous_path,
            arrival=arrival, arrival_report_path=arrival_path,
            checkpoint=checkpoint, shift_plan=plan,
            bounds=self.deviation_bounds,
            **dict(self.axial_replan_parameters))
        replan_path = self._report(
            runner, "postshift_arrival_axial_replans", replan_index)
        if replan.status != "sampled_arrival_20mm_axial_preflight_pass":
            self._unavailable(runner, "arrival_preflight_failed", replan_path)
            return
        runner.record_grounded_xy_retry(
            replan=replan, replan_report_path=replan_path,
            previous=previous, arrival=arrival, checkpoint=checkpoint,
            shift_plan=plan, shift_plan_report_path=plan_path,
            timestamp_s=self.decision_time())
        self._pending = dict(
            attempt_id=runner.active_attempt.attempt_id,
            replan=replan, replan_path=replan_path,
            previous=previous, arrival=arrival, checkpoint=checkpoint,
            shift_plan=plan)

    def insert_and_observe(self, runner, decision: SessionDecision) -> None:
        self._require(runner, decision,
                      "await_retry_execution_and_observation")
        pending = self._pending
        if pending is None or pending["attempt_id"] != runner.active_attempt.attempt_id:
            raise ValueError("retry insertion lacks this attempt's observed shift")
        handoff = runner.prepare_retry_axial_handoff(
            replan=pending["replan"],
            replan_report_path=pending["replan_path"],
            previous=pending["previous"], arrival=pending["arrival"],
            checkpoint=pending["checkpoint"],
            shift_plan=pending["shift_plan"],
            measured_start=self.read_state(),
            decision_timestamp_s=self.decision_time(),
            **dict(self.retry_handoff_parameters))
        before, _, _ = self._capture(
            runner, decision, self.capture_preinsert, "preinsert")
        executed = execute_bound_retry_guarded_insertion(
            runner=runner, expected=pending["replan"],
            previous=pending["previous"], arrival=pending["arrival"],
            checkpoint=pending["checkpoint"],
            shift_plan=pending["shift_plan"],
            handoff_report_path=handoff, adapter=self.insertion_adapter,
            pre_state=self.read_state(), read_post_state=self.read_state,
            limits=self.insertion_limits, contact_limits=self.contact_limits,
            watchdog_commissioning_path=self.insertion_watchdog_path,
            contact_commissioning_path=self.insertion_contact_path,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion,
            **dict(self.retry_execution_parameters))
        execution = (Path(runner._attempt_dir) / "retry_guarded_executions" /
                     handoff.parent.name / "execution.json")
        after, _, _ = self._capture(
            runner, decision, self.capture_final, "final_or_abort")
        decision_at = self.decision_time()
        depth_path = None
        if self.depth_limits is not None:
            metric_path = Path(executed["metric_record"]["path"])
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            saved, visible = prepare_live_depth_evidence(
                runner=runner, final_bundle=after, backend=self.backend,
                completed_at_s=metric["completed_at_s"],
                decision_timestamp_s=decision_at,
                limits=self.depth_limits,
                capture_parameters=self.depth_capture_parameters,
                slot="retry")
            if visible and self.depth_commissioning_path is not None:
                depth_path = saved
        runner.prepare_observed_retry_insertion_label(
            replan=pending["replan"], previous=pending["previous"],
            arrival=pending["arrival"], checkpoint=pending["checkpoint"],
            shift_plan=pending["shift_plan"],
            handoff_report_path=handoff, execution_log_path=execution,
            preinsert_bundle=before, final_bundle=after,
            backend=self.backend, decision_timestamp_s=decision_at,
            depth_report_path=depth_path,
            depth_commissioning_path=(
                self.depth_commissioning_path if depth_path is not None else None),
            **dict(self.final_parameters))
        self._pending = None

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "guarded_withdrawal_then_xy_assessment": self.assess_and_shift,
            "await_retry_execution_and_observation": self.insert_and_observe,
        }
