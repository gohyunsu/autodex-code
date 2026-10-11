"""Connect held transfer and first guarded insertion in one live session.

All planning, execution evidence and visual/metric labels come from the
existing demo modules. Robot controllers and camera acquisition are injected
from the AutoDex PC; this module does not emulate physical motion or sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .guarded_insertion_execution import execute_bound_guarded_insertion
from .live_robot_state import LiveRobotState
from .live_depth import prepare_live_depth_evidence
from .labeling import request_annotation
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from .recoverable_failure import acknowledged_failure
from .session_policy import SessionDecision
from .transfer_execution import execute_bound_held_transfer


@dataclass
class LiveTaskHandlers:
    """Execute transfer/20 mm stroke and ask the existing Qwen checkpoints.

    The two capture callbacks receive ``(live, runner, decision)`` and return
    a `RawCameraCapture` from the same calibrated, open AutoDex cameras.
    ``state_for_capture`` reads time-aligned measured joints from a buffer.
    The external adapters remain responsible for actual robot-side stops.
    """

    live: object
    transfer_adapter: object
    insertion_adapter: object
    read_state: Callable[[], LiveRobotState]
    state_for_capture: Callable[[RawCameraCapture], LiveRobotState]
    capture_preinsert: Callable
    capture_final: Callable
    backend: object
    motion_interlock: Callable[[], bool]
    transfer_limits: object
    insertion_limits: object
    contact_limits: object
    transfer_commissioning_path: Path
    insertion_watchdog_path: Path
    insertion_contact_path: Path
    preinsert_parameters: Mapping[str, object]
    handoff_parameters: Mapping[str, object]
    insertion_parameters: Mapping[str, object]
    final_parameters: Mapping[str, object]
    depth_limits: object | None = None
    depth_capture_parameters: Mapping[str, float] = field(default_factory=dict)
    depth_commissioning_path: Path | None = None
    label_mode: str = "evidence_fused"
    manual_label_provider: Callable | None = None
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time
    _preinsert_bundle: Path | None = field(default=None, init=False)
    _preinsert_attempt_id: str | None = field(default=None, init=False)

    def _require(self, runner, decision: SessionDecision, action: str):
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != action):
            raise ValueError(f"{action} belongs to another session or decision")
        if self.enable_robot_motion is not True:
            raise PermissionError(f"{action} robot motion is not enabled")
        attempt = runner.active_attempt
        if attempt is None or attempt.candidate_id != decision.candidate_id:
            raise ValueError(f"{action} has no bound selected attempt")
        return attempt

    def _capture(
        self, *, runner, decision, callback, phase: str,
    ) -> tuple[Path, RawCameraCapture]:
        capture = callback(self.live, runner, decision)
        if not isinstance(capture, RawCameraCapture):
            raise TypeError(f"{phase} needs same-session raw AutoDex frames")
        cameras = set(self.live.cameras.launch.camera_ids)
        if len(capture.images_bgr) < 2 or not set(capture.images_bgr) <= cameras:
            raise ValueError(f"{phase} frames differ from frozen session cameras")
        attempt = runner.active_attempt
        output = (Path(runner.output_dir) / "attempts" /
                  attempt.attempt_id / "raw_captures" / phase /
                  capture.capture_id)
        write_raw_camera_capture(capture, output, phase=phase)
        verify_raw_camera_capture(output, phase=phase)
        return output, capture

    def transfer_and_observe(self, runner, decision: SessionDecision) -> None:
        attempt = self._require(runner, decision,
                                "transfer_execution_gate_required")
        try:
            execute_bound_held_transfer(
                runner=runner, adapter=self.transfer_adapter,
                pre_state=self.read_state(), read_post_state=self.read_state,
                limits=self.transfer_limits,
                commissioning_record_path=self.transfer_commissioning_path,
                motion_interlock=self.motion_interlock,
                enable_robot_motion=self.enable_robot_motion)
        except Exception:
            # The executor writes this record only after a motion was requested
            # and stop_and_acknowledge() returned. No transfer checkpoint or
            # held-key assumption is fabricated from a failed controller call.
            failure_path = (Path(runner.output_dir) / "attempts" /
                            attempt.attempt_id / "transfer_failure.json")
            if not acknowledged_failure(
                    failure_path,
                    schema="precision_insertion_transfer_execution_v1",
                    attempt_id=attempt.attempt_id,
                    candidate_id=attempt.candidate_id):
                raise
            runner.observe_stage(
                "preinsert_reached", False,
                timestamp_s=self.decision_time(),
                evidence_refs={"transfer_failure": str(failure_path)})
            return
        transfer_log = (Path(runner.output_dir) / "attempts" /
                        attempt.attempt_id / "transfer_execution.json")
        raw, capture = self._capture(
            runner=runner, decision=decision,
            callback=self.capture_preinsert, phase="preinsert")
        state = self.state_for_capture(capture)
        if not isinstance(state, LiveRobotState):
            raise TypeError("pre-insertion exposure needs measured robot state")
        checkpoint = runner.prepare_observed_preinsert_label(
            raw_bundle=raw, transfer_execution_path=transfer_log,
            joint_sample=state, backend=self.backend,
            label_mode=self.label_mode,
            manual_annotation=request_annotation(
                mode=self.label_mode, provider=self.manual_label_provider,
                stage="preinsert_reached", evidence_dirs=(raw,),
                attempt_id=attempt.attempt_id,
                candidate_id=attempt.candidate_id),
            **dict(self.preinsert_parameters))
        report = runner._preinsert_report_path
        refs = {
            "postlift_preflight": str(runner._postlift_report_path),
            "preinsert_checkpoint": str(report),
            "trajectory": str(transfer_log),
            "grasp_state": str(transfer_log),
            "key_socket_pose": str(report),
            "preinsert_image": str(raw / "manifest.json"),
        }
        runner.observe_stage(
            "preinsert_reached", checkpoint.preinsert_reached,
            timestamp_s=self.decision_time(), evidence_refs=refs)
        if checkpoint.preinsert_reached is True:
            self._preinsert_bundle = raw
            self._preinsert_attempt_id = attempt.attempt_id

    def insert_and_observe(self, runner, decision: SessionDecision) -> None:
        attempt = self._require(runner, decision,
                                "await_guarded_insertion_and_observation")
        if (self._preinsert_attempt_id != attempt.attempt_id or
                self._preinsert_bundle is None):
            raise ValueError("guarded insertion lacks this attempt's pre-insertion frames")
        handoff = runner.prepare_guarded_axial_handoff(
            measured_start=self.read_state(),
            decision_timestamp_s=self.decision_time(),
            **dict(self.handoff_parameters))
        execution = execute_bound_guarded_insertion(
            runner=runner, adapter=self.insertion_adapter,
            handoff_report_path=handoff,
            pre_state=self.read_state(), read_post_state=self.read_state,
            limits=self.insertion_limits,
            contact_limits=self.contact_limits,
            watchdog_commissioning_path=self.insertion_watchdog_path,
            contact_commissioning_path=self.insertion_contact_path,
            motion_interlock=self.motion_interlock,
            enable_robot_motion=self.enable_robot_motion,
            **dict(self.insertion_parameters))
        metric = Path(execution["metric_record"]["path"])
        final, _capture = self._capture(
            runner=runner, decision=decision,
            callback=self.capture_final, phase="final_or_abort")
        decision_at = self.decision_time()
        depth_path = None
        if self.depth_limits is not None:
            metric_record = json.loads(metric.read_text(encoding="utf-8"))
            saved, visible = prepare_live_depth_evidence(
                runner=runner, final_bundle=final, backend=self.backend,
                completed_at_s=metric_record["completed_at_s"],
                decision_timestamp_s=decision_at,
                limits=self.depth_limits,
                capture_parameters=self.depth_capture_parameters,
                slot="first")
            if visible and self.depth_commissioning_path is not None:
                depth_path = saved
        runner.prepare_observed_insertion_label(
            preinsert_bundle=self._preinsert_bundle,
            final_bundle=final, metric_record_path=metric,
            backend=self.backend,
            decision_timestamp_s=decision_at,
            depth_report_path=depth_path,
            depth_commissioning_path=(
                self.depth_commissioning_path if depth_path is not None else None),
            label_mode=self.label_mode,
            manual_annotation=request_annotation(
                mode=self.label_mode, provider=self.manual_label_provider,
                stage="insertion_success",
                evidence_dirs=(self._preinsert_bundle, final),
                attempt_id=attempt.attempt_id,
                candidate_id=attempt.candidate_id),
            **dict(self.final_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "transfer_execution_gate_required": self.transfer_and_observe,
            "await_guarded_insertion_and_observation": self.insert_and_observe,
        }
