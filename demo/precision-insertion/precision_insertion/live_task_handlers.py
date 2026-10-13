"""Connect held transfer and first guarded insertion in one live session.

All planning, execution evidence and visual/metric labels come from the
existing demo modules. Robot controllers and camera acquisition are injected
from the AutoDex PC; this module does not emulate physical motion or sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .guarded_insertion_execution import execute_bound_guarded_insertion
from .live_robot_state import LiveRobotState
from .labeling import request_annotation
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from .recoverable_failure import acknowledged_failure
from .session_policy import SessionDecision
from .post_insertion_release import verify_post_insertion_release
from .transfer_execution import execute_bound_held_transfer
from .timing_profile import timed


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
    transfer_commissioning_path: Path | None
    preinsert_parameters: Mapping[str, object]
    handoff_parameters: Mapping[str, object]
    insertion_parameters: Mapping[str, object]
    final_parameters: Mapping[str, object]
    contact_limits: object | None = None
    insertion_watchdog_path: Path | None = None
    insertion_contact_path: Path | None = None
    depth_limits: object | None = None
    depth_capture_parameters: Mapping[str, float] = field(default_factory=dict)
    depth_commissioning_path: Path | None = None
    label_mode: str = "evidence_fused"
    manual_label_provider: Callable | None = None
    release_and_clear: Callable | None = None
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time
    _preinsert_bundle: Path | None = field(default=None, init=False)
    _preinsert_attempt_id: str | None = field(default=None, init=False)
    _withdrawn: tuple[str, LiveRobotState, Path] | None = field(
        default=None, init=False)

    def withdrawn_for_return(self):
        return self._withdrawn

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
        with timed(f"trial.{phase}_capture", phase="trial", kind="camera",
                   attempt_id=runner.active_attempt.attempt_id):
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
            with timed("trial.transfer_motion", phase="trial", kind="motion",
                       attempt_id=attempt.attempt_id):
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
        lift_checkpoint = getattr(runner, "_lift_checkpoint", None)
        after_lift_raw = getattr(lift_checkpoint, "after_bundle", None)
        if after_lift_raw is None:
            raise ValueError("preinsert VLM needs this attempt's raw after-lift capture")
        state = self.state_for_capture(capture)
        if not isinstance(state, LiveRobotState):
            raise TypeError("pre-insertion exposure needs measured robot state")
        with timed("trial.preinsert_label", phase="trial",
                   attempt_id=attempt.attempt_id):
            checkpoint = runner.prepare_observed_preinsert_label(
                raw_bundle=raw, transfer_execution_path=transfer_log,
                joint_sample=state, backend=self.backend,
                label_mode=self.label_mode,
                manual_annotation=request_annotation(
                    mode=self.label_mode, provider=self.manual_label_provider,
                    stage="preinsert_reached",
                    evidence_dirs=(Path(after_lift_raw), raw),
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
            "after_lift_image": str(Path(after_lift_raw) / "manifest.json"),
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
        with timed("trial.insertion_handoff", phase="planning", kind="planning",
                   attempt_id=attempt.attempt_id):
            handoff = runner.prepare_guarded_axial_handoff(
                measured_start=self.read_state(),
                decision_timestamp_s=self.decision_time(),
                **dict(self.handoff_parameters))
        prepare_release = getattr(self.release_and_clear, "prepare", None)
        if callable(prepare_release):
            prepare_release(self.live, runner, decision, handoff)
        with timed("trial.guarded_insertion_motion", phase="trial",
                   kind="motion", attempt_id=attempt.attempt_id):
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
        if (execution.get("safety_abort") is True and
                execution.get("retreat_completed") is True and
                execution.get("withdrawal_record_path")):
            withdrawal = Path(execution["withdrawal_record_path"])
            self._withdrawn = (
                attempt.attempt_id, self.read_state(), withdrawal)
            metric = Path(execution["metric_record"]["path"])
            completed = self.decision_time()
            runner.observe_stock_force_stop(
                metric_record_path=metric,
                withdrawal_record_path=withdrawal,
                timestamp_s=completed)
            runner.record_failure(
                "force_abort", timestamp_s=self.decision_time(),
                evidence_refs={"force_trace": str(metric),
                               "axial_withdrawal": str(withdrawal)})
            return
        if (execution.get("safety_abort") is not False or
                execution.get("grasp_held") is not True):
            raise RuntimeError(
                "insertion ended without a verified force-stop ascent; "
                "supervised recovery is required")
        metric = Path(execution["metric_record"]["path"])
        if not callable(self.release_and_clear):
            raise RuntimeError(
                "20 mm insertion needs a commissioned release_and_clear callback")
        with timed("trial.release_and_clear", phase="trial", kind="motion",
                   attempt_id=attempt.attempt_id):
            release_path = Path(self.release_and_clear(
                self.live, runner, decision, execution)).expanduser().resolve()
        verify_post_insertion_release(
            release_path, attempt_id=attempt.attempt_id,
            candidate_id=attempt.candidate_id,
            session_calibration_sha256=runner.session_sha256,
            guarded_execution_path=metric)
        # The release callback performs the physical open/exit; this measured
        # gate prevents a final image from being taken with the hand in view.
        with timed("trial.final_clear_view", phase="trial", kind="motion",
                   attempt_id=attempt.attempt_id):
            self.live.clear_view.before_key_capture()
        final, _capture = self._capture(
            runner=runner, decision=decision,
            callback=self.capture_final, phase="final_clear_view")
        decision_at = self.decision_time()
        initial = runner.initial_key_evidence_dir
        if initial is None:
            raise ValueError("released insertion lacks its initial key perception")
        with timed("trial.final_label", phase="trial",
                   attempt_id=attempt.attempt_id):
            runner.prepare_released_insertion_label(
                initial_key_bundle=initial,
                preinsert_bundle=self._preinsert_bundle,
                final_bundle=final, metric_record_path=metric,
                release_record_path=release_path,
                backend=self.backend,
                decision_timestamp_s=decision_at,
                label_mode=self.label_mode,
                manual_annotation=request_annotation(
                    mode=self.label_mode, provider=self.manual_label_provider,
                    stage="insertion_success",
                    evidence_dirs=(initial, final),
                    attempt_id=attempt.attempt_id,
                    candidate_id=attempt.candidate_id),
                **dict(self.final_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "transfer_execution_gate_required": self.transfer_and_observe,
            "await_guarded_insertion_and_observation": self.insert_and_observe,
        }
