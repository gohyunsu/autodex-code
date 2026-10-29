"""Connect held transfer and first guarded insertion in one live session.

All planning, execution evidence and visual/metric labels come from the
existing demo modules. Robot controllers and camera acquisition are injected
from the AutoDex PC; this module does not emulate physical motion or sensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
import json
import numpy as np
from pathlib import Path
import time
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .guarded_insertion_execution import execute_bound_guarded_insertion
from .live_robot_state import LiveRobotState
from .labeling import request_annotation
from .initial_visual import initial_visual_source
from .raw_camera_capture import (
    RawCameraCapture, write_raw_camera_capture,
)
from .recoverable_failure import acknowledged_failure
from .session_policy import SessionDecision
from .post_insertion_release import verify_post_insertion_release
from .transfer_execution import execute_bound_held_transfer
from .timing_profile import timed
from .same_trial_xy import (
    CorrectedAxialPlanUnavailable, HeldXYPlanUnavailable,
    choose_xy_move, measure_preinsert_xy,
    plan_and_execute_shift,
    plan_corrected_axial_handoff,
)


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
    planner: object | None = None
    path_limits: object | None = None
    alignment_limits: object | None = None
    same_trial_tip_height_error_m: float | None = None
    xy_estimator: str = "vlm"
    sam_segmentor: object | None = None
    same_trial_xy_tolerance_m: float | None = None
    same_trial_max_shifts: int | None = None
    after_lift_capture_provider: (
        Callable[[], tuple[str, RawCameraCapture] | None] | None) = None
    decision_time: Callable[[], float] = time.time
    _preinsert_bundle: Path | None = field(default=None, init=False)
    _preinsert_attempt_id: str | None = field(default=None, init=False)
    _withdrawn: tuple[str, LiveRobotState, Path] | None = field(
        default=None, init=False)
    _same_trial_handoff: Path | None = field(default=None, init=False)

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
                  capture.capture_id).resolve()
        write_raw_camera_capture(capture, output, phase=phase)
        # Downstream checkpoint readers perform the one required disk read.
        return output, capture

    def transfer_and_observe(self, runner, decision: SessionDecision) -> None:
        attempt = self._require(runner, decision,
                                "transfer_execution_gate_required")
        self._same_trial_handoff = None
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
        # Episode links under _pipeline_runs point to the same file as the
        # canonical attempt directory. Use the path saved by the checkpoint.
        transfer_log = (Path(runner.output_dir) / "attempts" /
                        attempt.attempt_id / "transfer_execution.json").resolve()
        raw, capture = self._capture(
            runner=runner, decision=decision,
            callback=self.capture_preinsert, phase="preinsert")
        lift_checkpoint = getattr(runner, "_lift_checkpoint", None)
        after_lift_raw = getattr(lift_checkpoint, "after_bundle", None)
        if after_lift_raw is None:
            raise ValueError("preinsert VLM needs this attempt's raw after-lift capture")
        recent_lift = (self.after_lift_capture_provider()
                       if self.after_lift_capture_provider is not None else None)
        lift_capture = (recent_lift[1] if recent_lift is not None and
                        recent_lift[0] == attempt.attempt_id else None)
        state = self.state_for_capture(capture)
        if not isinstance(state, LiveRobotState):
            raise TypeError("pre-insertion exposure needs measured robot state")
        with timed("trial.preinsert_label", phase="trial",
                   attempt_id=attempt.attempt_id):
            checkpoint = runner.prepare_observed_preinsert_label(
                raw_bundle=raw, transfer_execution_path=transfer_log,
                joint_sample=state, backend=self.backend,
                live_capture=capture,
                after_lift_live_capture=lift_capture,
                label_mode=self.label_mode,
                **({"allow_held_misalignment": True}
                   if self.same_trial_xy_tolerance_m is not None else {}),
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
            if self.same_trial_xy_tolerance_m is not None:
                self._refine_current_hold(runner, capture=capture)

    def _refine_current_hold(
        self, runner, *, capture: RawCameraCapture | None = None,
    ) -> None:
        """Reobserve after each planned shift before preparing any axial path."""
        if self.planner is None or self.path_limits is None or (
                self.alignment_limits is None or
                self.same_trial_tip_height_error_m is None or
                (self.same_trial_max_shifts is not None and
                 self.same_trial_max_shifts < 1)):
            raise ValueError("same-trial XY needs the session planner and limits")
        attempt = runner.active_attempt
        folder = Path(runner._attempt_dir) / "same_trial_xy"
        observations: list[Path] = []
        shifts: list[Path] = []
        total = [0., 0.]
        final_residual = None
        predicted_residual = None
        residual_source = None
        for index in count():
            with timed("trial.same_trial_xy_measure", phase="trial",
                       kind="vlm", attempt_id=attempt.attempt_id):
                observed = measure_preinsert_xy(
                    runner=runner, raw_bundle=self._preinsert_bundle,
                    backend=self.backend,
                    alignment_limits=self.alignment_limits, index=index,
                    nominal_tip_height_error_m=(
                        self.same_trial_tip_height_error_m),
                    nominal_key_xy_shift_socket_m=total,
                    xy_estimator=self.xy_estimator,
                    sam_segmentor=self.sam_segmentor,
                    capture=capture)
            choice = choose_xy_move(
                observed["alignment"],
                tolerance_m=self.same_trial_xy_tolerance_m)
            if (choice["status"] == "not_metric_multiview" and
                    predicted_residual is not None):
                predicted_error = float(np.linalg.norm(predicted_residual))
                step_fraction = min(1., .010 / max(predicted_error, 1e-12))
                choice = {
                    "status": ("aligned" if predicted_error <=
                               self.same_trial_xy_tolerance_m else "shift"),
                    "xy_error_m": predicted_error,
                    "increment_socket_m": (
                        [0., 0.] if predicted_error <=
                        self.same_trial_xy_tolerance_m else
                        (-np.asarray(predicted_residual) *
                         step_fraction).tolist()),
                    "source": "last_metric_plus_completed_held_motion",
                }
            report = folder / "observations" / f"{index:03d}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as stream:
                json.dump({**observed, "choice": choice}, stream,
                          indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            observations.append(report)
            if choice["status"] == "aligned":
                if choice.get("source") == "last_metric_plus_completed_held_motion":
                    final_residual = predicted_residual
                    residual_source = choice["source"]
                else:
                    final_residual = [-value for value in observed[
                        "alignment"]["xy_correction_socket_m"]]
                    residual_source = "current_metric_multiview"
                break
            if choice["status"] == "not_metric_multiview":
                # No XY observation was available. Replan from the measured
                # hold even in this case: the post-lift axial archive starts
                # at the planned transfer endpoint, not necessarily here.
                final_residual = ([0., 0.] if predicted_residual is None
                                  else predicted_residual)
                residual_source = ("nominal_unobserved" if
                                   predicted_residual is None else
                                   "last_metric_plus_completed_held_motion")
                break
            if (self.same_trial_max_shifts is not None and
                    index == self.same_trial_max_shifts):
                raise RuntimeError(
                    "same-trial XY did not reach its configured target")
            step = choice["increment_socket_m"]
            current_residual = (
                predicted_residual if choice.get("source") ==
                "last_metric_plus_completed_held_motion" else
                [-value for value in observed["alignment"][
                    "xy_correction_socket_m"]])
            with timed("trial.same_trial_xy_motion", phase="trial",
                       kind="motion", attempt_id=attempt.attempt_id):
                try:
                    shift = plan_and_execute_shift(
                        runner=runner, planner=self.planner,
                        adapter=self.transfer_adapter,
                        read_state=self.read_state,
                        motion_interlock=self.motion_interlock,
                        limits=self.transfer_limits,
                        path_limits=self.path_limits,
                        increment_socket_m=step,
                        residual_socket_m=current_residual,
                        residual_source=(choice.get("source") or
                                         "current_metric_multiview"),
                        index=index)
                except HeldXYPlanUnavailable as exc:
                    runner.record_failure(
                        "preinsert_path_unavailable",
                        timestamp_s=self.decision_time(),
                        evidence_refs={
                            "corrected_xy_plan_failure": str(exc.report_path),
                            "preinsert_snapshot": str(
                                self._preinsert_bundle / "manifest.json"),
                        })
                    return
            shifts.append(shift)
            shift_result = json.loads(shift.read_text(encoding="utf-8"))
            measured_step = shift_result["measured_increment_socket_m"]
            total = [a + b for a, b in zip(total, measured_step)]
            predicted_residual = [a + b for a, b in zip(
                current_residual, measured_step)]
            current = runner.current_decision()
            raw, capture = self._capture(
                runner=runner, decision=current,
                callback=self.capture_preinsert, phase="preinsert")
            self._preinsert_bundle = raw
        if final_residual is not None:
            with timed("trial.same_trial_axial_replan", phase="planning",
                       kind="planning", attempt_id=attempt.attempt_id):
                try:
                    self._same_trial_handoff = plan_corrected_axial_handoff(
                        runner=runner, planner=self.planner,
                        path_limits=self.path_limits,
                        read_state=self.read_state,
                        total_xy_m=total, residual_xy_m=final_residual,
                        residual_source=residual_source,
                        observations=observations,
                        shifts=shifts,
                        final_preinsert_bundle=self._preinsert_bundle)
                except CorrectedAxialPlanUnavailable as exc:
                    # The held key has not undergone an insertion stroke.
                    # Preserve that distinction and the exact failed plan;
                    # the session decision cannot proceed to contact motion.
                    runner.record_failure(
                        "preinsert_path_unavailable",
                        timestamp_s=self.decision_time(),
                        evidence_refs={
                            "corrected_axial_plan_failure": str(exc.report_path),
                            "preinsert_snapshot": str(
                                self._preinsert_bundle / "manifest.json"),
                        })

    def insert_and_observe(self, runner, decision: SessionDecision) -> None:
        attempt = self._require(runner, decision,
                                "await_guarded_insertion_and_observation")
        if (self._preinsert_attempt_id != attempt.attempt_id or
                self._preinsert_bundle is None):
            raise ValueError("guarded insertion lacks this attempt's pre-insertion frames")
        with timed("trial.insertion_handoff", phase="planning", kind="planning",
                   attempt_id=attempt.attempt_id):
            handoff = (self._same_trial_handoff or
                       runner.prepare_guarded_axial_handoff(
                           measured_start=self.read_state(),
                           decision_timestamp_s=self.decision_time(),
                           **dict(self.handoff_parameters)))
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
        # The release callback already measured the open-hand clear-view
        # endpoint. Capture the final image without a second pose gate.
        final, final_capture = self._capture(
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
                final_capture=final_capture,
                backend=self.backend,
                decision_timestamp_s=decision_at,
                label_mode=self.label_mode,
                manual_annotation=request_annotation(
                    mode=self.label_mode, provider=self.manual_label_provider,
                    stage="insertion_success",
                    evidence_dirs=(
                        (initial_visual_source(initial)[0], final)
                        if self.label_mode == "manual" else (final,)),
                    attempt_id=attempt.attempt_id,
                    candidate_id=attempt.candidate_id),
                **dict(self.final_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "transfer_execution_gate_required": self.transfer_and_observe,
            "await_guarded_insertion_and_observation": self.insert_and_observe,
        }
