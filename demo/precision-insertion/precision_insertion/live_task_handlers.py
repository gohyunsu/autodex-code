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
    choose_key_axis_move, choose_xy_move, measure_preinsert_xy,
    plan_and_execute_shift,
    plan_corrected_axial_handoff, rank_lateral_observation_offsets,
)
from .same_trial_visual_feedback import update_held_key_xy
from .assets import AssetPaths
from .capture_persistence import PendingCaptureWrite, begin_capture_write
from .world import validated_frozen_socket_pose


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
    same_trial_max_step_m: float = .003
    same_trial_axis_tolerance_rad: float | None = None
    same_trial_max_tilt_step_rad: float = np.deg2rad(3.)
    same_trial_max_shifts: int | None = None
    same_trial_observation_raise_m: float = 0.
    same_trial_observation_lateral_m: float = 0.
    same_trial_observation_lateral_attempts: int = 1
    same_trial_observation_yaw_rad: float = 0.
    on_preinsert_confirmed: Callable[[str, str], None] | None = None
    on_release_and_clear_completed: Callable[[str, str], None] | None = None
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

    def _request_capture(
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
        return output, capture

    def _capture(
        self, *, runner, decision, callback, phase: str,
    ) -> tuple[Path, RawCameraCapture]:
        output, capture = self._request_capture(
            runner=runner, decision=decision, callback=callback, phase=phase)
        write_raw_camera_capture(capture, output, phase=phase)
        # Downstream checkpoint readers perform the one required disk read.
        return output, capture

    def _capture_deferred(
        self, *, runner, decision, callback, phase: str,
    ) -> tuple[Path, RawCameraCapture, PendingCaptureWrite]:
        """Start persistence while the current pixels enter GPU inference."""
        output, capture = self._request_capture(
            runner=runner, decision=decision, callback=callback, phase=phase)
        return output, capture, begin_capture_write(
            capture, output, phase=phase, writer=write_raw_camera_capture)

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
            if self.on_preinsert_confirmed is not None:
                self.on_preinsert_confirmed(
                    attempt.attempt_id, attempt.candidate_id)
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
        axis_raise = 0.
        observation_raise_attempted = False
        observation_lateral_attempted = False
        observation_lateral_offset = None
        lateral_pending = []
        lateral_attempt_index = 0
        observation_yaw_attempted = False
        xy_shift_count = 0
        previous_observation = None
        previous_motion = None
        previous_camera_id = None
        previous_xy = None
        anchor_observation = None
        anchor_xy_motion = None
        anchor_axis_motion = None
        pending_capture_write = None
        temporal_feedback_enabled = self.xy_estimator == "sam_half_ellipse"
        if temporal_feedback_enabled:
            key_radius_m = float(json.loads(AssetPaths(
                Path(runner.shared_root), runner.mode).task_geometry.read_text(
                    encoding="utf-8"))["key_radius_m"])
            socket_robot = validated_frozen_socket_pose(
                mode=runner.mode, shared_root=runner.shared_root,
                calibration=runner.calibration)
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
                    nominal_key_axis_raise_socket_m=axis_raise,
                    xy_estimator=self.xy_estimator,
                    sam_segmentor=self.sam_segmentor,
                    capture=capture,
                    pending_capture_write=pending_capture_write)
            if pending_capture_write is not None:
                pending_capture_write.finish()
                pending_capture_write = None
            if temporal_feedback_enabled and previous_motion is not None:
                feedback = update_held_key_xy(
                    previous=previous_observation, current=observed,
                    measured_xy_socket_m=previous_motion[
                        "measured_increment_socket_m"],
                    measured_axis_socket_m=previous_motion[
                        "measured_axis_increment_socket_m"],
                    calibration=runner.calibration.record,
                    socket_robot=socket_robot,
                    previous_camera_id=previous_camera_id,
                    previous_xy_socket_m=previous_xy,
                    key_radius_m=key_radius_m,
                    anchor=anchor_observation,
                    cumulative_xy_socket_m=(
                        np.asarray(total) - anchor_xy_motion),
                    cumulative_axis_socket_m=(
                        axis_raise - anchor_axis_motion))
                observed["temporal_feedback"] = feedback
                if (feedback["status"] == "tracked" and
                        observed["alignment"]["status"] !=
                        "diagnostic_metric_xy_correction"):
                    estimate = feedback["fused_xy_socket_m"]
                    observed["alignment"].update({
                        "status": "diagnostic_planar_xy_hypothesis",
                        "reason": "measured_held_motion_consistent_face",
                        "xy_source": "temporal_motion_fused_planar_xy",
                        "inlier_cameras": [feedback["camera_id"]],
                        "tip_socket_m": [*estimate, observed["alignment"][
                            "nominal_tip_z_socket_m"]],
                        "xy_correction_socket_m":
                            [-estimate[0], -estimate[1]],
                        "xy_correction_norm_m": float(
                            np.linalg.norm(estimate)),
                    })
                elif (feedback["status"] == "lost" and
                      observed["alignment"]["status"] !=
                      "diagnostic_metric_xy_correction"):
                    observed["alignment"].update({
                        "status": "abstain",
                        "reason": "temporal_key_motion_inconsistent",
                        "xy_source": "none", "inlier_cameras": [],
                        "tip_socket_m": None,
                        "xy_correction_socket_m": None,
                    })
            choice = choose_xy_move(
                observed["alignment"],
                tolerance_m=self.same_trial_xy_tolerance_m,
                max_step_m=self.same_trial_max_step_m)
            axis_choice = (
                choose_key_axis_move(
                    observed["alignment"],
                    tolerance_rad=self.same_trial_axis_tolerance_rad,
                    max_step_rad=self.same_trial_max_tilt_step_rad)
                if self.same_trial_axis_tolerance_rad is not None else
                {"status": "axis_disabled"})
            if (choice["status"] == "not_metric_multiview" and
                    predicted_residual is not None):
                # A new visual association is useful for correcting the
                # previous estimate, but it is not required to finish a move
                # that was already grounded metrically.  The completed FR3
                # displacement is measured after every held-key command, so
                # propagate the last metric residual through that motion when
                # the next photograph cannot identify the same face.  This
                # keeps the loop moving toward its existing target without
                # inventing another image measurement or adding a
                # perception-only stop condition.
                predicted_error = float(np.linalg.norm(predicted_residual))
                step_fraction = min(1., self.same_trial_max_step_m /
                                    max(predicted_error, 1e-12))
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
            lateral_viewpoints = None
            if (choice["status"] == "not_metric_multiview" and
                    predicted_residual is None and
                    self.same_trial_observation_lateral_m > 0 and
                    not observation_lateral_attempted):
                lateral_viewpoints = rank_lateral_observation_offsets(
                    runner=runner, current_xy_socket_m=total,
                    axis_raise_m=axis_raise,
                    component_m=self.same_trial_observation_lateral_m)
                lateral_pending = list(lateral_viewpoints[
                    :self.same_trial_observation_lateral_attempts])
            report = folder / "observations" / f"{index:03d}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            with report.open("x", encoding="utf-8") as stream:
                json.dump({**observed, "choice": choice,
                           "axis_choice": axis_choice,
                           **({"lateral_viewpoint_ranking": lateral_viewpoints}
                              if lateral_viewpoints is not None else {})}, stream,
                          indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            observations.append(report)
            if axis_choice["status"] == "tilt":
                # Correct orientation first. Tilting about the observed key
                # tip leaves its lateral position unchanged; the mandatory
                # recapture then supplies the XY correction for that new axis.
                current_residual = np.asarray(
                    axis_choice["tip_socket_m"][:2], dtype=float)
                try:
                    with timed("trial.same_trial_axis_motion", phase="trial",
                               kind="motion", attempt_id=attempt.attempt_id):
                        tilted = plan_and_execute_shift(
                            runner=runner, planner=self.planner,
                            adapter=self.transfer_adapter,
                            read_state=self.read_state,
                            motion_interlock=self.motion_interlock,
                            limits=self.transfer_limits,
                            path_limits=self.path_limits,
                            increment_socket_m=(0., 0.),
                            residual_socket_m=current_residual,
                            residual_source="current_metric_tip_and_axis",
                            current_axis_raise_socket_m=axis_raise,
                            tilt_rotvec_socket_rad=axis_choice[
                                "tilt_rotvec_socket_rad"],
                            observed_tip_socket_m=axis_choice["tip_socket_m"],
                            observed_axis_socket=axis_choice[
                                "observed_axis_socket"],
                            index=index, action_tag="axis_alignment")
                except HeldXYPlanUnavailable as exc:
                    runner.record_failure(
                        "preinsert_path_unavailable",
                        timestamp_s=self.decision_time(),
                        evidence_refs={
                            "axis_alignment_plan_failure": str(exc.report_path),
                            "preinsert_snapshot": str(
                                self._preinsert_bundle / "manifest.json"),
                        })
                    return
                shifts.append(tilted)
                motion = json.loads(tilted.read_text(encoding="utf-8"))
                measured_step = np.asarray(
                    motion["measured_increment_socket_m"], dtype=float)
                total = (np.asarray(total) + measured_step).tolist()
                axis_raise += motion["measured_axis_increment_socket_m"]
                predicted_residual = (current_residual + measured_step).tolist()
                current = runner.current_decision()
                raw, capture, pending_capture_write = self._capture_deferred(
                    runner=runner, decision=current,
                    callback=self.capture_preinsert, phase="preinsert")
                self._preinsert_bundle = raw
                continue
            returned_lateral_this_iteration = False
            if observation_lateral_offset is not None:
                metric = choice["status"] in {"aligned", "shift"}
                current_residual = (
                    [-value for value in observed["alignment"][
                        "xy_correction_socket_m"]] if metric else total)
                source = ("current_metric_multiview" if metric else
                          "nominal_unobserved_lateral_view")
                try:
                    with timed("trial.same_trial_observation_lateral_return",
                               phase="trial", kind="motion",
                               attempt_id=attempt.attempt_id):
                        returned = plan_and_execute_shift(
                            runner=runner, planner=self.planner,
                            adapter=self.transfer_adapter,
                            read_state=self.read_state,
                            motion_interlock=self.motion_interlock,
                            limits=self.transfer_limits,
                            path_limits=self.path_limits,
                            increment_socket_m=-observation_lateral_offset,
                            residual_socket_m=current_residual,
                            residual_source=source,
                            current_axis_raise_socket_m=axis_raise,
                            index=index)
                except HeldXYPlanUnavailable as exc:
                    runner.record_failure(
                        "preinsert_path_unavailable",
                        timestamp_s=self.decision_time(),
                        evidence_refs={
                            "observation_return_plan_failure": str(exc.report_path),
                            "preinsert_snapshot": str(
                                self._preinsert_bundle / "manifest.json"),
                        })
                    return
                shifts.append(returned)
                motion = json.loads(returned.read_text(encoding="utf-8"))
                measured_return = np.asarray(
                    motion["measured_increment_socket_m"], float)
                total = (np.asarray(total) + measured_return).tolist()
                axis_raise += motion["measured_axis_increment_socket_m"]
                predicted_residual = (
                    (np.asarray(current_residual) + measured_return).tolist()
                    if metric else None)
                observation_lateral_offset = None
                returned_lateral_this_iteration = True
                if metric:
                    predicted_error = float(np.linalg.norm(predicted_residual))
                    step_fraction = min(1., self.same_trial_max_step_m /
                                        max(predicted_error, 1e-12))
                    choice = {
                        "status": ("aligned" if predicted_error <=
                                   self.same_trial_xy_tolerance_m else "shift"),
                        "xy_error_m": predicted_error,
                        "increment_socket_m": (
                            [0., 0.] if predicted_error <=
                            self.same_trial_xy_tolerance_m else
                            (-np.asarray(predicted_residual) *
                             step_fraction).tolist()),
                        "source": "lateral_metric_plus_completed_return",
                    }
                elif lateral_pending:
                    # The first planned viewpoint was still uninformative.
                    # Try the next ranked view now, without re-inferencing
                    # the unchanged raised hold after the return.
                    lateral_viewpoints = lateral_pending
            if (choice["status"] == "not_metric_multiview" and
                    predicted_residual is None and
                    self.same_trial_observation_raise_m > 0 and
                    not observation_raise_attempted):
                observation_raise_attempted = True
                try:
                    with timed("trial.same_trial_observation_raise",
                               phase="trial", kind="motion",
                               attempt_id=attempt.attempt_id):
                        raised = plan_and_execute_shift(
                            runner=runner, planner=self.planner,
                            adapter=self.transfer_adapter,
                            read_state=self.read_state,
                            motion_interlock=self.motion_interlock,
                            limits=self.transfer_limits,
                            path_limits=self.path_limits,
                            increment_socket_m=(0., 0.),
                            residual_socket_m=(0., 0.),
                            residual_source="nominal_unobserved",
                            axis_increment_socket_m=(
                                self.same_trial_observation_raise_m),
                            current_axis_raise_socket_m=0., index=index)
                except HeldXYPlanUnavailable:
                    # The optional viewpoint was unreachable; the measured
                    # original preinsert hold remains available for insertion.
                    pass
                else:
                    shifts.append(raised)
                    motion = json.loads(raised.read_text(encoding="utf-8"))
                    axis_raise = motion["measured_axis_increment_socket_m"]
                    total = [a + b for a, b in zip(
                        total, motion["measured_increment_socket_m"])]
                    current = runner.current_decision()
                    raw, capture, pending_capture_write = self._capture_deferred(
                        runner=runner, decision=current,
                        callback=self.capture_preinsert, phase="preinsert")
                    self._preinsert_bundle = raw
                    continue
            if lateral_viewpoints is not None:
                observation_lateral_attempted = True
                while lateral_pending:
                    viewpoint = lateral_pending.pop(0)
                    rank_index = lateral_attempt_index
                    lateral_attempt_index += 1
                    offset = np.asarray(viewpoint["offset_socket_m"], float)
                    try:
                        with timed("trial.same_trial_observation_lateral",
                                   phase="trial", kind="motion",
                                   attempt_id=attempt.attempt_id):
                            shifted = plan_and_execute_shift(
                                runner=runner, planner=self.planner,
                                adapter=self.transfer_adapter,
                                read_state=self.read_state,
                                motion_interlock=self.motion_interlock,
                                limits=self.transfer_limits,
                                path_limits=self.path_limits,
                                increment_socket_m=offset,
                                residual_socket_m=total,
                                residual_source="nominal_observation_view",
                                current_axis_raise_socket_m=axis_raise,
                                index=index,
                                action_tag=f"lateral_{rank_index}")
                    except HeldXYPlanUnavailable:
                        continue
                    else:
                        shifts.append(shifted)
                        motion = json.loads(shifted.read_text(encoding="utf-8"))
                        observation_lateral_offset = np.asarray(
                            motion["measured_increment_socket_m"], float)
                        total = (np.asarray(total) +
                                 observation_lateral_offset).tolist()
                        axis_raise += motion["measured_axis_increment_socket_m"]
                        current = runner.current_decision()
                        raw, capture, pending_capture_write = self._capture_deferred(
                            runner=runner, decision=current,
                            callback=self.capture_preinsert, phase="preinsert")
                        self._preinsert_bundle = raw
                        break
                if observation_lateral_offset is not None:
                    continue
            if (choice["status"] == "not_metric_multiview" and
                    self.same_trial_observation_yaw_rad != 0. and
                    not observation_yaw_attempted):
                observation_yaw_attempted = True
                current_residual = (total if predicted_residual is None
                                    else predicted_residual)
                try:
                    with timed("trial.same_trial_observation_yaw",
                               phase="trial", kind="motion",
                               attempt_id=attempt.attempt_id):
                        turned = plan_and_execute_shift(
                            runner=runner, planner=self.planner,
                            adapter=self.transfer_adapter,
                            read_state=self.read_state,
                            motion_interlock=self.motion_interlock,
                            limits=self.transfer_limits,
                            path_limits=self.path_limits,
                            increment_socket_m=(0., 0.),
                            residual_socket_m=current_residual,
                            residual_source=(
                                "nominal_unobserved" if predicted_residual is None
                                else "last_metric_plus_completed_held_motion"),
                            axis_increment_socket_m=0.,
                            current_axis_raise_socket_m=axis_raise,
                            yaw_increment_socket_rad=(
                                self.same_trial_observation_yaw_rad),
                            index=index)
                except HeldXYPlanUnavailable:
                    # A different view was not planned; keep the original
                    # insertion route available from the current hold.
                    pass
                else:
                    shifts.append(turned)
                    motion = json.loads(turned.read_text(encoding="utf-8"))
                    total = [a + b for a, b in zip(
                        total, motion["measured_increment_socket_m"])]
                    if predicted_residual is not None:
                        predicted_residual = [a + b for a, b in zip(
                            predicted_residual,
                            motion["measured_increment_socket_m"])]
                    axis_raise += motion["measured_axis_increment_socket_m"]
                    current = runner.current_decision()
                    raw, capture, pending_capture_write = self._capture_deferred(
                        runner=runner, decision=current,
                        callback=self.capture_preinsert, phase="preinsert")
                    self._preinsert_bundle = raw
                    continue
            if choice["status"] == "aligned":
                if choice.get("source") in {
                        "last_metric_plus_completed_held_motion",
                        "lateral_metric_plus_completed_return"}:
                    final_residual = predicted_residual
                    residual_source = choice["source"]
                else:
                    final_residual = [-value for value in observed[
                        "alignment"]["xy_correction_socket_m"]]
                    residual_source = choice.get(
                        "source", "current_metric_multiview")
                break
            if choice["status"] == "not_metric_multiview":
                # No XY observation was available. Replan from the measured
                # hold even in this case: the post-lift axial archive starts
                # at the planned transfer endpoint, not necessarily here.
                final_residual = (total if predicted_residual is None
                                  else predicted_residual)
                residual_source = (
                    ("nominal_unobserved" if total == [0., 0.] else
                     "nominal_plus_observation_motion")
                    if predicted_residual is None else
                    "last_metric_plus_completed_held_motion")
                break
            if (self.same_trial_max_shifts is not None and
                    xy_shift_count >= self.same_trial_max_shifts):
                raise RuntimeError(
                    "same-trial XY did not reach its configured target")
            step = choice["increment_socket_m"]
            current_residual = (
                predicted_residual if choice.get("source") in {
                    "last_metric_plus_completed_held_motion",
                    "lateral_metric_plus_completed_return"} else
                [-value for value in observed["alignment"][
                    "xy_correction_socket_m"]])
            with timed("trial.same_trial_xy_motion", phase="trial",
                       kind="motion", attempt_id=attempt.attempt_id):
                action = ("post_lateral_correction"
                          if returned_lateral_this_iteration else "")
                # If one XY step reaches the observed target, return from the
                # temporary observation height in the same planned motion.
                # This shortens the episode and avoids the independently
                # planned full-height return that failed in saved-state replay.
                combined_return = (axis_raise > 0. and
                                   np.linalg.norm(step) >=
                                   choice["xy_error_m"] - 1e-9)
                routes = (
                    [(-axis_raise, f"{action}_xy_and_axis_return".strip("_")),
                     (0., f"{action}_xy_only".strip("_"))]
                    if combined_return else [(0., action)])
                failure = None
                shift = None
                for axis_step, route_tag in routes:
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
                            axis_increment_socket_m=axis_step,
                            current_axis_raise_socket_m=axis_raise,
                            observed_tip_socket_m=(
                                axis_choice.get("tip_socket_m") if
                                axis_choice["status"] == "axis_aligned" else
                                None),
                            observed_axis_socket=(
                                axis_choice.get("observed_axis_socket") if
                                axis_choice["status"] == "axis_aligned" else
                                None),
                            index=index, action_tag=route_tag)
                    except HeldXYPlanUnavailable as exc:
                        failure = exc
                    else:
                        break
                if shift is None:
                    runner.record_failure(
                        "preinsert_path_unavailable",
                        timestamp_s=self.decision_time(),
                        evidence_refs={
                            "corrected_xy_plan_failure": str(
                                failure.report_path),
                            "preinsert_snapshot": str(
                                self._preinsert_bundle / "manifest.json"),
                        })
                    return
            shifts.append(shift)
            xy_shift_count += 1
            shift_result = json.loads(shift.read_text(encoding="utf-8"))
            measured_step = shift_result["measured_increment_socket_m"]
            total = [a + b for a, b in zip(total, measured_step)]
            axis_raise += shift_result["measured_axis_increment_socket_m"]
            predicted_residual = [a + b for a, b in zip(
                current_residual, measured_step)]
            if temporal_feedback_enabled:
                if anchor_observation is None:
                    anchor_observation = observed
                    anchor_xy_motion = (np.asarray(total) -
                                        np.asarray(measured_step))
                    anchor_axis_motion = (axis_raise - shift_result[
                        "measured_axis_increment_socket_m"])
                correction = observed["alignment"].get(
                    "xy_correction_socket_m")
                if correction is not None:
                    # A fresh image-grounded point becomes the next temporal
                    # reference.  This includes a coherently tracked point.
                    previous_observation = observed
                    previous_motion = shift_result
                    previous_camera_id = (observed["alignment"].get(
                        "inlier_cameras") or [None])[0]
                    previous_xy = [-value for value in correction]
                elif previous_motion is not None:
                    # The current image did not identify a key end, but the
                    # robot completed another measured continuation of the
                    # last metric target.  Keep that last image association
                    # and accumulate wrist motion from it instead of turning
                    # an absent image point into a zero-valued observation.
                    previous_motion = {
                        **previous_motion,
                        "measured_increment_socket_m": (
                            np.asarray(previous_motion[
                                "measured_increment_socket_m"], float) +
                            np.asarray(measured_step, float)).tolist(),
                        "measured_axis_increment_socket_m": float(
                            previous_motion[
                                "measured_axis_increment_socket_m"] +
                            shift_result[
                                "measured_axis_increment_socket_m"]),
                    }
            current = runner.current_decision()
            raw, capture, pending_capture_write = self._capture_deferred(
                runner=runner, decision=current,
                callback=self.capture_preinsert, phase="preinsert")
            self._preinsert_bundle = raw
        if axis_raise != 0.:
            with timed("trial.same_trial_observation_return",
                       phase="trial", kind="motion",
                       attempt_id=attempt.attempt_id):
                try:
                    returned = plan_and_execute_shift(
                        runner=runner, planner=self.planner,
                        adapter=self.transfer_adapter,
                        read_state=self.read_state,
                        motion_interlock=self.motion_interlock,
                        limits=self.transfer_limits,
                        path_limits=self.path_limits,
                        increment_socket_m=(0., 0.),
                        residual_socket_m=final_residual,
                        residual_source=residual_source,
                        axis_increment_socket_m=-axis_raise,
                        current_axis_raise_socket_m=axis_raise,
                        index=index + 1)
                except HeldXYPlanUnavailable as exc:
                    runner.record_failure(
                        "preinsert_path_unavailable",
                        timestamp_s=self.decision_time(),
                        evidence_refs={
                            "observation_return_plan_failure": str(exc.report_path),
                            "preinsert_snapshot": str(
                                self._preinsert_bundle / "manifest.json"),
                        })
                    return
            shifts.append(returned)
            motion = json.loads(returned.read_text(encoding="utf-8"))
            total = [a + b for a, b in zip(
                total, motion["measured_increment_socket_m"])]
            final_residual = [a + b for a, b in zip(
                final_residual, motion["measured_increment_socket_m"])]
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
        if self.on_release_and_clear_completed is not None:
            self.on_release_and_clear_completed(
                attempt.attempt_id, attempt.candidate_id)
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
