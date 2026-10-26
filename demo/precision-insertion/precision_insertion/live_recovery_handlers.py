"""Restore a failed attempt's key through the same live AutoDex run.

The active route starts from the key already released after the 20 mm stroke,
perceives it from clear view, and reuses the AutoDex reset/repose machinery to
restore the trial-start tabletop pose and position.  The historical held-key
return implementation remains available internally for recorded artifacts,
but is not exposed by the current action map.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
from autodex.utils.robot_config import FR3_CLEAR_VIEW

from .action_loop import ActionHandler
from .failed_return_evidence import verify_failed_return_execution
from .live_key_trial import capture_returned_key_for_reset
from .live_robot_state import LiveRobotState
from .original_pose_restoration import (
    original_center_release_xy, plan_original_pose_restoration,
)
from .repose_artifacts import write_repose_preflight_artifacts
from .repose_execution_evidence import verify_repose_execution_evidence
from .session_policy import SessionDecision
from .timing_profile import timed


_CAPTURE_KEYS = frozenset({
    "maximum_multiview_center_error_mm",
    "maximum_multiview_angle_error_deg",
    "maximum_socket_mask_overlap_fraction",
    "socket_projection_dilation_px", "minimum_refinement_iou",
    "max_key_state_skew_s", "capture_timeout_s", "image_write_timeout_s",
})
_CAPTURE_OPTIONAL = frozenset({
    "silhouette_iterations", "silhouette_loss_threshold",
    "registered_key_assistance", "request_id_factory",
})
_RETURN_KEYS = frozenset({
    "release_height_m", "minimum_rest_socket_clearance_m",
    "minimum_board_edge_clearance_m", "minimum_release_key_clearance_m",
    "max_withdrawal_state_gap_s", "max_arm_hand_skew_s",
    "max_hand_command_error_raw", "max_arm_velocity_rad_s",
})
_RETURN_OPTIONAL = frozenset({"max_reused_withdrawal_joint_drift_rad"})
_LANDING_KEYS = frozenset({
    "max_pose_error_deg", "max_return_center_shift_m",
    "support_tolerance_m", "minimum_rest_socket_clearance_m",
    "minimum_board_edge_clearance_m",
})


@dataclass
class LiveHeldReturnHandlers:
    """One action boundary from failed insertion to observed reset landing.

    The current action observes the released key and restores it with the
    existing reset/repose planner. A stock force-stop ascent also routes here
    with its saved withdrawal and the stock held-return execution recorder.
    The key captures reuse the session's stock cameras.
    """

    live: object
    planner: object
    limits: object
    state_for_observation: Callable
    key_prompt: str
    capture_parameters: Mapping[str, object]
    return_parameters: Mapping[str, object]
    landing_parameters: Mapping[str, object]
    withdraw_guarded: Callable | None = None
    record_execution: Callable | None = None
    capture_id_factory: Callable[[], str] = lambda: f"reset_{uuid4().hex}"
    read_state: Callable[[], LiveRobotState] | None = None
    reuse_withdrawal: Callable[[], tuple[str, LiveRobotState, Path] | None] | None = None
    execute_restoration: Callable | None = None
    restoration_height_cm: int = 12
    max_restoration_attempts: int = 3

    def __post_init__(self) -> None:
        if (not callable(self.state_for_observation) or
                not callable(self.capture_id_factory) or
                not isinstance(self.key_prompt, str) or
                not self.key_prompt.strip()):
            raise TypeError("held-return handler needs commissioned callbacks and prompt")
        if (self.restoration_height_cm not in (4, 8, 12) or
                type(self.max_restoration_attempts) is not int or
                self.max_restoration_attempts < 1):
            raise ValueError("restoration height/attempt budget is invalid")
        if self.execute_restoration is not None and not callable(
                self.execute_restoration):
            raise TypeError("restoration executor must be callable")
        for name, provided, required, optional in (
                ("capture", self.capture_parameters, _CAPTURE_KEYS,
                 _CAPTURE_OPTIONAL),
                ("return", self.return_parameters, _RETURN_KEYS,
                 _RETURN_OPTIONAL),
                ("landing", self.landing_parameters, _LANDING_KEYS, frozenset())):
            if not isinstance(provided, Mapping):
                raise TypeError(f"{name} limits must be a mapping")
            missing = required - provided.keys()
            unknown = provided.keys() - required - optional
            if missing or unknown:
                raise ValueError(
                    f"{name} limits missing {sorted(missing)} or unknown "
                    f"{sorted(unknown)}")

    def _capture_landing(self, runner, launch, decision):
        capture_id = self.capture_id_factory()
        with timed("recovery.key_capture", phase="recovery", kind="camera",
                   capture_id=capture_id):
            return capture_returned_key_for_reset(
                runner=runner, clear_view=self.live.clear_view,
                init_orchestrator=self.live.cameras.init_orchestrator,
                acquisition_metadata_for_request=(
                    self.live.cameras.acquisition_metadata_for_request),
                state_for_observation=self.state_for_observation,
                capture_root=launch.capture_root,
                key_evidence_dir=(runner.output_dir / "reset_key_captures" /
                                  capture_id),
                capture_id=capture_id,
                calibrated_camera_ids=set(launch.camera_ids),
                intrinsics_full=launch.intrinsics_full,
                extrinsics_full=launch.extrinsics_full,
                image_hw=launch.image_hw, key_prompt=self.key_prompt,
                view_limits=launch.view_limits,
                expected_action=decision.action,
                **dict(self.capture_parameters))

    def restore_until_original(self, runner, decision, landed, launch):
        """Correct a misplaced key inside the same failed physical attempt."""
        restoration_logs = []
        for index in range(self.max_restoration_attempts):
            geometry = runner.reset_landing_geometry(
                key_observation=landed.observation,
                max_pose_error_deg=self.landing_parameters["max_pose_error_deg"])
            if (geometry["classification"]["stem"] ==
                    geometry["target_pose_stem"] and
                    geometry["center_xy_displacement_m"] <=
                    self.landing_parameters["max_return_center_shift_m"]):
                break
            with timed("recovery.restore_attempt", phase="recovery",
                       restoration_index=index):
                restoration_log, completed_at = self._restore_original(
                    runner, decision, landed, index)
            restoration_logs.append(restoration_log)
            landed = self._capture_landing(runner, launch, decision)
            if landed.observation.acquisition_interval_s[0] <= completed_at:
                raise ValueError("restoration key image precedes placement release")
        return landed, tuple(restoration_logs)

    def _restore_original(self, runner, decision, landed, index: int
                          ) -> tuple[Path, float]:
        if not callable(self.execute_restoration):
            raise RuntimeError(
                "original-pose restoration needs a commissioned reset executor")
        geometry = runner.reset_landing_geometry(
            key_observation=landed.observation,
            max_pose_error_deg=self.landing_parameters["max_pose_error_deg"])
        with timed("recovery.restore_plan", phase="recovery", kind="planning",
                   restoration_index=index):
            plan, scene = plan_original_pose_restoration(
                planner=self.planner, shared_root=runner.shared_root,
                mode=runner.mode, calibration=runner.calibration,
                catalog=runner.catalog, key_observation=landed.observation,
                measured_start_q=landed.state.full_q,
                start_q_timestamp_s=landed.state.sample_timestamp_s,
                max_state_skew_s=self.capture_parameters["max_key_state_skew_s"],
                target_pose_stem=geometry["target_pose_stem"],
                target_xy_robot_m=original_center_release_xy(
                    shared_root=runner.shared_root, mode=runner.mode,
                    target_pose_stem=geometry["target_pose_stem"],
                    original_center_xy_robot_m=(
                        geometry["original_center_xy_robot_m"]),
                    key_center_local_m=geometry["key_center_local_m"]),
                release_height_cm=self.restoration_height_cm,
                max_pose_error_deg=self.landing_parameters["max_pose_error_deg"],
                minimum_rest_socket_clearance_m=(
                    self.landing_parameters["minimum_rest_socket_clearance_m"]),
                minimum_board_edge_clearance_m=(
                    self.landing_parameters["minimum_board_edge_clearance_m"]),
                minimum_release_key_clearance_m=(
                    self.return_parameters["minimum_release_key_clearance_m"]),
                retreat_goal_arm_q=np.asarray(FR3_CLEAR_VIEW, dtype=float),
                limits=self.limits,
                preferred_candidate_key=tuple(
                    runner.active_attempt.candidate_id.split("/")))
        if plan.status != "nominal_reset_preflight_pass_drop_unobserved":
            raise RuntimeError(f"original-pose restoration unavailable: {plan.status}")
        folder = Path(runner._attempt_dir) / "reset_restoration" / f"{index:03d}"
        inputs = folder / "inputs"
        inputs.mkdir(parents=True, exist_ok=False)
        key_file = inputs / "key_pose_world.npy"
        state_file = inputs / "live_start_q.npy"
        limits_file = inputs / "limits.json"
        np.save(key_file, landed.observation.pose_world)
        np.save(state_file, landed.state.full_q)
        limits_file.write_text(json.dumps(vars(self.limits), indent=2) + "\n",
                               encoding="utf-8")
        report = write_repose_preflight_artifacts(
            result=plan, trial_scene=scene, output_dir=folder / "preflight",
            source_files={
                "session": runner.output_dir / "frozen_session_calibration.json",
                "catalog": runner.output_dir / "endpoint_catalog.json",
                "key_pose_world": key_file, "live_start_q": state_file,
                "limits": limits_file}) / "report.json"
        with timed("recovery.restore_motion", phase="recovery", kind="motion",
                   restoration_index=index):
            execution = Path(self.execute_restoration(
                runner, decision, plan, report)).expanduser().resolve()
        verified = verify_repose_execution_evidence(
            path=execution, attempt_id=runner.active_attempt.attempt_id,
            attempt_started_at_s=runner.active_attempt.started_at_s,
            preflight_report_path=report,
            preflight_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            session_calibration_sha256=runner.session_sha256,
            selected_seed=plan.selected_seed)
        return execution, float(verified["release_completed_at_s"])

    def recover(self, runner, decision: SessionDecision) -> None:
        if not callable(self.record_execution):
            raise RuntimeError("held return needs an execution recorder")
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action != "recover_held_key_then_reobserve" or
                runner.current_decision() != decision):
            raise ValueError("held return belongs to another session or decision")
        withdrawal = None
        if callable(self.reuse_withdrawal) and callable(self.read_state):
            previous = self.reuse_withdrawal()
            if previous is not None:
                if (not isinstance(previous, tuple) or len(previous) != 3 or
                        previous[0] != runner.active_attempt.attempt_id or
                        not isinstance(previous[1], LiveRobotState)):
                    raise ValueError("reused withdrawal belongs to another attempt")
                current = self.read_state()
                if not isinstance(current, LiveRobotState):
                    raise TypeError("reused withdrawal needs measured robot state")
                drift = self.return_parameters.get(
                    "max_reused_withdrawal_joint_drift_rad", .02)
                if (type(drift) not in (int, float) or drift <= 0 or
                        not np.isfinite(drift) or
                        current.sample_timestamp_s <=
                        previous[1].sample_timestamp_s or
                        np.max(np.abs(current.full_q - previous[1].full_q)) >
                        drift):
                    raise ValueError("robot moved after the reusable withdrawal")
                withdrawal = (current, previous[2])
        if withdrawal is None:
            if not callable(self.withdraw_guarded):
                raise RuntimeError("held return lacks a completed axis withdrawal")
            withdrawal = self.withdraw_guarded(runner, decision)
        if (not isinstance(withdrawal, tuple) or len(withdrawal) != 2 or
                not isinstance(withdrawal[0], LiveRobotState)):
            raise TypeError("guarded withdrawal must return measured state and log")
        withdrawn_state, withdrawal_path = withdrawal
        return_parameters = dict(self.return_parameters)
        return_parameters.pop("max_reused_withdrawal_joint_drift_rad", None)
        held, release, report_path = runner.prepare_failed_insertion_return(
            planner=self.planner, withdrawn_state=withdrawn_state,
            withdrawal_evidence_path=Path(withdrawal_path),
            limits=self.limits, **return_parameters)
        invoked = False

        def run_planned_return(evidence_sink=None):
            nonlocal invoked
            if invoked:
                raise RuntimeError("held-key return motion can run only once")
            invoked = True
            return self.live.clear_view.complete_failed_insertion_return_with_autodex(
                held_plan=held, release_plan=release,
                planner=self.planner,
                **({"evidence_sink": evidence_sink}
                   if evidence_sink is not None else {}))

        execution_path = Path(self.record_execution(
            run_planned_return, report_path, runner, decision)).expanduser()
        if not invoked:
            raise ValueError("controller recorder skipped the physical return")
        attempt = runner.active_attempt
        if attempt is None:
            raise ValueError("held return lost its active attempt")
        verified = verify_failed_return_execution(
            path=execution_path, attempt_id=attempt.attempt_id,
            candidate_id=attempt.candidate_id,
            attempt_started_at_s=attempt.started_at_s,
            session_calibration_sha256=runner.session_sha256,
            catalog_sha256=runner.catalog_sha256)
        if verified["plan_report"]["path"] != str(Path(report_path).resolve()):
            raise ValueError("controller executed another failed-return plan")
        launch = self.live.cameras.launch
        landed = self._capture_landing(runner, launch, decision)
        landed, restoration_logs = self.restore_until_original(
            runner, decision, landed, launch)
        runner.observe_reset_landing(
            key_observation=landed.observation,
            key_evidence_dir=landed.key_evidence_dir,
            recovery_log_path=execution_path,
            timestamp_s=landed.observation.acquisition_interval_s[1],
            restoration_evidence_paths=restoration_logs,
            **dict(self.landing_parameters))

    def recover_released(self, runner, decision: SessionDecision) -> None:
        """Observe the released key, then repose/reorient it to trial start."""
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action != "recover_released_key_then_reobserve" or
                runner.current_decision() != decision or
                runner.active_attempt is None):
            raise ValueError("released-key recovery belongs to another attempt")
        if not callable(self.read_state):
            raise RuntimeError("released-key recovery needs measured clear-view state")
        insertion = next((
            event for event in reversed(runner.active_attempt.events)
            if event["stage"] == "insertion_success"), None)
        if insertion is None or insertion["value"] is not False:
            raise ValueError("released-key recovery requires observed insertion failure")
        refs = insertion["evidence_refs"]
        release_path = Path(refs.get("release_execution", "")).expanduser().resolve()
        if not release_path.is_file():
            raise ValueError("released-key recovery lacks the post-stroke release log")
        measured = self.read_state()
        if not isinstance(measured, LiveRobotState):
            raise TypeError("released-key recovery needs measured robot feedback")
        measured = self.live.clear_view.verify_state(
            measured, phase="released_failure_clear_view")
        verified_at_s = time.time()
        folder = Path(runner._attempt_dir)
        recovery_path = folder / "released_failure_clear_view.json"
        record = {
            "schema": "precision_insertion_released_failure_clear_view_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "candidate_id": runner.active_attempt.candidate_id,
            "completed_at_s": verified_at_s,
            "hand_open": True,
            "clear_view_reached": True,
            "measured_state": measured.to_record(),
            "poststroke_release": {
                "path": str(release_path),
                "sha256": hashlib.sha256(release_path.read_bytes()).hexdigest(),
            },
            "scope": "released_key_ready_for_fresh_perception_and_restoration",
            "robot_ready": False,
        }
        with recovery_path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        launch = self.live.cameras.launch
        landed = self._capture_landing(runner, launch, decision)
        landed, restoration_logs = self.restore_until_original(
            runner, decision, landed, launch)
        runner.observe_reset_landing(
            key_observation=landed.observation,
            key_evidence_dir=landed.key_evidence_dir,
            recovery_log_path=recovery_path,
            timestamp_s=landed.observation.acquisition_interval_s[1],
            restoration_evidence_paths=restoration_logs,
            **dict(self.landing_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "recover_held_key_then_reobserve": self.recover,
            "recover_released_key_then_reobserve": self.recover_released,
        }
