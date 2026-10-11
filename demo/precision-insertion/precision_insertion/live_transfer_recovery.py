"""Return a known held key after misalignment, or supervise an unknown one.

No key pose is inferred from an incomplete/aborted transfer. A completed held
transfer with a visible alignment failure can use the existing socket-aware
held return and release path. A visible slip/miss is not a held return. The
fallback producer must write a supervised-reset record for an unknown key
state after pickup, lift or transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .failed_return_evidence import verify_failed_return_execution
from .live_key_trial import capture_returned_key_for_reset
from .preinsert_checkpoint import _transfer_log
from .session_policy import SessionDecision


@dataclass
class LiveTransferRecoveryHandlers:
    live: object
    recover_to_tabletop: Callable
    state_for_observation: Callable
    key_prompt: str
    capture_parameters: Mapping[str, object]
    landing_parameters: Mapping[str, object]
    planner: object | None = None
    limits: object | None = None
    read_state: Callable | None = None
    record_execution: Callable | None = None
    return_parameters: Mapping[str, object] | None = None
    capture_id_factory: Callable[[], str] = lambda: f"transfer_reset_{uuid4().hex}"
    restore_original: Callable | None = None
    key_planning: object | None = None

    def _recover_missed_grasp(self, runner) -> Path:
        """Use AutoDex's selected-grasp release/reset, then observe the key.

        A visual miss does not establish that the key stayed at its start
        position. The subsequent fresh capture and original-pose restoration
        are part of this same failed attempt.
        """
        selected = getattr(self.key_planning, "latest", None)
        preflight = getattr(selected, "preflight", None)
        attempt = runner.active_attempt
        if (preflight is None or getattr(preflight, "pickup_plan", None) is None or
                getattr(preflight, "trial_scene", None) is None or
                getattr(preflight, "selected_candidate_key", None) !=
                tuple(attempt.candidate_id.split("/")) or
                self.planner is None):
            raise ValueError("missed grasp needs its selected AutoDex pickup plan")
        state = self.live.clear_view.prepare_failed_grasp_for_capture(
            pickup_plan=preflight.pickup_plan, planner=self.planner,
            trial_scene=preflight.trial_scene)
        completed = float(state.sample_timestamp_s)
        if completed <= attempt.events[-1]["timestamp_s"]:
            raise ValueError("missed-grasp clear view must follow the lift verdict")
        path = Path(runner._attempt_dir) / "missed_grasp_recovery.json"
        with path.open("x", encoding="utf-8") as output:
            json.dump({
                "schema": "precision_insertion_missed_grasp_clear_view_v1",
                "attempt_id": attempt.attempt_id,
                "candidate_id": attempt.candidate_id,
                "completed_at_s": completed,
                "clear_view_reached": True,
                "hand_open": True,
                "measured_state": state.to_record(),
                "scope": "robot_clear_view_only_key_landing_requires_fresh_capture",
            }, output, indent=2)
            output.write("\n")
        return path

    def _completed_held_transfer(self, runner) -> bool:
        attempt = runner.active_attempt
        if attempt is None or runner._attempt_dir is None:
            return False
        checkpoint = runner._preinsert_checkpoint
        if (checkpoint is None or
                checkpoint.preinsert_reached is not False or
                checkpoint.reason not in {
                    "measured_hold_pose_outside_limits",
                    "multiview_visible_preinsert_failure"} or
                checkpoint.visual.status not in {
                    "coarse_match", "gross_misalignment"}):
            return False
        path = runner._attempt_dir / "transfer_execution.json"
        if not path.is_file():
            return False
        return _transfer_log(path, attempt)["measurement"] == {
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True}

    def _return_held_key(self, runner, decision: SessionDecision) -> Path:
        if (self.planner is None or self.limits is None or
                not callable(self.read_state) or
                not callable(self.record_execution) or
                not isinstance(self.return_parameters, Mapping)):
            raise ValueError("held-transfer return needs planner, feedback and controller")
        parameters = dict(self.return_parameters)
        held, release, report_path = runner.prepare_failed_transfer_return(
            planner=self.planner, held_state=self.read_state(),
            limits=self.limits, **parameters)
        invoked = False

        def run_planned_return():
            nonlocal invoked
            if invoked:
                raise RuntimeError("held transfer return can run only once")
            invoked = True
            return self.live.clear_view.complete_failed_insertion_return_with_autodex(
                held_plan=held, release_plan=release)

        execution = Path(self.record_execution(
            run_planned_return, report_path, runner, decision)).expanduser().resolve()
        if not invoked:
            raise ValueError("controller recorder skipped the held return")
        attempt = runner.active_attempt
        verified = verify_failed_return_execution(
            path=execution, attempt_id=attempt.attempt_id,
            candidate_id=attempt.candidate_id,
            attempt_started_at_s=attempt.started_at_s,
            session_calibration_sha256=runner.session_sha256,
            catalog_sha256=runner.catalog_sha256)
        if verified["plan_report"]["path"] != str(Path(report_path).resolve()):
            raise ValueError("controller executed another transfer-return plan")
        return execution

    def recover(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "recover_key_then_reobserve"):
            raise ValueError("transfer recovery belongs to another decision")
        automatic = all((self.planner is not None, self.limits is not None,
                         callable(self.read_state),
                         callable(self.record_execution),
                         isinstance(self.return_parameters, Mapping)))
        if runner.active_attempt.labels["grasp_success"] is False:
            recovery_log = self._recover_missed_grasp(runner)
        elif automatic and self._completed_held_transfer(runner):
            recovery_log = self._return_held_key(runner, decision)
        else:
            recovery_log = Path(self.recover_to_tabletop(
                runner, decision)).expanduser().resolve()
        if not recovery_log.is_file():
            raise FileNotFoundError("transfer recovery returned no completion log")
        self.live.clear_view.before_key_capture()
        launch = self.live.cameras.launch
        capture_id = self.capture_id_factory()
        landed = capture_returned_key_for_reset(
            runner=runner, clear_view=self.live.clear_view,
            init_orchestrator=self.live.cameras.init_orchestrator,
            acquisition_metadata_for_request=(
                self.live.cameras.acquisition_metadata_for_request),
            state_for_observation=self.state_for_observation,
            capture_root=launch.capture_root,
            key_evidence_dir=(runner.output_dir / "transfer_reset_key_captures" /
                              capture_id),
            capture_id=capture_id,
            calibrated_camera_ids=set(launch.camera_ids),
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            image_hw=launch.image_hw, key_prompt=self.key_prompt,
            view_limits=launch.view_limits,
            expected_action="recover_key_then_reobserve",
            **dict(self.capture_parameters))
        restoration_logs = ()
        if callable(self.restore_original):
            landed, restoration_logs = self.restore_original(
                runner, decision, landed, launch)
        runner.observe_reset_landing(
            key_observation=landed.observation,
            key_evidence_dir=landed.key_evidence_dir,
            recovery_log_path=recovery_log,
            timestamp_s=landed.observation.acquisition_interval_s[1],
            restoration_evidence_paths=restoration_logs,
            **dict(self.landing_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {"recover_key_then_reobserve": self.recover}
