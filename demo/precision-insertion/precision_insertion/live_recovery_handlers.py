"""Route a failed held-key insertion back through the same live AutoDex run.

The camera, Franka executor and frozen SessionRunner are shared with the
trial. Only guarded withdrawal and independent controller logging are
injected: this module does not fabricate either physical observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .failed_return_evidence import verify_failed_return_execution
from .live_key_trial import capture_returned_key_for_reset
from .live_robot_state import LiveRobotState
from .session_policy import SessionDecision


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
_LANDING_KEYS = frozenset({
    "max_pose_error_deg", "max_return_center_shift_m",
    "support_tolerance_m", "minimum_rest_socket_clearance_m",
    "minimum_board_edge_clearance_m",
})


@dataclass
class LiveHeldReturnHandlers:
    """One action boundary from failed insertion to observed reset landing.

    ``withdraw_guarded`` must perform/log axial withdrawal and return its
    measured endpoint. ``record_execution`` receives a one-shot blocking
    function for the planned AutoDex held return and must emit the independent
    phase/source record accepted by ``verify_failed_return_execution``.
    The reset key capture itself reuses the session's stock camera context.
    """

    live: object
    planner: object
    limits: object
    state_for_observation: Callable
    withdraw_guarded: Callable
    record_execution: Callable
    key_prompt: str
    capture_parameters: Mapping[str, object]
    return_parameters: Mapping[str, object]
    landing_parameters: Mapping[str, object]
    capture_id_factory: Callable[[], str] = lambda: f"reset_{uuid4().hex}"

    def __post_init__(self) -> None:
        if (not callable(self.state_for_observation) or
                not callable(self.withdraw_guarded) or
                not callable(self.record_execution) or
                not callable(self.capture_id_factory) or
                not isinstance(self.key_prompt, str) or
                not self.key_prompt.strip()):
            raise TypeError("held-return handler needs commissioned callbacks and prompt")
        for name, provided, required, optional in (
                ("capture", self.capture_parameters, _CAPTURE_KEYS,
                 _CAPTURE_OPTIONAL),
                ("return", self.return_parameters, _RETURN_KEYS, frozenset()),
                ("landing", self.landing_parameters, _LANDING_KEYS, frozenset())):
            if not isinstance(provided, Mapping):
                raise TypeError(f"{name} limits must be a mapping")
            missing = required - provided.keys()
            unknown = provided.keys() - required - optional
            if missing or unknown:
                raise ValueError(
                    f"{name} limits missing {sorted(missing)} or unknown "
                    f"{sorted(unknown)}")

    def recover(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action != "recover_held_key_then_reobserve" or
                runner.current_decision() != decision):
            raise ValueError("held return belongs to another session or decision")
        withdrawal = self.withdraw_guarded(runner, decision)
        if (not isinstance(withdrawal, tuple) or len(withdrawal) != 2 or
                not isinstance(withdrawal[0], LiveRobotState)):
            raise TypeError("guarded withdrawal must return measured state and log")
        withdrawn_state, withdrawal_path = withdrawal
        held, release, report_path = runner.prepare_failed_insertion_return(
            planner=self.planner, withdrawn_state=withdrawn_state,
            withdrawal_evidence_path=Path(withdrawal_path),
            limits=self.limits, **dict(self.return_parameters))
        invoked = False

        def run_planned_return():
            nonlocal invoked
            if invoked:
                raise RuntimeError("held-key return motion can run only once")
            invoked = True
            return self.live.clear_view.complete_failed_insertion_return_with_autodex(
                held_plan=held, release_plan=release)

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
            key_evidence_dir=(runner.output_dir / "reset_key_captures" /
                              capture_id),
            capture_id=capture_id,
            calibrated_camera_ids=set(launch.camera_ids),
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            image_hw=launch.image_hw, key_prompt=self.key_prompt,
            view_limits=launch.view_limits,
            **dict(self.capture_parameters))
        runner.observe_reset_landing(
            key_observation=landed.observation,
            key_evidence_dir=landed.key_evidence_dir,
            recovery_log_path=execution_path,
            timestamp_s=landed.observation.acquisition_interval_s[1],
            **dict(self.landing_parameters))

    def handlers(self) -> dict[str, ActionHandler]:
        return {"recover_held_key_then_reobserve": self.recover}
