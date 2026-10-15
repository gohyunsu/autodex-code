"""Execute a directed v8 reset and reobserve the landed key on AutoDex.

The robot-PC callback owns physical pickup/lift/place/release. This module
keeps that single execution and the later same-session camera label together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Callable, Mapping
from uuid import uuid4

from .action_loop import ActionHandler
from .live_key_trial import capture_returned_key_for_reset
from .repose_execution_evidence import verify_repose_execution_evidence
from .session_policy import SessionDecision
from .timing_profile import timed


@dataclass
class LiveReposeExecutionHandlers:
    """Bind a commissioned v8 reset callback to fresh landing perception."""

    live: object
    execute_repose: Callable
    state_for_observation: Callable
    key_prompt: str
    capture_parameters: Mapping[str, object]
    landing_parameters: Mapping[str, object]
    attempt_id_factory: Callable[[], str] = lambda: f"repose_{uuid4().hex}"
    capture_id_factory: Callable[[], str] = lambda: f"repose_key_{uuid4().hex}"
    started_at_factory: Callable[[], float] = time.time
    decision_time: Callable[[], float] = time.time
    _execution_path: Path | None = field(default=None, init=False)
    _attempt_id: str | None = field(default=None, init=False)

    def begin(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "repose_execution_gate_required"):
            raise ValueError("repose execution belongs to another session or decision")
        attempt = runner.begin_repose_attempt(
            attempt_id=self.attempt_id_factory(),
            started_at_s=self.started_at_factory())
        plan = runner._repose_preflight
        with timed("repose.motion", phase="repose", kind="motion",
                   attempt_id=attempt.attempt_id):
            path = Path(self.execute_repose(
                runner, decision, plan)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError("repose controller returned no execution log")
        self._execution_path = path
        self._attempt_id = attempt.attempt_id

    def observe(self, runner, decision: SessionDecision) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                runner.current_decision() != decision or
                decision.action != "await_repose_observation" or
                self._execution_path is None or
                runner.active_attempt is None or
                self._attempt_id != runner.active_attempt.attempt_id):
            raise ValueError("repose landing needs this session's completed reset")
        report = Path(runner._repose_report_path).resolve()
        verified = verify_repose_execution_evidence(
            path=self._execution_path, attempt_id=self._attempt_id,
            attempt_started_at_s=runner.active_attempt.started_at_s,
            preflight_report_path=report,
            preflight_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            session_calibration_sha256=runner.session_sha256,
            selected_seed=runner._repose_preflight.selected_seed)
        with timed("repose.clear_view", phase="repose", kind="motion"):
            self.live.clear_view.before_key_capture()
        launch = self.live.cameras.launch
        capture_id = self.capture_id_factory()
        with timed("repose.key_capture", phase="repose", kind="camera",
                   capture_id=capture_id):
            landed = capture_returned_key_for_reset(
                runner=runner, clear_view=self.live.clear_view,
                init_orchestrator=self.live.cameras.init_orchestrator,
                acquisition_metadata_for_request=(
                    self.live.cameras.acquisition_metadata_for_request),
                state_for_observation=self.state_for_observation,
                capture_root=launch.capture_root,
                key_evidence_dir=(runner.output_dir / "repose_key_captures" /
                                  capture_id),
                capture_id=capture_id,
                calibrated_camera_ids=set(launch.camera_ids),
                intrinsics_full=launch.intrinsics_full,
                extrinsics_full=launch.extrinsics_full,
                image_hw=launch.image_hw, key_prompt=self.key_prompt,
                view_limits=launch.view_limits,
                expected_action="await_repose_observation",
                **dict(self.capture_parameters))
        with timed("repose.landing_assessment", phase="repose"):
            runner.observe_repose_landing(
                key_observation=landed.observation,
                key_evidence_dir=landed.key_evidence_dir,
                timestamp_s=self.decision_time(),
                release_completed_at_s=verified["release_completed_at_s"],
                release_evidence_path=self._execution_path,
                **dict(self.landing_parameters))
        self._execution_path = None
        self._attempt_id = None

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "repose_execution_gate_required": self.begin,
            "await_repose_observation": self.observe,
        }
