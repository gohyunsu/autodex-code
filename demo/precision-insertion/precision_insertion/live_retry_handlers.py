"""Observe failed insertion and save XY advice for the next pickup.

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
from .live_robot_state import LiveRobotState
from .observer import LabeledFrame
from .raw_camera_capture import RawCameraCapture
from .session_policy import SessionDecision


@dataclass
class LiveGroundedRetryHandlers:
    """Diagnose a failed insertion for the next physical attempt.

    The held key is withdrawn once, photographed, then returned to the table.
    No same-attempt XY movement or second insertion handler is exposed.
    """

    live: object
    backend: object
    state_for_capture: Callable[[RawCameraCapture], LiveRobotState]
    withdraw_guarded: Callable
    capture_withdrawn: Callable
    retry_limits: object
    alignment_limits: object
    diagnostic_parameters: Mapping[str, object]
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time
    _withdrawn: tuple[str, LiveRobotState, Path] | None = field(
        default=None, init=False)

    def withdrawn_for_return(self):
        """Expose a completed, still-stationary withdrawal to held return.

        The return handler checks that the robot has not moved since then.
        """
        return self._withdrawn

    def _require(self, runner, decision: SessionDecision, action: str) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action != action or
                runner.current_decision() != decision or
                runner.active_attempt is None or
                runner.active_attempt.candidate_id != decision.candidate_id):
            raise ValueError(f"{action} is not this session's held attempt")
        if self.enable_robot_motion is not True:
            raise PermissionError("failed-insertion withdrawal is not enabled")

    @staticmethod
    def _report(runner, folder: str, index: int) -> Path:
        return Path(runner._attempt_dir) / folder / f"{index:03d}" / "report.json"

    def _unavailable(self, runner, reason: str, report: Path) -> None:
        runner.record_xy_retry_unavailable(
            reason=reason, retry_assessment_path=report,
            timestamp_s=self.decision_time())

    def assess_for_next_trial(self, runner, decision: SessionDecision) -> None:
        self._require(runner, decision,
                      "guarded_withdrawal_then_xy_assessment")
        if runner._postlift_report_path is None:
            raise ValueError("retry needs this attempt's saved post-lift plan")
        withdrawal = self.withdraw_guarded(runner, decision)
        if (not isinstance(withdrawal, tuple) or len(withdrawal) != 2 or
                not isinstance(withdrawal[0], LiveRobotState)):
            raise TypeError("guarded withdrawal must return measured state and log")
        withdrawal_path = Path(withdrawal[1]).expanduser().resolve()
        completed = float(json.loads(withdrawal_path.read_text(
            encoding="utf-8"))["completed_at_s"])
        self._withdrawn = (runner.active_attempt.attempt_id,
                           withdrawal[0], withdrawal_path)
        capture = self.capture_withdrawn(self.live, runner, decision)
        if not isinstance(capture, RawCameraCapture):
            raise TypeError("withdrawn hold needs same-session camera frames")
        minimum_views = 1 if runner.mode.family == "cylinder" else 2
        if (len(capture.images_bgr) < minimum_views or
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
            square_translation_only=(runner.mode.family == "square"),
            **dict(self.diagnostic_parameters))
        diagnostic_path = self._report(
            runner, "xy_retry_assessments", diagnostic_index)
        if diagnostic.status != "diagnostic_metric_xy_correction":
            self._unavailable(runner, "vlm_abstain", diagnostic_path)
            return
        runner.record_next_trial_xy_advice(
            diagnostic=diagnostic, diagnostic_report_path=diagnostic_path,
            timestamp_s=self.decision_time())
        # No lateral command or guarded reinsertion belongs to this attempt.
        # The held-return handler now releases the key; a fresh pickup consumes
        # the saved candidate-specific wrist correction.

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "guarded_withdrawal_then_xy_assessment": self.assess_for_next_trial,
        }
