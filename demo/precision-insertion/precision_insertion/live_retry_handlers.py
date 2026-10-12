"""Replay the saved pre-insertion snapshot into next-attempt XY advice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping

from .action_loop import ActionHandler
from .session_policy import SessionDecision


@dataclass
class LiveGroundedRetryHandlers:
    """Diagnose failure without moving or recapturing the released key.

    Legacy adapter fields remain accepted so an existing robot-PC runtime
    configuration can upgrade without a simultaneous configuration rewrite;
    this action deliberately never calls them.
    """

    live: object
    backend: object
    retry_limits: object
    alignment_limits: object
    diagnostic_parameters: Mapping[str, object]
    state_for_capture: Callable | None = None
    withdraw_guarded: Callable | None = None
    capture_withdrawn: Callable | None = None
    enable_robot_motion: bool = False
    decision_time: Callable[[], float] = time.time

    def _require(self, runner, decision: SessionDecision, action: str) -> None:
        if (runner is not self.live.opened.runner or
                not isinstance(decision, SessionDecision) or
                decision.action != action or
                runner.current_decision() != decision or
                runner.active_attempt is None or
                runner.active_attempt.candidate_id != decision.candidate_id):
            raise ValueError(f"{action} is not this session's held attempt")

    @staticmethod
    def _report(runner, folder: str, index: int) -> Path:
        return Path(runner._attempt_dir) / folder / f"{index:03d}" / "report.json"

    def _unavailable(self, runner, reason: str, report: Path) -> None:
        runner.record_xy_retry_unavailable(
            reason=reason, retry_assessment_path=report,
            timestamp_s=self.decision_time())

    def assess_for_next_trial(self, runner, decision: SessionDecision) -> None:
        self._require(runner, decision, "assess_saved_preinsert_xy")
        if runner._postlift_report_path is None:
            raise ValueError("retry needs this attempt's saved post-lift plan")
        diagnostic_index = runner._retry_assessment_index
        diagnostic = runner.prepare_saved_preinsert_xy_diagnostic(
            backend=self.backend,
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
        # A fresh pickup in the next attempt consumes this cumulative update.

    def handlers(self) -> dict[str, ActionHandler]:
        return {
            "assess_saved_preinsert_xy": self.assess_for_next_trial,
        }
