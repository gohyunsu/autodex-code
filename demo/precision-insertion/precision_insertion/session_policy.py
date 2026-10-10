"""Fail-closed next-step policy for the independent insertion demo.

This module joins existing *observations* and planning reports; it never
captures a camera frame, treats a preflight as a motor command, or authorizes
contact motion. A live runner must satisfy the named evidence gate before it
can execute the next phase and must save a fresh observation afterward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .records import AttemptRecord


@dataclass(frozen=True)
class SessionDecision:
    action: str
    reason: str
    candidate_id: str | None
    required_before_motion: tuple[str, ...] = ()

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_session_decision_v1",
            "action": self.action,
            "reason": self.reason,
            "candidate_id": self.candidate_id,
            "required_before_motion": list(self.required_before_motion),
            "scope": "decision_only_not_robot_motion_authorization",
            "robot_ready": False,
        }


def decide_after_trial_preflight(report: Mapping) -> SessionDecision:
    """Route a saved fresh-key v8 planning report without moving the robot."""
    if report.get("schema") != "precision_insertion_trial_preflight_v2":
        raise ValueError("expected a current fresh-key trial preflight report")
    status = report.get("status")
    selected = report.get("selected_candidate_key")
    if status == "sampled_planning_pass":
        plan = report.get("insertion_plan")
        if (not isinstance(selected, list) or len(selected) != 3 or
                not all(isinstance(part, str) and part for part in selected) or
                not isinstance(plan, Mapping) or
                plan.get("sampled_planning_pass") is not True):
            raise ValueError("selected grasp lacks its matching insertion preflight")
        return SessionDecision(
            "execution_gate_required", "nominal_pickup_and_insertion_paths_only",
            "/".join(selected), (
                "commissioned_robot_and_force_limits",
                "fresh_synchronized_key_and_robot_state",
                "measured_post_lift_key_hand_relation",
                "guarded_insertion_controller",
            ))
    if selected is not None:
        raise ValueError("nonpassing trial cannot select a grasp")
    if status == "candidate_budget_exhausted":
        return SessionDecision(
            "continue_candidate_preflight", "pilot_prefix_is_not_pose_exhaustion",
            None)
    if status == "repose_required_unplanned":
        targets = report.get("repose_target_stems")
        if not isinstance(targets, list) or not targets:
            raise ValueError("repose decision lacks another eligible tabletop pose")
        return SessionDecision(
            "preflight_repose", "current_pose_has_no_remaining_plannable_grasp",
            None, ("fresh_key_and_joint_observation",
                   "directed_v8_reset_seed", "socket_aware_reset_preflight"))
    if status in {
            "catalog_unavailable", "no_eligible_pose_in_catalog",
            "planning_exhausted_current_pose", "repose_assets_unavailable",
            "repose_staged_only_unplanned",
    }:
        return SessionDecision("stop_for_review", str(status), None)
    raise ValueError(f"unrecognized trial preflight status: {status}")


def decide_after_attempt(
    attempt: AttemptRecord, *, max_xy_retries: int,
) -> SessionDecision:
    """Choose the next *evidence gate* after one observed physical phase.

    A failed grasp is never followed by a transfer. A failed insertion can
    reach a 1 mm VLM proposal only if the key is still held, there was no
    safety abort, and the finite retry budget remains. The separate retry
    module still requires guarded withdrawal, exact endpoint screening,
    multi-view consensus and a new live-state preflight.
    """
    if type(max_xy_retries) is not int or max_xy_retries < 0:
        raise ValueError("max_xy_retries must be a nonnegative integer")
    if not isinstance(attempt, AttemptRecord):
        raise TypeError("expected an observed AttemptRecord")
    labels = attempt.labels
    candidate_id = attempt.candidate_id

    def result(action: str, reason: str, *gates: str) -> SessionDecision:
        return SessionDecision(action, reason, candidate_id, gates)

    if labels["reset_success"] is False or attempt.failure_code in {
            "force_abort", "reset_failed"}:
        return result("stop_for_review", "safety_or_reset_failure")
    if attempt.failure_code in {"perception_unreliable", "slip"}:
        return result("stop_for_review", "key_state_unreliable_or_grasp_lost")
    failure_stage = {
        "grasp_miss": "grasp_success",
        "transfer_unreachable": "preinsert_reached",
        "transfer_collision": "preinsert_reached",
        "preinsert_misaligned": "preinsert_reached",
        "rim_jam": "insertion_success",
        "depth_shortfall": "insertion_success",
    }.get(attempt.failure_code)
    if failure_stage is not None and labels[failure_stage] is not False:
        return result("stop_for_review", "failure_code_lacks_matching_observed_stage")
    if labels["reset_success"] is True:
        return result(
            "reobserve_key_and_preflight", "reset_observed",
            "fresh_key_pose", "fresh_robot_state")
    if labels["insertion_success"] is True:
        return result(
            "hold_for_supervised_completion", "20mm_task_verified_extraction_deferred",
            "commissioned_extraction_or_release_and_reset")
    if labels["insertion_success"] is False:
        if attempt._pending_retry:
            return result(
                "await_retry_execution_and_observation",
                "1mm_choice_withdrawal_and_replan_recorded_not_executed",
                "commissioned_guarded_controller", "new_insertion_observation")
        insertion = next(
            event for event in reversed(attempt.events)
            if event["stage"] == "insertion_success")
        evidence = insertion["detail"]["input"]
        if (evidence["grasp_held"] is not True or
                evidence["safety_abort"] is not False or
                attempt.failure_code == "slip"):
            return result("stop_for_review", "grasp_lost_or_insertion_abort")
        retry_count = sum(event["stage"] == "xy_retry"
                          for event in attempt.events)
        if retry_count >= max_xy_retries:
            return result(
                "recover_held_key_then_reobserve", "xy_retry_budget_exhausted",
                "guarded_axial_withdrawal", "verified_key_rest_or_human_recovery")
        return result(
            "guarded_withdrawal_then_xy_assessment", "observed_insertion_failure",
            "guarded_axial_withdrawal", "observed_post_withdrawal_key_hand_pose",
            "calibrated_multiview_1mm_endpoint_and_vote",
            "fresh_held_transfer_and_axial_preflight")
    if any(event["stage"] == "insertion_success" for event in attempt.events):
        return result("stop_for_review", "insertion_verdict_unknown")
    if labels["preinsert_reached"] is False:
        return result(
            "recover_key_then_reobserve", "transfer_or_hold_failed",
            "safe_key_state_or_human_recovery", "fresh_key_pose")
    if labels["preinsert_reached"] is True:
        return result(
            "await_guarded_insertion_and_observation", "preinsert_hold_observed",
            "commissioned_contact_limits", "fresh_alignment_and_grip_check")
    if any(event["stage"] == "preinsert_reached" for event in attempt.events):
        return result("stop_for_review", "preinsert_verdict_unknown")
    if labels["grasp_success"] is False:
        return result(
            "reobserve_key_and_preflight", "grasp_failed_try_next_candidate",
            "fresh_key_pose", "fresh_robot_state", "exclude_attempted_grasp")
    if labels["grasp_success"] is True:
        return result(
            "postlift_observed_preflight_required", "lifted_key_observed",
            "fresh_multiview_key_pose", "measured_franka_inspire_state",
            "observed_key_hand_endpoint_and_held_path_preflight")
    if any(event["stage"] == "grasp_success" for event in attempt.events):
        return result("stop_for_review", "grasp_verdict_unknown")
    return result("await_lift_observation", "no_physical_stage_recorded")
