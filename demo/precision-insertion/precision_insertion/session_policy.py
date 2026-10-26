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
                "stock_franka_force_stop_12n",
                "fresh_synchronized_key_and_robot_state",
                "nominal_v8_key_hand_relation_for_cylinder",
                "bound_socket_axis_path",
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
            "repose_staged_only_unplanned", "repeat_grasp_unavailable",
    }:
        return SessionDecision("stop_for_review", str(status), None)
    raise ValueError(f"unrecognized trial preflight status: {status}")


def decide_after_attempt(
    attempt: AttemptRecord, *, max_xy_retries: int | None,
) -> SessionDecision:
    """Choose the next *evidence gate* after one observed physical phase.

    A failed grasp is never followed by a transfer. After insertion the hand
    is already open and the robot is at clear view. A visual XY proposal is
    therefore replayed from the snapshot saved at the pre-insertion hold; no
    post-failure lift/capture or same-attempt reinsertion is allowed.
    """
    if max_xy_retries is not None and (
            type(max_xy_retries) is not int or max_xy_retries < 0):
        raise ValueError("max_xy_retries must be a nonnegative integer or None")
    if not isinstance(attempt, AttemptRecord):
        raise TypeError("expected an observed AttemptRecord")
    labels = attempt.labels
    candidate_id = attempt.candidate_id

    def result(action: str, reason: str, *gates: str) -> SessionDecision:
        return SessionDecision(action, reason, candidate_id, gates)

    if labels["reset_success"] is False or attempt.failure_code == "reset_failed":
        return result("stop_for_review", "safety_or_reset_failure")
    if labels["reset_success"] is True:
        return result(
            "reobserve_key_and_preflight", "reset_observed",
            "fresh_key_pose", "fresh_robot_state")
    if attempt.failure_code == "force_abort":
        insertion = next((event for event in reversed(attempt.events)
                          if event["stage"] == "insertion_success"), None)
        if (labels["insertion_success"] is False and insertion is not None and
                insertion["evidence_refs"].get("axial_withdrawal")):
            return result(
                "recover_held_key_then_reobserve",
                "stock_force_stop_withdrawn_restore_original_pose",
                "measured_reverse_axis_ascent", "fresh_key_pose",
                "restore_original_pose_and_position")
        return result("stop_for_review", "force_stop_without_verified_ascent")
    if attempt.failure_code == "perception_unreliable":
        return result("stop_for_review", "key_state_unobservable")
    if attempt.failure_code == "slip":
        return result(
            "recover_key_then_reobserve", "slipped_key_restore_original_pose",
            "supervised_key_recovery", "fresh_key_pose",
            "restore_original_pose_and_position")
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
    if labels["reorient_success"] is not None and candidate_id is not None:
        return result("stop_for_review", "repose_label_on_insertion_attempt")
    if labels["reorient_success"] is False:
        return result(
            "reobserve_key_and_preflight",
            "repose_landed_in_different_supported_pose",
            "fresh_key_pose", "fresh_robot_state")
    if labels["reorient_success"] is True:
        return result(
            "reobserve_key_and_preflight", "repose_landing_observed",
            "fresh_key_pose", "fresh_robot_state")
    if labels["insertion_success"] is True:
        return result(
            "session_success", "observed_task_success_end_session")
    if attempt.failure_code in {
            "pickup_execution_failed", "post_squeeze_preflight_failed",
            "lift_execution_failed"}:
        if (labels["grasp_success"] is not None or
                not any(event["stage"] == "grasp_success" and
                        event["value"] is None for event in attempt.events)):
            return result("stop_for_review", "motor_failure_lacks_unknown_grasp")
        return result(
            "recover_key_then_reobserve", attempt.failure_code,
            "supervised_key_recovery", "fresh_key_pose")
    if labels["insertion_success"] is False:
        if attempt._pending_retry:
            return result(
                "stop_for_review", "legacy_same_attempt_retry_is_unsupported")
        if attempt.events[-1]["stage"] == "next_trial_xy_advice":
            return result(
                "recover_released_key_then_reobserve",
                "xy_advice_saved_for_next_physical_attempt",
                "fresh_released_key_pose", "restore_original_pose_and_position")
        if attempt.events[-1]["stage"] == "xy_retry_unavailable":
            return result(
                "recover_released_key_then_reobserve",
                attempt.events[-1]["value"],
                "fresh_released_key_pose", "restore_original_pose_and_position")
        insertion = next(
            event for event in reversed(attempt.events)
            if event["stage"] == "insertion_success")
        evidence = insertion["detail"]["input"]
        if (evidence["grasp_held"] is not True or
                evidence["safety_abort"] is not False or
                attempt.failure_code == "slip"):
            return result("stop_for_review", "grasp_lost_or_insertion_abort")
        if attempt.mode.family == "square":
            return result(
                "recover_released_key_then_reobserve",
                "square_mode_no_xy_error_correction",
                "fresh_released_key_pose", "restore_original_pose_and_position")
        retry_count = sum(event["stage"] == "xy_retry"
                          for event in attempt.events)
        grounded_retry_used = any(
            event["stage"] == "xy_retry" and
            event["value"] == "grounded_continuous_xy"
            for event in attempt.events)
        if ((max_xy_retries is not None and retry_count >= max_xy_retries) or
                grounded_retry_used):
            return result(
                "recover_released_key_then_reobserve",
                ("grounded_retry_step_exhausted" if grounded_retry_used else
                 "xy_retry_budget_exhausted"),
                "fresh_released_key_pose", "restore_original_pose_and_position")
        return result(
            "assess_saved_preinsert_xy", "observed_released_insertion_failure",
            "saved_preinsert_raw_multiview_tip_observation",
            "calibrated_xy_only_up_to_10mm_next_trial_update",
            "next_attempt_only_no_lateral_motion")
    if any(event["stage"] == "insertion_success" for event in attempt.events):
        return result("stop_for_review", "insertion_verdict_unknown")
    if labels["preinsert_reached"] is False:
        return result(
            "recover_key_then_reobserve", "transfer_or_hold_failed",
            "safe_key_state_or_human_recovery", "fresh_key_pose")
    if labels["preinsert_reached"] is True:
        insertion_gates = (
            ("stock_franka_force_stop_12n", "fresh_preinsert_hold")
            if attempt.mode.family == "cylinder" else
            ("commissioned_contact_limits", "fresh_alignment_and_grip_check"))
        return result(
            "await_guarded_insertion_and_observation", "preinsert_hold_observed",
            *insertion_gates)
    if any(event["stage"] == "preinsert_reached" for event in attempt.events):
        return result(
            "recover_key_then_reobserve", "preinsert_verdict_unknown",
            "supervised_key_recovery", "fresh_key_pose")
    if labels["grasp_success"] is False:
        return result(
            "recover_key_then_reobserve", "grasp_miss_reobserve_current_pose",
            "clear_view_open_hand", "fresh_supported_key_pose",
            "exclude_attempted_grasp")
    if labels["grasp_success"] is True:
        return result(
            "postlift_observed_preflight_required", "lifted_key_observed",
            "raw_multiview_lift_label", "measured_franka_inspire_state",
            "nominal_v8_key_hand_endpoint_and_held_path_preflight")
    if any(event["stage"] == "grasp_success" for event in attempt.events):
        return result(
            "recover_key_then_reobserve", "grasp_verdict_unknown",
            "supervised_key_recovery", "fresh_key_pose")
    if candidate_id is None:
        if any(event["stage"] == "reorient_success" for event in attempt.events):
            return result("stop_for_review", "repose_verdict_unknown")
        return result(
            "await_repose_observation", "no_repose_outcome_recorded",
            "observed_landed_key_pose", "tabletop_classification")
    return result("await_lift_observation", "no_physical_stage_recorded")
