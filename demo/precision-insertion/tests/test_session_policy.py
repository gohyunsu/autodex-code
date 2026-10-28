"""The demo's next-step decisions do not authorize robot motion."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.action_loop import run_session_actions  # noqa: E402
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion.session_policy import (  # noqa: E402
    decide_after_attempt, decide_after_trial_preflight,
)
from precision_insertion.xy_voting import ChoiceDecision  # noqa: E402


MODE = select_mode("cylinder", 20)
SESSION = {
    "schema": "precision_insertion_session_calibration_v1",
    "mode": {"family": MODE.family, "gap_mm": MODE.gap_mm,
             "key_object": MODE.key_object,
             "socket_object": MODE.socket_object},
}


def _attempt():
    return begin_attempt(
        attempt_id="trial_001", mode=MODE, session_record=SESSION,
        candidate_id="table/0/3", tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0)


def _grasp(record, value=True):
    record.record_stage(
        "grasp_success", value, timestamp_s=2.0,
        evidence_refs={"vlm_observation": "vlm/lift.json",
                       "key_wrist_check": "sensors/grip.json"})


def _hold(record, value=True):
    record.record_stage(
        "preinsert_reached", value, timestamp_s=3.0,
        evidence_refs={"trajectory": "plan/transfer.json",
                       "key_socket_pose": "sensors/hold.json",
                       "grasp_state": "sensors/grip.json",
                       "postlift_preflight": "plan/postlift/report.json"})


def _insert(record, *, vlm="partial", grasp_held=True, safety_abort=False,
            timestamp=4.0, depth=(0.005, 0.008)):
    return record.record_insertion_evidence(
        InsertionEvidence(
            vlm_class=vlm, key_depth_interval_m=depth,
            key_depth_source="key_pose_multiview",
            alignment_within_limits=True, safety_abort=safety_abort,
            grasp_held=grasp_held),
        timestamp_s=timestamp,
        evidence_refs={"vlm_observation": "vlm/insertion.json",
                       "key_depth": "sensors/depth.json",
                       "alignment": "sensors/alignment.json",
                       "force_trace": "sensors/wrench.json"})


def test_corrected_axial_plan_failure_stops_with_unknown_insertion():
    record = _attempt()
    _grasp(record)
    _hold(record)
    record.record_failure(
        "preinsert_path_unavailable", timestamp_s=4.,
        evidence_refs={"corrected_axial_plan_failure": "plan_failure.json"})
    assert record.labels["insertion_success"] is None
    decision = decide_after_attempt(record, max_xy_retries=None)
    assert decision.action == "stop_for_review"
    assert decision.reason == "corrected_axial_plan_unavailable_while_key_held"


def _decision(*, choice="x_plus_1mm", cameras=("cam0", "cam1")):
    return ChoiceDecision(
        "propose", "two_view_consensus", choice, (0.001, 0.0),
        cameras, {choice: 2})


def _retry(record, decision=None):
    record.record_retry(
        _decision() if decision is None else decision,
        timestamp_s=5.0,
        evidence_refs={"axial_withdrawal": "robot/withdrawal.json",
                       "live_preflight": "plan/retry.json",
                       "xy_vlm_vote": "vlm/votes.json"})


def test_preflight_routes_pass_budget_and_repose_without_motion_permission():
    base = {"schema": "precision_insertion_trial_preflight_v2",
            "status": "sampled_planning_pass",
            "selected_candidate_key": ["table", "0", "3"],
            "insertion_plan": {"sampled_planning_pass": True}}
    decision = decide_after_trial_preflight(base)
    assert decision.action == "execution_gate_required"
    assert decision.candidate_id == "table/0/3"
    assert decision.to_record()["robot_ready"] is False
    with pytest.raises(ValueError, match="matching insertion preflight"):
        decide_after_trial_preflight({**base, "insertion_plan": None})
    budget = {**base, "status": "candidate_budget_exhausted",
              "selected_candidate_key": None, "insertion_plan": None}
    assert decide_after_trial_preflight(budget).action == (
        "continue_candidate_preflight")
    repose = {**budget, "status": "repose_required_unplanned",
              "repose_target_stems": ["001"]}
    assert decide_after_trial_preflight(repose).action == "preflight_repose"
    assert decide_after_trial_preflight(
        {**budget, "status": "catalog_unavailable"}).action == "stop_for_review"


def test_grasp_transfer_and_unknown_observations_branch_separately():
    missed = _attempt()
    _grasp(missed, False)
    decision = decide_after_attempt(missed, max_xy_retries=2)
    assert decision.action == "recover_key_then_reobserve"
    assert "fresh_supported_key_pose" in decision.required_before_motion
    assert "exclude_attempted_grasp" in decision.required_before_motion
    assert missed.labels["preinsert_reached"] is None
    failed_transfer = _attempt()
    _grasp(failed_transfer)
    assert decide_after_attempt(
        failed_transfer, max_xy_retries=2).action == (
            "postlift_observed_preflight_required")
    _hold(failed_transfer, False)
    assert decide_after_attempt(
        failed_transfer, max_xy_retries=2).action == "recover_key_then_reobserve"
    unknown = _attempt()
    unknown.record_stage(
        "grasp_success", None, timestamp_s=2.0,
        evidence_refs={"vlm_observation": "vlm/occluded.json"})
    assert decide_after_attempt(
        unknown, max_xy_retries=2).action == "recover_key_then_reobserve"


@pytest.mark.parametrize("code", [
    "pickup_execution_failed", "post_squeeze_preflight_failed",
    "lift_execution_failed",
])
def test_unknown_grasp_after_stopped_failure_requires_supervised_return(code):
    attempt = _attempt()
    attempt.record_stage(
        "grasp_success", None, timestamp_s=2.0,
        evidence_refs={"aborted_attempt": "robot/abort.json"})
    attempt.record_failure(
        code, timestamp_s=2.0,
        evidence_refs={"aborted_attempt": "robot/abort.json"})
    decision = decide_after_attempt(attempt, max_xy_retries=1)
    assert decision.action == "recover_key_then_reobserve"
    assert "supervised_key_recovery" in decision.required_before_motion


def test_unknown_preinsert_does_not_repeat_same_transfer():
    attempt = _attempt()
    _grasp(attempt)
    attempt.record_stage(
        "preinsert_reached", None, timestamp_s=3.0,
        evidence_refs={"preinsert_checkpoint": "vlm/unknown.json"})
    assert decide_after_attempt(
        attempt, max_xy_retries=1).action == "recover_key_then_reobserve"


def test_failed_insertion_advice_returns_key_before_next_physical_attempt():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    assert decide_after_attempt(
        record, max_xy_retries=1).action == (
            "assess_saved_preinsert_xy")
    record.record_next_trial_xy_advice(
        increment_socket_xy_m=(0.0006, -0.0008),
        update_weight=1.0,
        applied_step_socket_xy_m=(0.0006, -0.0008),
        next_wrist_correction_socket_m=(0.0006, -0.0008),
        supporting_cameras=("cam0", "cam1"), timestamp_s=5.0,
        evidence_refs={"preinsert_snapshot": "camera/preinsert.json",
                       "grounded_xy": "vlm/tip_axis.json"})
    assert decide_after_attempt(
        record, max_xy_retries=1).action == (
            "recover_released_key_then_reobserve")


def test_xy_advice_can_use_gain_above_one_within_ten_mm_step():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    record.record_next_trial_xy_advice(
        increment_socket_xy_m=(0.004, -0.002),
        update_weight=1.5,
        applied_step_socket_xy_m=(0.006, -0.003),
        next_wrist_correction_socket_m=(0.006, -0.003),
        supporting_cameras=("cam0", "cam1"), timestamp_s=5.0,
        evidence_refs={"preinsert_snapshot": "camera/preinsert.json",
                       "grounded_xy": "vlm/tip_axis.json"})
    advice = record.events[-1]
    assert advice["stage"] == "next_trial_xy_advice"
    assert advice["applied_step_socket_xy_m"] == [0.006, -0.003]
    assert advice["next_wrist_correction_socket_m"] == [0.006, -0.003]


def test_stock_force_stop_returns_held_key_only_after_logged_ascent():
    record = _attempt()
    _grasp(record)
    _hold(record)
    record.record_insertion_evidence(
        InsertionEvidence("unobservable", None, None, None, True, True),
        timestamp_s=4.,
        evidence_refs={"force_trace": "robot/force.json",
                       "axial_withdrawal": "robot/ascent.json"})
    record.record_failure(
        "force_abort", timestamp_s=4.1,
        evidence_refs={"force_trace": "robot/force.json"})
    assert decide_after_attempt(
        record, max_xy_retries=1).action == "recover_held_key_then_reobserve"
    assert not any(event["stage"] == "next_trial_xy_advice"
                   for event in record.events)
    record.events[2]["evidence_refs"].pop("axial_withdrawal")
    assert decide_after_attempt(
        record, max_xy_retries=1).action == "stop_for_review"


def test_force_stop_recovery_diagnoses_saved_xy_before_next_key():
    record = _attempt()
    _grasp(record)
    _hold(record)
    record.record_insertion_evidence(
        InsertionEvidence("unobservable", None, None, None, True, True),
        timestamp_s=4.0,
        evidence_refs={"force_trace": "robot/force.json",
                       "axial_withdrawal": "robot/ascent.json"})
    record.record_failure(
        "force_abort", timestamp_s=4.1,
        evidence_refs={"force_trace": "robot/force.json"})
    assert decide_after_attempt(
        record, max_xy_retries=1).action == "recover_held_key_then_reobserve"

    class Runner:
        def current_decision(self):
            return decide_after_attempt(record, max_xy_retries=1)

    def recover(_runner, _decision):
        record.record_stage(
            "reset_success", True, timestamp_s=5.0,
            evidence_refs={"key_pose": "fresh_landed_key.json"})

    def diagnose(_runner, _decision):
        record.record_xy_retry_unavailable(
            reason="sam_half_ellipse_unavailable", timestamp_s=5.1,
            evidence_refs={"retry_assessment": "xy/sam_abstain.json"})

    outcome = run_session_actions(
        Runner(), handlers={
            "recover_held_key_then_reobserve": recover,
            "assess_saved_preinsert_xy": diagnose},
        stop_before_actions={"reobserve_key_and_preflight": "recovery_complete"})
    assert outcome.status == "recovery_complete"
    assert [decision.action for decision in outcome.visited] == [
        "recover_held_key_then_reobserve", "assess_saved_preinsert_xy",
        "reobserve_key_and_preflight"]
    assert outcome.decision.reason == "reset_observed"
    assert decide_after_attempt(record, max_xy_retries=0).action == (
        "reobserve_key_and_preflight")


def test_slipped_key_recovery_advances_after_fresh_reset_observation():
    record = _attempt()
    _grasp(record, False)
    record.record_failure(
        "slip", timestamp_s=3.0,
        evidence_refs={"camera": "images/slip.json"})
    assert decide_after_attempt(
        record, max_xy_retries=None).action == "recover_key_then_reobserve"
    record.record_stage(
        "reset_success", True, timestamp_s=4.0,
        evidence_refs={"key_pose": "fresh_landed_key.json"})
    assert decide_after_attempt(
        record, max_xy_retries=None).action == "reobserve_key_and_preflight"


def test_next_trial_xy_has_10mm_step_but_no_cumulative_limit():
    record = begin_attempt(
        attempt_id="trial_large_offset", mode=MODE, session_record=SESSION,
        candidate_id="table/0/3", tabletop_pose_stem="000",
        xy_offset_socket_m=(0., 0.), started_at_s=1.,
        wrist_correction_socket_m=(.025, 0.))
    _grasp(record)
    _hold(record)
    _insert(record)
    record.record_next_trial_xy_advice(
        increment_socket_xy_m=(.01, 0.), update_weight=1.,
        applied_step_socket_xy_m=(.01, 0.),
        next_wrist_correction_socket_m=(.035, 0.),
        supporting_cameras=(), timestamp_s=5.,
        evidence_refs={"axial_withdrawal": "robot/withdrawal.json",
                       "grounded_xy": "vlm/search.json"})
    assert record.events[-1]["next_wrist_correction_socket_m"] == [.035, 0.]


def test_retry_count_cap_is_optional_not_an_implicit_single_trial_limit():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    assert decide_after_attempt(
        record, max_xy_retries=None).action == (
            "assess_saved_preinsert_xy")
    assert decide_after_attempt(
        record, max_xy_retries=0).reason == "xy_retry_budget_exhausted"


def test_square_failed_insertion_returns_without_xy_advice():
    square = select_mode("square", 1.5)
    session = {"schema": "precision_insertion_session_calibration_v1",
               "mode": {"family": square.family, "gap_mm": square.gap_mm,
                        "key_object": square.key_object,
                        "socket_object": square.socket_object}}
    record = begin_attempt(
        attempt_id="square_001", mode=square, session_record=session,
        candidate_id="table/0/3", tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0)
    _grasp(record)
    _hold(record)
    _insert(record)
    decision = decide_after_attempt(record, max_xy_retries=5)
    assert decision.action == "recover_released_key_then_reobserve"
    assert decision.reason == "square_mode_no_xy_error_correction"
    with pytest.raises(ValueError, match="square mode has no XY"):
        record.record_next_trial_xy_advice(
            increment_socket_xy_m=(0.001, 0.0), update_weight=1.0,
            applied_step_socket_xy_m=(0.001, 0.0),
            next_wrist_correction_socket_m=(0.001, 0.0),
            supporting_cameras=("cam0", "cam1"), timestamp_s=5.0,
            evidence_refs={"axial_withdrawal": "robot/withdrawal.json",
                           "grounded_xy": "vlm/tip_axis.json"})
    with pytest.raises(ValueError, match="square mode has no XY"):
        begin_attempt(
            attempt_id="square_002", mode=square, session_record=session,
            candidate_id="table/0/3", tabletop_pose_stem="000",
            xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0,
            wrist_correction_socket_m=(0.001, 0.0))
    assert record.xy_offset_socket_m == (0.0, 0.0)
    assert not record._pending_retry


def test_abstained_grounded_retry_returns_held_key_without_false_motion():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    record.record_xy_retry_unavailable(
        reason="vlm_abstain", timestamp_s=5.0,
        evidence_refs={"retry_assessment": "vlm/abstain.json"})
    assert record.xy_offset_socket_m == (0.0, 0.0)
    assert not record._pending_retry
    decision = decide_after_attempt(record, max_xy_retries=2)
    assert decision.action == "recover_released_key_then_reobserve"
    assert decision.reason == "vlm_abstain"


def test_unknown_lost_grasp_and_safety_abort_never_offer_xy_retry():
    for kwargs in (
            {"vlm": "unobservable", "depth": (0.0201, 0.021)},
            {"grasp_held": False},
            {"safety_abort": True}):
        record = _attempt()
        _grasp(record)
        _hold(record)
        outcome = _insert(record, **kwargs)
        decision = decide_after_attempt(record, max_xy_retries=2)
        assert decision.action == "stop_for_review"
        if outcome["insertion_success"] is None:
            with pytest.raises(ValueError, match="observed failure"):
                _retry(record)
        else:
            with pytest.raises(ValueError, match="held grasp"):
                _retry(record)


def test_retry_record_rejects_duplicate_view_and_mismatched_direction():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    with pytest.raises(ValueError, match="two-view"):
        _retry(record, _decision(cameras=("cam0", "cam0")))
    with pytest.raises(ValueError, match="does not describe"):
        _retry(record, _decision(choice="y_plus_1mm"))
    malformed = ChoiceDecision(
        "propose", "two_view_consensus", "y_minus_1mm", (math.nan, 0.0),
        ("cam0", "cam1"), {"y_minus_1mm": 2})
    with pytest.raises(ValueError, match="finite socket-frame offsets"):
        _retry(record, malformed)


def test_grounded_continuous_xy_retry_is_distinct_from_cardinal_vote():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    refs = {
        "axial_withdrawal": "robot/withdrawal.json",
        "grounded_xy": "vlm/tip_axis.json",
        "lateral_preflight": "planner/xy_shift.json",
        "lateral_execution": "robot/xy_shift.json",
        "postshift_arrival": "vision/arrival.json",
        "arrival_axial_preflight": "planner/new_axial.json",
    }
    with pytest.raises(ValueError, match="multi-view inliers"):
        record.record_grounded_retry(
            increment_socket_xy_m=(.0006, -.0008),
            supporting_cameras=("cam0", "cam0"), timestamp_s=5.,
            evidence_refs=refs)
    with pytest.raises(ValueError, match="at most 1 mm"):
        record.record_grounded_retry(
            increment_socket_xy_m=(.0011, 0.),
            supporting_cameras=("cam0", "cam1"), timestamp_s=5.,
            evidence_refs=refs)
    record.record_grounded_retry(
        increment_socket_xy_m=(.0006, -.0008),
        supporting_cameras=("cam0", "cam1"), timestamp_s=5.,
        evidence_refs=refs)
    assert record.xy_offset_socket_m == pytest.approx((.0006, -.0008))
    assert record.events[-1]["value"] == "grounded_continuous_xy"
    assert decide_after_attempt(record, max_xy_retries=1).action == (
        "stop_for_review")
    with pytest.raises(ValueError, match="no pending retry"):
        record.record_grounded_retry(
            increment_socket_xy_m=(.0001, .0001),
            supporting_cameras=("cam0", "cam1"), timestamp_s=5.1,
            evidence_refs=refs)


def test_verified_insertion_ends_session():
    record = _attempt()
    _grasp(record)
    _hold(record)
    record.record_insertion_evidence(
        InsertionEvidence("normal_appearance", (0.0201, 0.021),
                          "key_pose_multiview", True, False, True),
        timestamp_s=4.0,
        evidence_refs={"vlm_observation": "vlm/final.json",
                       "key_depth": "sensors/depth.json",
                       "alignment": "sensors/alignment.json",
                       "force_trace": "sensors/wrench.json"})
    decision = decide_after_attempt(record, max_xy_retries=2)
    assert decision.action == "session_success"
    assert decision.required_before_motion == ()


def test_unmatched_failure_codes_cannot_resolve_to_a_motion_phase():
    unknown_key = _attempt()
    unknown_key.record_failure(
        "perception_unreliable", timestamp_s=2.0,
        evidence_refs={"camera": "images/occluded.json"})
    assert decide_after_attempt(
        unknown_key, max_xy_retries=2).action == "stop_for_review"
    unmatched = _attempt()
    unmatched.record_failure(
        "rim_jam", timestamp_s=2.0,
        evidence_refs={"camera": "images/jam.json"})
    assert decide_after_attempt(
        unmatched, max_xy_retries=2).reason == (
            "failure_code_lacks_matching_observed_stage")
    slipped = _attempt()
    _grasp(slipped, False)
    slipped.record_failure(
        "slip", timestamp_s=3.0,
        evidence_refs={"camera": "images/slip.json"})
    assert decide_after_attempt(
        slipped, max_xy_retries=2).action == "recover_key_then_reobserve"


def test_repose_has_separate_observed_landing_label():
    repose = begin_attempt(
        attempt_id="repose_1", mode=MODE, session_record=SESSION,
        candidate_id=None, tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0)
    assert decide_after_attempt(repose, max_xy_retries=0).action == (
        "await_repose_observation")
    repose.record_stage(
        "reorient_success", True, timestamp_s=2.0,
        evidence_refs={"key_pose": "pose/landed.json"})
    assert decide_after_attempt(repose, max_xy_retries=0).action == (
        "reobserve_key_and_preflight")
    bad = _attempt()
    bad.record_stage(
        "reorient_success", True, timestamp_s=2.0,
        evidence_refs={"key_pose": "pose/landed.json"})
    assert decide_after_attempt(bad, max_xy_retries=0).reason == (
        "repose_label_on_insertion_attempt")
