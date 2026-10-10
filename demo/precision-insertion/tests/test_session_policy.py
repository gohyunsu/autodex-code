"""The demo's next-step decisions do not authorize robot motion."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
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
                       "grasp_state": "sensors/grip.json"})


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
    assert decision.action == "reobserve_key_and_preflight"
    assert "exclude_attempted_grasp" in decision.required_before_motion
    assert missed.labels["preinsert_reached"] is None
    failed_transfer = _attempt()
    _grasp(failed_transfer)
    _hold(failed_transfer, False)
    assert decide_after_attempt(
        failed_transfer, max_xy_retries=2).action == "recover_key_then_reobserve"
    unknown = _attempt()
    unknown.record_stage(
        "grasp_success", None, timestamp_s=2.0,
        evidence_refs={"vlm_observation": "vlm/occluded.json"})
    assert decide_after_attempt(
        unknown, max_xy_retries=2).action == "stop_for_review"


def test_failed_insertion_requires_withdrawal_then_bounded_1mm_retry():
    record = _attempt()
    _grasp(record)
    _hold(record)
    _insert(record)
    assert decide_after_attempt(
        record, max_xy_retries=1).action == (
            "guarded_withdrawal_then_xy_assessment")
    _retry(record)
    assert decide_after_attempt(
        record, max_xy_retries=1).action == (
            "await_retry_execution_and_observation")
    _insert(record, timestamp=6.0)
    assert decide_after_attempt(
        record, max_xy_retries=1).action == (
            "recover_held_key_then_reobserve")


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


def test_verified_insertion_waits_for_supervised_completion():
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
    assert decide_after_attempt(
        record, max_xy_retries=2).action == "hold_for_supervised_completion"
