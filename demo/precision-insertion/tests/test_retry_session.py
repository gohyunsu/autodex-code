"""A one-millimetre retry must bind one failed attempt to held-key frames."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.sync import convert_inspire_raw  # noqa: E402
from precision_insertion.held_relation import HeldRelation  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion.retry_session import (  # noqa: E402
    RetrySessionLimits, assess_and_plan_observed_xy_retry,
    assess_unobserved_xy_diagnostic,
    write_retry_session_artifacts,
)
from precision_insertion import retry_session  # noqa: E402
from precision_insertion.xy_retry import XYRetryAssessment  # noqa: E402
from precision_insertion.xy_voting import ChoiceDecision  # noqa: E402
from test_xy_retry import _setup  # noqa: E402


def _inputs(tmp_path, monkeypatch):
    args = _setup(tmp_path)
    mode = args["mode"]
    candidate_dir = (tmp_path / "AutoDex/candidates/inspire/v8" /
                     mode.key_object / "table/0/1")
    nominal = np.load(candidate_dir / "wrist_se3.npy", allow_pickle=False)
    catalog_hash = hashlib.sha256(json.dumps(
        args["catalog"], sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    attempt = begin_attempt(
        attempt_id="trial_1", mode=mode,
        session_record=args["calibration"].record,
        candidate_id="table/0/1", tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=99.0)
    attempt.record_stage(
        "grasp_success", True, timestamp_s=99.2,
        evidence_refs={"vlm_observation": "vlm/lift.json",
                       "key_wrist_check": "pose/lift.json"})
    postlift_file = tmp_path / "postlift/report.json"
    postlift_file.parent.mkdir()
    postlift_file.write_text(json.dumps({
        "schema": "precision_insertion_postlift_preflight_v1",
        "status": "sampled_postlift_preflight_pass",
        "attempt_id": attempt.attempt_id,
        "candidate_key": ["table", "0", "1"],
        "session_calibration_sha256": attempt.session_calibration_sha256,
        "catalog_sha256": catalog_hash,
        "planning": {"sampled_planning_pass": True},
        "observed_held_relation": {"T_key_hand": nominal.tolist()},
    }), encoding="utf-8")
    attempt.record_stage(
        "preinsert_reached", True, timestamp_s=99.4,
        evidence_refs={"trajectory": "plan/transfer.json",
                       "key_socket_pose": "pose/hold.json",
                       "grasp_state": "pose/grip.json",
                       "postlift_preflight": str(postlift_file)})
    attempt.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=99.6,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"})
    trial = SimpleNamespace(
        status="sampled_planning_pass",
        selected_candidate_key=args["candidate_key"],
        insertion_plan=SimpleNamespace(sampled_planning_pass=True),
        pose_class={"stem": "000"},
        session_calibration_sha256=attempt.session_calibration_sha256,
        catalog_sha256=catalog_hash,
        trial_scene={"mesh": {"target": {}}})
    frame_evidence = args["acquisition_metadata"]["frames"]
    held = KeyPoseObservation(
        capture_id="held_1", request_id=args["frame_request_id"],
        key_object=mode.key_object, family=mode.family,
        pose_world=np.eye(4), selected_camera_id="a",
        selected_acquisition_timestamp_s=100.0,
        acquisition_interval_s=(99.999, 100.001),
        frame_evidence=frame_evidence,
        per_view={}, consistency={
            "accepted_views": ["a", "b"],
            "held_pose_prior": {
                "source": "measured_wrist_plus_observed_held_relation",
                "timestamp_s": 100.0,
                "pose_world": np.eye(4).tolist()}},
        selection={}, source_capture_dir=tmp_path / "held_source",
        phase="held_preinsert")
    evidence = tmp_path / "held_evidence"
    evidence.mkdir()
    (evidence / "key_observation.json").write_text(
        json.dumps(held.to_record()), encoding="utf-8")
    (evidence / "evidence_manifest.json").write_text(
        json.dumps({"capture_id": "held_1", "request_id": 15}),
        encoding="utf-8")
    monkeypatch.setattr(
        retry_session, "verify_key_capture_artifacts",
        lambda path: json.loads((path / "evidence_manifest.json")
                               .read_text(encoding="utf-8")))
    raw = np.full(6, 500.0)
    q = np.zeros(13)
    q[2] = 0.30
    q[7:] = convert_inspire_raw(raw[None, :])[0]
    joints = LiveRobotState(
        full_q=q, arm_qvel=np.zeros(7), sample_timestamp_s=100.0,
        arm_timestamp_s=100.0, arm_robot_uptime_s=10.0,
        hand_timestamp_s=100.0, hand_raw_measured=raw,
        hand_raw_commanded=raw, max_hand_command_error_raw=0.0,
        wrench=np.zeros(6))
    path_limits = PathAuditLimits(
        max_joint_step_rad=0.02, max_wrist_step_m=0.005,
        max_wrist_rotation_deg=1.0, goal_position_tolerance_m=0.001,
        goal_rotation_tolerance_deg=1.0,
        axial_lateral_tolerance_m=0.001,
        axial_rotation_tolerance_deg=1.0,
        minimum_hand_clearance_m=0.001)
    limits = RetrySessionLimits(
        max_state_skew_s=0.02, max_arm_hand_skew_s=0.02,
        max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05,
        max_grasp_translation_drift_m=0.002,
        max_grasp_rotation_drift_deg=5.0,
        max_total_offset_m=0.002,
        minimum_anchor_separation_px=3.0, crop_width_px=320,
        max_frame_age_s=0.2, max_capture_skew_s=0.02,
        axial_waypoint_step_m=0.005, path_limits=path_limits)
    withdrawal = tmp_path / "withdrawal.json"
    withdrawal.write_text(json.dumps({
        "schema": "precision_insertion_guarded_withdrawal_v1",
        "attempt_id": "trial_1", "candidate_id": "table/0/1",
        "status": "withdrawn_to_preinsert_hold",
        "completed_at_s": 99.8, "key_still_held": True,
        "safety_abort": False,
        "source": "commissioned_guarded_controller",
    }), encoding="utf-8")
    return dict(
        planner=SimpleNamespace(fk_wrist=lambda _q: nominal.copy()),
        mode=mode, shared_root=tmp_path, calibration=args["calibration"],
        catalog=args["catalog"], trial=trial, attempt=attempt,
        held_key_observation=held, held_key_evidence_dir=evidence,
        joint_sample=joints, frames=args["frames"],
        intrinsics_full=args["intrinsics_full"],
        extrinsics_full=args["extrinsics_full"],
        frame_request_id=args["frame_request_id"],
        frame_ids=args["frame_ids"],
        acquisition_metadata=args["acquisition_metadata"],
        backend=args["backend"], withdrawal_completed_at_s=99.8,
        withdrawal_evidence_path=withdrawal,
        postlift_preflight_report_path=postlift_file,
        decision_timestamp_s=100.05, limits=limits)


def test_same_frame_held_pose_and_measured_hand_gate_retry_before_vote(
        tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    nominal = np.load(
        tmp_path / "AutoDex/candidates/inspire/v8" /
        kwargs["mode"].key_object / "table/0/1/wrist_se3.npy",
        allow_pickle=False)
    relation = HeldRelation(
        np.eye(4), nominal, np.zeros(3), 0.0, 0.0, "identity")
    monkeypatch.setattr(
        retry_session, "resolve_postlift_held_relation",
        lambda **_kwargs: relation)
    seen = {}
    choice = ChoiceDecision(
        "propose", "two_view_consensus", "x_plus_1mm", (0.001, 0.0),
        ("a", "b"), {"x_plus_1mm": 2})
    assessment = XYRetryAssessment(
        "proposal_requires_live_preflight", {}, None, (), choice,
        "two_view_consensus")
    def vote(**inputs):
        seen["hand_q"] = inputs["held_hand_q_measured"].copy()
        return assessment
    def plan(**inputs):
        seen["plan_q"] = inputs["live_start_q"].copy()
        assert inputs["held_hand_source"] == "measured"
        return SimpleNamespace(status="sampled_retry_preflight_pass")
    monkeypatch.setattr(retry_session, "assess_xy_retry", vote)
    monkeypatch.setattr(
        retry_session, "plan_xy_retry_from_withdrawn_hold", plan)
    result = assess_and_plan_observed_xy_retry(**kwargs)
    assert result.status == "ready_to_record_pending_retry"
    assert np.allclose(seen["hand_q"], kwargs["joint_sample"].full_q[7:])
    assert np.allclose(seen["plan_q"], kwargs["joint_sample"].full_q)
    assert kwargs["backend"].calls == 0  # fake vote, no actual VLM call
    changed = replace(kwargs["held_key_observation"], request_id=16)
    with pytest.raises(ValueError, match="held-key pose differs"):
        assess_and_plan_observed_xy_retry(**{
            **kwargs, "held_key_observation": changed})
    assert "plan_q" in seen


def test_retry_rejects_prewithdrawal_frames_or_missing_log(
        tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="preceding logged withdrawal"):
        assess_and_plan_observed_xy_retry(**{
            **kwargs, "withdrawal_completed_at_s": 100.01})
    kwargs["withdrawal_evidence_path"].unlink()
    with pytest.raises(ValueError, match="preceding logged withdrawal"):
        assess_and_plan_observed_xy_retry(**kwargs)
    kwargs["withdrawal_evidence_path"].write_text(
        '{"withdrawn": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="does not confirm a held safe return"):
        assess_and_plan_observed_xy_retry(**kwargs)


def test_held_pose_prior_must_derive_from_saved_postlift_relation(
        tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    bad_prior = kwargs["held_key_observation"].to_record()["consistency"]
    bad_prior["held_pose_prior"]["pose_world"][0][3] = 0.01
    changed = replace(kwargs["held_key_observation"], consistency=bad_prior)
    (kwargs["held_key_evidence_dir"] / "key_observation.json").write_text(
        json.dumps(changed.to_record()), encoding="utf-8")
    with pytest.raises(ValueError, match="prior differs from saved grasp/wrist"):
        assess_and_plan_observed_xy_retry(**{
            **kwargs, "held_key_observation": changed})


def test_retry_artifacts_preserve_full_verified_camera_pixels(tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    from precision_insertion.retry_session import RetrySessionResult
    from precision_insertion.held_relation import HeldRelation
    frame_binding = kwargs["acquisition_metadata"]["frames"]
    result = RetrySessionResult(
        "visual_abstain", "trial_1", "table/0/1", "held_1",
        kwargs["held_key_evidence_dir"], "a" * 64, 99.8,
        kwargs["withdrawal_evidence_path"], "b" * 64,
        kwargs["postlift_preflight_report_path"], "c" * 64,
        frame_binding, kwargs["joint_sample"],
        HeldRelation(np.eye(4), np.eye(4), np.zeros(3), 0.0, 0.0,
                     "identity"), None, None)
    output = write_retry_session_artifacts(
        result, kwargs["frames"], tmp_path / "retry_artifacts")
    assert (output / "frames/a.png").is_file()
    assert len(json.loads((output / "report.json").read_text())[
        "artifacts_sha256"]) == 2
    with pytest.raises(FileExistsError):
        write_retry_session_artifacts(result, kwargs["frames"], output)


def test_unobserved_squeeze_diagnostic_is_saved_but_not_preflighted(
        tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    kwargs.pop("planner")
    kwargs.pop("held_key_observation")
    kwargs.pop("held_key_evidence_dir")
    seen = []

    def hand_clear_only(**screen_args):
        seen.append(screen_args["xy_offset_socket_m"])
        assert screen_args["override_source"] == "v8_nominal_unobserved_key"
        return {
            "endpoint_pass": False,
            "hand_socket_clear_at_20mm": True,
            "xy_offset_socket_m": list(screen_args["xy_offset_socket_m"]),
            "verification_depth_m": kwargs["mode"].target_depth_m,
        }

    result = assess_unobserved_xy_diagnostic(**kwargs, screen=hand_clear_only)
    assert len(seen) == 5
    assert result.status == "diagnostic_xy_hypothesis_only"
    assert result.preflight is None
    assert result.assessment.decision.choice_id == "x_plus_1mm"
    assert result.to_record()["robot_ready"] is False
    assert kwargs["attempt"]._pending_retry is False
    output = write_retry_session_artifacts(
        result, kwargs["frames"], tmp_path / "unobserved_diagnostic")
    saved = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert saved["schema"] == "precision_insertion_unobserved_xy_diagnostic_v1"
    assert saved["preflight"] is None
    assert (output / "frames/a.png").is_file()


def test_unobserved_diagnostic_rejects_missing_withdrawal_and_wrong_attempt(
        tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    kwargs.pop("planner")
    kwargs.pop("held_key_observation")
    kwargs.pop("held_key_evidence_dir")
    kwargs["withdrawal_evidence_path"].unlink()
    with pytest.raises(ValueError, match="preceding logged withdrawal"):
        assess_unobserved_xy_diagnostic(**kwargs)
    assert kwargs["backend"].calls == 0
    kwargs["withdrawal_evidence_path"].write_text(json.dumps({
        "schema": "precision_insertion_guarded_withdrawal_v1",
        "attempt_id": "wrong", "candidate_id": "table/0/1",
        "status": "withdrawn_to_preinsert_hold",
        "completed_at_s": 99.8, "key_still_held": True,
        "safety_abort": False, "source": "commissioned_guarded_controller",
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="does not confirm a held safe return"):
        assess_unobserved_xy_diagnostic(**kwargs)
    assert kwargs["backend"].calls == 0


def test_unobserved_diagnostic_rejects_stale_hand_feedback(tmp_path, monkeypatch):
    kwargs = _inputs(tmp_path, monkeypatch)
    kwargs.pop("planner")
    kwargs.pop("held_key_observation")
    kwargs.pop("held_key_evidence_dir")
    kwargs["joint_sample"] = replace(
        kwargs["joint_sample"], sample_timestamp_s=99.9,
        arm_timestamp_s=99.9, hand_timestamp_s=99.9)
    with pytest.raises(ValueError, match="not synchronized with retry frames"):
        assess_unobserved_xy_diagnostic(**kwargs)
    assert kwargs["backend"].calls == 0
