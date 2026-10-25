"""Session-level preflight/observation bookkeeping never commands a robot."""

from __future__ import annotations

from dataclasses import replace
import json
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.trial_preflight import (  # noqa: E402
    TrialPreflight, write_trial_preflight_artifacts,
)
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.retry_preflight import XYRetryPreflight  # noqa: E402
from precision_insertion.retry_session import (  # noqa: E402
    GroundedXYDiagnostic, PreinsertXYDiagnostic,
)
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion import session_runner  # noqa: E402
from precision_insertion.xy_retry import XYRetryAssessment  # noqa: E402
from precision_insertion.xy_voting import ChoiceDecision  # noqa: E402
from autodex.utils.sync import convert_inspire_raw  # noqa: E402


MODE = select_mode("cylinder", 20)
KEY_A = ("table", "0", "3")
KEY_B = ("table", "0", "4")


def _observation(capture_id: str, timestamp: float) -> KeyPoseObservation:
    return KeyPoseObservation(
        capture_id=capture_id, request_id=int(timestamp),
        key_object=MODE.key_object, family=MODE.family,
        pose_world=np.eye(4), selected_camera_id="cam0",
        selected_acquisition_timestamp_s=timestamp,
        acquisition_interval_s=(timestamp - 0.01, timestamp + 0.01),
        frame_evidence={}, per_view={}, consistency={}, selection={},
        source_capture_dir=Path("/tmp/fake_key_capture"))


def _report(runner, observation, *, status="sampled_planning_pass",
            key=KEY_A, attempted=(), repose_targets=()):
    class Report(SimpleNamespace):
        def to_record(self):
            return {
                "schema": "precision_insertion_trial_preflight_v2",
                "status": self.status,
                "selected_candidate_key": (
                    list(self.selected_candidate_key)
                    if self.selected_candidate_key else None),
                "insertion_plan": (
                    {"sampled_planning_pass": True}
                    if self.selected_candidate_key else None),
                "repose_target_stems": list(self.repose_target_stems),
            }

    return Report(
        status=status, selected_candidate_key=(
            key if status == "sampled_planning_pass" else None),
        attempted_candidates=attempted, pose_class={"stem": "000"},
        repose_target_stems=repose_targets,
        trial_scene={"mesh": {"target": {}, "fixture_socket": {}}},
        key_observation_id=observation.capture_id,
        key_pose_world=observation.pose_world,
        start_q_acquisition_timestamp_s=(
            observation.selected_acquisition_timestamp_s),
        session_calibration_sha256=runner.session_sha256,
        catalog_sha256=runner.catalog_sha256)


def _runner(monkeypatch, tmp_path):
    monkeypatch.setattr(session_runner, "validate_catalog_session",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        session_runner, "verify_key_capture_artifacts",
        lambda path: json.loads((path / "evidence_manifest.json")
                               .read_text(encoding="utf-8")))
    def save_report(_report, output_dir):
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "report.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(session_runner, "write_trial_preflight_artifacts",
                        save_report)
    calibration = SessionCalibration(
        board={}, socket_pose_robot=np.eye(4), socket_diagnostics={},
        collision_scene={}, record={
            "schema": "precision_insertion_session_calibration_v1",
            "mode": {"family": MODE.family, "gap_mm": MODE.gap_mm,
                     "key_object": MODE.key_object,
                     "socket_object": MODE.socket_object},
            "c2r": np.eye(4).tolist()})
    catalog = {"shared_root": str(tmp_path / "shared"),
               "complete_scan": True}
    return session_runner.SessionRunner(
        mode=MODE, calibration=calibration, catalog=catalog,
        shared_root=tmp_path / "shared", output_dir=tmp_path / "session",
        max_xy_retries=1)


def test_square_session_effective_xy_weight_and_budget_are_zero(
        monkeypatch, tmp_path):
    monkeypatch.setattr(session_runner, "validate_catalog_session",
                        lambda *_args, **_kwargs: None)
    square = select_mode("square", 1.5)
    record = {
        "schema": "precision_insertion_session_calibration_v1",
        "mode": {"family": square.family, "gap_mm": square.gap_mm,
                 "key_object": square.key_object,
                 "socket_object": square.socket_object},
        "c2r": np.eye(4).tolist(),
    }
    calibration = SessionCalibration(
        board={}, socket_pose_robot=np.eye(4), socket_diagnostics={},
        collision_scene={}, record=record)
    runner = session_runner.SessionRunner(
        mode=square, calibration=calibration,
        catalog={"shared_root": str(tmp_path), "complete_scan": True},
        shared_root=tmp_path, output_dir=tmp_path / "square_session",
        max_xy_retries=5, xy_update_weight=0.0)
    assert runner.max_xy_retries == 0
    assert runner.xy_update_weight == 0.0
    saved = json.loads((runner.output_dir / "session_run.json").read_text())
    assert saved["max_xy_retries"] == 0
    assert saved["xy_update_weight"] == 0.0


def test_stock_experiment_episode_layout_and_result_index(monkeypatch, tmp_path):
    source = _runner(monkeypatch, tmp_path)
    object_root = (tmp_path / "experiment" / "precision-insertion" /
                   "inspire" / MODE.key_object)
    runner = session_runner.SessionRunner(
        mode=MODE, calibration=source.calibration, catalog=source.catalog,
        shared_root=source.shared_root,
        output_dir=object_root / "_pipeline_runs" / "run_001",
        max_xy_retries=1)
    attempt = begin_attempt(
        attempt_id="episode_001", mode=MODE,
        session_record=runner.calibration.record,
        candidate_id="/".join(KEY_A), tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0)
    episode_dir = runner._new_attempt_dir(attempt.attempt_id)
    assert episode_dir == object_root / "episode_001"
    assert (runner.output_dir / "attempts" / "episode_001").resolve() == episode_dir
    runner._attempt = attempt
    runner._attempt_dir = episode_dir
    runner._attempt_index = 0
    first_state = attempt.write_new(episode_dir / "state_000.json")
    runner._save_episode_result(attempt, first_state)
    first_result = json.loads((episode_dir / "result.json").read_text())
    assert first_result["exp_name"] == "precision-insertion"
    assert first_result["status"] == "in_progress"
    assert first_result["latest_state"] == "state_000.json"
    assert json.loads((object_root / "summary.json").read_text())[0][
        "episode_id"] == "episode_001"

    runner._record(lambda row: row.record_stage(
        "grasp_success", False, timestamp_s=2.0,
        evidence_refs={"vlm_observation": "lift.json"}))
    second_result = json.loads((episode_dir / "result.json").read_text())
    assert second_result["latest_state"] == "state_001.json"
    assert second_result["labels"]["grasp_success"] is False
    assert (episode_dir / "state_000.json").is_file()
    assert (episode_dir / "state_001.json").is_file()


def _save_evidence(runner, observation):
    evidence_dir = runner.output_dir.parent / "key_evidence" / observation.capture_id
    if not evidence_dir.exists():
        evidence_dir.mkdir(parents=True)
        (evidence_dir / "key_observation.json").write_text(
            json.dumps(observation.to_record()), encoding="utf-8")
        (evidence_dir / "evidence_manifest.json").write_text(
            json.dumps({"capture_id": observation.capture_id,
                        "request_id": observation.request_id}),
            encoding="utf-8")
    return evidence_dir


def _plan(runner, observation):
    evidence_dir = _save_evidence(runner, observation)
    return runner.preflight_next_key(
        planner=object(), key_observation=observation,
        key_evidence_dir=evidence_dir,
        live_start_q=np.zeros(13),
        start_q_acquisition_timestamp_s=(
            observation.selected_acquisition_timestamp_s),
        max_key_state_skew_s=0.05, limits=object(),
        max_pose_error_deg=5.0, axial_waypoint_step_m=0.002)


def test_xy_advice_survives_intermediate_key_reorientation(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    advice = {"candidate_id": "/".join(KEY_A), "tabletop_pose_stem": "000"}
    runner._pending_next_trial_xy = advice

    def plan_lying(**kwargs):
        assert kwargs["retry_candidate"] == KEY_A
        report = _report(
            runner, kwargs["key_observation"],
            status="repose_required_unplanned", repose_targets=("000",))
        report.pose_class = {"stem": "001"}
        return report

    monkeypatch.setattr(session_runner, "plan_admitted_key_trial", plan_lying)
    _plan(runner, _observation("lying_key", 7.0))
    assert runner.current_decision().action == "preflight_repose"
    assert runner._pending_next_trial_xy is advice


@pytest.mark.parametrize("xy_source", [
    "multiview_tip_and_axis",  # Qwen line estimate
    "multiview_tip_xy_vertical_axis_assumption",  # tracked/SAM face point
])
def test_grounded_xy_is_saved_then_consumed_by_fresh_same_grasp_attempt(
        monkeypatch, tmp_path, xy_source):
    runner = _runner(monkeypatch, tmp_path)
    runner.max_xy_retries = 2
    runner.xy_update_weight = 0.5
    first = begin_attempt(
        attempt_id="attempt_1", mode=MODE,
        session_record=runner.calibration.record,
        candidate_id="/".join(KEY_A), tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0)
    first.record_stage(
        "grasp_success", True, timestamp_s=2.0,
        evidence_refs={"vlm_observation": "lift.json",
                       "raw_lift_visual": "lift.png"})
    first.record_stage(
        "preinsert_reached", True, timestamp_s=3.0,
        evidence_refs={"trajectory": "transfer.json",
                       "key_socket_pose": "hold.json",
                       "grasp_state": "hand.json",
                       "postlift_preflight": "postlift.json"})
    first.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=4.0,
        evidence_refs={"vlm_observation": "insert.json",
                       "key_depth": "depth.json", "alignment": "axis.json",
                       "force_trace": "force.json"})
    runner._attempt = first
    runner._attempt_dir = runner.output_dir / "attempts" / first.attempt_id
    runner._attempt_dir.mkdir(parents=True)
    assert runner.current_decision().action == "assess_saved_preinsert_xy"
    preinsert = tmp_path / "preinsert.json"
    preinsert.write_text("{}", encoding="utf-8")
    bundle = tmp_path / "preinsert_bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    diagnostic = PreinsertXYDiagnostic(
        status="diagnostic_metric_xy_correction",
        attempt_id=first.attempt_id, candidate_id=first.candidate_id,
        session_calibration_sha256=runner.session_sha256,
        camera_calibration_sha256="camera_hash",
        task_geometry_sha256="geometry_hash", frame_request_id=1,
        preinsert_checkpoint_path=preinsert,
        preinsert_checkpoint_sha256=hashlib.sha256(
            preinsert.read_bytes()).hexdigest(),
        preinsert_bundle_path=bundle,
        preinsert_manifest_sha256=hashlib.sha256(
            (bundle / "manifest.json").read_bytes()).hexdigest(),
        postlift_preflight_path=tmp_path / "postlift.json",
        postlift_preflight_sha256="postlift_hash", frame_binding={},
        joint_sample=SimpleNamespace(to_record=lambda: {}),
        alignment={"bounded_xy_increment_socket_m": [0.0006, -0.0008],
                   "inlier_cameras": ["cam0", "cam1"],
                   "status": "diagnostic_metric_xy_correction",
                   "xy_source": xy_source},
        vlm_observations=())
    source = runner._attempt_dir / "xy_retry_assessments/000/report.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(diagnostic.to_record()), encoding="utf-8")
    runner.record_next_trial_xy_advice(
        diagnostic=diagnostic, diagnostic_report_path=source,
        timestamp_s=5.0)
    assert runner.current_decision().action == "recover_released_key_then_reobserve"
    assert runner.active_attempt.xy_offset_socket_m == (0.0, 0.0)
    assert all(event["stage"] != "xy_retry" for event in runner.active_attempt.events)
    runner._record(lambda row: row.record_stage(
        "reset_success", True, timestamp_s=6.0,
        evidence_refs={"key_pose": "fresh_key.json"}))
    captured = []

    def plan_next(**kwargs):
        captured.append(kwargs)
        return _report(runner, kwargs["key_observation"])

    monkeypatch.setattr(session_runner, "plan_admitted_key_trial", plan_next)
    _plan(runner, _observation("key_2", 7.0))
    assert captured[0]["preferred_candidate"] == KEY_A
    assert captured[0]["retry_candidate"] == KEY_A
    assert captured[0]["retry_pose_stem"] == "000"
    assert KEY_A not in captured[0]["attempted"]
    runner.begin_selected_attempt(attempt_id="attempt_2", started_at_s=7.2)
    assert runner.active_attempt.wrist_correction_socket_m == pytest.approx(
        (0.0003, -0.0004))
    assert runner.active_attempt.xy_offset_socket_m == (0.0, 0.0)
    assert runner._pending_next_trial_xy is None
    assert runner.current_decision().action == "await_lift_observation"

    second = runner._attempt
    second.record_stage(
        "grasp_success", True, timestamp_s=8.0,
        evidence_refs={"vlm_observation": "lift2.json",
                       "raw_lift_visual": "lift2.png"})
    second.record_stage(
        "preinsert_reached", True, timestamp_s=9.0,
        evidence_refs={"trajectory": "transfer2.json",
                       "key_socket_pose": "hold2.json",
                       "grasp_state": "hand2.json",
                       "postlift_preflight": "postlift2.json"})
    second.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.0,
        evidence_refs={"vlm_observation": "insert2.json",
                       "key_depth": "depth2.json", "alignment": "axis2.json",
                       "force_trace": "force2.json"})
    assert runner.current_decision().action == "assess_saved_preinsert_xy"
    preinsert2 = tmp_path / "preinsert2.json"
    preinsert2.write_text("{}", encoding="utf-8")
    bundle2 = tmp_path / "preinsert_bundle2"
    bundle2.mkdir()
    (bundle2 / "manifest.json").write_text("{}", encoding="utf-8")
    diagnostic2 = PreinsertXYDiagnostic(
        status="diagnostic_exploratory_xy_step",
        attempt_id=second.attempt_id, candidate_id=second.candidate_id,
        session_calibration_sha256=runner.session_sha256,
        camera_calibration_sha256="camera_hash",
        task_geometry_sha256="geometry_hash", frame_request_id=2,
        preinsert_checkpoint_path=preinsert2,
        preinsert_checkpoint_sha256=hashlib.sha256(
            preinsert2.read_bytes()).hexdigest(),
        preinsert_bundle_path=bundle2,
        preinsert_manifest_sha256=hashlib.sha256(
            (bundle2 / "manifest.json").read_bytes()).hexdigest(),
        postlift_preflight_path=tmp_path / "postlift2.json",
        postlift_preflight_sha256="postlift_hash", frame_binding={},
        joint_sample=SimpleNamespace(to_record=lambda: {}),
        alignment={"bounded_xy_increment_socket_m": [0.002, 0.0],
                   "inlier_cameras": [],
                   "status": "diagnostic_exploratory_xy_step",
                   "xy_source": "no_metric_measurement_exploratory_search"},
        vlm_observations=())
    source2 = runner._attempt_dir / "xy_retry_assessments/000/report.json"
    source2.parent.mkdir(parents=True)
    source2.write_text(json.dumps(diagnostic2.to_record()), encoding="utf-8")
    runner.record_next_trial_xy_advice(
        diagnostic=diagnostic2, diagnostic_report_path=source2,
        timestamp_s=11.0)
    advice2 = json.loads((runner._attempt_dir /
                          "next_trial_xy_advice.json").read_text())
    assert advice2["prior_wrist_correction_socket_m"] == pytest.approx(
        [0.0003, -0.0004])
    assert advice2["advice_kind"] == "diagnostic_exploratory_xy_step"
    assert advice2["xy_source"] == "no_metric_measurement_exploratory_search"
    assert advice2["supporting_cameras"] == []
    assert advice2["next_wrist_correction_socket_m"] == pytest.approx(
        [0.0013, -0.0004])
    runner._record(lambda row: row.record_stage(
        "reset_success", True, timestamp_s=12.0,
        evidence_refs={"key_pose": "fresh_key3.json"}))
    _plan(runner, _observation("key_3", 13.0))
    runner.begin_selected_attempt(attempt_id="attempt_3", started_at_s=13.2)
    assert runner.active_attempt.wrist_correction_socket_m == pytest.approx(
        (0.0013, -0.0004))


def test_grounded_lateral_session_only_accepts_own_saved_diagnostic(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._attempt = object()
    runner._preflight = object()
    runner._attempt_dir = runner.output_dir / "attempts" / "attempt_1"
    runner._attempt_dir.mkdir(parents=True)
    postlift = runner._attempt_dir / "postlift.json"
    postlift.write_text("{}", encoding="utf-8")
    runner._postlift_report_path = postlift
    monkeypatch.setattr(runner, "current_decision", lambda: SimpleNamespace(
        action="guarded_withdrawal_then_xy_assessment"))
    diagnostic = SimpleNamespace(postlift_preflight_path=postlift)
    source = (runner._attempt_dir / "xy_retry_assessments" / "000" /
              "report.json")
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    calls = []
    result = object()
    monkeypatch.setattr(session_runner,
                        "plan_grounded_lateral_from_withdrawal",
                        lambda **kwargs: calls.append(kwargs) or result)
    def save(_result, output):
        output.mkdir(parents=True, exist_ok=False)
        return output
    monkeypatch.setattr(session_runner, "write_grounded_lateral_preflight",
                        save)
    args = dict(
        planner=object(), diagnostic=diagnostic,
        diagnostic_report_path=source, joint_sample=object(),
        decision_timestamp_s=2., limits=object(),
        max_hold_joint_drift_rad=.01,
        max_grounded_tip_error_m=.001,
        max_grounded_axis_error_deg=1.,
        max_path_deviation_m=.0001,
        max_hold_height_deviation_m=.0001,
        max_hold_rotation_deg=1.)
    with pytest.raises(ValueError, match="not from this attempt"):
        runner.prepare_grounded_lateral_hold_preflight(**{
            **args, "diagnostic_report_path": tmp_path / "outside.json"})
    assert calls == []
    assert runner.prepare_grounded_lateral_hold_preflight(**args) is result
    assert len(calls) == 1
    assert calls[0]["diagnostic_report_path"] == source.resolve()
    assert (runner._attempt_dir / "lateral_hold_preflights/000").is_dir()
    assert runner._lateral_preflight_index == 1
    with pytest.raises(ValueError, match="already used"):
        runner.prepare_grounded_lateral_hold_preflight(**args)


def test_postshift_session_consumes_new_capture_without_attempt_label(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3")
    runner._attempt_dir = runner.output_dir / "attempts" / "attempt_1"
    runner._attempt_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "current_decision", lambda: SimpleNamespace(
        action="guarded_withdrawal_then_xy_assessment"))
    diagnostic = (runner._attempt_dir / "xy_retry_assessments" /
                  "000/report.json").resolve()
    runner._lateral_used_diagnostic_reports.add(diagnostic)
    source = (runner._attempt_dir / "lateral_hold_preflights" /
              "000/report.json").resolve()
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    capture = tmp_path / "new_camera_capture"
    plan = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3",
        diagnostic_report_path=diagnostic)
    calls = []
    result = object()
    monkeypatch.setattr(session_runner, "assess_postshift_alignment",
                        lambda **kwargs: calls.append(kwargs) or result)
    monkeypatch.setattr(session_runner, "write_postshift_checkpoint",
                        lambda _result, output: output.mkdir(
                            parents=True, exist_ok=False))
    args = dict(
        plan=plan, plan_report_path=source,
        execution_log_path=tmp_path / "execution.json",
        capture_dir=capture, joint_sample=object(),
        decision_timestamp_s=3., planner=object(),
        intrinsics_full={}, extrinsics_full={}, backend=object(),
        alignment_limits=object(), max_capture_skew_s=.01,
        max_execution_observation_gap_s=.2, max_frame_age_s=.2,
        max_joint_frame_skew_s=.02, max_arm_hand_skew_s=.02,
        max_hand_command_error_raw=30., max_arm_velocity_rad_s=.05,
        max_joint_goal_error_rad=.01,
        max_goal_translation_error_m=.002,
        max_goal_rotation_error_deg=1.,
        max_visual_lateral_error_m=.005,
        max_visual_axis_tilt_deg=2., max_grounded_tip_error_m=.001)
    assert runner.assess_postshift_lateral_alignment(**args) is result
    assert len(calls) == 1
    assert (runner._attempt_dir / "postshift_checkpoints/000").is_dir()
    assert runner._postshift_checkpoint_index == 1
    with pytest.raises(ValueError, match="not a new session event"):
        runner.assess_postshift_lateral_alignment(**args)


def test_postshift_20mm_session_requires_its_saved_checkpoint(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3")
    runner._preflight = object()
    runner._attempt_dir = runner.output_dir / "attempts" / "attempt_1"
    runner._attempt_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "current_decision", lambda: SimpleNamespace(
        action="guarded_withdrawal_then_xy_assessment"))
    checks = []
    monkeypatch.setattr(runner, "verify_current_preflight_evidence",
                        lambda: checks.append("trial") or {})
    source = (runner._attempt_dir / "postshift_checkpoints" /
              "000/report.json").resolve()
    source.parent.mkdir(parents=True)
    source.write_text("{}", encoding="utf-8")
    capture = tmp_path / "new_camera_capture"
    runner._postshift_saved_reports.add(source)
    runner._postshift_used_capture_dirs.add(capture.resolve())
    checkpoint = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3",
        capture_dir=capture)
    plan = SimpleNamespace(attempt_id="attempt_1",
                           candidate_id="table/0/3")
    calls = []
    result = object()
    monkeypatch.setattr(session_runner,
                        "plan_postshift_insertion_preflight",
                        lambda **kwargs: calls.append(kwargs) or result)
    monkeypatch.setattr(session_runner,
                        "write_postshift_insertion_preflight",
                        lambda _result, output: output.mkdir(
                            parents=True, exist_ok=False))
    monkeypatch.setattr(session_runner,
                        "verify_postshift_insertion_preflight",
                        lambda *_args, **_kwargs: {})
    args = dict(
        planner=object(), shift_plan=plan, checkpoint=checkpoint,
        checkpoint_report_path=source, bounds=object(),
        max_visual_tip_error_m=.001,
        max_visual_axis_error_deg=1.,
        max_axis_prior_residual_deg=2.,
        max_hold_height_delta_m=.005,
        max_preinsert_hand_rotation_deg=1.)
    assert runner.prepare_postshift_insertion_preflight(**args) is result
    assert checks == ["trial"]
    assert calls[0]["checkpoint_report_path"] == source
    assert runner._postshift_20mm_index == 1
    assert (runner._attempt_dir / "postshift_20mm_preflights/000").is_dir()
    with pytest.raises(ValueError, match="not a new session checkpoint"):
        runner.prepare_postshift_insertion_preflight(**args)
    with pytest.raises(ValueError, match="not a new session checkpoint"):
        runner.prepare_postshift_insertion_preflight(**{
            **args, "checkpoint_report_path": tmp_path / "outside.json"})


def test_postshift_handoff_session_uses_only_its_saved_plan(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3")
    runner._attempt_dir = runner.output_dir / "attempts" / "attempt_1"
    runner._attempt_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "current_decision", lambda: SimpleNamespace(
        action="guarded_withdrawal_then_xy_assessment"))
    checks = []
    monkeypatch.setattr(runner, "verify_current_preflight_evidence",
                        lambda: checks.append("trial") or {})
    checkpoint_report = (runner._attempt_dir / "postshift_checkpoints" /
                         "000/report.json").resolve()
    checkpoint_report.parent.mkdir(parents=True)
    checkpoint_report.write_text("{}", encoding="utf-8")
    runner._postshift_20mm_used_reports.add(checkpoint_report)
    plan_report = (runner._attempt_dir / "postshift_20mm_preflights" /
                   "000/report.json").resolve()
    plan_report.parent.mkdir(parents=True)
    plan_report.write_text("{}", encoding="utf-8")
    preflight = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3",
        status="sampled_postshift_20mm_preflight_pass",
        postshift_report_path=checkpoint_report)
    calls = []

    def write(*, output_dir, **kwargs):
        calls.append(kwargs)
        output_dir.mkdir(parents=True, exist_ok=False)
        path = output_dir / "report.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(session_runner, "write_postshift_path_handoff", write)
    monkeypatch.setattr(session_runner, "verify_postshift_path_handoff",
                        lambda *_args, **_kwargs: {})
    args = dict(
        preflight=preflight, preflight_report_path=plan_report,
        checkpoint=object(), shift_plan=object(), measured_start=object(),
        decision_timestamp_s=100., max_state_age_s=.1,
        max_start_joint_error_rad=.01, max_hand_drift_raw=5.,
        max_arm_hand_skew_s=.01, max_hand_command_error_raw=10.,
        max_arm_velocity_rad_s=.01)
    output = runner.prepare_postshift_path_handoff(**args)
    assert output == (runner._attempt_dir /
                      "postshift_path_handoffs/000/report.json")
    assert calls[0]["preflight_report_path"] == plan_report
    assert checks == ["trial"]
    with pytest.raises(ValueError, match="not this attempt's new plan"):
        runner.prepare_postshift_path_handoff(**args)
    with pytest.raises(ValueError, match="not this attempt's new plan"):
        runner.prepare_postshift_path_handoff(**{
            **args, "preflight_report_path": tmp_path / "outside.json"})


def test_postshift_arrival_session_requires_new_transfer_capture(
        monkeypatch, tmp_path):
    from precision_insertion import postshift_arrival_checkpoint as arrival

    runner = _runner(monkeypatch, tmp_path)
    runner._attempt = SimpleNamespace(
        attempt_id="attempt_1", candidate_id="table/0/3")
    runner._attempt_dir = runner.output_dir / "attempts" / "attempt_1"
    runner._attempt_dir.mkdir(parents=True)
    monkeypatch.setattr(runner, "current_decision", lambda: SimpleNamespace(
        action="guarded_withdrawal_then_xy_assessment"))
    monkeypatch.setattr(runner, "verify_current_preflight_evidence",
                        lambda: {})
    handoff = (runner._attempt_dir / "postshift_path_handoffs" /
               "000/report.json").resolve()
    handoff.parent.mkdir(parents=True)
    handoff.write_text("{}", encoding="utf-8")
    runner._postshift_saved_handoffs.add(handoff)
    transfer = (runner._attempt_dir / "postshift_transfer_executions" /
                "000/execution.json").resolve()
    capture = tmp_path / "new_retry_capture"
    result = object()
    calls = []
    monkeypatch.setattr(arrival, "assess_postshift_arrival",
                        lambda **kwargs: calls.append(kwargs) or result)

    def write(_result, output):
        output.mkdir(parents=True, exist_ok=False)
        path = output / "report.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr(arrival, "write_postshift_arrival_checkpoint", write)
    monkeypatch.setattr(arrival, "verify_postshift_arrival_checkpoint",
                        lambda *_args, **_kwargs: {})
    args = dict(
        preflight=SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/3"),
        checkpoint=object(), shift_plan=object(),
        handoff_report_path=handoff, transfer_execution_path=transfer,
        capture_dir=capture, joint_sample=object(),
        decision_timestamp_s=101., planner=object(),
        intrinsics_full={}, extrinsics_full={}, backend=object(),
        alignment_limits=object(), max_capture_skew_s=.01,
        max_execution_observation_gap_s=.2, max_frame_age_s=.2,
        max_joint_frame_skew_s=.02, max_arm_hand_skew_s=.02,
        max_hand_command_error_raw=10., max_arm_velocity_rad_s=.01,
        max_joint_goal_error_rad=.01,
        max_goal_translation_error_m=.005,
        max_goal_rotation_error_deg=1.,
        max_visual_lateral_error_m=.001,
        max_visual_axis_tilt_deg=1.,
        max_grounded_tip_error_m=.001)
    assert runner.assess_postshift_transfer_arrival(**args) is result
    assert calls[0]["capture_dir"] == capture.resolve()
    assert runner._postshift_arrival_index == 1
    with pytest.raises(ValueError, match="not a new session event"):
        runner.assess_postshift_transfer_arrival(**args)


def test_attempt_rejects_changed_preflight_report_and_binding(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    report = runner._preflight_report_path
    report.write_text('{"selected_candidate_key": ["fake"]}\n',
                      encoding="utf-8")
    with pytest.raises(ValueError, match="preflight report changed"):
        runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    assert not (runner.output_dir / "attempts").exists()

    report.write_text("{}\n", encoding="utf-8")
    binding = runner._preflight_binding_path
    binding.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="binding changed"):
        runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    assert not (runner.output_dir / "attempts").exists()


def test_measured_start_state_is_hashed_and_checked_before_attempt(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    observation = _observation("key_1", 10.0)
    evidence = _save_evidence(runner, observation)
    raw = np.full(6, 500., dtype=float)
    q = np.concatenate((np.zeros(7), convert_inspire_raw(raw[None, :])[0]))
    state = LiveRobotState(
        q, np.zeros(7), 10., 10., 500., 10., raw.copy(), raw.copy(),
        0., np.zeros(6))
    state_limits = {"max_arm_hand_skew_s": .02,
                    "max_hand_command_error_raw": 20.,
                    "max_arm_velocity_rad_s": .1}
    arguments = dict(
        planner=object(), key_observation=observation,
        key_evidence_dir=evidence, live_start_q=q.copy(),
        start_q_acquisition_timestamp_s=10., max_key_state_skew_s=.05,
        limits=object(), max_pose_error_deg=5., axial_waypoint_step_m=.002,
        measured_start_state=state, measured_state_limits=state_limits)
    wrong_q = q.copy()
    wrong_q[0] = .1
    with pytest.raises(ValueError, match="differ from measured feedback"):
        runner.preflight_next_key(**{**arguments, "live_start_q": wrong_q})
    runner.preflight_next_key(**arguments)
    verified = runner.verify_current_preflight_evidence()
    assert verified["measured_start_state_sha256"]
    state_file = runner._preflight_report_path.parent / "measured_start_state.json"
    state_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="measured start state changed"):
        runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    assert not (runner.output_dir / "attempts").exists()


def test_actual_trial_scene_hash_is_rechecked_before_repose(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    observation = _observation("key_1", 10.0)
    _save_evidence(runner, observation)
    limits = PathAuditLimits(
        max_joint_step_rad=.02, max_wrist_step_m=.005,
        max_wrist_rotation_deg=1., goal_position_tolerance_m=.001,
        goal_rotation_tolerance_deg=1., axial_lateral_tolerance_m=.001,
        axial_rotation_tolerance_deg=1., minimum_hand_clearance_m=.001)

    def real_report(**kwargs):
        return TrialPreflight(
            status="no_eligible_pose_in_catalog", pose_class={"stem": "000"},
            attempted_candidates=(), selected_candidate_key=None,
            repose_target_stems=(), insertion_plan=None, pickup_plan=None,
            trial_scene={"mesh": {"target": {"pose": [0, 0, 0, 0, 0, 0, 1]}}},
            key_observation_id=observation.capture_id,
            key_capture_timestamp_s=10.,
            start_q_acquisition_timestamp_s=10.,
            max_key_state_skew_s=.05, key_pose_world=np.eye(4),
            live_start_q=np.zeros(13), attempted_before_trial=(),
            covered_scenes=(), limits=limits, max_pose_error_deg=5.,
            axial_waypoint_step_m=.002, max_candidate_attempts=None,
            session_calibration_sha256=runner.session_sha256,
            catalog_sha256=runner.catalog_sha256)

    monkeypatch.setattr(session_runner, "plan_admitted_key_trial", real_report)
    monkeypatch.setattr(session_runner, "write_trial_preflight_artifacts",
                        write_trial_preflight_artifacts)
    _plan(runner, observation)
    runner.verify_current_preflight_evidence()
    scene = runner._preflight_report_path.parent / "trial_scene.json"
    scene.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trial_scene artifact changed"):
        runner.verify_current_preflight_evidence()


def _bind_postlift(runner):
    """Existing label tests start after post-lift planning; bind that gate."""
    report = runner._attempt_dir / "postlift_preflight/report.json"
    report.parent.mkdir()
    report.write_text('{"status":"sampled_postlift_preflight_pass"}\n',
                      encoding="utf-8")
    runner._postlift_preflight = SimpleNamespace(
        status="sampled_postlift_preflight_pass",
        key_capture_timestamp_s=10.35, joint_timestamp_s=10.35)
    runner._postlift_report_path = report
    runner._postlift_report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    return str(report)


def _bind_preinsert(runner, monkeypatch):
    """Unit-test the runner's checkpoint binding without GPU/camera hardware."""
    report = runner._attempt_dir / "preinsert_assessments/000/report.json"
    report.parent.mkdir(parents=True)
    transfer = runner._attempt_dir / "transfer_execution.json"
    transfer.write_text('{"completed_at_s":10.35}\n', encoding="utf-8")
    raw_bundle = runner._attempt_dir / "preinsert_raw"
    raw_bundle.mkdir()
    (raw_bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    payload = {
        "attempt_id": runner._attempt.attempt_id,
        "candidate_id": runner._attempt.candidate_id,
        "preinsert_reached": True,
        "observation_completed_at_s": 10.36,
        "transfer_execution_path": str(transfer),
        "raw_bundle": str(raw_bundle),
    }
    report.write_text(json.dumps(payload), encoding="utf-8")
    runner._preinsert_checkpoint = SimpleNamespace(**payload)
    runner._preinsert_report_path = report
    runner._preinsert_report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    monkeypatch.setattr(
        session_runner, "verify_preinsert_checkpoint",
        lambda path: json.loads(path.read_text(encoding="utf-8")))
    return {
        "trajectory": str(transfer), "grasp_state": str(transfer),
        "key_socket_pose": str(report), "preinsert_checkpoint": str(report),
        "preinsert_image": str(raw_bundle / "manifest.json"),
        "postlift_preflight": str(runner._postlift_report_path),
    }


def _bind_lift(runner):
    """Existing stage tests inject an already checked VLM lift bundle."""
    report = runner._attempt_dir / "lift_assessments/000/report.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"grasp_success":true}\n', encoding="utf-8")
    runner._lift_checkpoint = SimpleNamespace(
        attempt_id=runner._attempt.attempt_id,
        candidate_id=runner._attempt.candidate_id,
        lift_completed_at_s=10.3, grasp_success=True)
    runner._lift_report_path = report
    runner._lift_report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
    lift_log = runner._attempt_dir / "lift_execution.json"
    lift_log.write_text('{"completed_at_s":10.3}\n', encoding="utf-8")
    runner._lift_execution_path = lift_log
    runner._lift_execution_sha256 = hashlib.sha256(
        lift_log.read_bytes()).hexdigest()
    return {"vlm_observation": str(report), "key_wrist_check": str(report),
            "lift_execution": str(lift_log)}


def test_observed_failed_grasp_is_excluded_only_after_saved_event(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    seen = []
    def fake_plan(**kwargs):
        seen.append(kwargs["attempted"])
        key = KEY_A if len(seen) == 1 else KEY_B
        return _report(runner, kwargs["key_observation"], key=key)
    monkeypatch.setattr(session_runner, "plan_admitted_key_trial", fake_plan)
    _plan(runner, _observation("key_1", 10.0))
    assert runner.current_decision().action == "execution_gate_required"
    with pytest.raises(ValueError, match="cannot start before"):
        runner.begin_selected_attempt(attempt_id="too_early", started_at_s=10.0)
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    assert runner.attempted_candidates == ()
    with pytest.raises(ValueError, match="no verified return"):
        _plan(runner, _observation("premature_key_2", 11.0))
    runner.observe_stage(
        "grasp_success", False, timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/miss.json"})
    assert runner.attempted_candidates == (KEY_A,)
    assert runner.current_decision().action == "recover_key_then_reobserve"
    snapshot = tmp_path / "session/attempts/attempt_1/state_001.json"
    assert json.loads(snapshot.read_text())["grasp_success"] is False
    binding = json.loads((tmp_path / "session/attempts/attempt_1/"
                          "preflight_binding.json").read_text())
    assert binding["candidate_id"] == "table/0/3"
    with pytest.raises(ValueError, match="no verified return"):
        _plan(runner, _observation("still_premature_key_2", 11.0))
    monkeypatch.setattr(
        session_runner, "classify_key_tabletop_pose",
        lambda **_kwargs: {"stem": "000", "rotation_error_deg": 0.0})
    monkeypatch.setattr(
        session_runner, "validate_repose_rest_target",
        lambda **_kwargs: {"key_socket_clearance_m": 0.1})
    monkeypatch.setattr(
        session_runner, "_load_mesh",
        lambda _path: SimpleNamespace(bounds=np.array([
            [-0.01, -0.01, 0.0], [0.01, 0.01, 0.08]])))
    recovery = tmp_path / "missed_grasp_recovery.json"
    recovery.write_text(json.dumps({
        "schema": "precision_insertion_missed_grasp_clear_view_v1",
        "attempt_id": "attempt_1", "candidate_id": "table/0/3",
        "completed_at_s": 10.7, "clear_view_reached": True,
        "hand_open": True, "measured_state": {
            "schema": "precision_insertion_live_robot_state_v1",
            "sample_timestamp_s": 10.7},
    }), encoding="utf-8")
    landed = _observation("reset_key", 11.0)
    runner.observe_reset_landing(
        key_observation=landed, key_evidence_dir=_save_evidence(runner, landed),
        recovery_log_path=recovery, timestamp_s=11.05,
        max_pose_error_deg=5.0, max_return_center_shift_m=0.01,
        support_tolerance_m=0.002,
        minimum_rest_socket_clearance_m=0.005,
        minimum_board_edge_clearance_m=0.005)
    assert runner.current_decision().action == "reobserve_key_and_preflight"
    with pytest.raises(ValueError, match="predates the previous attempt"):
        _plan(runner, _observation("stale_key_2", 10.3))
    _plan(runner, _observation("key_2", 12.0))
    assert seen == [(), (KEY_A,)]
    assert runner.current_decision().candidate_id == "table/0/4"
    assert json.loads((tmp_path / "session/session_run.json").read_text())[
        "robot_ready"] is False


def test_first_postlift_capture_binds_candidate_prior_and_gates_arrival(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", True, timestamp_s=10.3,
        evidence_refs=_bind_lift(runner))
    arrival_refs = {"trajectory": "path/transfer.json",
                    "key_socket_pose": "pose/hold.json",
                    "grasp_state": "pose/grip.json",
                    "postlift_preflight": "path/spoofed.json"}
    with pytest.raises(ValueError, match="passing post-lift plan"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.7,
                             evidence_refs=arrival_refs)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    np.save(candidate_dir / "wrist_se3.npy", np.eye(4))
    monkeypatch.setattr(session_runner, "select_pose_candidates",
                        lambda *_args, **_kwargs: {
                            "status": "candidates_available",
                            "candidates": [{"key": list(KEY_A),
                                            "candidate_dir": str(candidate_dir)}]})
    planner = SimpleNamespace(fk_wrist=lambda _q: np.eye(4))
    hand_raw = np.zeros(6)
    q = np.zeros(13)
    q[7:] = convert_inspire_raw(hand_raw[None, :])[0]
    measured = LiveRobotState(
        q, np.zeros(7), 10.5, 10.5, 12.0, 10.5,
        hand_raw, hand_raw.copy(), 0.0, np.zeros(6))
    feedback_limits = dict(
        max_arm_hand_skew_s=0.05, max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05)
    prior = runner.postlift_candidate_pose_prior(
        planner=planner, joint_sample=measured, **feedback_limits)
    np.testing.assert_allclose(prior, np.eye(4))
    held = replace(
        _observation("held_2", 10.5), phase="held_postlift",
        consistency={"held_pose_prior": {
            "source": "measured_wrist_plus_candidate_grasp",
            "timestamp_s": 10.5, "pose_world": prior.tolist()}})
    evidence_dir = _save_evidence(runner, held)
    wrong_prior = prior.copy()
    wrong_prior[0, 3] = 0.01
    wrong_held = replace(
        held, capture_id="held_wrong", request_id=53,
        consistency={"held_pose_prior": {
            "source": "measured_wrist_plus_candidate_grasp",
            "timestamp_s": 10.5, "pose_world": wrong_prior.tolist()}})
    wrong_evidence = _save_evidence(runner, wrong_held)
    common = dict(
        planner=planner, joint_sample=measured, max_state_skew_s=0.05,
        max_grasp_translation_drift_m=0.003,
        max_grasp_rotation_drift_deg=5.0,
        limits=object(), axial_waypoint_step_m=0.002,
        **feedback_limits)
    with pytest.raises(ValueError, match="differs from selected v8 grasp"):
        runner.prepare_postlift_transfer(
            key_observation=wrong_held, key_evidence_dir=wrong_evidence,
            **common)
    calls = []
    def fake_plan(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            status="sampled_postlift_preflight_pass",
            key_capture_timestamp_s=10.5, joint_timestamp_s=10.5)
    def fake_write(result, output):
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(
            json.dumps({"status": result.status}) + "\n", encoding="utf-8")
    monkeypatch.setattr(session_runner, "plan_postlift_observed_transfer",
                        fake_plan)
    monkeypatch.setattr(session_runner, "write_postlift_preflight", fake_write)
    result = runner.prepare_postlift_transfer(
        key_observation=held, key_evidence_dir=evidence_dir, **common)
    assert result.status == "sampled_postlift_preflight_pass"
    assert runner.current_decision().action == "transfer_execution_gate_required"
    assert calls[0]["key_observation_id"] == "held_2"
    assert calls[0]["joint_sample"] is measured
    report = runner._postlift_report_path
    assert report is not None
    with pytest.raises(ValueError, match="passing post-lift plan"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.7,
                             evidence_refs=arrival_refs)
    arrival_refs["postlift_preflight"] = str(report)
    report.write_text("{}\n", encoding="utf-8")
    assert runner.current_decision().action == "stop_for_review"
    with pytest.raises(ValueError, match="passing post-lift plan"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.7,
                             evidence_refs=arrival_refs)
    report.write_text('{"status": "sampled_postlift_preflight_pass"}\n',
                      encoding="utf-8")
    with pytest.raises(ValueError, match="observed checkpoint"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.7,
                             evidence_refs=arrival_refs)
    arrival_refs = _bind_preinsert(runner, monkeypatch)
    assert runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.7,
        evidence_refs=arrival_refs).labels["preinsert_reached"] is True


def test_candidate_pose_prior_is_available_before_vlm_grasp_label(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    assert runner.current_decision().action == "await_lift_observation"
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    np.save(candidate_dir / "wrist_se3.npy", np.eye(4))
    monkeypatch.setattr(session_runner, "select_pose_candidates",
                        lambda *_args, **_kwargs: {
                            "status": "candidates_available",
                            "candidates": [{"key": list(KEY_A),
                                            "candidate_dir": str(candidate_dir)}]})
    hand_raw = np.zeros(6)
    q = np.zeros(13)
    q[7:] = convert_inspire_raw(hand_raw[None, :])[0]
    measured = LiveRobotState(
        q, np.zeros(7), 10.5, 10.5, 12.0, 10.5,
        hand_raw, hand_raw.copy(), 0.0, np.zeros(6))
    inputs = dict(
        planner=SimpleNamespace(fk_wrist=lambda _q: np.eye(4)),
        joint_sample=measured, max_arm_hand_skew_s=0.05,
        max_hand_command_error_raw=30.0, max_arm_velocity_rad_s=0.05)
    np.testing.assert_allclose(
        runner.postlift_candidate_pose_prior(**inputs), np.eye(4))
    assert runner.active_attempt.labels["grasp_success"] is None
    runner.observe_stage(
        "grasp_success", False, timestamp_s=10.6,
        evidence_refs={"vlm_observation": "vlm/miss.json"})
    with pytest.raises(ValueError, match="active selected lift"):
        runner.postlift_candidate_pose_prior(**inputs)


def test_lift_checkpoint_abstains_then_records_hash_bound_positive_label(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    with pytest.raises(ValueError, match="bound lift checkpoint"):
        runner.observe_stage(
            "grasp_success", True, timestamp_s=10.4,
            evidence_refs={"vlm_observation": "vlm.json",
                           "key_wrist_check": "pose.json"})
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    np.save(candidate_dir / "wrist_se3.npy", np.eye(4))
    monkeypatch.setattr(session_runner, "select_pose_candidates",
                        lambda *_args, **_kwargs: {
                            "status": "candidates_available",
                            "candidates": [{"key": list(KEY_A),
                                            "candidate_dir": str(candidate_dir)}]})
    raw = np.zeros(6)
    q = np.zeros(13)
    q[7:] = convert_inspire_raw(raw[None, :])[0]
    joint = LiveRobotState(
        q, np.zeros(7), 10.6, 10.6, 12.0, 10.6,
        raw, raw.copy(), 0.0, np.zeros(6))
    lift_log = tmp_path / "lift_execution.json"
    valid_log = {
        "schema": "precision_insertion_lift_execution_v1",
        "attempt_id": "attempt_1", "candidate_id": "table/0/3",
        "trajectory_complete": True, "force_abort": False,
        "completed_at_s": 10.4,
    }
    lift_log.write_text(json.dumps({**valid_log, "force_abort": True}),
                        encoding="utf-8")
    labels = [None, True]
    calls = []
    def fake_assess(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/3",
            lift_completed_at_s=10.4, grasp_success=labels.pop(0),
            after_capture_id=kwargs["after_observation"].capture_id,
            evidence_kind="foundpose_key_rise")
    def fake_write(result, output):
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(
            json.dumps({"grasp_success": result.grasp_success}) + "\n",
            encoding="utf-8")
    monkeypatch.setattr(session_runner, "assess_lift_checkpoint", fake_assess)
    monkeypatch.setattr(session_runner, "write_lift_checkpoint", fake_write)
    monkeypatch.setattr(session_runner, "verify_lift_checkpoint",
                        lambda path: json.loads(path.read_text()))
    common = dict(
        planner=SimpleNamespace(fk_wrist=lambda _q: np.eye(4)),
        joint_sample=joint, backend=object(),
        lift_execution_log_path=lift_log,
        decision_timestamp_s=10.7,
        max_state_skew_s=0.05, max_phase_skew_s=0.05,
        min_center_rise_m=0.03,
        max_arm_hand_skew_s=0.05,
        max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05)
    unknown = replace(_observation("held_unknown", 10.6),
                      phase="held_postlift")
    with pytest.raises(ValueError, match="non-aborted execution log"):
        runner.prepare_observed_lift_label(
            after_observation=unknown, after_key_evidence_dir=tmp_path,
            **common)
    assert calls == []
    lift_log.write_text(json.dumps(valid_log), encoding="utf-8")
    result = runner.prepare_observed_lift_label(
        after_observation=unknown, after_key_evidence_dir=tmp_path,
        **common)
    assert result.grasp_success is None
    assert runner.active_attempt.labels["grasp_success"] is None
    assert runner.current_decision().action == "await_lift_observation"
    with pytest.raises(ValueError, match="new held-key capture"):
        runner.prepare_observed_lift_label(
            after_observation=unknown, after_key_evidence_dir=tmp_path,
            **common)
    held = replace(_observation("held_good", 10.8), phase="held_postlift")
    fresh_joint = replace(
        joint, sample_timestamp_s=10.8,
        arm_timestamp_s=10.8, hand_timestamp_s=10.8)
    result = runner.prepare_observed_lift_label(
        after_observation=held, after_key_evidence_dir=tmp_path,
        **{**common, "joint_sample": fresh_joint})
    assert result.grasp_success is True
    assert runner.active_attempt.labels["grasp_success"] is True
    assert runner.active_attempt.events[0]["timestamp_s"] == 10.4
    assert runner.current_decision().action == (
        "postlift_observed_preflight_required")
    assert calls[1]["before_capture_id"] == "key_1"
    assert calls[1]["lift_completed_at_s"] == 10.4
    assert (runner._attempt_dir /
            "lift_assessments/001/execution_binding.json").is_file()


def test_budgeted_preflight_continues_same_capture_but_new_capture_resets_rejects(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    seen = []
    def fake_plan(**kwargs):
        seen.append(kwargs["attempted"])
        observation = kwargs["key_observation"]
        if len(seen) == 1:
            return _report(
                runner, observation, status="candidate_budget_exhausted",
                attempted=({"key": list(KEY_A),
                            "pickup_preflight_pass": False},))
        return _report(runner, observation, key=KEY_B)
    monkeypatch.setattr(session_runner, "plan_admitted_key_trial", fake_plan)
    first = _observation("key_1", 10.0)
    _plan(runner, first)
    assert runner.current_decision().action == "continue_candidate_preflight"
    _plan(runner, first)
    assert seen[1] == (KEY_A,)
    with pytest.raises(ValueError, match="pilot prefix"):
        _plan(runner, first)
    _plan(runner, _observation("key_2", 11.0))
    assert seen[2] == ()


def test_initial_preferred_v8_candidate_only_applies_to_first_capture(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner._initial_preferred_candidate = KEY_B
    seen = []

    def fake_plan(**kwargs):
        seen.append(kwargs["preferred_candidate"])
        return _report(runner, kwargs["key_observation"], key=KEY_A)

    monkeypatch.setattr(session_runner, "plan_admitted_key_trial", fake_plan)
    _plan(runner, _observation("key_1", 10.0))
    _plan(runner, _observation("key_2", 11.0))
    assert seen == [KEY_B, None]


def test_verified_insertion_ends_session_without_reset_or_next_trial(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", True, timestamp_s=10.3,
        evidence_refs=_bind_lift(runner))
    postlift_report = _bind_postlift(runner)
    arrival_refs = _bind_preinsert(runner, monkeypatch)
    runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.4,
        evidence_refs=arrival_refs)
    runner._record(lambda row: row.record_insertion_evidence(
        InsertionEvidence("normal_appearance", (0.0201, 0.021),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"}))
    assert runner.current_decision().action == "session_success"
    with pytest.raises(ValueError, match="no verified return"):
        _plan(runner, _observation("key_2", 11.0))
    with pytest.raises(ValueError, match="observe_reset_landing"):
        runner.observe_stage("reset_success", True, timestamp_s=10.8,
                             evidence_refs={"key_pose": "pose/reset.json"})
    with pytest.raises(ValueError, match="ended this session"):
        runner.observe_reset_landing(
            key_observation=_observation("reset_key", 11.0),
            key_evidence_dir=tmp_path / "unused",
            recovery_log_path=tmp_path / "unused.json",
            timestamp_s=11.05, max_pose_error_deg=5.0,
            max_return_center_shift_m=0.01, support_tolerance_m=0.002,
            minimum_rest_socket_clearance_m=0.005,
            minimum_board_edge_clearance_m=0.005)
    with pytest.raises(ValueError, match="ended this session"):
        runner.observe_stage(
            "release_success", False, timestamp_s=11.1,
            evidence_refs={"release_observation": "vlm/release.json"})
    assert not (tmp_path / "session/attempts/attempt_1/"
                "reset_landing.json").exists()


@pytest.mark.parametrize(
    ("landed_stem", "shift_m"), [("001", 0.0), ("000", 0.03)])
def test_supervised_reset_rejects_wrong_class_or_displaced_key(
        monkeypatch, tmp_path, landed_stem, shift_m):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", False, timestamp_s=10.3,
        evidence_refs={"vlm_observation": "vlm/miss.json"})
    monkeypatch.setattr(
        session_runner, "classify_key_tabletop_pose",
        lambda **_kwargs: {"stem": landed_stem,
                           "rotation_error_deg": 0.0})
    monkeypatch.setattr(
        session_runner, "validate_repose_rest_target",
        lambda **_kwargs: {"key_socket_clearance_m": 0.1})
    monkeypatch.setattr(
        session_runner, "_load_mesh",
        lambda _path: SimpleNamespace(bounds=np.array([
            [-0.01, -0.01, 0.0], [0.01, 0.01, 0.08]])))
    recovery = tmp_path / "manual_return.json"
    recovery.write_text(json.dumps({
        "schema": "precision_insertion_supervised_reset_v1",
        "attempt_id": "attempt_1", "method": "supervised_manual_return",
        "reviewed_by": "operator", "completed_at_s": 10.7,
        "socket_clear": True, "hand_open": True,
        "key_removed_from_socket": False,
    }), encoding="utf-8")
    pose = np.eye(4)
    pose[0, 3] = shift_m
    landed = replace(_observation("reset_key", 11.0), pose_world=pose)
    evidence_dir = _save_evidence(runner, landed)
    result = runner.observe_reset_landing(
        key_observation=landed, key_evidence_dir=evidence_dir,
        recovery_log_path=recovery, timestamp_s=11.05,
        max_pose_error_deg=5.0, max_return_center_shift_m=0.01,
        support_tolerance_m=0.002,
        minimum_rest_socket_clearance_m=0.005,
        minimum_board_edge_clearance_m=0.005)
    assert result.labels["reset_success"] is False
    assert runner.current_decision().action == "stop_for_review"


def test_missed_grasp_retries_from_fresh_supported_pose_without_restoration(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", False, timestamp_s=10.3,
        evidence_refs={"vlm_observation": "vlm/miss.json"})
    assert runner.current_decision().action == "recover_key_then_reobserve"
    monkeypatch.setattr(
        session_runner, "classify_key_tabletop_pose",
        lambda **_kwargs: {"stem": "001", "rotation_error_deg": 0.0})
    monkeypatch.setattr(
        session_runner, "validate_repose_rest_target",
        lambda **_kwargs: {"key_socket_clearance_m": 0.1})
    monkeypatch.setattr(
        session_runner, "_load_mesh",
        lambda _path: SimpleNamespace(bounds=np.array([
            [-0.01, -0.01, 0.0], [0.01, 0.01, 0.08]])))
    recovery = tmp_path / "missed_grasp_recovery.json"
    recovery.write_text(json.dumps({
        "schema": "precision_insertion_missed_grasp_clear_view_v1",
        "attempt_id": "attempt_1",
        "candidate_id": runner.active_attempt.candidate_id,
        "completed_at_s": 10.7,
        "clear_view_reached": True, "hand_open": True,
        "measured_state": {
            "schema": "precision_insertion_live_robot_state_v1",
            "sample_timestamp_s": 10.7},
    }), encoding="utf-8")
    pose = np.eye(4)
    pose[0, 3] = 0.03
    landed = replace(_observation("reset_key", 11.0), pose_world=pose)
    evidence_dir = _save_evidence(runner, landed)
    result = runner.observe_reset_landing(
        key_observation=landed, key_evidence_dir=evidence_dir,
        recovery_log_path=recovery, timestamp_s=11.05,
        max_pose_error_deg=5.0, max_return_center_shift_m=0.01,
        support_tolerance_m=0.002,
        minimum_rest_socket_clearance_m=0.005,
        minimum_board_edge_clearance_m=0.005)
    assert result.labels["reset_success"] is True
    assert runner.current_decision().action == "reobserve_key_and_preflight"
    saved = json.loads((runner._attempt_dir / "reset_landing.json").read_text())
    assert saved["recovery_mode"] == "autodex_missed_grasp_clear_view"
    assert saved["reset_basis"] == "observed_supported_key_pose"
    assert saved["center_xy_displacement_m"] > 0.01


def test_failed_insertion_auto_return_observation_restarts_with_fresh_key(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner.max_xy_retries = 0
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage("grasp_success", True, timestamp_s=10.3,
                         evidence_refs=_bind_lift(runner))
    postlift = Path(_bind_postlift(runner))
    runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.4,
        evidence_refs=_bind_preinsert(runner, monkeypatch))
    release = tmp_path / "poststroke_release.json"
    release.write_text('{"released":true}\n', encoding="utf-8")
    runner._record(lambda row: row.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json",
                       "release_execution": str(release.resolve())}))
    assert runner.current_decision().action == (
        "recover_released_key_then_reobserve")
    raw = np.full(6, 500.)
    q = np.r_[np.zeros(7), convert_inspire_raw(raw[None])[0]]
    state = LiveRobotState(q, np.zeros(7), 10.7, 10.7, 100., 10.7,
                           raw, raw.copy(), 0., np.zeros(6))
    execution = tmp_path / "released_recovery.json"
    execution.write_text(json.dumps({
        "schema": "precision_insertion_released_failure_clear_view_v1",
        "attempt_id": "attempt_1", "candidate_id": "table/0/3",
        "completed_at_s": 10.7, "hand_open": True,
        "clear_view_reached": True, "measured_state": state.to_record(),
        "poststroke_release": {
            "path": str(release.resolve()),
            "sha256": hashlib.sha256(release.read_bytes()).hexdigest()},
    }), encoding="utf-8")
    monkeypatch.setattr(
        session_runner, "classify_key_tabletop_pose",
        lambda **_kwargs: {"stem": "000", "rotation_error_deg": 0.0})
    monkeypatch.setattr(
        session_runner, "validate_repose_rest_target",
        lambda **_kwargs: {"key_socket_clearance_m": 0.1})
    monkeypatch.setattr(
        session_runner, "_load_mesh",
        lambda _path: SimpleNamespace(bounds=np.array([
            [-0.01, -0.01, 0.0], [0.01, 0.01, 0.08]])))
    landed = _observation("reset_key", 11.0)
    evidence_dir = _save_evidence(runner, landed)
    result = runner.observe_reset_landing(
        key_observation=landed, key_evidence_dir=evidence_dir,
        recovery_log_path=execution, timestamp_s=11.05,
        max_pose_error_deg=5.0, max_return_center_shift_m=0.01,
        support_tolerance_m=0.002,
        minimum_rest_socket_clearance_m=0.005,
        minimum_board_edge_clearance_m=0.005)
    assert result.labels["reset_success"] is True
    assert runner.current_decision().action == "reobserve_key_and_preflight"
    saved = json.loads((runner._attempt_dir / "reset_landing.json").read_text())
    assert saved["recovery_mode"] == "released_key_clear_view_then_restore"


def test_unknown_preinsert_checkpoint_routes_to_recovery_not_repeat_transfer(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage("grasp_success", True, timestamp_s=10.3,
                         evidence_refs=_bind_lift(runner))
    _bind_postlift(runner)
    runner.observe_stage(
        "preinsert_reached", None, timestamp_s=10.4,
        evidence_refs={"preinsert_checkpoint": "vlm/unknown.json"})
    assert runner.current_decision().action == "recover_key_then_reobserve"


@pytest.mark.skip(reason="held-key return was replaced by released-key restoration")
def test_failed_return_plan_reuses_postlift_relation_and_frozen_world(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    runner.max_xy_retries = 0
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage("grasp_success", True, timestamp_s=10.3,
                         evidence_refs=_bind_lift(runner))
    _bind_postlift(runner)
    runner._postlift_preflight.relation = SimpleNamespace(T_key_hand=np.eye(4))
    runner.observe_stage("preinsert_reached", True, timestamp_s=10.4,
                         evidence_refs=_bind_preinsert(runner, monkeypatch))
    runner._record(lambda row: row.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"}))
    withdrawal = tmp_path / "withdrawal.json"
    withdrawal.write_text(json.dumps({
        "schema": "precision_insertion_guarded_withdrawal_v1",
        "attempt_id": "attempt_1", "candidate_id": "table/0/3",
        "status": "withdrawn_to_preinsert_hold",
        "source": "commissioned_guarded_controller",
        "key_still_held": True, "safety_abort": False,
        "completed_at_s": 10.6,
    }), encoding="utf-8")
    raw = np.full(6, 500.)
    q = np.r_[np.zeros(7), convert_inspire_raw(raw[None])[0]]
    state = LiveRobotState(q, np.zeros(7), 10.7, 10.7, 100., 10.7,
                           raw, raw.copy(), 0., np.zeros(6))
    limits = PathAuditLimits(.1, .01, 5., .001, 2., .001, 2., .001)
    held = SimpleNamespace(status="sampled_held_path_pass_release_unplanned")
    release = SimpleNamespace(
        status="nominal_release_exit_path_pass_drop_unobserved")
    calls = []

    def plan_held(**kwargs):
        calls.append(("held", kwargs))
        return held

    def plan_release(**kwargs):
        calls.append(("release", kwargs))
        return release

    report = tmp_path / "saved_return_plan.json"
    monkeypatch.setattr(session_runner, "plan_repose_held_from_state", plan_held)
    monkeypatch.setattr(session_runner, "plan_repose_release_exit", plan_release)
    monkeypatch.setattr(
        session_runner, "write_failed_return_plan",
        lambda **kwargs: calls.append(("write", kwargs)) or report)
    result = runner.prepare_failed_insertion_return(
        planner=object(), withdrawn_state=state,
        withdrawal_evidence_path=withdrawal, limits=limits,
        release_height_m=.03, minimum_rest_socket_clearance_m=.005,
        minimum_board_edge_clearance_m=.005,
        minimum_release_key_clearance_m=.001,
        max_withdrawal_state_gap_s=.2,
        max_arm_hand_skew_s=.02, max_hand_command_error_raw=20.,
        max_arm_velocity_rad_s=.05)
    assert result == (held, release, report)
    assert [name for name, _ in calls] == ["held", "release", "write"]
    np.testing.assert_allclose(calls[0][1]["T_key_hand"], np.eye(4))
    np.testing.assert_allclose(calls[0][1]["T_robot_key_rest"], np.eye(4))
    assert calls[0][1]["trial_scene"] is runner._preflight.trial_scene
    assert calls[2][1]["postlift_report_path"] == runner._postlift_report_path
    withdrawal.write_text(withdrawal.read_text(encoding="utf-8").replace(
        '"key_still_held": true', '"key_still_held": false'), encoding="utf-8")
    with pytest.raises(ValueError, match="held withdrawal"):
        runner.prepare_failed_insertion_return(
            planner=object(), withdrawn_state=state,
            withdrawal_evidence_path=withdrawal, limits=limits,
            release_height_m=.03, minimum_rest_socket_clearance_m=.005,
            minimum_board_edge_clearance_m=.005,
            minimum_release_key_clearance_m=.001,
            max_withdrawal_state_gap_s=.2,
            max_arm_hand_skew_s=.02, max_hand_command_error_raw=20.,
            max_arm_velocity_rad_s=.05)


@pytest.mark.skip(reason="same-attempt lateral reinsertion was removed by policy")
def test_retry_requires_matching_two_view_vote_and_live_replan(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", True, timestamp_s=10.3,
        evidence_refs=_bind_lift(runner))
    postlift_report = _bind_postlift(runner)
    arrival_refs = _bind_preinsert(runner, monkeypatch)
    runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.4,
        evidence_refs=arrival_refs)
    runner._record(lambda row: row.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"}))
    assert runner.current_decision().action == "guarded_withdrawal_then_xy_assessment"
    choice = ChoiceDecision(
        "propose", "two_view_consensus", "x_plus_1mm", (0.001, 0.0),
        ("cam0", "cam1"), {"x_plus_1mm": 2})
    assessment = XYRetryAssessment(
        "proposal_requires_live_preflight", {}, None, (), choice,
        "two_view_consensus")
    def retry_plan(candidate):
        return XYRetryPreflight(
            "sampled_retry_preflight_pass", candidate, "x_plus_1mm",
            SimpleNamespace(xy_offset_socket_m=(0.001, 0.0)),
            SimpleNamespace(sampled_planning_pass=True),
            {"endpoint_pass": True}, np.zeros(13), np.eye(4), 10.6, 10.6)
    refs = {"axial_withdrawal": "robot/withdrawal.json",
            "live_preflight": "plan/retry.json",
            "xy_vlm_vote": "vlm/votes.json"}
    with pytest.raises(ValueError, match="does not match"):
        runner.record_retry(assessment, retry_plan(KEY_B), timestamp_s=10.7,
                            evidence_refs=refs)
    withdrawal = tmp_path / "withdrawal.json"
    withdrawal.write_text('{"withdrawn": true}\n', encoding="utf-8")
    prepared = SimpleNamespace(
        status="ready_to_record_pending_retry", assessment=assessment,
        preflight=retry_plan(KEY_A), withdrawal_evidence_path=withdrawal)
    abstained = SimpleNamespace(
        status="visual_abstain", assessment=None, preflight=None,
        withdrawal_evidence_path=withdrawal)
    outcomes = [abstained, prepared]
    def fake_prepare(**kwargs):
        assert kwargs["attempt"].candidate_id == "table/0/3"
        assert kwargs["trial"].selected_candidate_key == KEY_A
        return outcomes.pop(0)
    def save_retry(_result, _frames, output):
        output.mkdir(parents=True, exist_ok=False)
        (output / "preflight").mkdir()
        (output / "preflight/report.json").write_text("{}", encoding="utf-8")
        (output / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(session_runner, "assess_and_plan_observed_xy_retry",
                        fake_prepare)
    monkeypatch.setattr(session_runner, "write_retry_session_artifacts",
                        save_retry)
    monkeypatch.setattr(session_runner.time, "time", lambda: 10.8)
    retry_inputs = dict(
        planner=object(), held_key_observation=_observation("held", 10.6),
        held_key_evidence_dir=tmp_path, joint_sample=object(), frames=(),
        intrinsics_full={}, extrinsics_full={}, frame_request_id=1,
        frame_ids={}, acquisition_metadata={}, backend=object(),
        withdrawal_completed_at_s=10.6,
        withdrawal_evidence_path=withdrawal,
        postlift_preflight_report_path=tmp_path / "postlift.json",
        decision_timestamp_s=10.7, limits=object())
    runner.prepare_observed_xy_retry(**retry_inputs)
    assert runner.current_decision().action == (
        "guarded_withdrawal_then_xy_assessment")
    assert runner.active_attempt.events[-1]["stage"] == "insertion_success"
    runner.prepare_observed_xy_retry(**retry_inputs)
    assert runner.current_decision().action == "stop_for_review"
    assert (tmp_path / "session/attempts/attempt_1/state_004.json").is_file()
    assert (tmp_path / "session/attempts/attempt_1/"
            "xy_retry_assessments/000/report.json").is_file()
    assert (tmp_path / "session/attempts/attempt_1/"
            "xy_retry_assessments/001/report.json").is_file()
    assert runner.active_attempt.events[-1]["timestamp_s"] == 10.8


@pytest.mark.skip(reason="withdrawn-key diagnostic was replaced by saved preinsert frames")
def test_unobserved_xy_advice_is_saved_without_pending_retry(monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", True, timestamp_s=10.3,
        evidence_refs=_bind_lift(runner))
    postlift_report = _bind_postlift(runner)
    runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.4,
        evidence_refs=_bind_preinsert(runner, monkeypatch))
    runner._record(lambda row: row.record_insertion_evidence(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"}))
    choice = ChoiceDecision(
        "propose", "two_view_consensus", "x_plus_1mm", (0.001, 0.0),
        ("cam0", "cam1"), {"x_plus_1mm": 2})
    assessment = XYRetryAssessment(
        "diagnostic_xy_hypothesis_only", {}, None, (), choice,
        "two_view_consensus")
    prepared = SimpleNamespace(
        status="diagnostic_xy_hypothesis_only", assessment=assessment)
    monkeypatch.setattr(
        session_runner, "assess_unobserved_xy_diagnostic",
        lambda **_kwargs: prepared)
    def save_diagnostic(_result, _frames, output):
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        session_runner, "write_retry_session_artifacts", save_diagnostic)
    prior_events = len(runner.active_attempt.events)
    result = runner.prepare_unobserved_xy_diagnostic(
        joint_sample=object(), frames=(), intrinsics_full={},
        extrinsics_full={}, frame_request_id=1, frame_ids={},
        acquisition_metadata={}, backend=object(),
        withdrawal_completed_at_s=10.6,
        withdrawal_evidence_path=tmp_path / "withdrawal.json",
        postlift_preflight_report_path=Path(postlift_report),
        decision_timestamp_s=10.7, limits=object())
    assert result is prepared
    assert len(runner.active_attempt.events) == prior_events
    assert runner.active_attempt._pending_retry is False
    assert runner.current_decision().action == (
        "guarded_withdrawal_then_xy_assessment")
    assert (runner._attempt_dir / "xy_retry_assessments/000/report.json").is_file()
    with pytest.raises(ValueError, match="voted proposal"):
        runner.record_retry(
            assessment, object(), timestamp_s=10.8,
            evidence_refs={"xy_vlm_vote": "diagnostic_only"})


@pytest.mark.parametrize(
    ("landing_stem", "next_action", "support_ok"),
    [("001", "reobserve_key_and_preflight", True),
     ("000", "reobserve_key_and_preflight", True),
     ("001", "await_repose_observation", False)],
)
def test_pose_exhaustion_routes_to_separate_repose_and_observed_landing(
        monkeypatch, tmp_path, landing_stem, next_action, support_ok):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(
            runner, kwargs["key_observation"],
            status="repose_required_unplanned", repose_targets=("001",)))
    observation = _observation("key_1", 10.0)
    _plan(runner, observation)
    assert runner.current_decision().action == "preflight_repose"
    def fake_reset(**kwargs):
        assert kwargs["from_pose_stem"] == "000"
        assert kwargs["to_pose_stem"] == "001"
        assert kwargs["attempted_insertion"] == ()
        return SimpleNamespace(
            status="nominal_reset_preflight_pass_drop_unobserved",
            observation_id=observation.capture_id,
            from_pose_stem="000", to_pose_stem="001", height_cm=8,
            start_q_acquisition_timestamp_s=10.0,
            attempted_seeds=({"seed_id": "7", "status": "pass"},),
            selected_seed={"seed_id": "7"},
            pickup_plan=SimpleNamespace(success=True),
            held_plan=SimpleNamespace(
                status="sampled_held_path_pass_release_unplanned"),
            release_plan=SimpleNamespace(
                status="nominal_release_exit_path_pass_drop_unobserved"))
    def save_reset(*, result, trial_scene, output_dir, source_files):
        assert set(source_files) == {
            "session", "catalog", "key_pose_world", "live_start_q", "limits"}
        assert all(path.is_file() for path in source_files.values())
        assert "fixture_socket" in trial_scene["mesh"]
        output_dir.mkdir(parents=True, exist_ok=False)
        scene = output_dir / "trial_scene.json"
        scene.write_text(json.dumps(trial_scene), encoding="utf-8")
        trajectories = output_dir / "planned_trajectories.npz"
        np.savez_compressed(trajectories, pickup_approach=np.zeros((2, 13)))
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        (output_dir / "report.json").write_text(json.dumps({
            "status": result.status, "robot_ready": False,
            "selected_seed": result.selected_seed,
            "artifacts": {
                "trial_scene": scene.name,
                "trial_scene_sha256": sha(scene),
                "planned_trajectories": trajectories.name,
                "planned_trajectories_sha256": sha(trajectories),
                "input_files": {
                    name: {"path": str(path.resolve()), "sha256": sha(path)}
                    for name, path in source_files.items()},
            },
        }), encoding="utf-8")
    monkeypatch.setattr(session_runner, "preflight_v8_repose_transition",
                        fake_reset)
    monkeypatch.setattr(session_runner, "write_repose_preflight_artifacts",
                        save_reset)
    class Limits:
        def validate(self):
            pass
    kwargs = dict(
        planner=object(), key_observation=observation,
        to_pose_stem="001", height_cm=8,
        release_xy_robot_m=(0.1, 0.1), live_start_q=np.zeros(13),
        start_q_acquisition_timestamp_s=10.0, max_state_skew_s=0.05,
        max_pose_error_deg=5.0, max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=5.0,
        minimum_rest_socket_clearance_m=0.01,
        minimum_board_edge_clearance_m=0.01, limits=Limits(),
        retreat_goal_arm_q=np.zeros(7),
        minimum_release_key_clearance_m=0.005)
    with pytest.raises(ValueError, match="target has no eligible"):
        runner.preflight_repose(**{**kwargs, "to_pose_stem": "002"})
    runner.preflight_repose(**kwargs)
    assert runner.current_decision().action == "repose_execution_gate_required"
    assert ("nominal_v8_key_hand_relation_not_postgrasp_measurement" in
            runner.current_decision().required_before_motion)
    frozen_report = runner._repose_report_path
    saved_report = frozen_report.read_bytes()
    frozen_report.write_bytes(saved_report + b" ")
    with pytest.raises(ValueError, match="frozen repose preflight"):
        runner.begin_repose_attempt(attempt_id="repose_1", started_at_s=10.2)
    frozen_report.write_bytes(saved_report)
    runner.begin_repose_attempt(attempt_id="repose_1", started_at_s=10.2)
    assert runner.current_decision().action == "await_repose_observation"
    with pytest.raises(ValueError, match="labels must stay separate"):
        runner.observe_stage("grasp_success", True, timestamp_s=10.3,
                             evidence_refs={"vlm_observation": "lift.json",
                                            "key_wrist_check": "grip.json"})
    with pytest.raises(ValueError, match="observe_repose_landing"):
        runner.observe_stage("reorient_success", True, timestamp_s=10.5,
                             evidence_refs={"key_pose": "pose/landed.json"})
    monkeypatch.setattr(
        session_runner, "classify_key_tabletop_pose",
        lambda **_kwargs: {"stem": landing_stem})
    if support_ok:
        monkeypatch.setattr(
            session_runner, "validate_repose_rest_target",
            lambda **_kwargs: {"key_bottom_z_m": 0.0,
                               "table_surface_z_m": 0.0})
    else:
        def no_support(**_kwargs):
            raise ValueError("key is not supported on the board")
        monkeypatch.setattr(
            session_runner, "validate_repose_rest_target", no_support)
    landing = _observation("landed_key", 10.6)
    release_log = tmp_path / "repose_release.json"
    release_log.write_text('{"released": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to this safe directed plan"):
        runner.observe_repose_landing(
            key_observation=landing,
            key_evidence_dir=_save_evidence(runner, landing),
            timestamp_s=10.7, release_completed_at_s=10.4,
            release_evidence_path=release_log,
            max_pose_error_deg=5.0, support_tolerance_m=0.003,
            minimum_rest_socket_clearance_m=0.01,
            minimum_board_edge_clearance_m=0.01)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = runner._repose_report_path
    report_data = json.loads(report.read_text(encoding="utf-8"))
    sources = {}
    for name in ("trajectory_feedback", "safety", "grasp_state",
                 "hand_feedback"):
        source = tmp_path / f"repose_{name}.json"
        source.write_text(json.dumps({"producer": name}), encoding="utf-8")
        sources[name] = {"path": str(source), "sha256": sha(source)}
    phase_names = (
        "pickup_squeeze", "held_lift", "held_transfer", "held_descent",
        "release_open", "post_release_lift", "post_release_retract")
    boundaries = (10.21, 10.24, 10.28, 10.32, 10.36, 10.40, 10.45, 10.50)
    phases = []
    for index, name in enumerate(phase_names):
        row = {"name": name, "started_at_s": boundaries[index],
               "completed_at_s": boundaries[index + 1],
               "complete": True, "safety_abort": False}
        if name in {"held_lift", "held_transfer", "held_descent"}:
            row["grasp_held"] = True
        if name == "release_open":
            row["hand_open_feedback"] = True
        phases.append(row)
    release_log.write_text(json.dumps({
        "schema": "precision_insertion_repose_execution_evidence_v1",
        "source": "commissioned_external_controller",
        "attempt_id": "repose_1", "preflight_report_sha256": sha(report),
        "planned_trajectories_sha256": report_data["artifacts"][
            "planned_trajectories_sha256"],
        "session_calibration_sha256": runner.session_sha256,
        "selected_seed": {"seed_id": "7"},
        "safety_abort": False, "grip_loss_before_release": False,
        "phases": phases, "source_records": sources,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="prior completed release and exit"):
        runner.observe_repose_landing(
            key_observation=landing,
            key_evidence_dir=_save_evidence(runner, landing),
            timestamp_s=10.7, release_completed_at_s=10.65,
            release_evidence_path=release_log,
            max_pose_error_deg=5.0, support_tolerance_m=0.003,
            minimum_rest_socket_clearance_m=0.01,
            minimum_board_edge_clearance_m=0.01)
    landing_kwargs = dict(
        key_observation=landing,
        key_evidence_dir=_save_evidence(runner, landing),
        timestamp_s=10.7, release_completed_at_s=10.4,
        release_evidence_path=release_log, max_pose_error_deg=5.0,
        support_tolerance_m=0.003,
        minimum_rest_socket_clearance_m=0.01,
        minimum_board_edge_clearance_m=0.01)
    if support_ok:
        runner.observe_repose_landing(**landing_kwargs)
    else:
        with pytest.raises(ValueError, match="not supported"):
            runner.observe_repose_landing(**landing_kwargs)
    assert runner.current_decision().action == next_action
    assert runner.attempted_candidates == ()
    if next_action == "reobserve_key_and_preflight":
        _plan(runner, _observation("key_2", 11.0))


def test_reset_seed_rejections_are_scoped_to_directed_height_cell(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(
            runner, kwargs["key_observation"],
            status="repose_required_unplanned", repose_targets=("001",)))
    observation = _observation("key_1", 10.0)
    _plan(runner, observation)
    seen = []
    def fake_reset(**kwargs):
        seen.append((kwargs["height_cm"], kwargs["to_pose_stem"],
                     kwargs["attempted_reset_ids"]))
        return SimpleNamespace(
            status="reset_seed_budget_exhausted",
            observation_id=observation.capture_id,
            from_pose_stem="000", to_pose_stem="001",
            height_cm=kwargs["height_cm"],
            attempted_seeds=({"seed_id": "7"},), selected_seed=None)
    def save_reset(*, output_dir, **_kwargs):
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "report.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(session_runner, "preflight_v8_repose_transition",
                        fake_reset)
    monkeypatch.setattr(session_runner, "write_repose_preflight_artifacts",
                        save_reset)
    class Limits:
        def validate(self):
            pass
    kwargs = dict(
        planner=object(), key_observation=observation,
        to_pose_stem="001", release_xy_robot_m=(0.1, 0.1),
        live_start_q=np.zeros(13), start_q_acquisition_timestamp_s=10.0,
        max_state_skew_s=0.05, max_pose_error_deg=5.0,
        max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=5.0,
        minimum_rest_socket_clearance_m=0.01,
        minimum_board_edge_clearance_m=0.01, limits=Limits())
    for height in (8, 12, 8):
        runner.preflight_repose(**{**kwargs, "height_cm": height})
        assert runner.current_decision().action == "continue_repose_seed_preflight"
    assert seen == [(8, "001", ()), (12, "001", ()), (8, "001", ("7",))]


def test_preinsert_assessment_is_saved_before_positive_arrival_label(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", True, timestamp_s=10.3,
        evidence_refs=_bind_lift(runner))
    postlift = _bind_postlift(runner)
    transfer = runner._attempt_dir / "transfer_execution.json"
    transfer.write_text('{"completed_at_s":10.4}\n', encoding="utf-8")
    raw_bundle = runner._attempt_dir / "preinsert_raw"
    raw_bundle.mkdir()
    (raw_bundle / "manifest.json").write_text("{}\n", encoding="utf-8")
    # This composition test substitutes a raw lift checkpoint without
    # building camera pixels; the checkpoint verifier has separate tests.
    runner._lift_checkpoint.evidence_kind = "raw_visual"
    runner._lift_checkpoint.after_bundle = runner._attempt_dir / "after_lift_raw"
    monkeypatch.setattr(session_runner, "LiftCheckpoint", SimpleNamespace)
    seen = []

    def assess(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(
            attempt_id="attempt_1", candidate_id="table/0/3",
            preinsert_reached=True)

    def write(result, output):
        output.mkdir(parents=True)
        report = output / "report.json"
        report.write_text(json.dumps({
            "attempt_id": result.attempt_id,
            "candidate_id": result.candidate_id,
            "preinsert_reached": True,
            "observation_completed_at_s": 10.45,
            "transfer_execution_path": str(transfer),
            "raw_bundle": str(raw_bundle),
        }), encoding="utf-8")
        return report

    monkeypatch.setattr(session_runner, "assess_preinsert_checkpoint", assess)
    monkeypatch.setattr(session_runner, "write_preinsert_checkpoint", write)
    monkeypatch.setattr(
        session_runner, "verify_raw_camera_capture",
        lambda _path, phase: {
            "frame_evidence": {
                "cam_a": {"frame_id": 31, "image_sha256": "a" * 64,
                          "timestamp_s": 10.45, "max_error_s": 0.001},
                "cam_b": {"frame_id": 32, "image_sha256": "b" * 64,
                          "timestamp_s": 10.455, "max_error_s": 0.001},
            }})
    monkeypatch.setattr(
        session_runner, "verify_preinsert_checkpoint",
        lambda path: json.loads(path.read_text(encoding="utf-8")))
    assessment_kwargs = dict(
        raw_bundle=tmp_path / "saved_raw", transfer_execution_path=transfer,
        joint_sample=object(), backend=object(),
        max_capture_skew_s=0.02, max_joint_frame_skew_s=0.02,
        max_arm_hand_skew_s=0.02,
        max_hand_command_error_raw=30.0,
        max_arm_velocity_rad_s=0.05)
    result = runner.prepare_observed_preinsert_label(**assessment_kwargs)
    assert result.preinsert_reached is True
    assert seen[0]["attempt"].attempt_id == "attempt_1"
    assert seen[0]["after_lift_raw_bundle"] == (
        runner._attempt_dir / "after_lift_raw")
    with pytest.raises(ValueError, match="newer camera exposures"):
        runner.prepare_observed_preinsert_label(**assessment_kwargs)
    report = runner._preinsert_report_path
    refs = {"trajectory": str(transfer), "grasp_state": str(transfer),
            "key_socket_pose": str(report),
            "preinsert_image": str(raw_bundle / "manifest.json"),
            "postlift_preflight": postlift,
            "preinsert_checkpoint": str(report)}
    wrong_image = dict(refs, preinsert_image="unrelated/manifest.json")
    with pytest.raises(ValueError, match="verified checkpoint"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.5,
                             evidence_refs=wrong_image)
    original_report = report.read_bytes()
    report.write_bytes(original_report + b" ")
    with pytest.raises(ValueError, match="observed checkpoint"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.5,
                             evidence_refs=refs)
    report.write_bytes(original_report)
    with pytest.raises(ValueError, match="verified checkpoint"):
        runner.observe_stage("preinsert_reached", True, timestamp_s=10.42,
                             evidence_refs=refs)
    assert runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.5,
        evidence_refs=refs).labels["preinsert_reached"] is True
