"""Session-level preflight/observation bookkeeping never commands a robot."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.retry_preflight import XYRetryPreflight  # noqa: E402
from precision_insertion import session_runner  # noqa: E402
from precision_insertion.xy_retry import XYRetryAssessment  # noqa: E402
from precision_insertion.xy_voting import ChoiceDecision  # noqa: E402


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
        _plan(runner, _observation("key_2", 11.0))
    runner.observe_stage(
        "grasp_success", False, timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/miss.json"})
    assert runner.attempted_candidates == (KEY_A,)
    assert runner.current_decision().action == "reobserve_key_and_preflight"
    snapshot = tmp_path / "session/attempts/attempt_1/state_001.json"
    assert json.loads(snapshot.read_text())["grasp_success"] is False
    binding = json.loads((tmp_path / "session/attempts/attempt_1/"
                          "preflight_binding.json").read_text())
    assert binding["candidate_id"] == "table/0/3"
    with pytest.raises(ValueError, match="predates the previous attempt"):
        _plan(runner, _observation("stale_key_2", 10.3))
    _plan(runner, _observation("key_2", 11.0))
    assert seen == [(), (KEY_A,)]
    assert runner.current_decision().candidate_id == "table/0/4"
    assert json.loads((tmp_path / "session/session_run.json").read_text())[
        "robot_ready"] is False


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


def test_verified_insertion_cannot_start_new_trial_until_reset_observed(
        monkeypatch, tmp_path):
    runner = _runner(monkeypatch, tmp_path)
    monkeypatch.setattr(
        session_runner, "plan_admitted_key_trial",
        lambda **kwargs: _report(runner, kwargs["key_observation"]))
    _plan(runner, _observation("key_1", 10.0))
    runner.begin_selected_attempt(attempt_id="attempt_1", started_at_s=10.2)
    runner.observe_stage(
        "grasp_success", True, timestamp_s=10.3,
        evidence_refs={"vlm_observation": "vlm/lift.json",
                       "key_wrist_check": "pose/lift.json"})
    runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.4,
        evidence_refs={"trajectory": "path/transfer.json",
                       "key_socket_pose": "pose/hold.json",
                       "grasp_state": "pose/grip.json",
                       "postlift_preflight": "path/postlift.json"})
    runner.observe_insertion(
        InsertionEvidence("normal_appearance", (0.0201, 0.021),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"})
    assert runner.current_decision().action == "hold_for_supervised_completion"
    with pytest.raises(ValueError, match="no verified return"):
        _plan(runner, _observation("key_2", 11.0))
    runner.observe_stage("reset_success", True, timestamp_s=10.8,
                         evidence_refs={"key_pose": "pose/reset.json"})
    assert runner.current_decision().action == "reobserve_key_and_preflight"
    _plan(runner, _observation("key_2", 11.0))


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
        evidence_refs={"vlm_observation": "vlm/lift.json",
                       "key_wrist_check": "pose/lift.json"})
    runner.observe_stage(
        "preinsert_reached", True, timestamp_s=10.4,
        evidence_refs={"trajectory": "path/transfer.json",
                       "key_socket_pose": "pose/hold.json",
                       "grasp_state": "pose/grip.json",
                       "postlift_preflight": "path/postlift.json"})
    runner.observe_insertion(
        InsertionEvidence("partial", (0.005, 0.008),
                          "key_pose_multiview", True, False, True),
        timestamp_s=10.5,
        evidence_refs={"vlm_observation": "vlm/insert.json",
                       "key_depth": "pose/depth.json",
                       "alignment": "pose/axis.json",
                       "force_trace": "wrench/trace.json"})
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
    runner.record_retry(assessment, retry_plan(KEY_A), timestamp_s=10.7,
                        evidence_refs=refs)
    assert runner.current_decision().action == "await_retry_execution_and_observation"


@pytest.mark.parametrize(
    ("landing_stem", "next_action", "support_ok"),
    [("001", "reobserve_key_and_preflight", True),
     ("000", "stop_for_review", True),
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
        (output_dir / "report.json").write_text(
            json.dumps({"status": result.status}), encoding="utf-8")
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
    with pytest.raises(ValueError, match="prior logged physical release"):
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
        timestamp_s=10.7, release_completed_at_s=10.5,
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
