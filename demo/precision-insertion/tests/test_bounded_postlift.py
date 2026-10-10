"""A hidden key may use a physical grasp relation, never a nominal pose."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import coal  # noqa: F401 -- load its C++ runtime before trimesh on this host
import trimesh


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.bounded_postlift import (  # noqa: E402
    BoundedHeldRelation, BoundedPostLiftPreflight,
    plan_bounded_postlift_transfer, verify_bounded_postlift_preflight,
    write_bounded_postlift_preflight,
)
from precision_insertion import bounded_postlift as bounded_module  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion import postlift_preflight as observed_postlift  # noqa: E402
from precision_insertion.lift_checkpoint import LiftCheckpoint  # noqa: E402
from precision_insertion.observer import VLMObservation  # noqa: E402
from precision_insertion import preinsert_checkpoint as arrival_module  # noqa: E402
from precision_insertion import session_runner as runner_module  # noqa: E402
from precision_insertion.physical_grasp_calibration import (  # noqa: E402
    calibrate_physical_held_relation,
)
from precision_insertion.uncertainty_margin import (  # noqa: E402
    SurfaceDeviationBounds,
)
from test_physical_grasp_calibration import _sample  # noqa: E402
from test_postlift_preflight import _setup  # noqa: E402
from test_preinsert_checkpoint import _setup as _arrival_setup  # noqa: E402


def _case(tmp_path, monkeypatch):
    original, screen_calls = _setup(tmp_path, monkeypatch)
    key = original["trial"].selected_candidate_key
    candidate_dir = AssetPaths(tmp_path, original["mode"]).candidate_dir / Path(*key)
    nominal = np.load(candidate_dir / "wrist_se3.npy")
    calibration = calibrate_physical_held_relation(
        mode=original["mode"], shared_root=tmp_path,
        candidate_key=key, candidate_T_key_hand=nominal,
        samples=[_sample(tmp_path, i, key) for i in range(5)],
        minimum_independent_trials=5,
        max_nominal_translation_drift_m=0.003,
        max_nominal_rotation_drift_deg=5.0)
    calibration_path = tmp_path / "physical_grasp_calibration.json"
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    raw_state = original["joint_sample"]
    visual = VLMObservation(
        "post_lift", {"class": "held", "evidence_views": ["cam_a", "cam_b"],
                      "evidence": "visible key follows hand"},
        "raw", "prompt", ("before_grasp/cam_a@3.000000",
                          "after_lift/cam_a@5.000000"), None,
        "test-model", 0.1)
    lift = LiftCheckpoint(
        original["attempt"].attempt_id, original["attempt"].candidate_id,
        True, "multiview_visible_held_key", 4.0, 5.1,
        original["trial"].key_observation_id, "raw-lift-1",
        tmp_path, tmp_path, "a" * 64, "b" * 64, (),
        None, None, 2, raw_state, visual, "raw_visual", {})
    lift_path = tmp_path / "raw_lift.json"
    lift_path.write_text(json.dumps(lift.to_record()), encoding="utf-8")
    monkeypatch.setattr(
        "precision_insertion.bounded_postlift.verify_lift_checkpoint",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    monkeypatch.setattr(
        "precision_insertion.bounded_postlift._load_mesh",
        lambda _path: trimesh.creation.box((0.02, 0.02, 0.04)))
    monkeypatch.setattr(
        "precision_insertion.bounded_postlift.build_rigid_insertion_targets",
        observed_postlift.build_rigid_insertion_targets)
    monkeypatch.setattr(
        "precision_insertion.bounded_postlift.audit_sampled_uncertainty_margins",
        lambda **_: {"sampled_margin_pass": True})
    now = time.time()
    monkeypatch.setattr(
        "precision_insertion.bounded_postlift._wall_time", lambda: now)
    state = replace(raw_state, sample_timestamp_s=now - 0.1,
                    arm_timestamp_s=now - 0.1,
                    hand_timestamp_s=now - 0.1)
    arguments = dict(
        planner=original["planner"], trial=original["trial"],
        attempt=original["attempt"], calibration=original["calibration"],
        catalog=original["catalog"], mode=original["mode"],
        shared_root=tmp_path, raw_lift=lift,
        raw_lift_report_path=lift_path,
        physical_calibration_path=calibration_path,
        joint_sample=state,
        bounds=SurfaceDeviationBounds(
            0.001, 0.001, "commissioned_future_trial_surface_bound"),
        max_state_age_s=1.0, max_arm_hand_skew_s=0.05,
        max_hand_command_error_raw=30,
        max_arm_velocity_rad_s=0.05,
        max_postlift_arm_drift_rad=0.01,
        max_postlift_hand_drift_raw=1.0,
        max_calibration_hand_excess_rad=0.01,
        limits=original["limits"], axial_waypoint_step_m=0.005,
        screen=original["screen"])
    return arguments, candidate_dir, screen_calls


def test_physical_relation_replans_and_verifies_saved_bundle(tmp_path, monkeypatch):
    args, candidate_dir, screens = _case(tmp_path, monkeypatch)
    result = plan_bounded_postlift_transfer(**args)
    assert result.status == "sampled_postlift_preflight_pass"
    assert result.to_record()["robot_ready"] is False
    assert result.relation.hand_calibration_excess_rad == pytest.approx(0.0)
    assert screens[0]["override_source"] == (
        "physical_grasp_medoid_plus_measured_Inspire")
    np.testing.assert_allclose(
        screens[0]["hand_poses_override"]["measured_post_lift"],
        args["joint_sample"].full_q[7:])
    output = write_bounded_postlift_preflight(result, tmp_path / "bounded")
    report = output / "report.json"
    saved = verify_bounded_postlift_preflight(
        report, mode=args["mode"], shared_root=tmp_path,
        candidate_dir=candidate_dir)
    assert saved["status"] == "sampled_postlift_preflight_pass"
    with pytest.raises(FileExistsError):
        write_bounded_postlift_preflight(result, output)
    trajectory_file = output / "planned_trajectories.npz"
    trajectory_bytes = trajectory_file.read_bytes()
    trajectory_file.write_bytes(trajectory_bytes + b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        verify_bounded_postlift_preflight(
            report, mode=args["mode"], shared_root=tmp_path,
            candidate_dir=candidate_dir)
    trajectory_file.write_bytes(trajectory_bytes)
    args["physical_calibration_path"].write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration file changed"):
        verify_bounded_postlift_preflight(
            report, mode=args["mode"], shared_root=tmp_path,
            candidate_dir=candidate_dir)


def test_empirical_scatter_or_unmatched_hand_cannot_be_promoted(tmp_path, monkeypatch):
    args, _, screens = _case(tmp_path, monkeypatch)
    args["bounds"] = SurfaceDeviationBounds(
        0.00001, 0.001, "commissioned_future_trial_surface_bound")
    with pytest.raises(ValueError, match="smaller than observed scatter"):
        plan_bounded_postlift_transfer(**args)
    assert screens == []
    args, _, screens = _case(tmp_path / "other", monkeypatch)
    args["max_calibration_hand_excess_rad"] = 0.001
    measured = args["joint_sample"].full_q.copy()
    measured[7] += 0.02
    raw = args["joint_sample"].hand_raw_measured.copy()
    # Reconstruct consistent raw feedback; a hand-range rejection must not
    # merely be a robot-state contract failure.
    from autodex.utils.sync import convert_inspire_raw  # noqa: E402

    raw[0] += 20.0
    measured[7:] = convert_inspire_raw(raw[None, :])[0]
    args["joint_sample"] = replace(
        args["joint_sample"], full_q=measured,
        hand_raw_measured=raw, hand_raw_commanded=raw)
    with pytest.raises(ValueError, match="outside calibrated grasp range"):
        plan_bounded_postlift_transfer(**args)
    assert screens == []


def test_bounded_relation_reaches_preinsert_visual_gate_not_pose_claim(
    tmp_path, monkeypatch,
):
    args = _arrival_setup(tmp_path)
    observed = args["postlift"]
    relation = BoundedHeldRelation(
        np.eye(4), tmp_path / "physical_calibration.json", "a" * 64,
        "physical_trial_1", 0.001, 0.0, 0.01,
        SurfaceDeviationBounds(
            0.002, 0.002, "commissioned_future_trial_surface_bound"))
    bounded = BoundedPostLiftPreflight(
        observed.status, observed.attempt_id, observed.candidate_key,
        observed.session_calibration_sha256, observed.catalog_sha256,
        observed.trial_key_observation_id, "raw-lift-1", 10.5, 10.5,
        observed.live_start_q, observed.joint_feedback,
        tmp_path / "raw_lift.json", "b" * 64, relation,
        None, observed.targets, observed.planning, {"sampled_margin_pass": True})
    args["postlift"] = bounded
    saved = json.loads(args["postlift_report_path"].read_text())
    saved["schema"] = "precision_insertion_bounded_postlift_preflight_v1"
    saved["bounded_held_relation"] = saved.pop("observed_held_relation")
    args["postlift_report_path"].write_text(json.dumps(saved), encoding="utf-8")
    calls = []

    def verify(path, **kwargs):
        calls.append((path, kwargs["candidate_dir"]))
        return saved

    monkeypatch.setattr(arrival_module, "verify_bounded_postlift_preflight", verify)
    result = arrival_module.assess_preinsert_checkpoint(**args)
    assert result.preinsert_reached is True
    assert result.comparison.prediction.relation_source == (
        "verified_physical_grasp_calibration")
    report = arrival_module.write_preinsert_checkpoint(result, tmp_path / "arrival")
    assert arrival_module.verify_preinsert_checkpoint(report)[
        "preinsert_reached"] is True
    assert len(calls) == 2


def test_session_runner_keeps_bounded_path_separate_and_source_bound(
    tmp_path, monkeypatch,
):
    args, _, _ = _case(tmp_path, monkeypatch)
    source_calibration = args["calibration"]
    calibration = SessionCalibration(
        board={}, socket_pose_robot=source_calibration.socket_pose_robot,
        socket_diagnostics={},
        collision_scene=source_calibration.collision_scene,
        record=source_calibration.record)
    runner = runner_module.SessionRunner(
        mode=args["mode"], calibration=calibration,
        catalog=args["catalog"], shared_root=tmp_path,
        output_dir=tmp_path / "session", max_xy_retries=1)
    runner._attempt = args["attempt"]
    runner._attempt_dir = runner.output_dir / "attempts" / args["attempt"].attempt_id
    runner._attempt_dir.mkdir(parents=True)
    runner._preflight = args["trial"]
    runner._lift_checkpoint = args["raw_lift"]
    runner._lift_report_path = args["raw_lift_report_path"]
    runner._lift_report_sha256 = hashlib.sha256(
        runner._lift_report_path.read_bytes()).hexdigest()
    assert runner.current_decision().action == "held_relation_evidence_required"
    monkeypatch.setattr(
        runner_module, "plan_bounded_postlift_transfer",
        lambda **kwargs: bounded_module.plan_bounded_postlift_transfer(
            **kwargs, screen=args["screen"]))
    outcome = runner.prepare_bounded_postlift_transfer(
        planner=args["planner"],
        physical_calibration_path=args["physical_calibration_path"],
        joint_sample=args["joint_sample"], bounds=args["bounds"],
        max_state_age_s=args["max_state_age_s"],
        max_arm_hand_skew_s=args["max_arm_hand_skew_s"],
        max_hand_command_error_raw=args["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=args["max_arm_velocity_rad_s"],
        max_postlift_arm_drift_rad=args["max_postlift_arm_drift_rad"],
        max_postlift_hand_drift_raw=args["max_postlift_hand_drift_raw"],
        max_calibration_hand_excess_rad=args["max_calibration_hand_excess_rad"],
        limits=args["limits"],
        axial_waypoint_step_m=args["axial_waypoint_step_m"])
    assert outcome.status == "sampled_postlift_preflight_pass"
    assert runner.current_decision().action == "transfer_execution_gate_required"
    assert runner.current_decision().reason == (
        "bounded_postlift_path_planned_arrival_unobserved")
    binding = runner._postlift_report_path.parent / "physical_relation_binding.json"
    assert json.loads(binding.read_text())["physical_calibration_sha256"] == (
        outcome.relation.calibration_sha256)
    args["physical_calibration_path"].write_text("{}", encoding="utf-8")
    assert runner.current_decision().reason == "bounded_postlift_evidence_changed"
    runner._postlift_report_path.write_text("{}", encoding="utf-8")
    assert runner.current_decision().action == "stop_for_review"
