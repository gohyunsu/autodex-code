"""An occluded cylinder may use an explicit *nominal* grasp hypothesis."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import nominal_postlift as nominal_module  # noqa: E402
from precision_insertion import preinsert_checkpoint as arrival_module  # noqa: E402
from precision_insertion import retry_session as retry_module  # noqa: E402
from precision_insertion import session_runner as runner_module  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.grounded_alignment import AlignmentLimits  # noqa: E402
from precision_insertion.lift_checkpoint import LiftCheckpoint  # noqa: E402
from precision_insertion.nominal_postlift import (  # noqa: E402
    NominalHeldRelation, NominalPostLiftPreflight,
    plan_nominal_postlift_transfer, verify_nominal_postlift_preflight,
    write_nominal_postlift_preflight,
)
from precision_insertion.observer import VLMObservation  # noqa: E402
from precision_insertion.retry_session import assess_grounded_xy_diagnostic  # noqa: E402
from precision_insertion import postlift_preflight as observed_module  # noqa: E402
import test_postlift_preflight as observed_fixture  # noqa: E402
from test_candidates import _candidate_fixture  # noqa: E402
from test_preinsert_checkpoint import _setup as _arrival_setup  # noqa: E402
from test_retry_session import _inputs as _retry_inputs  # noqa: E402


def _case(tmp_path, monkeypatch):
    mode = select_mode("cylinder", 20)

    def cylinder_fixture(root):
        result = _candidate_fixture(root, mode=mode)
        geometry = AssetPaths(root, mode).task_geometry
        geometry.write_text(json.dumps({
            "T_socket_key_verification": np.eye(4).tolist(),
        }), encoding="utf-8")
        return result

    monkeypatch.setattr(observed_fixture, "_candidate_fixture", cylinder_fixture)
    original, calls = observed_fixture._setup(tmp_path, monkeypatch)
    original_targets = observed_module.build_rigid_insertion_targets
    monkeypatch.setattr(
        nominal_module, "build_rigid_insertion_targets",
        lambda **kwargs: replace(original_targets(**kwargs), mode=mode))
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
        nominal_module, "verify_lift_checkpoint",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    now = time.time()
    state = replace(raw_state, sample_timestamp_s=now - .1,
                    arm_timestamp_s=now - .1, hand_timestamp_s=now - .1)
    args = dict(
        planner=original["planner"], trial=original["trial"],
        attempt=original["attempt"], calibration=original["calibration"],
        catalog=original["catalog"], mode=mode, shared_root=tmp_path,
        raw_lift=lift, raw_lift_report_path=lift_path,
        joint_sample=state, max_state_age_s=1.0,
        max_arm_hand_skew_s=.05, max_hand_command_error_raw=30,
        max_arm_velocity_rad_s=.05, max_postlift_arm_drift_rad=.01,
        max_postlift_hand_drift_raw=1.0,
        limits=original["limits"], axial_waypoint_step_m=.005,
        screen=original["screen"])
    candidate_dir = (AssetPaths(tmp_path, mode).candidate_dir /
                     Path(*original["trial"].selected_candidate_key))
    return args, candidate_dir, calls


def test_nominal_relation_is_marked_hypothesis_and_source_verified(
    tmp_path, monkeypatch,
):
    args, candidate_dir, calls = _case(tmp_path, monkeypatch)
    args["attempt"].wrist_correction_socket_m = (.0006, -.0008)
    result = plan_nominal_postlift_transfer(**args)
    assert result.status == "sampled_postlift_preflight_pass"
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["nominal_held_relation"]["source"] == (
        "selected_v8_wrist_hypothesis_not_observed_key_pose")
    assert calls[0]["override_source"] == (
        "nominal_v8_relation_plus_measured_Inspire")
    assert result.relation.T_key_hand_command[0, 3] == pytest.approx(
        result.relation.T_key_hand[0, 3] + .0006)
    output = write_nominal_postlift_preflight(result, tmp_path / "nominal")
    report = output / "report.json"
    monkeypatch.setattr(
        nominal_module, "verify_lift_checkpoint",
        lambda path: json.loads(Path(path).read_text(encoding="utf-8")))
    saved = verify_nominal_postlift_preflight(
        report, mode=args["mode"], shared_root=tmp_path,
        candidate_dir=candidate_dir)
    assert saved["status"] == "sampled_postlift_preflight_pass"
    with pytest.raises(FileExistsError):
        write_nominal_postlift_preflight(result, output)

    source = candidate_dir / "wrist_se3.npy"
    original = source.read_bytes()
    changed = np.eye(4)
    changed[0, 3] = .020
    np.save(source, changed)
    with pytest.raises(ValueError, match="nominal grasp source changed"):
        verify_nominal_postlift_preflight(
            report, mode=args["mode"], shared_root=tmp_path,
            candidate_dir=candidate_dir)
    source.write_bytes(original)
    trajectories = output / "planned_trajectories.npz"
    trajectories.write_bytes(trajectories.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        verify_nominal_postlift_preflight(
            report, mode=args["mode"], shared_root=tmp_path,
            candidate_dir=candidate_dir)


def test_nominal_route_rejects_incompatible_tabletop_grasp(tmp_path, monkeypatch):
    args, _, calls = _case(tmp_path, monkeypatch)
    args["attempt"].candidate_id = "table/1/3"
    with pytest.raises(ValueError, match="positive raw lift"):
        plan_nominal_postlift_transfer(**args)
    assert calls == []


def test_session_runner_dispatches_nominal_plan_without_physical_file(
    tmp_path, monkeypatch,
):
    args, _, _ = _case(tmp_path, monkeypatch)
    source = args["calibration"]
    calibration = SessionCalibration(
        board={}, socket_pose_robot=source.socket_pose_robot,
        socket_diagnostics={}, collision_scene=source.collision_scene,
        record=source.record)
    runner = runner_module.SessionRunner(
        mode=args["mode"], calibration=calibration,
        catalog=args["catalog"], shared_root=tmp_path,
        output_dir=tmp_path / "session", max_xy_retries=None)
    runner._attempt = args["attempt"]
    runner._attempt_dir = runner.output_dir / "attempts" / args["attempt"].attempt_id
    runner._attempt_dir.mkdir(parents=True)
    runner._preflight = args["trial"]
    runner._lift_checkpoint = args["raw_lift"]
    runner._lift_report_path = args["raw_lift_report_path"]
    runner._lift_report_sha256 = nominal_module._sha(runner._lift_report_path)
    assert runner.current_decision().action == "held_relation_evidence_required"
    monkeypatch.setattr(
        runner_module, "plan_nominal_postlift_transfer",
        lambda **kwargs: nominal_module.plan_nominal_postlift_transfer(
            **kwargs, screen=args["screen"]))
    result = runner.prepare_nominal_postlift_transfer(
        planner=args["planner"], joint_sample=args["joint_sample"],
        max_state_age_s=args["max_state_age_s"],
        max_arm_hand_skew_s=args["max_arm_hand_skew_s"],
        max_hand_command_error_raw=args["max_hand_command_error_raw"],
        max_arm_velocity_rad_s=args["max_arm_velocity_rad_s"],
        max_postlift_arm_drift_rad=args["max_postlift_arm_drift_rad"],
        max_postlift_hand_drift_raw=args["max_postlift_hand_drift_raw"],
        limits=args["limits"],
        axial_waypoint_step_m=args["axial_waypoint_step_m"])
    assert result.status == "sampled_postlift_preflight_pass"
    decision = runner.current_decision()
    assert decision.action == "transfer_execution_gate_required"
    assert decision.reason == (
        "nominal_postlift_path_planned_relation_unmeasured")
    assert "nominal_key_hand_is_an_unmeasured_hypothesis" in (
        decision.required_before_motion)
    wrist_source = result.relation.wrist_source_path
    wrist_source.write_bytes(wrist_source.read_bytes() + b"changed")
    assert runner.current_decision().reason == "nominal_postlift_evidence_changed"


def test_nominal_relation_reaches_preinsert_visual_checkpoint(
    tmp_path, monkeypatch,
):
    args = _arrival_setup(tmp_path)
    observed = args["postlift"]
    relation = NominalHeldRelation(
        np.eye(4), np.eye(4), tmp_path / "wrist_se3.npy", "a" * 64,
        (0.0, 0.0))
    nominal = NominalPostLiftPreflight(
        observed.status, observed.attempt_id, observed.candidate_key,
        observed.session_calibration_sha256, observed.catalog_sha256,
        observed.trial_key_observation_id, "raw-lift-1", 10.5, 10.5,
        observed.live_start_q, args["joint_sample"],
        tmp_path / "raw_lift.json", "b" * 64, relation,
        None, observed.targets, observed.planning)
    args["postlift"] = nominal
    saved = json.loads(args["postlift_report_path"].read_text())
    saved["schema"] = "precision_insertion_nominal_postlift_preflight_v1"
    saved["nominal_held_relation"] = saved.pop("observed_held_relation")
    args["postlift_report_path"].write_text(json.dumps(saved), encoding="utf-8")
    calls = []

    def verify(path, **kwargs):
        calls.append((path, kwargs["candidate_dir"]))
        return saved

    monkeypatch.setattr(arrival_module,
                        "verify_nominal_postlift_preflight", verify)
    result = arrival_module.assess_preinsert_checkpoint(**args)
    assert result.preinsert_reached is True
    assert result.comparison.prediction.relation_source == (
        "nominal_v8_grasp_hypothesis")
    report = arrival_module.write_preinsert_checkpoint(result, tmp_path / "arrival")
    assert arrival_module.verify_preinsert_checkpoint(report)[
        "preinsert_reached"] is True
    assert len(calls) == 2


def test_grounded_failed_trial_accepts_source_verified_nominal_report(
    tmp_path, monkeypatch,
):
    kwargs = _retry_inputs(tmp_path, monkeypatch)
    postlift = kwargs["postlift_preflight_report_path"]
    saved = json.loads(postlift.read_text(encoding="utf-8"))
    saved["schema"] = "precision_insertion_nominal_postlift_preflight_v1"
    saved["nominal_held_relation"] = saved.pop("observed_held_relation")
    postlift.write_text(json.dumps(saved), encoding="utf-8")
    geometry = next((tmp_path / "AutoDex/precision_insertion/fixtures").rglob(
        "task_geometry.json"))
    geometry.write_text(json.dumps({
        "key_frame": {"tip_z_m": .08},
        "verification_depth_m": .02,
        "insertion_direction_socket": [0., 0., -1.],
        "socket_entry_plane_z_m": .055,
    }), encoding="utf-8")
    candidate_dir = (AssetPaths(tmp_path, kwargs["mode"]).candidate_dir /
                     "table/0/1")
    monkeypatch.setattr(retry_module, "select_pose_candidates",
                        lambda *_args, **_kwargs: {
                            "status": "candidates_available",
                            "candidates": [{"key": ["table", "0", "1"],
                                            "candidate_dir": str(candidate_dir)}]})
    verification = []
    monkeypatch.setattr(retry_module, "verify_nominal_postlift_preflight",
                        lambda path, **_kwargs: verification.append(path))
    alignment_limits = AlignmentLimits(
        pixel_sigma_px=1., max_reprojection_px=3.,
        min_parallax_deg=5., max_axis_tilt_deg=3.,
        max_20mm_axis_sweep_m=.001,
        max_lateral_uncertainty_95_m=.0005,
        systematic_lateral_sigma_m=.0001,
        cad_spacing_sigma_m=.0005)
    diagnostic_inputs = {key: value for key, value in kwargs.items()
                         if key not in {"planner", "held_key_observation",
                                        "held_key_evidence_dir"}}
    result = assess_grounded_xy_diagnostic(
        **diagnostic_inputs, alignment_limits=alignment_limits,
        axis_reference_key_z_m=.03)
    assert result.status == "abstain"  # synthetic square has unresolved yaw
    assert verification == [postlift.resolve()]
