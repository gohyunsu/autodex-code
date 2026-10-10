"""Post-shift 20 mm replan is source-bound and never a retry command."""

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

from precision_insertion import postshift_checkpoint, postshift_insertion  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.uncertainty_margin import SurfaceDeviationBounds  # noqa: E402
from test_postshift_checkpoint import _case  # noqa: E402


def _setup(tmp_path, monkeypatch):
    call, alignment = _case(tmp_path, monkeypatch)
    alignment.update({
        "schema": "precision_insertion_grounded_alignment_v2",
        "inlier_cameras": ["a", "b"],
        "insertion_axis_socket": [0., 0., -1.],
        "robot_ready": False,
    })
    checkpoint = postshift_checkpoint.assess_postshift_alignment(**call)
    assert checkpoint.status == "visual_alignment_within_budget"
    checkpoint_path = postshift_checkpoint.write_postshift_checkpoint(
        checkpoint, tmp_path / "postshift_checkpoint")
    plan = call["plan"]
    postlift = {
        "schema": "precision_insertion_bounded_postlift_preflight_v1",
        "bounded_held_relation": {
            "source": "physical_grasp_calibration_medoid_not_runtime_key_pose",
            "T_key_hand": plan.lateral.T_key_hand.tolist(),
            "surface_bounds": {"key_surface_m": .003,
                               "hand_surface_m": .001},
        },
    }
    monkeypatch.setattr(
        postshift_insertion, "_validated_retry_trial_context",
        lambda **_kwargs: ({}, postlift, plan.postlift_report_path,
                           tmp_path))
    monkeypatch.setattr(
        postshift_insertion, "validated_frozen_socket_pose",
        lambda **_kwargs: np.eye(4))
    monkeypatch.setattr(
        postshift_insertion, "_load_mesh",
        lambda _path: SimpleNamespace(
            vertices=np.array([[0., 0., 0.], [0., 0., .08]])))
    monkeypatch.setattr(
        postshift_insertion, "select_pose_candidates",
        lambda *_args, **_kwargs: {
            "status": "candidates_available",
            "candidates": [{"key": list(plan.candidate_id.split("/")),
                            "candidate_dir": str(tmp_path / "candidate")}]})
    wrist = call["planner"].fk_wrist(checkpoint.joint_sample.full_q)
    fake_targets = SimpleNamespace(
        T_robot_key_preinsert=np.eye(4),
        T_robot_hand_preinsert=wrist,
        T_robot_key_verification=np.eye(4),
        to_record=lambda: {"schema": "test_target"})
    fake_targets.T_robot_key_preinsert[2, 3] = .30
    fake_targets.T_robot_key_verification[2, 3] = .115
    monkeypatch.setattr(postshift_insertion,
                        "build_rigid_insertion_targets",
                        lambda **_kwargs: fake_targets)
    planning = InsertionPreflight(
        "sampled_planning_pass", None,
        np.zeros((2, 13)), np.zeros((2, 13)),
        {"sampled_clear": True}, 1,
        checkpoint.joint_sample.full_q[7:], "measured", ())
    seen = {}

    def plan_path(**kwargs):
        seen["path"] = kwargs
        return planning

    monkeypatch.setattr(postshift_insertion,
                        "plan_held_transfer_and_axial", plan_path)
    monkeypatch.setattr(postshift_insertion,
                        "audit_sampled_uncertainty_margins",
                        lambda **_kwargs: {"sampled_margin_pass": True})

    def screen(**kwargs):
        seen["screen"] = kwargs
        return {"endpoint_pass": True,
                "T_socket_key_tested":
                fake_targets.T_robot_key_verification.tolist()}

    attempt = SimpleNamespace(
        attempt_id=plan.attempt_id, candidate_id=plan.candidate_id,
        tabletop_pose_stem="pose_0")
    trial = SimpleNamespace(
        selected_candidate_key=tuple(plan.candidate_id.split("/")),
        trial_scene={},
        limits=SimpleNamespace(minimum_hand_clearance_m=.0002),
        axial_waypoint_step_m=.005)
    args = dict(
        planner=call["planner"], mode=call["mode"], shared_root=tmp_path,
        calibration=call["calibration"], catalog={}, trial=trial,
        attempt=attempt, shift_plan=plan, checkpoint=checkpoint,
        checkpoint_report_path=checkpoint_path,
        bounds=SurfaceDeviationBounds(
            .006, .002, "commissioned_future_trial_surface_bound"),
        max_visual_tip_error_m=.0002,
        max_visual_axis_error_deg=1.,
        max_axis_prior_residual_deg=2.,
        max_hold_height_delta_m=.2,
        max_preinsert_hand_rotation_deg=5.,
        screen=screen)
    return args, seen


def test_postshift_replan_reuses_measured_hand_and_saved_sources(
        tmp_path, monkeypatch):
    args, seen = _setup(tmp_path, monkeypatch)
    result = postshift_insertion.plan_postshift_insertion_preflight(**args)
    assert result.status == "sampled_postshift_20mm_preflight_pass"
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["insertion_replan_allowed"] is False
    assert result.required_key_surface_bound_m > .003
    np.testing.assert_allclose(
        seen["screen"]["hand_poses_override"]["measured_postshift"],
        args["checkpoint"].joint_sample.full_q[7:])
    assert seen["screen"]["cylinder_yaw_gauge_socket_rad"] == (
        result.yaw_gauge_socket_rad)
    assert seen["path"]["start_q"] is args["checkpoint"].joint_sample.full_q
    report = postshift_insertion.write_postshift_insertion_preflight(
        result, tmp_path / "postshift_20mm")
    saved = json.loads(report.read_text())
    archive = report.parent / saved["planned_trajectories"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == (
        saved["planned_trajectories_sha256"])


def test_postshift_replan_rejects_visual_abstain_and_small_bound(
        tmp_path, monkeypatch):
    args, seen = _setup(tmp_path, monkeypatch)
    abstain = replace(args["checkpoint"], status="visual_abstain")
    with pytest.raises(ValueError, match="visually aligned"):
        postshift_insertion.plan_postshift_insertion_preflight(
            **{**args, "checkpoint": abstain})
    bounds = SurfaceDeviationBounds(
        .0031, .002, "commissioned_future_trial_surface_bound")
    with pytest.raises(ValueError, match="omit post-shift visual error"):
        postshift_insertion.plan_postshift_insertion_preflight(
            **{**args, "bounds": bounds})
    assert "screen" not in seen


def test_postshift_replan_rejects_changed_capture_or_endpoint_target(
        tmp_path, monkeypatch):
    args, seen = _setup(tmp_path, monkeypatch)
    def wrong_screen(**_kwargs):
        bad = np.eye(4)
        bad[0, 3] = .001
        return {"endpoint_pass": True,
                "T_socket_key_tested": bad.tolist()}
    with pytest.raises(ValueError, match="different yaw/pose"):
        postshift_insertion.plan_postshift_insertion_preflight(
            **{**args, "screen": wrong_screen})
    image = args["checkpoint"].capture_dir / "images/a.png"
    with image.open("ab") as stream:
        stream.write(b"mutated")
    with pytest.raises(ValueError, match="raw camera PNG changed"):
        postshift_insertion.plan_postshift_insertion_preflight(**args)
    assert "screen" not in seen


def test_saved_postshift_20mm_rechecks_paths_and_cad(
        tmp_path, monkeypatch):
    args, _seen = _setup(tmp_path, monkeypatch)
    result = postshift_insertion.plan_postshift_insertion_preflight(**args)
    mode = args["mode"]
    paths = AssetPaths(tmp_path, mode)
    candidate = paths.candidate_dir / Path(*result.candidate_id.split("/"))
    files = {
        "key_mesh": paths.raw_mesh(mode.key_object),
        "socket_mesh": paths.socket_collision_mesh,
        "task_geometry": paths.task_geometry,
        "robot_urdf": paths.robot_urdf,
        "wrist_se3": candidate / "wrist_se3.npy",
        "pregrasp_pose": candidate / "pregrasp_pose.npy",
        "grasp_pose": candidate / "grasp_pose.npy",
    }
    for name, file in files.items():
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            file.write_bytes(name.encode("ascii"))
    endpoint = {
        **result.endpoint,
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "T_key_hand": result.hypothesis.T_key_hand.tolist(),
        "cylinder_yaw_gauge_socket_rad": result.yaw_gauge_socket_rad,
        "input_sha256": {
            name: hashlib.sha256(file.read_bytes()).hexdigest()
            for name, file in files.items()},
    }
    start = args["checkpoint"].joint_sample.full_q.copy()
    middle = start.copy()
    middle[0] += .0001
    end = middle.copy()
    end[0] += .0001
    planning = replace(
        result.planning,
        transfer_trajectory=np.stack([start, middle]),
        axial_trajectory=np.stack([middle, end]))
    result = replace(result, endpoint=endpoint, candidate_dir=candidate,
                     planning=planning)
    monkeypatch.setattr(postshift_insertion,
                        "verify_bounded_postlift_preflight",
                        lambda *_args, **_kwargs: {})
    report = postshift_insertion.write_postshift_insertion_preflight(
        result, tmp_path / "saved_20mm")
    verify = lambda: postshift_insertion.verify_postshift_insertion_preflight(
        report, expected=result, checkpoint=args["checkpoint"],
        shift_plan=args["shift_plan"], mode=mode,
        shared_root=tmp_path, calibration=args["calibration"])
    assert verify()["status"] == "sampled_postshift_20mm_preflight_pass"
    archive = report.parent / "planned_trajectories.npz"
    with archive.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="trajectory bytes changed"):
        verify()
    # The original archive bytes are not recoverable from this test report;
    # use a second immutable report to isolate the CAD-source rejection.
    other = postshift_insertion.write_postshift_insertion_preflight(
        result, tmp_path / "saved_20mm_other")
    files["key_mesh"].write_bytes(b"new CAD bytes")
    with pytest.raises(ValueError, match="CAD/candidate inputs changed"):
        postshift_insertion.verify_postshift_insertion_preflight(
            other, expected=result, checkpoint=args["checkpoint"],
            shift_plan=args["shift_plan"], mode=mode,
            shared_root=tmp_path, calibration=args["calibration"])
