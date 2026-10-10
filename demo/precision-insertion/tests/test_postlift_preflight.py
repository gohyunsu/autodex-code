"""Replan from the observed held key, not the pre-squeeze BODex transform."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.conversion import se32cart  # noqa: E402
from precision_insertion.candidates import build_endpoint_catalog  # noqa: E402
from precision_insertion.postlift_preflight import (  # noqa: E402
    plan_postlift_observed_transfer, write_postlift_preflight,
)
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402
from precision_insertion.trial_preflight import (  # noqa: E402
    TrialPreflight, _canonical_sha256,
)
from test_candidates import _candidate_fixture, _session_record  # noqa: E402
from test_preflight import _Planner, _fixture  # noqa: E402


def _setup(tmp_path, monkeypatch):
    mode, paths, candidates, catalog_screen = _candidate_fixture(tmp_path)
    catalog = build_endpoint_catalog(
        shared_root=tmp_path, mode=mode,
        minimum_hand_clearance_m=0.0002, screen=catalog_screen)
    record = _session_record(mode, paths)
    record["c2r"] = np.eye(4).tolist()
    frozen = {
        "mesh": {"fixture_socket": {
            "pose": se32cart(np.eye(4)).tolist(),
            "file_path": str(paths.socket_collision_mesh.resolve())}},
        "cuboid": {},
    }
    calibration = SimpleNamespace(
        record=record, socket_pose_robot=np.eye(4), collision_scene=frozen)
    scene = {"mesh": {**frozen["mesh"], "target": {
        "pose": se32cart(np.eye(4)).tolist(),
        "file_path": str(paths.raw_mesh(mode.key_object))}},
        "cuboid": frozen["cuboid"]}
    initial = InsertionPreflight(
        "sampled_planning_pass", None, None, None, {"sampled_clear": True},
        1, np.ones(6) * 0.2, "commanded_nominal", ())
    _, _, _, fixture_targets, limits = _fixture()
    trial = TrialPreflight(
        status="sampled_planning_pass", pose_class={"stem": "000"},
        attempted_candidates=(),
        selected_candidate_key=("table", "0", "1"),
        repose_target_stems=(), insertion_plan=initial,
        pickup_plan=object(), trial_scene=scene,
        key_observation_id="table-key-1", key_capture_timestamp_s=3.0,
        start_q_acquisition_timestamp_s=3.01,
        max_key_state_skew_s=0.05,
        key_pose_world=np.eye(4), live_start_q=np.zeros(13),
        attempted_before_trial=(), covered_scenes=(), limits=limits,
        max_pose_error_deg=10.0, axial_waypoint_step_m=0.005,
        max_candidate_attempts=None,
        session_calibration_sha256=_canonical_sha256(record),
        catalog_sha256=_canonical_sha256(catalog))
    attempt = begin_attempt(
        attempt_id="trial-1", mode=mode, session_record=record,
        candidate_id="table/0/1", tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=3.0)
    attempt.record_stage(
        "grasp_success", True, timestamp_s=4.0,
        evidence_refs={"vlm_observation": "vlm/lift.json",
                       "key_wrist_check": "sensors/lift_relation.json"})
    observed = np.eye(4)
    observed[2, 3] = 0.1
    start = np.zeros(13)
    start[0] = 0.001
    start[2] = 0.1
    start[7:] = 0.2
    screen_calls = []

    def screen(**kwargs):
        screen_calls.append(kwargs)
        return {
            "endpoint_pass": True,
            "T_key_hand": kwargs["T_key_hand_override"].tolist(),
            "xy_offset_socket_m": [0.0, 0.0],
            "hold_pose_screens": {"measured_post_lift": {
                "hand_q": kwargs["hand_poses_override"][
                    "measured_post_lift"].tolist()}},
        }

    def targets(*, T_key_hand, **_kwargs):
        return replace(
            fixture_targets, T_key_hand=T_key_hand,
            T_robot_hand_preinsert=(
                fixture_targets.T_robot_key_preinsert @ T_key_hand),
            T_robot_hand_entry=(
                fixture_targets.T_robot_key_entry @ T_key_hand),
            T_robot_hand_verification=(
                fixture_targets.T_robot_key_verification @ T_key_hand))

    monkeypatch.setattr(
        "precision_insertion.postlift_preflight.build_rigid_insertion_targets",
        targets)
    monkeypatch.setattr(
        "precision_insertion.preflight.audit_held_joint_paths",
        lambda **_: {"sampled_clear": True})
    return {
        "planner": _Planner(), "trial": trial, "attempt": attempt,
        "calibration": calibration, "catalog": catalog,
        "mode": mode, "shared_root": tmp_path,
        "key_pose_world": observed,
        "key_observation_id": "lift-key-2",
        "key_capture_timestamp_s": 5.0,
        "key_pose_source": "multiview_foundpose",
        "live_start_q": start, "joint_timestamp_s": 5.01,
        "joint_state_source": "robot_joint_feedback",
        "max_state_skew_s": 0.05,
        "max_grasp_translation_drift_m": 0.003,
        "max_grasp_rotation_drift_deg": 5.0,
        "limits": limits, "axial_waypoint_step_m": 0.005,
        "screen": screen,
    }, screen_calls


def test_measured_postlift_relation_rescreens_then_plans(tmp_path, monkeypatch):
    args, screen_calls = _setup(tmp_path, monkeypatch)
    result = plan_postlift_observed_transfer(**args)
    assert result.status == "sampled_postlift_preflight_pass"
    assert result.relation.translation_drift_m < 1e-9
    assert result.planning.held_hand_source == "measured"
    assert result.planning.lift_trajectory is None
    assert len(args["planner"].calls) == 11
    assert screen_calls[0]["override_source"] == (
        "multiview_key_pose_plus_live_wrist")
    np.testing.assert_allclose(
        screen_calls[0]["hand_poses_override"]["measured_post_lift"],
        args["live_start_q"][7:])
    assert result.to_record()["robot_ready"] is False
    bundle = write_postlift_preflight(result, tmp_path / "postlift")
    report = json.loads((bundle / "report.json").read_text())
    assert report["planned_trajectories_sha256"]
    with np.load(bundle / "planned_trajectories.npz") as trajectories:
        assert trajectories["transfer"].shape == (5, 13)
        assert trajectories["axial"].shape == (41, 13)
    with pytest.raises(FileExistsError):
        write_postlift_preflight(result, bundle)


def test_slipped_key_or_failed_endpoint_cannot_reach_planner(tmp_path, monkeypatch):
    args, calls = _setup(tmp_path, monkeypatch)
    pose = args["key_pose_world"].copy()
    pose[0, 3] += 0.006
    args["key_pose_world"] = pose
    result = plan_postlift_observed_transfer(**args)
    assert result.status == "observed_grasp_relation_drift_exceeded"
    assert calls == []
    assert args["planner"].calls == []

    args, calls = _setup(tmp_path / "second", monkeypatch)
    valid_screen = args["screen"]

    def blocked(**kwargs):
        return {**valid_screen(**kwargs), "endpoint_pass": False}

    args["screen"] = blocked
    result = plan_postlift_observed_transfer(**args)
    assert result.status == "observed_20mm_endpoint_rejected"
    assert len(calls) == 1
    assert args["planner"].calls == []


def test_stale_state_wrong_candidate_and_catalog_are_rejected(tmp_path, monkeypatch):
    args, _ = _setup(tmp_path, monkeypatch)
    args["joint_timestamp_s"] = 5.2
    with pytest.raises(ValueError, match="stale or asynchronous"):
        plan_postlift_observed_transfer(**args)
    args, _ = _setup(tmp_path / "second", monkeypatch)
    args["attempt"].candidate_id = "table/0/2"
    with pytest.raises(ValueError, match="this grasp"):
        plan_postlift_observed_transfer(**args)
    args, _ = _setup(tmp_path / "third", monkeypatch)
    args["catalog"] = {**args["catalog"], "minimum_hand_clearance_m": 0.004}
    with pytest.raises(ValueError, match="catalogue or frozen"):
        plan_postlift_observed_transfer(**args)
