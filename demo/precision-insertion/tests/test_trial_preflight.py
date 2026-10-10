"""Read-only per-trial candidate sequencing and honest repose status."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.preflight import InsertionPreflight  # noqa: E402
from precision_insertion.trial_preflight import (  # noqa: E402
    plan_admitted_key_trial, plan_fresh_key_trial,
    write_trial_preflight_artifacts,
)


class _Planner:
    def __init__(self):
        self.starts = []
        self.pickups = []

    def set_start_state(self, q):
        self.starts.append(np.asarray(q).copy())

    def plan(self, scene, name, version, **kwargs):
        key = tuple(kwargs["candidate_override"][3][0])
        self.pickups.append(key)
        if key[-1] == "1":
            return SimpleNamespace(success=False)
        return SimpleNamespace(
            success=True, scene_info=key,
            pregrasp_pose=np.zeros(6), grasp_pose=np.ones(6) * 0.2,
        )


def _fixture(tmp_path, monkeypatch):
    mode = select_mode("square", 1.5)
    record = {
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "socket_observations": [{"timestamp_s": 2.0}],
    }
    calibration = SimpleNamespace(record=record, collision_scene={})
    scene = {"mesh": {"target": {"pose": [0, 0, 0.1, 1, 0, 0, 0]}}}
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.validate_catalog_session",
        lambda _catalog, **_: None)
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.build_trial_scene_from_session",
        lambda **_: scene)
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.classify_key_tabletop_pose",
        lambda **_: {"stem": "000", "rotation_error_deg": 0.0})
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.planner_candidate_override",
        lambda selected, **_: (
            np.eye(4)[None], np.zeros((1, 6)), np.ones((1, 6)) * 0.2,
            [tuple(selected[0]["key"])], [None]))
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.build_rigid_insertion_targets",
        lambda **_: object())

    def select(catalog, *, tabletop_pose_stem, attempted=(), **_):
        rows = [row for row in catalog["candidates"]
                if row["tabletop_pose_stem"] == tabletop_pose_stem and
                tuple(row["key"]) not in set(attempted)]
        return {"status": ("candidates_available" if rows else
                           "no_eligible_in_screened_pool"),
                "candidates": rows}

    monkeypatch.setattr(
        "precision_insertion.trial_preflight.select_pose_candidates", select)
    rows = []
    for grasp_id in ("1", "2", "3"):
        directory = tmp_path / "candidates" / grasp_id
        directory.mkdir(parents=True)
        np.save(directory / "wrist_se3.npy", np.eye(4))
        rows.append({"key": ["table", "0", grasp_id],
                     "tabletop_pose_stem": "000", "eligible": True,
                     "candidate_dir": str(directory)})
    catalog = {"shared_root": str(tmp_path), "candidates": rows}
    limits = PathAuditLimits(
        max_joint_step_rad=0.02, max_wrist_step_m=0.005,
        max_wrist_rotation_deg=1.0, goal_position_tolerance_m=0.001,
        goal_rotation_tolerance_deg=1.0,
        axial_lateral_tolerance_m=0.001,
        axial_rotation_tolerance_deg=1.0,
        minimum_hand_clearance_m=0.001)
    return mode, calibration, catalog, limits


def _run(tmp_path, fixture, planner, **overrides):
    mode, calibration, catalog, limits = fixture
    arguments = dict(
        planner=planner, mode=mode, shared_root=tmp_path,
        calibration=calibration, catalog=catalog,
        key_pose_world=np.eye(4), key_observation_id="key-capture-3",
        key_capture_timestamp_s=3.0, live_start_q=np.zeros(13),
        start_q_acquisition_timestamp_s=3.001,
        max_key_state_skew_s=0.02,
        limits=limits, max_pose_error_deg=10.0,
        axial_waypoint_step_m=0.005,
    )
    arguments.update(overrides)
    return plan_fresh_key_trial(**arguments)


def test_live_wrapper_checks_every_key_camera_time_before_planning(monkeypatch):
    mode = select_mode("square", 1.5)
    observation = KeyPoseObservation(
        capture_id="key_1", request_id=31, key_object=mode.key_object,
        family=mode.family, pose_world=np.eye(4),
        selected_camera_id="cam_a", selected_acquisition_timestamp_s=100.0,
        acquisition_interval_s=(99.999, 100.006), frame_evidence={},
        per_view={}, consistency={}, selection={},
        source_capture_dir=Path("/unused"))
    received = {}

    def fake_plan(**kwargs):
        received.update(kwargs)
        return "planned"

    monkeypatch.setattr(
        "precision_insertion.trial_preflight.plan_fresh_key_trial", fake_plan)
    with pytest.raises(ValueError, match="every key view"):
        plan_admitted_key_trial(
            key_observation=observation,
            start_q_acquisition_timestamp_s=100.02,
            max_key_state_skew_s=0.01, mode=mode)
    assert not received
    assert plan_admitted_key_trial(
        key_observation=observation,
        start_q_acquisition_timestamp_s=100.003,
        max_key_state_skew_s=0.01, mode=mode) == "planned"
    assert received["key_observation_id"] == "key_1"
    assert received["key_capture_timestamp_s"] == 100.0
    assert received["key_pose_world"] == pytest.approx(np.eye(4))


def test_fresh_trial_tries_next_grasp_after_pickup_or_insertion_preflight_failure(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    calls = []

    def plan_after(*, pickup_plan, **kwargs):
        del kwargs
        calls.append(tuple(pickup_plan.scene_info))
        status = ("transfer_hand_drift" if calls[-1][-1] == "2"
                  else "sampled_planning_pass")
        queries = (({"stage": "transfer", "success": True,
                     "held_hand_lock_verified": False,
                     "max_abs_hand_delta_rad": 0.03725},)
                   if status == "transfer_hand_drift" else ())
        return InsertionPreflight(
            status, None, None, None, {"sampled_clear": status ==
                                      "sampled_planning_pass"},
            0, np.zeros(6), "commanded_nominal", queries)

    monkeypatch.setattr(
        "precision_insertion.trial_preflight.plan_insertion_after_pickup",
        plan_after)
    planner = _Planner()
    result = _run(tmp_path, fixture, planner)
    assert result.status == "sampled_planning_pass"
    assert result.selected_candidate_key == ("table", "0", "3")
    assert planner.pickups == [("table", "0", "1"),
                               ("table", "0", "2"),
                               ("table", "0", "3")]
    assert len(planner.starts) == 3
    assert [row["insertion_preflight_status"]
            for row in result.attempted_candidates] == [
                None, "transfer_hand_drift", "sampled_planning_pass"]
    assert result.attempted_candidates[1]["held_hand_drift"] == {
        "stage": "transfer", "max_abs_delta_rad": 0.03725}
    assert result.to_record()["robot_ready"] is False


def test_fresh_trial_rejects_hand_drift_without_query_evidence(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.plan_insertion_after_pickup",
        lambda **_: InsertionPreflight(
            "transfer_hand_drift", None, None, None, None, 0,
            np.zeros(6), "commanded_nominal", ()))
    with pytest.raises(ValueError, match="lacks planner evidence"):
        _run(tmp_path, fixture, _Planner())


def test_no_current_pose_grasp_requests_repose_only_if_other_stem_has_one(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    catalog = fixture[2]
    catalog["candidates"] = [{**catalog["candidates"][0],
                              "tabletop_pose_stem": "001"}]
    planner = _Planner()
    result = _run(tmp_path, fixture, planner)
    assert result.status == "repose_required_unplanned"
    assert result.repose_target_stems == ("001",)
    assert planner.pickups == []
    catalog["candidates"] = []
    result = _run(tmp_path, fixture, planner)
    assert result.status == "no_eligible_pose_in_catalog"
    assert result.repose_target_stems == ()


def test_trial_records_reset_seed_assessment_without_authorizing_repose(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    catalog = fixture[2]
    catalog["candidates"] = [{**catalog["candidates"][0],
                              "tabletop_pose_stem": "001"}]
    seen = []

    def assess(**kwargs):
        seen.append(kwargs)
        return {"status": "reset_seed_available_requires_full_chain_preflight",
                "targets": [{"target_pose_stem": "001"}], "robot_ready": False}

    monkeypatch.setattr(
        "precision_insertion.trial_preflight.assess_repose_options", assess)
    result = _run(
        tmp_path, fixture, _Planner(),
        max_reset_center_drift_m=0.003,
        max_reset_axis_tilt_deg=8.0,
        reset_candidate_root=tmp_path / "handoff")
    assert result.status == "repose_required_unplanned"
    assert result.repose_assessment["robot_ready"] is False
    assert result.to_record()["schema"] == "precision_insertion_trial_preflight_v2"
    assert result.to_record()["repose_assessment"]["targets"][0][
        "target_pose_stem"] == "001"
    assert seen[0]["target_stems"] == ("001",)
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.assess_repose_options",
        lambda **_: {"status": "no_executable_repose_path",
                      "targets": [], "robot_ready": False})
    unavailable = _run(
        tmp_path, fixture, _Planner(),
        max_reset_center_drift_m=0.003, max_reset_axis_tilt_deg=8.0)
    assert unavailable.status == "repose_assets_unavailable"
    monkeypatch.setattr(
        "precision_insertion.trial_preflight.assess_repose_options",
        lambda **_: {
            "status": "staged_reset_seed_requires_install_and_full_chain_preflight",
            "targets": [], "robot_ready": False})
    staged = _run(
        tmp_path, fixture, _Planner(),
        max_reset_center_drift_m=0.003, max_reset_axis_tilt_deg=8.0)
    assert staged.status == "repose_staged_only_unplanned"
    with pytest.raises(ValueError, match="both reset pose-fidelity limits"):
        _run(tmp_path, fixture, _Planner(), max_reset_axis_tilt_deg=8.0)
    with pytest.raises(ValueError, match="finite and positive"):
        _run(tmp_path, fixture, _Planner(),
             max_reset_center_drift_m=-0.003,
             max_reset_axis_tilt_deg=8.0)


def test_candidate_budget_and_stale_key_observation_are_fail_closed(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    result = _run(tmp_path, fixture, planner, max_candidate_attempts=1)
    assert result.status == "candidate_budget_exhausted"
    assert len(result.attempted_candidates) == 1
    with pytest.raises(ValueError, match="follow frozen socket"):
        _run(tmp_path, fixture, planner, key_capture_timestamp_s=1.9)
    with pytest.raises(ValueError, match="not time-aligned"):
        _run(tmp_path, fixture, planner,
             start_q_acquisition_timestamp_s=3.1)


def test_trial_artifacts_are_new_and_do_not_claim_robot_readiness(
    tmp_path, monkeypatch,
):
    fixture = _fixture(tmp_path, monkeypatch)
    fixture[2]["candidates"] = []
    result = _run(tmp_path, fixture, _Planner())
    output = write_trial_preflight_artifacts(result, tmp_path / "run_001")
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["robot_ready"] is False
    assert report["cartesian_planner_mode"] == "default"
    assert (output / "trial_scene.json").is_file()
    assert "planned_trajectories" not in report["artifacts"]
    with pytest.raises(FileExistsError):
        write_trial_preflight_artifacts(result, output)

    q = np.zeros((2, 13))
    plan = InsertionPreflight(
        "sampled_planning_pass", q, q, q, {"sampled_clear": True},
        1, np.zeros(6), "commanded_nominal", ())
    planned = replace(
        result, status="sampled_planning_pass", insertion_plan=plan,
        selected_candidate_key=("table", "0", "3"),
        pickup_plan=SimpleNamespace(
            success=True, scene_info=("table", "0", "3"), traj=q,
            pregrasp_pose=np.zeros(6), grasp_pose=np.ones(6) * .2,
            wrist_se3=np.eye(4)))
    output = write_trial_preflight_artifacts(planned, tmp_path / "run_002")
    with np.load(output / "planned_trajectories.npz") as paths:
        assert paths["axial"].shape == (2, 13)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert len(report["artifacts"]["planned_trajectories_sha256"]) == 64
