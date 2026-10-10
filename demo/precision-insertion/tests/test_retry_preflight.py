"""A VLM proposal gets a fresh held-state path, never old stroke replay."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.retry_preflight import (  # noqa: E402
    plan_xy_retry_from_withdrawn_hold, write_xy_retry_preflight,
)
from precision_insertion.targets import InsertionTargets  # noqa: E402
from precision_insertion.xy_retry import assess_xy_retry  # noqa: E402
from test_preflight import _Planner  # noqa: E402
from test_xy_retry import _setup  # noqa: E402


def _pose(x, z):
    pose = np.eye(4)
    pose[0, 3] = x
    pose[2, 3] = z
    return pose


def _inputs(tmp_path, monkeypatch):
    args = _setup(tmp_path)
    assessment = assess_xy_retry(**args)
    mode = args["mode"]
    calibration = args["calibration"]
    start = np.zeros(13)
    start[2] = 0.30
    start[7:] = 0.2
    scene = {
        "cuboid": calibration.collision_scene["cuboid"],
        "mesh": {**calibration.collision_scene["mesh"],
                 "target": {"pose": [0, 0, 0.1, 1, 0, 0, 0],
                            "file_path": "key.obj"}},
    }

    def targets(*, xy_offset_socket_m, T_key_hand, **_kwargs):
        x = xy_offset_socket_m[0]
        preinsert = _pose(x, 0.30)
        entry = _pose(x, 0.27)
        verification = _pose(x, 0.25)
        return InsertionTargets(
            mode, "a" * 64, "b" * 64, T_key_hand,
            preinsert, entry, verification,
            preinsert @ T_key_hand, entry @ T_key_hand,
            verification @ T_key_hand,
            np.array([0, 0, -1.0]), tuple(xy_offset_socket_m), 0.03)

    monkeypatch.setattr(
        "precision_insertion.retry_preflight.build_rigid_insertion_targets",
        targets)
    monkeypatch.setattr(
        "precision_insertion.retry_preflight.screen_grasp_endpoint",
        lambda **_kwargs: {"endpoint_pass": True})
    monkeypatch.setattr(
        "precision_insertion.preflight.audit_held_joint_paths",
        lambda **_kwargs: {"sampled_clear": True})
    limits = PathAuditLimits(
        max_joint_step_rad=0.02, max_wrist_step_m=0.005,
        max_wrist_rotation_deg=1.0, goal_position_tolerance_m=0.001,
        goal_rotation_tolerance_deg=1.0, axial_lateral_tolerance_m=0.001,
        axial_rotation_tolerance_deg=1.0,
        minimum_hand_clearance_m=0.001)
    return {
        "planner": _Planner(), "assessment": assessment,
        "calibration": calibration, "catalog": args["catalog"],
        "mode": mode, "shared_root": tmp_path,
        "candidate_key": args["candidate_key"],
        "tabletop_pose_stem": "000",
        "observed_T_key_hand": args["observed_T_key_hand"],
        "observed_relation_timestamp_s": 100.0,
        "live_start_q": start, "start_q_timestamp_s": 100.01,
        "max_state_skew_s": 0.05,
        "held_hand_source": "commanded_nominal",
        "trial_scene": scene, "limits": limits,
        "axial_waypoint_step_m": 0.005,
    }


def test_retry_plans_fresh_transfer_and_axial_from_withdrawn_state(
    tmp_path, monkeypatch,
):
    args = _inputs(tmp_path, monkeypatch)
    result = plan_xy_retry_from_withdrawn_hold(**args)
    assert result.status == "sampled_retry_preflight_pass"
    assert result.choice_id == "x_plus_1mm"
    assert len(args["planner"].calls) == 11
    assert result.planning.lift_trajectory is None
    assert result.targets.xy_offset_socket_m == pytest.approx((0.001, 0.0))
    assert result.to_record()["robot_ready"] is False
    saved = write_xy_retry_preflight(result, tmp_path / "retry_report")
    assert (saved / "report.json").is_file()
    with np.load(saved / "planned_trajectories.npz") as arrays:
        assert arrays["transfer"].shape == (5, 13)
        assert arrays["axial"].shape == (41, 13)
    with pytest.raises(FileExistsError):
        write_xy_retry_preflight(result, saved)


def test_retry_rejects_stale_or_nonwithdrawn_state(tmp_path, monkeypatch):
    args = _inputs(tmp_path, monkeypatch)
    args["start_q_timestamp_s"] = 101.0
    with pytest.raises(ValueError, match="stale"):
        plan_xy_retry_from_withdrawn_hold(**args)
    args = _inputs(tmp_path / "other", monkeypatch)
    start = args["live_start_q"].copy()
    start[2] = 0.25
    args["live_start_q"] = start
    with pytest.raises(ValueError, match="not at the verified withdrawn hold"):
        plan_xy_retry_from_withdrawn_hold(**args)


def test_fresh_endpoint_failure_never_calls_planner(tmp_path, monkeypatch):
    args = _inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "precision_insertion.retry_preflight.screen_grasp_endpoint",
        lambda **_kwargs: {"endpoint_pass": False})
    with pytest.raises(ValueError, match="fresh exact 20 mm endpoint"):
        plan_xy_retry_from_withdrawn_hold(**args)
    assert args["planner"].calls == []
