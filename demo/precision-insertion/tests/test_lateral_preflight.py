"""Grounded 1 mm hold shifts remain read-only and geometry-audited."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import coal  # noqa: F401 -- load before trimesh on the AutoDex host
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.lateral_preflight import (  # noqa: E402
    plan_lateral_hold_shift, write_lateral_hold_preflight,
)
from precision_insertion.uncertainty_margin import (  # noqa: E402
    SurfaceDeviationBounds,
)
from test_path_audit import _fixture, _limits  # noqa: E402


class _Planner:
    _n_arm = 7
    _hand = "fr3_inspire"
    _robot_cfg = {"kinematics": {"ee_link": "base_link"}}

    def __init__(self):
        self.queries = []

    def fk_wrist(self, q):
        T = np.eye(4)
        T[:3, 3] = q[:3]
        return T

    def plan_cartesian_pose(self, start, goal, **kwargs):
        self.queries.append((start.copy(), goal.copy(), kwargs))
        end = start.copy()
        end[:3] = goal[:3, 3]
        path = np.linspace(start, end, 6)
        return SimpleNamespace(success=True, trajectory=path,
                               constraint_mode="test", failure_stage=None)


def _inputs(tmp_path, monkeypatch):
    calibration, targets, _paths = _fixture(tmp_path, monkeypatch)
    start = np.zeros(13)
    start[:3] = [-0.002, 0.0, 0.135]
    expected = np.eye(4)
    expected[:3, 3] = start[:3]
    scene = {"mesh": dict(calibration.collision_scene["mesh"]),
             "cuboid": calibration.collision_scene["cuboid"]}
    scene["mesh"]["target"] = {"file_path": "held-key"}
    return dict(
        planner=_Planner(), mode=targets.mode, shared_root=tmp_path,
        calibration=calibration, trial_scene=scene,
        start_q=start, expected_hold_pose=expected,
        T_key_hand=np.eye(4), increment_socket_xy_m=(0.001, 0.0),
        bounds=SurfaceDeviationBounds(
            0.0001, 0.0001, "commissioned_future_trial_surface_bound"),
        limits=_limits(), max_path_deviation_m=0.0001,
        max_hold_height_deviation_m=0.0001,
        max_hold_rotation_deg=1.0)


def test_lateral_plan_reuses_stock_planner_and_never_authorizes_insertion(
        tmp_path, monkeypatch):
    args = _inputs(tmp_path, monkeypatch)
    result = plan_lateral_hold_shift(**args)
    assert result.status == "sampled_lateral_hold_shift_pass"
    assert result.to_record()["insertion_replan_allowed"] is False
    assert result.to_record()["robot_ready"] is False
    assert result.sampled_audit["sampled_clear"] is True
    start, goal, query = args["planner"].queries[0]
    assert np.allclose(start, args["start_q"])
    assert np.allclose(goal[:3, 3], [-0.001, 0, 0.135])
    assert "target" not in query["scene_cfg"]["mesh"]
    assert "fixture_socket" in query["scene_cfg"]["mesh"]
    assert query["lock_hand"] is True
    output = write_lateral_hold_preflight(result, tmp_path / "lateral")
    saved = json.loads((output / "report.json").read_text())
    assert saved["status"] == result.status
    assert (output / "lateral_trajectory.npy").is_file()
    assert len(saved["lateral_trajectory_sha256"]) == 64
    with pytest.raises(FileExistsError):
        write_lateral_hold_preflight(result, output)


def test_lateral_plan_rejects_wrong_start_and_uncommissioned_bounds(
        tmp_path, monkeypatch):
    args = _inputs(tmp_path, monkeypatch)
    bad = args["expected_hold_pose"].copy()
    bad[0, 3] += 0.01
    with pytest.raises(ValueError, match="not at the saved withdrawn hold"):
        plan_lateral_hold_shift(**{**args, "expected_hold_pose": bad})
    with pytest.raises(ValueError, match="<= 1 mm"):
        plan_lateral_hold_shift(**{
            **args, "increment_socket_xy_m": (0.0011, 0.0)})
    with pytest.raises(ValueError, match="future-trial surface bound"):
        plan_lateral_hold_shift(**{
            **args, "bounds": SurfaceDeviationBounds(
                .0001, .0001, "empirical_scatter")})
    assert args["planner"].queries == []


def test_lateral_plan_rejects_sampled_future_margin(
        tmp_path, monkeypatch):
    args = _inputs(tmp_path, monkeypatch)
    args["bounds"] = SurfaceDeviationBounds(
        1.0, .0001, "commissioned_future_trial_surface_bound")
    result = plan_lateral_hold_shift(**args)
    assert result.status == "sampled_lateral_hold_shift_rejected"
    assert any(row["reason"] ==
               "future_surface_bound_exceeds_sampled_clearance"
               for row in result.sampled_audit["failures"])
