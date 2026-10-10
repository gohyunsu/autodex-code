"""Composition of original pickup/lift and cuRobo calls, without robot I/O."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.preflight import (  # noqa: E402
    plan_held_transfer_and_axial, plan_insertion_after_pickup,
)
from precision_insertion.targets import InsertionTargets  # noqa: E402


def _pose(z):
    pose = np.eye(4)
    pose[2, 3] = z
    return pose


class _Planner:
    _n_arm = 7
    _hand = "fr3_inspire"
    _robot_cfg = {"kinematics": {"ee_link": "base_link"}}

    def __init__(self, fail_query=None, hand_drift_query=None,
                 lift_hand_drift=False, start_mismatch_query=None):
        self.fail_query = fail_query
        self.hand_drift_query = hand_drift_query
        self.lift_hand_drift = lift_hand_drift
        self.start_mismatch_query = start_mismatch_query
        self.calls = []
        self.lift_start = None

    def fk_wrist(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3]
        return pose

    def plan_lift_preflight(self, start, scene, lift_h):
        self.lift_start = np.asarray(start).copy()
        end = self.lift_start.copy()
        end[2] += lift_h
        traj = np.linspace(start, end, 11)
        if self.lift_hand_drift:
            traj[1:, 7] += 0.01
        return SimpleNamespace(traj=traj)

    def plan_cartesian_pose(self, start, goal, **kwargs):
        self.calls.append((np.asarray(start).copy(), np.asarray(goal).copy(),
                           kwargs))
        if self.fail_query == len(self.calls):
            return SimpleNamespace(success=False, trajectory=None)
        end = np.asarray(start).copy()
        end[:3] = goal[:3, 3]
        traj = np.linspace(start, end, 5)
        if self.hand_drift_query == len(self.calls):
            traj[1:, 7] += 0.01
        if self.start_mismatch_query == len(self.calls):
            traj[0, 0] += 0.01
        return SimpleNamespace(success=True, trajectory=traj)


def _fixture():
    mode = select_mode("square", 1.5)
    hand_in_key = np.eye(4)
    preinsert = _pose(0.30)
    entry = _pose(0.27)
    verification = _pose(0.25)
    targets = InsertionTargets(
        mode=mode, task_geometry_sha256="x" * 64,
        socket_collision_mesh_sha256="y" * 64,
        T_key_hand=hand_in_key,
        T_robot_key_preinsert=preinsert,
        T_robot_key_entry=entry,
        T_robot_key_verification=verification,
        T_robot_hand_preinsert=preinsert,
        T_robot_hand_entry=entry,
        T_robot_hand_verification=verification,
        insertion_axis_robot=np.array([0.0, 0.0, -1.0]),
        xy_offset_socket_m=(0.0, 0.0), preinsert_clearance_m=0.030,
    )
    frozen = {"cuboid": {"table": {"dims": [1, 1, 1],
                                    "pose": [0, 0, 0, 1, 0, 0, 0]}},
              "mesh": {"fixture_socket": {"pose": [0, 0, 0, 1, 0, 0, 0],
                                           "file_path": "socket.obj"}}}
    scene = {"cuboid": frozen["cuboid"],
             "mesh": {**frozen["mesh"],
                      "target": {"pose": [0, 0, 0.1, 1, 0, 0, 0],
                                 "file_path": "key.obj"}}}
    calibration = SimpleNamespace(collision_scene=frozen)
    pickup_q = np.zeros((2, 13))
    pickup_q[-1, 2] = 0.1
    pickup = SimpleNamespace(success=True, lift_preflight=object(),
                             traj=pickup_q, wrist_se3=_pose(0.1))
    limits = PathAuditLimits(
        max_joint_step_rad=0.02, max_wrist_step_m=0.005,
        max_wrist_rotation_deg=1.0, goal_position_tolerance_m=0.001,
        goal_rotation_tolerance_deg=1.0, axial_lateral_tolerance_m=0.001,
        axial_rotation_tolerance_deg=1.0,
        minimum_hand_clearance_m=0.001)
    return pickup, scene, calibration, targets, limits


def _run(planner, fixture, **overrides):
    pickup, scene, calibration, targets, limits = fixture
    arguments = dict(
        planner=planner, pickup_plan=pickup, trial_scene=scene,
        shared_root=Path("/tmp"), calibration=calibration, targets=targets,
        held_hand_q=np.ones(6) * 0.2,
        held_hand_source="commanded_nominal", limits=limits,
        axial_waypoint_step_m=0.005,
    )
    arguments.update(overrides)
    return plan_insertion_after_pickup(**arguments)


def test_preflight_replans_held_lift_and_all_axial_waypoints(monkeypatch):
    fixture = _fixture()
    planner = _Planner()
    captured = {}

    def audit(**kwargs):
        captured.update(kwargs)
        return {"sampled_clear": True, "failures": []}

    monkeypatch.setattr("precision_insertion.preflight.audit_held_joint_paths", audit)
    result = _run(planner, fixture)
    assert result.status == "sampled_planning_pass"
    assert result.axial_waypoint_count == 10
    assert len(planner.calls) == 11  # one transfer + ten <=5 mm axial goals
    assert planner.lift_start[7:] == pytest.approx([0.2] * 6)
    assert all(call[2]["include_obj_obstacle"] is False and
               call[2]["lock_hand"] is True and
               call[2]["return_result"] is True for call in planner.calls)
    np.testing.assert_allclose(planner.calls[-1][1], fixture[3].T_robot_hand_verification)
    assert captured["transfer_trajectory"].shape == (5, 13)
    assert captured["descent_trajectory"].shape == (41, 13)
    assert captured["lift_trajectory"].shape == (11, 13)
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["held_hand_source"] == "commanded_nominal"
    assert len(result.to_record()["planner_query_records"]) == 12


def test_query_or_sampled_geometry_failure_does_not_promote_plan(monkeypatch):
    fixture = _fixture()
    monkeypatch.setattr(
        "precision_insertion.preflight.audit_held_joint_paths",
        lambda **_: {"sampled_clear": False,
                    "failures": [{"reason": "held_geometry_collision_or_clearance"}]})
    result = _run(_Planner(), fixture)
    assert result.status == "sampled_held_path_rejected"
    assert result.sampled_planning_pass is False
    result = _run(_Planner(fail_query=1), fixture)
    assert result.status == "transfer_unreachable"
    result = _run(_Planner(fail_query=4), fixture)
    assert result.status == "axial_waypoint_unreachable"
    assert result.axial_waypoint_count == 2
    assert result.axial_trajectory is None
    assert result.to_record()["planner_query_records"][-1]["success"] is False


@pytest.mark.parametrize(
    ("planner_kwargs", "status", "last_stage", "completed_waypoints"), [
        ({"lift_hand_drift": True}, "held_lift_hand_drift", "held_lift", 0),
        ({"hand_drift_query": 1}, "transfer_hand_drift", "transfer", 0),
        ({"hand_drift_query": 4}, "axial_waypoint_hand_drift",
         "axial_waypoint", 2),
    ])
def test_planner_hand_drift_rejects_only_this_candidate(
    planner_kwargs, status, last_stage, completed_waypoints,
):
    result = _run(_Planner(**planner_kwargs), _fixture())
    assert result.status == status
    assert result.sampled_planning_pass is False
    assert result.axial_waypoint_count == completed_waypoints
    assert result.axial_trajectory is None
    last_query = result.to_record()["planner_query_records"][-1]
    assert last_query["stage"] == last_stage
    assert last_query["success"] is True
    assert last_query["held_hand_lock_verified"] is False
    assert last_query["max_abs_hand_delta_rad"] == pytest.approx(0.01)


def test_planner_start_state_corruption_still_raises():
    with pytest.raises(ValueError, match="does not start at the previous"):
        _run(_Planner(start_mismatch_query=1), _fixture())


def test_preflight_rejects_wrong_world_frame_or_untrusted_hold_source(monkeypatch):
    fixture = _fixture()
    wrong_scene = {"cuboid": fixture[1]["cuboid"],
                   "mesh": {"target": fixture[1]["mesh"]["target"]}}
    with pytest.raises(ValueError, match="differs from frozen"):
        _run(_Planner(), fixture, trial_scene=wrong_scene)
    planner = _Planner()
    planner._robot_cfg = {"kinematics": {"ee_link": "fr3_link7"}}
    with pytest.raises(ValueError, match="base_link planner"):
        _run(planner, fixture)
    with pytest.raises(ValueError, match="held_hand_source"):
        _run(_Planner(), fixture, held_hand_source="assumed_measured")
    with pytest.raises(ValueError, match="<= 5 mm"):
        _run(_Planner(), fixture, axial_waypoint_step_m=0.01)


def test_fresh_arrival_plans_axial_without_another_transfer(monkeypatch):
    fixture = _fixture()
    _pickup, scene, calibration, targets, limits = fixture
    planner = _Planner()
    captured = {}

    def audit(**kwargs):
        captured.update(kwargs)
        return {"sampled_clear": True, "failures": []}

    monkeypatch.setattr("precision_insertion.preflight.audit_held_joint_paths",
                        audit)
    start = np.zeros(13)
    start[:3] = targets.T_robot_hand_preinsert[:3, 3]
    start[7:] = .2
    arguments = dict(
        planner=planner, trial_scene=scene, shared_root=Path("/tmp"),
        calibration=calibration, targets=targets, start_q=start,
        held_hand_q=start[7:], held_hand_source="measured", limits=limits,
        axial_waypoint_step_m=.005, start_at_preinsert=True)
    result = plan_held_transfer_and_axial(**arguments)
    assert result.status == "sampled_planning_pass"
    assert len(planner.calls) == 10  # axial waypoints only
    assert result.planner_query_records[0] == {
        "stage": "arrival_hold", "success": True,
        "planner_api": "measured_fk_no_transfer",
        "executable_transfer": False,
    }
    np.testing.assert_array_equal(result.transfer_trajectory[0], start)
    np.testing.assert_array_equal(result.transfer_trajectory[1], start)
    np.testing.assert_array_equal(captured["transfer_trajectory"],
                                  result.transfer_trajectory)
    assert result.axial_trajectory[0].tolist() == start.tolist()

    displaced = start.copy()
    displaced[0] = .003
    rejected = plan_held_transfer_and_axial(
        **{**arguments, "start_q": displaced})
    assert rejected.status == "arrival_hold_goal_residual"
    assert len(planner.calls) == 10  # a shifted hold never reaches cuRobo
