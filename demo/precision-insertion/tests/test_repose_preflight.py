"""Planning-only reset composition retains the frozen socket."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import coal  # noqa: F401 -- AutoDex host loads Coal before Trimesh
import numpy as np
import pytest
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodex.utils.conversion import se32cart  # noqa: E402
from autodex.utils.tabletop_geometry import table_cuboid  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.path_audit import PathAuditLimits  # noqa: E402
from precision_insertion.repose_preflight import (  # noqa: E402
    build_v8_repose_rest_pose, plan_repose_held_chain,
    validate_repose_rest_target,
)
from precision_insertion.repose_transition import (  # noqa: E402
    preflight_v8_repose_transition,
)
from precision_insertion.repose_release import (  # noqa: E402
    build_repose_release_world, plan_repose_release_exit,
)
from precision_insertion.repose_artifacts import (  # noqa: E402
    write_repose_preflight_artifacts,
)
from precision_insertion.world import add_fixed_mesh_fixtures  # noqa: E402


MODE = select_mode("square", 1.5)


def _pose(x: float, z: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 3] = (x, 0.0, z)
    return pose


class _Planner:
    _n_arm = 7
    _hand = "fr3_inspire"
    _robot_cfg = {"kinematics": {"ee_link": "base_link"}}

    def __init__(self):
        self.calls = []
        self.reject_transfer = False
        self.reject_descent = False
        self.reject_postrelease = False
        self.pickup_plan = None

    def set_start_state(self, q):
        self.calls.append(("start", tuple(np.asarray(q).shape)))

    def plan(self, scene, *, candidate_override, **kwargs):
        self.calls.append(("pickup", sorted(scene["mesh"])))
        assert candidate_override[0].shape == (1, 4, 4)
        assert candidate_override[3][0][0] == "reset"
        return self.pickup_plan

    def fk_wrist(self, q):
        return _pose(float(q[0]), float(q[2]))

    def plan_lift_preflight(self, start, scene, lift_h):
        self.calls.append(("lift", sorted(scene["mesh"])))
        result = np.repeat(np.asarray(start)[None], 11, axis=0)
        result[:, 2] = np.linspace(start[2], start[2] + lift_h, 11)
        return SimpleNamespace(traj=result)

    def plan_cartesian_pose(self, start, goal, *, scene_cfg, **kwargs):
        self.calls.append(("transfer", sorted(scene_cfg["mesh"])))
        if self.reject_transfer:
            return SimpleNamespace(success=False, trajectory=None,
                                   constraint_mode="test",
                                   failure_stage="blocked")
        path = np.repeat(np.asarray(start)[None], 11, axis=0)
        path[:, 0] = np.linspace(start[0], goal[0, 3], 11)
        path[:, 2] = np.linspace(start[2], goal[2, 3], 11)
        return SimpleNamespace(success=True, trajectory=path,
                               constraint_mode="test", failure_stage=None)

    def plan_vertical_stroke(self, start, wrist_start, wrist_end, *,
                             scene_cfg, **kwargs):
        stage = ("post_release_lift" if "released_rest_key" in scene_cfg["mesh"]
                 else "descent")
        self.calls.append((stage, sorted(scene_cfg["mesh"])))
        if stage == "descent" and self.reject_descent:
            return SimpleNamespace(
                success=False, trajectory=None,
                failure_code="jacobian_segment_robot_collision",
                failure_detail=(
                    "jacobian_segment_robot_collision:"
                    "MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION"),
                step_records=[{"step": 17, "target_z_m": 0.125}])
        if stage == "post_release_lift" and self.reject_postrelease:
            return SimpleNamespace(success=False, trajectory=None,
                                   failure_code="test_rejected")
        assert np.allclose(wrist_start[:3, 3], self.fk_wrist(start)[:3, 3])
        path = np.repeat(np.asarray(start)[None], 11, axis=0)
        path[:, 2] = np.linspace(start[2], wrist_end[2, 3], 11)
        return SimpleNamespace(success=True, trajectory=path,
                               failure_code=None)

    def plan_js_to_init(self, scene, start_arm_qpos, *,
                        start_hand_qpos, goal_arm_qpos):
        self.calls.append(("retract", sorted(scene["mesh"])))
        start = np.concatenate([start_arm_qpos, start_hand_qpos])
        end = np.concatenate([goal_arm_qpos, start_hand_qpos])
        return np.linspace(start, end, 11)


def _fixture(tmp_path, monkeypatch):
    assets = AssetPaths(tmp_path, MODE)
    key_path = assets.raw_mesh(MODE.key_object)
    socket_path = assets.socket_collision_mesh
    key_path.parent.mkdir(parents=True)
    socket_path.parent.mkdir(parents=True)
    trimesh.creation.box(extents=(0.006, 0.006, 0.006)).export(key_path)
    socket = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    socket.apply_translation([0.0, 0.0, 0.02])
    socket.export(socket_path)
    assets.robot_urdf.parent.mkdir(parents=True)
    assets.robot_urdf.write_text("test URDF stand-in", encoding="utf-8")
    assets.key_tabletop_dir.mkdir(parents=True)
    np.save(assets.key_tabletop_dir / "001.npy", _pose(0.0, 0.003))
    hand = trimesh.creation.box(extents=(0.006, 0.006, 0.006))
    hand.apply_translation([0.0, 0.0, 0.06])
    monkeypatch.setattr(
        "precision_insertion.repose_path_audit._hand_link_meshes",
        lambda *_: {"hand": hand})
    monkeypatch.setattr(
        "precision_insertion.repose_release._hand_link_meshes",
        lambda *_: {"hand": hand})
    board = {
        "table_surface_z_m": 0.0,
        "corners_robot_m": [
            [-0.40, -0.40, 0.0], [-0.40, 0.40, 0.0],
            [0.40, -0.40, 0.0], [0.40, 0.40, 0.0],
        ],
    }
    socket_pose = _pose(0.30, 0.0)
    frozen = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {"table": table_cuboid(board)}},
        {"fixture_socket": {"pose_robot": socket_pose,
                            "collision_mesh": socket_path}})
    record = {
        "schema": "precision_insertion_session_calibration_v1",
        "mode": {"family": MODE.family, "gap_mm": MODE.gap_mm,
                 "key_object": MODE.key_object,
                 "socket_object": MODE.socket_object},
        "socket_pose_robot": socket_pose.tolist(),
        "socket_collision_mesh": str(socket_path),
        "socket_collision_mesh_sha256": hashlib.sha256(
            socket_path.read_bytes()).hexdigest(),
    }
    calibration = SessionCalibration(board, socket_pose, {}, frozen, record)
    initial = _pose(-0.10, 0.003)
    rest = _pose(-0.05, 0.003)
    scene = {"mesh": dict(frozen["mesh"]), "cuboid": frozen["cuboid"]}
    scene["mesh"]["target"] = {
        "file_path": str(key_path), "pose": se32cart(initial).tolist()}
    pickup_q = np.zeros((2, 13))
    pickup_q[:, 0] = -0.10
    pickup_q[:, 2] = 0.003
    pickup = SimpleNamespace(
        success=True, lift_preflight=object(), traj=pickup_q,
        wrist_se3=initial, pregrasp_pose=np.zeros(6))
    limits = PathAuditLimits(
        max_joint_step_rad=0.02, max_wrist_step_m=0.02,
        max_wrist_rotation_deg=1.0,
        goal_position_tolerance_m=0.0005,
        goal_rotation_tolerance_deg=1.0,
        axial_lateral_tolerance_m=0.001,
        axial_rotation_tolerance_deg=1.0,
        minimum_hand_clearance_m=0.001)
    return calibration, scene, pickup, rest, limits


def _plan(tmp_path, calibration, scene, pickup, rest, limits, planner):
    return plan_repose_held_chain(
        planner=planner, pickup_plan=pickup, trial_scene=scene,
        shared_root=tmp_path, calibration=calibration, mode=MODE,
        T_key_hand=np.eye(4), T_robot_key_rest=rest,
        release_height_m=0.10, minimum_rest_socket_clearance_m=0.01,
        minimum_board_edge_clearance_m=0.01,
        held_hand_q=np.zeros(6), held_hand_source="commanded_nominal",
        limits=limits)


def test_repose_held_chain_keeps_socket_and_does_not_authorize_release(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, rest, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    result = _plan(tmp_path, calibration, scene, pickup, rest, limits, planner)
    assert result.status == "sampled_held_path_pass_release_unplanned"
    assert result.sampled_held_path_audit["sampled_clear"] is True
    assert result.to_record()["robot_ready"] is False
    assert result.to_record()["sampled_held_path_pass"] is True
    assert planner.calls == [
        ("lift", ["fixture_socket", "target"]),
        ("transfer", ["fixture_socket"]),
        ("descent", ["fixture_socket", "target"]),
    ]
    assert "target" not in calibration.collision_scene["mesh"]


def test_repose_rejects_rest_key_on_socket_before_planning(tmp_path, monkeypatch):
    calibration, scene, pickup, _, limits = _fixture(tmp_path, monkeypatch)
    rest_on_socket = _pose(0.30, 0.003)
    with pytest.raises(ValueError, match="collides with or approaches socket"):
        validate_repose_rest_target(
            shared_root=tmp_path, mode=MODE, calibration=calibration,
            T_robot_key_rest=rest_on_socket,
            support_tolerance_m=limits.goal_position_tolerance_m,
            minimum_rest_socket_clearance_m=0.01,
            minimum_board_edge_clearance_m=0.01)
    planner = _Planner()
    with pytest.raises(ValueError, match="collides with or approaches socket"):
        _plan(tmp_path, calibration, scene, pickup, rest_on_socket, limits,
              planner)
    assert planner.calls == []


def test_repose_rejects_key_footprint_outside_measured_board(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, _, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    outside = _pose(0.405, 0.003)
    with pytest.raises(ValueError, match="footprint exceeds"):
        _plan(tmp_path, calibration, scene, pickup, outside, limits, planner)
    assert planner.calls == []


def test_repose_reports_transfer_infeasible_without_running_descent(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, rest, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    planner.reject_transfer = True
    result = _plan(tmp_path, calibration, scene, pickup, rest, limits, planner)
    assert result.status == "held_transfer_unreachable"
    assert result.descent_trajectory is None
    assert result.to_record()["robot_ready"] is False
    assert [row[0] for row in planner.calls] == ["lift", "transfer"]


def test_v8_target_builder_uses_measured_table_and_rejects_bad_asset(
    tmp_path, monkeypatch,
):
    calibration, _, _, _, limits = _fixture(tmp_path, monkeypatch)
    result = build_v8_repose_rest_pose(
        shared_root=tmp_path, mode=MODE, calibration=calibration,
        target_pose_stem="001", release_xy_robot_m=(-0.05, 0.02),
        asset_support_tolerance_m=limits.goal_position_tolerance_m)
    assert np.allclose(result[:3, 3], [-0.05, 0.02, 0.003])
    with pytest.raises(ValueError, match="three-digit"):
        build_v8_repose_rest_pose(
            shared_root=tmp_path, mode=MODE, calibration=calibration,
            target_pose_stem="1", release_xy_robot_m=(-0.05, 0.02),
            asset_support_tolerance_m=limits.goal_position_tolerance_m)
    bad = _pose(0.0, 0.02)
    np.save(AssetPaths(tmp_path, MODE).key_tabletop_dir / "001.npy", bad)
    with pytest.raises(ValueError, match="does not rest"):
        build_v8_repose_rest_pose(
            shared_root=tmp_path, mode=MODE, calibration=calibration,
            target_pose_stem="001", release_xy_robot_m=(-0.05, 0.02),
            asset_support_tolerance_m=limits.goal_position_tolerance_m)


def _transition(tmp_path, calibration, scene, limits, planner, **overrides):
    catalog = {
        "shared_root": str(tmp_path),
        "mode": {"family": MODE.family, "gap_mm": MODE.gap_mm,
                 "key_object": MODE.key_object,
                 "socket_object": MODE.socket_object,
                 "target_depth_m": MODE.target_depth_m},
    }
    q = np.zeros(13)
    q[0], q[2] = -0.10, 0.003
    args = dict(
        planner=planner, shared_root=tmp_path, mode=MODE,
        calibration=calibration, trial_scene=scene, catalog=catalog,
        from_pose_stem="000", to_pose_stem="001", height_cm=12,
        release_xy_robot_m=(-0.05, 0.0), live_start_q=q,
        observation_id="camera-seq-42", key_capture_timestamp_s=1.0,
        start_q_acquisition_timestamp_s=1.01, max_state_skew_s=0.1,
        max_pose_error_deg=10.0, max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=8.0,
        minimum_rest_socket_clearance_m=0.01,
        minimum_board_edge_clearance_m=0.01, limits=limits)
    args.update(overrides)
    return preflight_v8_repose_transition(**args)


def test_v8_reset_seed_is_screened_then_planned_in_frozen_socket_world(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, _, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    planner.pickup_plan = pickup
    compat_calls = []
    monkeypatch.setattr(
        "precision_insertion.repose_transition.install_curobo_planner_compat",
        lambda: compat_calls.append("installed"))
    monkeypatch.setattr(
        "precision_insertion.repose_transition.select_pose_candidates",
        lambda *_, **__: {"status": "candidates_available",
                        "candidates": [object()]})
    monkeypatch.setattr(
        "precision_insertion.repose_transition.classify_key_tabletop_pose",
        lambda **_: {"stem": "000"})
    initial = _pose(-0.10, 0.003)
    monkeypatch.setattr(
        "precision_insertion.repose_transition.load_v8_reset_seeds",
        lambda **_: {
            "n_total": 1,
            "wrist_se3": initial[None],
            "pregrasp": np.zeros((1, 6)),
            "grasp": np.zeros((1, 6)),
            "openpose_start": [None],
            "scene_info": [{"grasp_idx": "191", "source": "/verified/191",
                            "cell": "0_1"}],
        })
    result = _transition(tmp_path, calibration, scene, limits, planner)
    assert result.status == "held_reset_path_available_release_unplanned"
    assert compat_calls == ["installed"]
    assert result.selected_seed["seed_id"] == "191"
    assert result.to_record()["robot_ready"] is False
    assert [call[0] for call in planner.calls] == [
        "start", "pickup", "lift", "transfer", "descent"]
    assert planner.calls[3] == ("transfer", ["fixture_socket"])
    planner.reject_descent = True
    rejected = _transition(tmp_path, calibration, scene, limits, planner)
    assert rejected.status == "no_held_reset_path"
    descent_query = rejected.attempted_seeds[0]["held_planner_queries"][-1]
    assert descent_query["failure_detail"].endswith(
        "INVALID_START_STATE_WORLD_COLLISION")
    assert descent_query["failed_step"] == 17
    assert descent_query["failed_target_z_m"] == 0.125


def test_repose_does_not_plan_if_target_has_no_insertable_grasp(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, _, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    planner.pickup_plan = pickup
    monkeypatch.setattr(
        "precision_insertion.repose_transition.select_pose_candidates",
        lambda *_, **__: {"status": "no_eligible_in_screened_pool",
                        "candidates": []})
    result = _transition(tmp_path, calibration, scene, limits, planner)
    assert result.status == "target_without_insertable_grasp"
    assert result.to_record()["robot_ready"] is False
    assert planner.calls == []


def test_v8_reset_seed_requires_release_exit_when_requested(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, _, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    planner.pickup_plan = pickup
    monkeypatch.setattr(
        "precision_insertion.repose_transition.select_pose_candidates",
        lambda *_, **__: {"status": "candidates_available",
                        "candidates": [object()]})
    monkeypatch.setattr(
        "precision_insertion.repose_transition.classify_key_tabletop_pose",
        lambda **_: {"stem": "000"})
    initial = _pose(-0.10, 0.003)
    monkeypatch.setattr(
        "precision_insertion.repose_transition.load_v8_reset_seeds",
        lambda **_: {
            "n_total": 1, "wrist_se3": initial[None],
            "pregrasp": np.zeros((1, 6)), "grasp": np.zeros((1, 6)),
            "openpose_start": [None],
            "scene_info": [{"grasp_idx": "191", "source": "/verified/191",
                            "cell": "0_1"}],
        })
    retreat = np.zeros(7)
    retreat[0], retreat[2] = -0.10, 0.30
    result = _transition(
        tmp_path, calibration, scene, limits, planner,
        retreat_goal_arm_q=retreat, minimum_release_key_clearance_m=0.001)
    assert result.status == "nominal_reset_preflight_pass_drop_unobserved"
    assert result.release_plan.status == (
        "nominal_release_exit_path_pass_drop_unobserved")
    assert result.to_record()["robot_ready"] is False
    assert result.attempted_seeds[0]["release_status"] == result.release_plan.status


def test_release_world_and_exit_preflight_keep_socket_and_two_key_poses(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, rest, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    held = _plan(tmp_path, calibration, scene, pickup, rest, limits, planner)
    world, floating = build_repose_release_world(
        trial_scene=scene, calibration=calibration,
        T_robot_key_rest=rest, release_height_m=held.release_height_m)
    assert sorted(world["mesh"]) == [
        "fixture_socket", "released_rest_key", "target"]
    assert np.isclose(floating[2, 3], rest[2, 3] + 0.10)
    goal = held.descent_trajectory[-1, :7].copy()
    goal[0] -= 0.05
    result = plan_repose_release_exit(
        planner=planner, trial_scene=scene, shared_root=tmp_path,
        calibration=calibration, mode=MODE, held_plan=held,
        release_hand_q=np.zeros(6), retreat_goal_arm_q=goal,
        minimum_release_key_clearance_m=0.001, limits=limits)
    assert result.status == "nominal_release_exit_path_pass_drop_unobserved"
    assert result.release_geometry_audit["sampled_clear"] is True
    assert result.to_record()["robot_ready"] is False
    assert planner.calls[-2:] == [
        ("post_release_lift", ["fixture_socket", "released_rest_key", "target"]),
        ("retract", ["fixture_socket", "released_rest_key", "target"]),
    ]


def test_release_exit_rejects_open_hand_inside_floating_key(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, rest, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    held = _plan(tmp_path, calibration, scene, pickup, rest, limits, planner)
    hand = trimesh.creation.box(extents=(0.004, 0.004, 0.004))
    monkeypatch.setattr(
        "precision_insertion.repose_release._hand_link_meshes",
        lambda *_: {"hand": hand})
    result = plan_repose_release_exit(
        planner=planner, trial_scene=scene, shared_root=tmp_path,
        calibration=calibration, mode=MODE, held_plan=held,
        release_hand_q=np.zeros(6),
        retreat_goal_arm_q=np.zeros(7),
        minimum_release_key_clearance_m=0.001, limits=limits)
    assert result.status == "sampled_release_exit_rejected"
    assert any(row["obstacle"] == "key/floating_release"
               for row in result.release_geometry_audit["failures"]
               if row["reason"] == "release_geometry_collision_or_clearance")
    assert not any(call[0] == "retract" for call in planner.calls)


def test_release_exit_rejects_mutated_held_path_evidence(tmp_path, monkeypatch):
    calibration, scene, pickup, rest, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    held = _plan(tmp_path, calibration, scene, pickup, rest, limits, planner)
    held.descent_trajectory[1, 0] += 0.01
    with pytest.raises(ValueError, match="evidence changed"):
        plan_repose_release_exit(
            planner=planner, trial_scene=scene, shared_root=tmp_path,
            calibration=calibration, mode=MODE, held_plan=held,
            release_hand_q=np.zeros(6),
            retreat_goal_arm_q=np.zeros(7),
            minimum_release_key_clearance_m=0.001, limits=limits)
    assert not any(call[0] == "post_release_lift" for call in planner.calls)


def test_repose_report_saves_paths_and_input_hashes_without_overwrite(
    tmp_path, monkeypatch,
):
    calibration, scene, pickup, _, limits = _fixture(tmp_path, monkeypatch)
    planner = _Planner()
    planner.pickup_plan = pickup
    monkeypatch.setattr(
        "precision_insertion.repose_transition.select_pose_candidates",
        lambda *_, **__: {"status": "candidates_available",
                        "candidates": [object()]})
    monkeypatch.setattr(
        "precision_insertion.repose_transition.classify_key_tabletop_pose",
        lambda **_: {"stem": "000"})
    initial = _pose(-0.10, 0.003)
    monkeypatch.setattr(
        "precision_insertion.repose_transition.load_v8_reset_seeds",
        lambda **_: {
            "n_total": 1, "wrist_se3": initial[None],
            "pregrasp": np.zeros((1, 6)), "grasp": np.zeros((1, 6)),
            "openpose_start": [None],
            "scene_info": [{"grasp_idx": "191", "source": "/verified/191",
                            "cell": "0_1"}],
        })
    retreat = np.zeros(7)
    retreat[0], retreat[2] = -0.10, 0.30
    result = _transition(
        tmp_path, calibration, scene, limits, planner,
        retreat_goal_arm_q=retreat, minimum_release_key_clearance_m=0.001)
    inputs = {}
    for name in ("session", "catalog", "key_pose_world", "live_start_q",
                 "limits"):
        path = tmp_path / f"{name}.input"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    output = tmp_path / "new_reset_report"
    write_repose_preflight_artifacts(
        result=result, trial_scene=scene, output_dir=output,
        source_files=inputs)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["robot_ready"] is False
    assert report["artifacts"]["input_files"]["session"]["sha256"] == (
        hashlib.sha256(b"session").hexdigest())
    with np.load(output / "planned_trajectories.npz") as saved:
        assert {"pickup_approach", "held_lift", "held_transfer",
                "held_descent", "post_release_lift",
                "post_release_retract"} <= set(saved.files)
    with pytest.raises(FileExistsError):
        write_repose_preflight_artifacts(
            result=result, trial_scene=scene, output_dir=output,
            source_files=inputs)


def test_repose_cli_exposes_planning_only_command():
    cli = Path(__file__).resolve().parents[1] / "run_pipeline.py"
    completed = subprocess.run(
        [sys.executable, str(cli), "preflight-repose", "--help"],
        capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert "--reset-candidate-dir" in completed.stdout
    assert "--retreat-goal-q-npy" in completed.stdout
