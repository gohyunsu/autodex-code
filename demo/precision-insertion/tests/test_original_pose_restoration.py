"""Original-tabletop-target geometry and no unmeasured state substitution."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.original_pose_restoration import (  # noqa: E402
    original_center_release_xy, plan_original_pose_restoration,
)
from autodex.utils.conversion import se32cart  # noqa: E402


def test_canonical_release_origin_preserves_original_cad_center(tmp_path):
    mode = select_mode("cylinder", 20)
    folder = AssetPaths(tmp_path, mode).key_tabletop_dir
    folder.mkdir(parents=True)
    canonical = np.eye(4)
    canonical[:3, :3] = np.array([[0., 0., 1.],
                                 [0., 1., 0.],
                                 [-1., 0., 0.]])
    np.save(folder / "001.npy", canonical)
    center = (0., 0., .04)
    xy = original_center_release_xy(
        shared_root=tmp_path, mode=mode, target_pose_stem="001",
        original_center_xy_robot_m=(.32, -.08),
        key_center_local_m=center)
    assert xy == pytest.approx((.28, -.08))
    assert np.asarray(xy) + (canonical[:3, :3] @ center)[:2] == pytest.approx(
        (.32, -.08))


def test_restoration_requires_exposure_aligned_measured_joints(tmp_path):
    with pytest.raises(ValueError, match="camera-synchronized measured joints"):
        plan_original_pose_restoration(
            planner=object(), shared_root=tmp_path, mode=select_mode("cylinder", 20),
            calibration=object(), catalog={},
            key_observation=SimpleNamespace(selected_acquisition_timestamp_s=10.),
            measured_start_q=np.zeros(13), start_q_timestamp_s=11.,
            max_state_skew_s=.02, target_pose_stem="000",
            target_xy_robot_m=(0., 0.), release_height_cm=12,
            max_pose_error_deg=10., minimum_rest_socket_clearance_m=.005,
            minimum_board_edge_clearance_m=.005,
            minimum_release_key_clearance_m=.005,
            retreat_goal_arm_q=np.zeros(7), limits=object())


def test_same_pose_restoration_reuses_v8_grasp_and_fully_opens_hand(
        tmp_path, monkeypatch):
    from precision_insertion import original_pose_restoration as module

    candidate = tmp_path / "candidate"
    candidate.mkdir()
    np.save(candidate / "wrist_se3.npy", np.eye(4))
    row = {"key": ["table", "0", "3"], "candidate_dir": str(candidate)}
    other = {"key": ["table", "0", "1"], "candidate_dir": str(candidate)}
    scene = {"mesh": {"target": {"pose": se32cart(np.eye(4)).tolist()}}}
    monkeypatch.setattr(module, "build_trial_scene_from_session",
                        lambda **_: scene)
    monkeypatch.setattr(module, "classify_key_tabletop_pose",
                        lambda **_: {"stem": "000"})
    monkeypatch.setattr(module, "build_v8_repose_rest_pose",
                        lambda **_: np.eye(4))
    monkeypatch.setattr(module, "select_pose_candidates",
                        lambda *_args, **_kwargs: {
                            "status": "candidates_available",
                            "candidates": [other, row]})
    monkeypatch.setattr(module, "planner_candidate_override",
                        lambda **kwargs: tuple(kwargs["selected"][0]["key"]))
    monkeypatch.setattr(module, "install_curobo_planner_compat", lambda: None)
    held_requests = []

    def plan_held(**kwargs):
        held_requests.append(kwargs)
        return SimpleNamespace(
            status="sampled_held_path_pass_release_unplanned")

    monkeypatch.setattr(module, "plan_repose_held_chain", plan_held)
    release_requests = []

    def release(**kwargs):
        release_requests.append(kwargs)
        return SimpleNamespace(
            status="nominal_release_exit_path_pass_drop_unobserved")

    monkeypatch.setattr(module, "plan_repose_release_exit", release)

    class Planner:
        def set_start_state(self, _state):
            pass

        def plan(self, _scene, _object, _version, **kwargs):
            assert kwargs["candidate_override"] == ("table", "0", "3")
            return SimpleNamespace(
                success=True, lift_preflight=object(),
                scene_info=("table", "0", "3"),
                pregrasp_pose=np.full(6, 0.2),
                grasp_pose=np.full(6, 0.3))

    observation = SimpleNamespace(
        selected_acquisition_timestamp_s=10.0,
        pose_world=np.eye(4), capture_id="landing_001")
    result, frozen_scene = plan_original_pose_restoration(
        planner=Planner(), shared_root=tmp_path,
        mode=select_mode("cylinder", 20),
        calibration=SimpleNamespace(record={"c2r": np.eye(4).tolist()}),
        catalog={}, key_observation=observation,
        measured_start_q=np.zeros(13), start_q_timestamp_s=10.01,
        max_state_skew_s=0.1, target_pose_stem="000",
        target_xy_robot_m=(0.1, 0.2), release_height_cm=12,
        max_pose_error_deg=10., minimum_rest_socket_clearance_m=.005,
        minimum_board_edge_clearance_m=.005,
        minimum_release_key_clearance_m=.005,
        retreat_goal_arm_q=np.zeros(7),
        limits=SimpleNamespace(goal_position_tolerance_m=.001),
        preferred_candidate_key=("table", "0", "3"))
    assert result.status == "nominal_reset_preflight_pass_drop_unobserved"
    assert result.selected_seed["seed_id"] == "table/0/3"
    assert frozen_scene is scene
    assert len(release_requests) == 1
    assert held_requests[0]["release_height_m"] == 0.0
    assert np.array_equal(release_requests[0]["release_hand_q"], np.zeros(6))
