"""Same-episode XY correction uses measured multiview residuals and replans."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.guarded_axial_handoff import (  # noqa: E402
    verify_guarded_axial_handoff,
)
from precision_insertion.same_trial_xy import (  # noqa: E402
    CorrectedAxialPlanUnavailable, HeldXYPlanUnavailable,
    _alignment_frame_inputs, _held_wrist_goal, _shifted_targets,
    choose_key_axis_move, choose_xy_move,
)
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.retry_session import (  # noqa: E402
    _cylinder_tip_and_rois, _nominal_key_for_xy_capture,
)
from precision_insertion.targets import InsertionTargets  # noqa: E402
from precision_insertion.xy_overlay import CalibratedXYFrame  # noqa: E402


def test_same_trial_move_requires_metric_multiview_and_caps_one_motion():
    metric = {
        "status": "diagnostic_metric_xy_correction",
        "inlier_cameras": ["cam1", "cam2"],
        "xy_correction_socket_m": [.024, -.018],
    }
    move = choose_xy_move(metric, tolerance_m=.0025)
    assert move["status"] == "shift"
    np.testing.assert_allclose(
        np.linalg.norm(move["increment_socket_m"]), .010)
    assert choose_xy_move({**metric, "inlier_cameras": ["cam1"]},
                          tolerance_m=.0025)["status"] == "not_metric_multiview"
    planar = choose_xy_move({
        **metric, "status": "diagnostic_planar_xy_hypothesis",
        "inlier_cameras": ["cam1"],
        "xy_source": "single_view_nominal_tip_height_plane",
    }, tolerance_m=.0025)
    assert planar["status"] == "shift"
    assert planar["source"] == "single_view_nominal_tip_height_plane"
    short = choose_xy_move(metric, tolerance_m=.0025, max_step_m=.003)
    np.testing.assert_allclose(
        np.linalg.norm(short["increment_socket_m"]), .003)
    assert choose_xy_move({**metric,
                           "xy_correction_socket_m": [.001, -.001]},
                          tolerance_m=.0025)["status"] == "aligned"


def test_cylinder_axis_move_removes_only_tilt_and_keeps_tip_pivot():
    angle = np.deg2rad(8.)
    axis = np.array([np.sin(angle), 0., -np.cos(angle)])
    alignment = {
        "inlier_cameras": ["a", "b", "c"],
        "tip_socket_m": [.004, -.002, .05],
        "insertion_axis_socket": axis.tolist(),
        "estimator": "sam3_tip_and_shaft_axis",
    }
    choice = choose_key_axis_move(
        alignment, tolerance_rad=np.deg2rad(1.),
        max_step_rad=np.deg2rad(3.))
    assert choice["status"] == "tilt"
    assert choice["axis_error_deg"] == pytest.approx(8.)
    assert np.linalg.norm(choice["tilt_rotvec_socket_rad"]) == pytest.approx(
        np.deg2rad(3.))
    rotated = (cv2.Rodrigues(np.asarray(
        choice["tilt_rotvec_socket_rad"]))[0] @ axis)
    assert np.rad2deg(np.arccos(np.clip(-rotated[2], -1., 1.))) == pytest.approx(5.)

    pivot = np.array([.2, .3, .4])
    wrist = np.eye(4)
    wrist[:3, 3] = pivot + [.04, -.01, .08]
    pivot_in_hand = np.linalg.inv(wrist) @ np.r_[pivot, 1.]
    goal = _held_wrist_goal(
        wrist, pivot, np.eye(4), np.zeros(2), 0., 0.,
        choice["tilt_rotvec_socket_rad"])
    np.testing.assert_allclose((goal @ pivot_in_hand)[:3], pivot, atol=1e-12)


def test_lateral_observation_ranking_uses_each_calibrated_camera(
        tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    mode = select_mode("cylinder", 5)
    geometry = same_trial.AssetPaths(tmp_path, mode).task_geometry
    geometry.parent.mkdir(parents=True)
    geometry.write_text(json.dumps({
        "key_frame": {"tip_z_m": 0.}, "socket_rim_z_m": 1.,
    }))
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: np.eye(4))
    key = np.eye(4)
    key[2, 3] = 1.
    intrinsics = {}
    for camera, focal in (("one", 100.), ("two", 200.)):
        intrinsics[camera] = {
            "K_undist": [[focal, 0., 500.], [0., focal, 500.],
                         [0., 0., 1.]],
            "width": 1000, "height": 1000,
        }
    camera_two = np.eye(4)
    camera_two[0, 3] = .2
    calibration = SimpleNamespace(record={
        "c2r": np.eye(4).tolist(),
        "camera_calibration": {
            "intrinsics_full": intrinsics,
            "extrinsics_full": {
                "one": np.eye(4).tolist(), "two": camera_two.tolist()},
        },
    })
    runner = SimpleNamespace(
        mode=mode, shared_root=tmp_path, calibration=calibration,
        _postlift_preflight=SimpleNamespace(targets=SimpleNamespace(
            T_robot_key_preinsert=key)))
    ranked = same_trial.rank_lateral_observation_offsets(
        runner=runner, current_xy_socket_m=(0., 0.),
        axis_raise_m=0., component_m=.03)
    assert len(ranked) == 8
    row = next(item for item in ranked
               if item["offset_socket_m"] == [.03, 0.])
    assert row["projected_separation_px_by_camera"]["one"] == pytest.approx(3.)
    assert row["projected_separation_px_by_camera"]["two"] == pytest.approx(6.)
    assert row["best_parallax_pair"] == ["one", "two"]
    assert row["pair_separation_parallax_score_px"] > 0


def test_live_xy_measurement_reuses_just_saved_pixels(tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    frames = {"100": np.zeros((3, 4, 3), dtype=np.uint8),
              "200": np.ones((3, 4, 3), dtype=np.uint8)}
    evidence = {camera: {
        "frame_id": index,
        "image_sha256": image_sha256(frames[camera]),
        "timestamp_s": 1., "max_error_s": .001,
        "timestamp_method": "hardware_exposure",
        "clock_domain": "unix_utc",
    } for index, camera in enumerate(frames, 1)}
    raw = RawCameraCapture(
        capture_id="current", request_id=17, images_bgr=frames,
        frame_ids={"100": 1, "200": 2},
        acquisition_metadata={"request_id": 17,
                              "source": "camera_acquisition",
                              "frames": evidence})
    monkeypatch.setattr(same_trial, "verify_raw_camera_capture",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(
                            AssertionError("live path reread saved PNGs")))
    rows, images = _alignment_frame_inputs(tmp_path, raw)
    assert set(rows) == set(images) == {"100", "200"}
    assert images["100"] is frames["100"]
    assert images["200"] is frames["200"]


def test_repeat_xy_crop_prediction_follows_measured_socket_frame_shift(tmp_path):
    postlift = tmp_path / "postlift.json"
    key = np.eye(4)
    key[:3, 3] = [.1, .2, .3]
    postlift.write_text(json.dumps({
        "targets": {"T_robot_key_preinsert": key.tolist()}}))
    socket = np.eye(4)
    socket[:3, :3] = np.array([
        [0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shifted = _nominal_key_for_xy_capture(
        postlift, socket, [.006, -.004])
    np.testing.assert_allclose(shifted[:3, 3], [.104, .206, .3])
    np.testing.assert_allclose(shifted[:3, :3], key[:3, :3])
    raised = _nominal_key_for_xy_capture(
        postlift, socket, [.006, -.004], axis_raise_socket_m=.04)
    np.testing.assert_allclose(raised[:3, 3], [.104, .206, .34])


def test_repeat_xy_vlm_roi_moves_with_completed_lateral_motion(tmp_path):
    postlift = tmp_path / "postlift.json"
    key = np.eye(4)
    key[2, 3] = 1.
    postlift.write_text(json.dumps({
        "targets": {"T_robot_key_preinsert": key.tolist()}}))
    frame = CalibratedXYFrame(
        "camera1", 1., Image.new("RGB", (2048, 1536)), np.eye(4),
        np.array([[1000., 0., 1024.], [0., 1000., 768.], [0., 0., 1.]]))
    backend = SimpleNamespace(native_pixel_coordinates=True)
    original_tip, original_rois = _cylinder_tip_and_rois(
        postlift_file=postlift, socket=np.eye(4), tip_z_m=.08,
        socket_rim_z_m=.5, frames=[frame], backend=backend,
        roi_xyxy_by_camera=None)
    shifted_tip, shifted_rois = _cylinder_tip_and_rois(
        postlift_file=postlift, socket=np.eye(4), tip_z_m=.08,
        socket_rim_z_m=.5, frames=[frame], backend=backend,
        roi_xyxy_by_camera=None,
        nominal_key_xy_shift_socket_m=(.03, 0.))
    np.testing.assert_allclose(shifted_tip[:2] - original_tip[:2], [.03, 0.])
    assert shifted_rois["camera1"][0] > original_rois["camera1"][0]


def test_observation_yaw_keeps_held_key_origin_on_target_axis():
    socket = np.eye(4)
    socket[:3, :3] = np.array([
        [0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    pivot = np.array([.2, .3, .4])
    wrist = np.eye(4)
    wrist[:3, 3] = pivot + np.array([.05, -.02, .01])
    key = np.eye(4)
    key[:3, 3] = pivot
    key_in_wrist = np.linalg.inv(wrist) @ key
    goal = _held_wrist_goal(
        wrist, pivot, socket, np.array([.003, -.004]), .04,
        np.deg2rad(150.))
    held_key_goal = goal @ key_in_wrist
    np.testing.assert_allclose(
        held_key_goal[:3, 3],
        pivot + socket[:3, :2] @ np.array([.003, -.004]) +
        socket[:3, 2] * .04, atol=1e-12)
    np.testing.assert_allclose(held_key_goal[:3, 2], key[:3, 2], atol=1e-12)


@pytest.mark.parametrize("collision", [False, True])
def test_held_xy_plan_failure_keeps_key_held_for_reverse_return(
        tmp_path, monkeypatch, collision):
    import precision_insertion.same_trial_xy as same_trial

    pose = np.eye(4)
    state = LiveRobotState(
        np.zeros(13), np.zeros(7), 20., 20., 10., 20.,
        np.zeros(6), np.zeros(6), 0., np.zeros(6))
    attempted = SimpleNamespace(attempt_id="a1", candidate_id="table/0/42")
    runner = SimpleNamespace(
        active_attempt=attempted, _attempt_dir=tmp_path,
        shared_root=tmp_path, mode=object(), calibration=object(),
        _preflight=SimpleNamespace(trial_scene={}),
        _postlift_preflight=SimpleNamespace(targets=SimpleNamespace(
            T_robot_key_preinsert=pose)))
    planner = SimpleNamespace(fk_wrist=lambda _q: pose)
    monkeypatch.setattr(
        same_trial, "plan_jacobian_linear_stroke",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=collision,
            trajectory=(np.zeros((2, 13)) if collision else None),
            failure_code=(None if collision else "linear_unreachable"),
            step_records=([{}] if collision else []), timing={}))
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: pose)
    monkeypatch.setattr(same_trial, "build_held_scene_from_trial",
                        lambda **_kwargs: {})
    monkeypatch.setattr(same_trial, "_audit_held_geometry_samples",
                        lambda **_kwargs: (["collision"], {}, None))
    with pytest.raises(HeldXYPlanUnavailable) as stopped:
        same_trial.plan_and_execute_shift(
            runner=runner, planner=planner,
            adapter=SimpleNamespace(follow_transfer=lambda *_a, **_k:
                                    pytest.fail("XY motion was commanded")),
            read_state=lambda: state, motion_interlock=lambda: True,
            limits=object(),
            path_limits=SimpleNamespace(minimum_hand_clearance_m=0.),
            increment_socket_m=(.002, 0.), residual_socket_m=(.003, 0.),
            residual_source="current_metric_multiview", index=0)
    report = json.loads(stopped.value.report_path.read_text())
    assert report["schema"] == "precision_insertion_same_trial_xy_plan_failure_v1"
    assert report["scope"] == "failed_plan_no_xy_command"
    assert report["measured_start"]["full_q"] == [0.] * 13
    assert not (tmp_path / "same_trial_xy" / "shifts").exists()


def test_held_xy_uses_one_local_jacobian_plan_and_records_backend(
        tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    pose = np.eye(4)
    start_q = np.zeros(13)
    end_q = start_q.copy()
    end_q[0] = .002
    trajectory = np.stack((start_q, end_q))
    states = iter((
        LiveRobotState(start_q, np.zeros(7), 20., 20., 10., 20.,
                       np.zeros(6), np.zeros(6), 0., np.zeros(6)),
        LiveRobotState(end_q, np.zeros(7), 20., 20., 10., 20.,
                       np.zeros(6), np.zeros(6), 0., np.zeros(6)),
    ))

    def fk(q):
        wrist = pose.copy()
        wrist[0, 3] = q[0]
        return wrist

    calls = []

    def linear(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            success=True, trajectory=trajectory, failure_code=None,
            step_records=[{}, {}], timing={"total_s": .01})

    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(
            attempt_id="a1", candidate_id="table/0/42"),
        _attempt_dir=tmp_path, shared_root=tmp_path, mode=object(),
        calibration=object(), _preflight=SimpleNamespace(trial_scene={}),
        _postlift_preflight=SimpleNamespace(targets=SimpleNamespace(
            T_robot_key_preinsert=pose)))
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: pose)
    monkeypatch.setattr(same_trial, "build_held_scene_from_trial",
                        lambda **_kwargs: {})
    monkeypatch.setattr(same_trial, "plan_jacobian_linear_stroke", linear)
    monkeypatch.setattr(same_trial, "_path",
                        lambda *_args, **_kwargs: trajectory)
    monkeypatch.setattr(same_trial, "_audit_held_geometry_samples",
                        lambda **_kwargs: ([], {}, None))
    monkeypatch.setattr(same_trial, "verify_transfer_controller_result",
                        lambda *_args, **_kwargs: (10., 11., {}))
    report_path = same_trial.plan_and_execute_shift(
        runner=runner, planner=SimpleNamespace(fk_wrist=fk),
        adapter=SimpleNamespace(
            follow_transfer=lambda *_args, **_kwargs: {}),
        read_state=lambda: next(states), motion_interlock=lambda: True,
        limits=SimpleNamespace(max_execution_duration_s=5.),
        path_limits=SimpleNamespace(minimum_hand_clearance_m=0.),
        increment_socket_m=(.002, 0.), residual_socket_m=(0., 0.),
        residual_source="current_metric_multiview", index=0)
    assert len(calls) == 1
    assert calls[0][1]["step_m"] == .001
    assert calls[0][1]["expected_travel_m"] == pytest.approx(.002)
    report = json.loads(report_path.read_text())
    assert report["planning_backend"] == "damped_jacobian_linear"
    assert report["cartesian_target_segments"] == 2
    assert report["linear_plan_timing"] == {"total_s": .01}


def test_held_axis_alignment_uses_tip_pivot_jacobian_and_fixed_hand(
        tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    start_q = np.zeros(13)
    start_q[7:] = .25
    end_q = start_q.copy()
    end_q[4] = np.deg2rad(2.)
    trajectory = np.stack((start_q, end_q))
    states = iter((
        LiveRobotState(start_q, np.zeros(7), 20., 20., 10., 20.,
                       np.zeros(6), np.zeros(6), 0., np.zeros(6)),
        LiveRobotState(end_q, np.zeros(7), 20., 20., 10., 20.,
                       np.zeros(6), np.zeros(6), 0., np.zeros(6)),
    ))

    def fk(q):
        pose = np.eye(4)
        pose[:3, 3] = [.1, 0., .3]
        pose[:3, :3] = Rotation.from_rotvec(
            [0., np.asarray(q)[4], 0.]).as_matrix()
        return pose

    calls = []
    def align(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            success=True, trajectory=trajectory, failure_code=None,
            step_records=[{}, {}], timing={"total_s": .02})

    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(
            attempt_id="a1", candidate_id="table/0/42"),
        _attempt_dir=tmp_path, shared_root=tmp_path, mode=object(),
        calibration=object(), _preflight=SimpleNamespace(trial_scene={}),
        _postlift_preflight=SimpleNamespace(targets=SimpleNamespace(
            T_robot_key_preinsert=np.eye(4))))
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: np.eye(4))
    monkeypatch.setattr(same_trial, "build_held_scene_from_trial",
                        lambda **_kwargs: {})
    monkeypatch.setattr(same_trial, "plan_jacobian_tip_axis_alignment", align)
    monkeypatch.setattr(same_trial, "_path",
                        lambda *_args, **_kwargs: trajectory)
    monkeypatch.setattr(same_trial, "_audit_held_geometry_samples",
                        lambda **_kwargs: ([], {}, None))
    monkeypatch.setattr(same_trial, "verify_transfer_controller_result",
                        lambda *_args, **_kwargs: (10., 11., {}))
    report_path = same_trial.plan_and_execute_shift(
        runner=runner, planner=SimpleNamespace(fk_wrist=fk),
        adapter=SimpleNamespace(follow_transfer=lambda *_args, **_kwargs: {}),
        read_state=lambda: next(states), motion_interlock=lambda: True,
        limits=SimpleNamespace(max_execution_duration_s=5.),
        path_limits=SimpleNamespace(minimum_hand_clearance_m=0.),
        increment_socket_m=(0., 0.), residual_socket_m=(.003, -.002),
        residual_source="current_metric_tip_and_axis", index=0,
        tilt_rotvec_socket_rad=(0., np.deg2rad(2.), 0.),
        observed_tip_socket_m=(.003, -.002, .05))
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][1]["pivot_robot"], [.003, -.002, .05])
    np.testing.assert_allclose(calls[0][1]["translation_robot"], 0.)
    report = json.loads(report_path.read_text())
    assert report["planning_backend"] == "damped_jacobian_tip_axis"
    assert report["tilt_rotvec_socket_rad"][1] == pytest.approx(np.deg2rad(2.))


def test_observation_raise_moves_wrist_on_socket_axis_and_records_feedback(
        tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    socket = np.eye(4)
    socket[:3, :3] = np.array([
        [1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    trajectory = np.zeros((2, 13))
    trajectory[-1, 0] = .04
    states = iter((
        LiveRobotState(np.zeros(13), np.zeros(7), 10., 10., 10., 10.,
                       np.zeros(6), np.zeros(6), 0., np.zeros(6)),
        LiveRobotState(trajectory[-1], np.zeros(7), 11., 11., 11., 11.,
                       np.zeros(6), np.zeros(6), 0., np.zeros(6)),
    ))

    def fk(q):
        wrist = np.eye(4)
        wrist[:3, 3] = socket[:3, 2] * q[0]
        return wrist

    goals = []

    def plan(_start, goal, **_kwargs):
        goals.append(goal.copy())
        return SimpleNamespace(success=True, trajectory=trajectory)

    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(attempt_id="a1", candidate_id="table/0/42"),
        _attempt_dir=tmp_path, shared_root=tmp_path, mode=object(),
        calibration=object(), _preflight=SimpleNamespace(trial_scene={}),
        _postlift_preflight=SimpleNamespace(targets=SimpleNamespace(
            T_robot_key_preinsert=np.eye(4))))
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: socket)
    monkeypatch.setattr(same_trial, "build_held_scene_from_trial",
                        lambda **_kwargs: {})
    monkeypatch.setattr(same_trial, "_path",
                        lambda *_args, **_kwargs: trajectory)
    monkeypatch.setattr(same_trial, "_audit_held_geometry_samples",
                        lambda **_kwargs: ([], {}, None))
    monkeypatch.setattr(same_trial, "verify_transfer_controller_result",
                        lambda *_args, **_kwargs: (10., 11., {}))
    result = same_trial.plan_and_execute_shift(
        runner=runner,
        planner=SimpleNamespace(fk_wrist=fk, plan_cartesian_pose=plan),
        adapter=SimpleNamespace(follow_transfer=lambda *_args, **_kwargs: {}),
        read_state=lambda: next(states), motion_interlock=lambda: True,
        limits=SimpleNamespace(max_execution_duration_s=5.),
        path_limits=SimpleNamespace(minimum_hand_clearance_m=0.),
        increment_socket_m=(0., 0.), residual_socket_m=(0., 0.),
        residual_source="nominal_unobserved", index=0,
        axis_increment_socket_m=.04)
    np.testing.assert_allclose(goals[0][:3, 3], [0., -.04, 0.])
    report = json.loads(result.read_text())
    assert report["axis_increment_socket_m"] == .04
    assert report["measured_axis_increment_socket_m"] == .04
    np.testing.assert_allclose(report["measured_increment_socket_m"], [0., 0.])


def test_failed_raise_and_yaw_at_same_observation_keep_separate_reports(
        tmp_path, monkeypatch):
    import precision_insertion.same_trial_xy as same_trial

    pose = np.eye(4)
    state = LiveRobotState(
        np.zeros(13), np.zeros(7), 20., 20., 10., 20.,
        np.zeros(6), np.zeros(6), 0., np.zeros(6))
    runner = SimpleNamespace(
        active_attempt=SimpleNamespace(attempt_id="a1", candidate_id="table/0/42"),
        _attempt_dir=tmp_path, shared_root=tmp_path, mode=object(),
        calibration=object(), _preflight=SimpleNamespace(trial_scene={}),
        _postlift_preflight=SimpleNamespace(targets=SimpleNamespace(
            T_robot_key_preinsert=pose)))
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: pose)
    monkeypatch.setattr(same_trial, "build_held_scene_from_trial",
                        lambda **_kwargs: {})
    planner = SimpleNamespace(
        fk_wrist=lambda _q: pose,
        plan_cartesian_pose=lambda *_args, **_kwargs: SimpleNamespace(
            success=False, trajectory=None))
    paths = []
    for movement in ({"axis_increment_socket_m": .04},
                     {"yaw_increment_socket_rad": np.deg2rad(60.)}):
        with pytest.raises(HeldXYPlanUnavailable) as stopped:
            same_trial.plan_and_execute_shift(
                runner=runner, planner=planner, adapter=object(),
                read_state=lambda: state, motion_interlock=lambda: True,
                limits=object(), path_limits=object(),
                increment_socket_m=(0., 0.), residual_socket_m=(0., 0.),
                residual_source="nominal_unobserved", index=0, **movement)
        paths.append(stopped.value.report_path)
    assert {path.name for path in paths} == {"000_axis.json", "000_yaw.json"}


def test_corrected_axial_keeps_key_at_observed_residual_and_moves_wrist():
    mode = select_mode("cylinder", 5)
    key = np.eye(4)
    key[2, 3] = .05
    hand = key.copy()
    hand[0, 3] = .02
    entry = key.copy()
    entry[2, 3] = .04
    verification = key.copy()
    verification[2, 3] = .02
    original = InsertionTargets(
        mode, "a" * 64, "b" * 64, np.linalg.inv(key) @ hand,
        key, entry, verification, hand,
        hand.copy(), hand.copy(), np.array([0., 0., -1.]),
        (0., 0.), .01)
    socket = np.eye(4)
    corrected = _shifted_targets(
        original, socket, np.array([.003, -.001]),
        np.array([-.002, .001]))
    np.testing.assert_allclose(
        corrected.T_robot_key_preinsert[:2, 3], [-.002, .001])
    np.testing.assert_allclose(
        corrected.T_robot_hand_preinsert[:2, 3], [.023, -.001])
    np.testing.assert_allclose(
        corrected.T_robot_key_preinsert @ corrected.T_key_hand,
        corrected.T_robot_hand_preinsert)
    np.testing.assert_allclose(corrected.xy_offset_socket_m, [-.002, .001])


def test_corrected_handoff_dispatch_binds_latest_snapshot(tmp_path):
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source = tmp_path / "guarded_axial_handoffs" / "000"
    source.mkdir(parents=True)
    postlift = tmp_path / "postlift.json"
    preinsert = tmp_path / "preinsert.json"
    final = tmp_path / "last_preinsert"
    final.mkdir()
    manifest = final / "manifest.json"
    for path in (postlift, preinsert, manifest):
        path.write_text("{}", encoding="utf-8")
    axial = np.stack((np.zeros(13), np.array([.01] + [0.] * 12)))
    archive = source / "planned_trajectories.npz"
    np.savez_compressed(archive, axial=axial)
    report = source / "report.json"
    report.write_text(json.dumps({
        "schema": "precision_insertion_same_trial_axial_handoff_v1",
        "robot_ready": False,
        "scope": "same_trial_replanned_axial_path_not_physical_key_depth",
        "postlift_report_path": str(postlift),
        "postlift_report_sha256": digest(postlift),
        "preinsert_report_path": str(preinsert),
        "preinsert_report_sha256": digest(preinsert),
        "final_preinsert_bundle_path": str(final),
        "final_preinsert_manifest_sha256": digest(manifest),
        "trajectory_archive_path": str(archive),
        "trajectory_archive_sha256": digest(archive),
        "axial_sample_count": len(axial),
        "axial_start_q": axial[0].tolist(),
        "axial_end_q": axial[-1].tolist(),
        "measured_start": {"full_q": axial[0].tolist()},
        "planning": {"sampled_planning_pass": True,
                     "sampled_held_path_audit": {"sampled_clear": True}},
        "observations": [], "shifts": [],
    }), encoding="utf-8")
    assert verify_guarded_axial_handoff(report)["axial_sample_count"] == 2
    manifest.write_text('{"changed": true}', encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="capture changed"):
        verify_guarded_axial_handoff(report)


def test_corrected_axial_screens_measured_hand_endpoint_before_planning(
    tmp_path, monkeypatch,
):
    import precision_insertion.same_trial_xy as same_trial

    mode = select_mode("cylinder", 5)
    pose = np.eye(4)
    original = InsertionTargets(
        mode, "a" * 64, "b" * 64, pose, pose, pose, pose,
        pose, pose, pose, np.array([0., 0., -1.]), (0., 0.), .03)
    state = SimpleNamespace(
        full_q=np.zeros(13), to_record=lambda: {"full_q": [0.] * 13})
    runner = SimpleNamespace(
        mode=mode, shared_root=tmp_path, calibration=None,
        _attempt_dir=tmp_path / "attempt", _guarded_axial_index=0,
        _postlift_preflight=SimpleNamespace(targets=original),
        _preflight=SimpleNamespace(selected_candidate_key=("table", "0", "42")),
        active_attempt=SimpleNamespace(attempt_id="attempt_1", candidate_id="table/0/42"),
    )
    (tmp_path / "attempt").mkdir()
    monkeypatch.setattr(same_trial, "validated_frozen_socket_pose",
                        lambda **_kwargs: pose)
    monkeypatch.setattr(same_trial, "build_rigid_insertion_targets",
                        lambda **_kwargs: original)
    monkeypatch.setattr(same_trial, "screen_grasp_endpoint",
                        lambda **_kwargs: {
                            "endpoint_pass": False,
                            "T_socket_key_tested": pose.tolist(),
                            "hand_socket_clear_at_20mm": False,
                        })
    monkeypatch.setattr(same_trial, "plan_held_transfer_and_axial",
                        lambda **_kwargs: (_ for _ in ()).throw(
                            AssertionError("collision endpoint reached axial planner")))
    with pytest.raises(CorrectedAxialPlanUnavailable) as stopped:
        same_trial.plan_corrected_axial_handoff(
            runner=runner,
            planner=SimpleNamespace(fk_wrist=lambda _q: pose),
            path_limits=SimpleNamespace(minimum_hand_clearance_m=0.),
            read_state=lambda: state, total_xy_m=(0., 0.),
            residual_xy_m=(0., 0.), residual_source="test",
            observations=[], shifts=[], final_preinsert_bundle=tmp_path)
    failure = json.loads((tmp_path / "attempt/guarded_axial_handoffs/000/plan_failure.json").read_text())
    assert stopped.value.report_path == (
        tmp_path / "attempt/guarded_axial_handoffs/000/plan_failure.json")
    assert stopped.value.status == "corrected_20mm_endpoint_rejected"
    assert failure["status"] == "corrected_20mm_endpoint_rejected"
    assert failure["endpoint"]["hand_socket_clear_at_20mm"] is False
