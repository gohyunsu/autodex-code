"""Fresh key pose must use bound frames and agreeing AutoDex views."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.key_perception import (  # noqa: E402
    admit_held_key_capture, admit_key_capture, admit_postlift_key_capture,
    verify_key_capture_artifacts, write_key_capture_artifacts,
)
from precision_insertion.live_capture import KeyCaptureInput  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion.world import add_fixed_mesh_fixtures  # noqa: E402
from autodex.utils.conversion import se32cart  # noqa: E402


CAMERAS = ("cam_a", "cam_b")


class SelectorStub:
    intrinsics_undist = {
        camera: np.array([[100, 0, 16], [0, 100, 12], [0, 0, 1]], dtype=float)
        for camera in CAMERAS}
    extrinsics = {}
    for _camera in CAMERAS:
        _extrinsic = np.eye(4)
        _extrinsic[2, 3] = 1.0
        extrinsics[_camera] = _extrinsic

    def __init__(self, object_name):
        self.obj_name = object_name
        self.calls = 0

    def refine_from_payloads(self, masks, poses, **kwargs):
        self.calls += 1
        assert kwargs["selection_mode"] == "iou"
        assert set(kwargs["subset_serials"]) == set(CAMERAS)
        return poses["cam_a"]["pose_world"], {
            "best_serial": "cam_a", "best_iou": 0.8,
            "sil_loss": 0.001, "sil_skipped": False,
        }


def _capture(poses, tmp_path):
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:15, 8:20] = True
    times = {"cam_a": 100.0, "cam_b": 100.005}
    frame_ids = {"cam_a": 21, "cam_b": 22}
    return KeyCaptureInput(
        "key_001", 51, "blue cylindrical key",
        {serial: image.copy() for serial in CAMERAS},
        {serial: {"mask": mask.copy(), "frame_id": frame_ids[serial]}
         for serial in CAMERAS},
        {serial: {"ok": True, "pose_world": poses[serial],
                  "quality": 0.8, "inliers": 20, "mask_pixels": 108,
                  "frame_id": frame_ids[serial]}
         for serial in CAMERAS},
        times, "camera_acquisition",
        {serial: {"frame_id": frame_ids[serial],
                  "image_sha256": image_sha256(image),
                  "timestamp_s": times[serial], "max_error_s": 0.001,
                  "timestamp_method": "hardware_exposure",
                  "clock_domain": "unix_utc"}
         for serial in CAMERAS},
        tmp_path / "key_001_request_51")


def _session(root, mode, *, socket_x=0.4):
    mesh = (root / "object_processing" / mode.socket_object /
            "processed_data" / "mesh" / "static_collision.obj")
    mesh.parent.mkdir(parents=True, exist_ok=True)
    vertices = [
        (socket_x + dx, dy, dz)
        for dx in (-0.02, 0.02)
        for dy in (-0.02, 0.02)
        for dz in (-0.02, 0.02)
    ]
    mesh.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in vertices),
                    encoding="utf-8")
    pose = np.eye(4)
    world = add_fixed_mesh_fixtures(
        {"mesh": {}, "cuboid": {}},
        {"fixture_socket": {"pose_robot": pose,
                            "collision_mesh": mesh}})
    record = {
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "c2r": np.eye(4).tolist(),
        "socket_pose_robot": pose.tolist(),
        "socket_collision_mesh": str(mesh.resolve()),
        "socket_collision_mesh_sha256": hashlib.sha256(
            mesh.read_bytes()).hexdigest(),
    }
    camera_snapshot = {
        "intrinsics_full": {
            serial: {"K_undist": SelectorStub.intrinsics_undist[serial].tolist()}
            for serial in CAMERAS},
        "extrinsics_full": {
            serial: SelectorStub.extrinsics[serial].tolist()
            for serial in CAMERAS},
    }
    record["camera_calibration"] = camera_snapshot
    record["camera_calibration_sha256"] = hashlib.sha256(json.dumps(
        camera_snapshot, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    assert world["mesh"]["fixture_socket"]["pose"] == se32cart(pose).tolist()
    return SessionCalibration({}, pose, {}, world, record)


def _admit(capture, mode, root, selector, *, socket_x=0.4):
    return admit_key_capture(
        capture=capture, init_orchestrator=selector,
        mode=mode, shared_root=root,
        calibration=_session(root, mode, socket_x=socket_x),
        calibrated_camera_ids=set(CAMERAS),
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        maximum_multiview_center_error_mm=2.0,
        maximum_multiview_angle_error_deg=5.0,
        maximum_socket_mask_overlap_fraction=0.1,
        socket_projection_dilation_px=0,
        minimum_refinement_iou=0.5)


def test_square_key_uses_agreeing_iou_selected_pose_and_full_capture_interval(
        tmp_path):
    mode = select_mode("square", 1.5)
    first = np.eye(4)
    second = np.eye(4)
    second[0, 3] = 0.001
    capture = _capture({"cam_a": first, "cam_b": second}, tmp_path)
    selector = SelectorStub(mode.key_object)
    result = _admit(capture, mode, tmp_path, selector)
    assert selector.calls == 1
    assert result.selected_camera_id == "cam_a"
    assert result.selection["minimum_refinement_iou"] == 0.5
    assert result.consistency["max_center_residual_mm"] == pytest.approx(1.0)
    assert result.acquisition_interval_s == pytest.approx((99.999, 100.006))
    assert result.to_record()["robot_ready"] is False
    with pytest.raises(ValueError, match="timing differs"):
        write_key_capture_artifacts(
            capture, replace(result, acquisition_interval_s=(99.0, 100.006)),
            tmp_path / "bad_key_evidence")
    bundle = write_key_capture_artifacts(
        capture, result, tmp_path / "key_evidence")
    assert verify_key_capture_artifacts(bundle)["robot_ready"] is False
    with pytest.raises(FileExistsError):
        write_key_capture_artifacts(capture, result, bundle)
    result.require_state_alignment(state_timestamp_s=100.003,
                                   maximum_skew_s=0.01)
    with pytest.raises(ValueError, match="every key view"):
        result.require_state_alignment(state_timestamp_s=100.02,
                                       maximum_skew_s=0.01)
    (bundle / "images" / "cam_a.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="key evidence changed"):
        verify_key_capture_artifacts(bundle)


def test_tabletop_key_rejects_low_or_missing_refinement_iou(tmp_path):
    mode = select_mode("square", 1.5)
    poses = {serial: np.eye(4) for serial in CAMERAS}
    capture = _capture(poses, tmp_path)
    selector = SelectorStub(mode.key_object)
    selector.refine_from_payloads = lambda *_args, **_kwargs: (
        np.eye(4), {"best_serial": "cam_a", "best_iou": 0.2,
                    "sil_loss": 0.0001})
    with pytest.raises(ValueError, match="IoU is below commissioned limit"):
        _admit(capture, mode, tmp_path, selector)
    selector.refine_from_payloads = lambda *_args, **_kwargs: (
        np.eye(4), {"best_serial": "cam_a", "best_iou": None,
                    "sil_loss": 0.0001})
    with pytest.raises(ValueError, match="IoU is below commissioned limit"):
        _admit(capture, mode, tmp_path, selector)


def test_rejects_mismatched_pixels_or_disagreeing_square_key_pose(tmp_path):
    mode = select_mode("square", 1.5)
    poses = {serial: np.eye(4) for serial in CAMERAS}
    capture = _capture(poses, tmp_path)
    capture.frame_evidence["cam_b"]["image_sha256"] = "0" * 64
    selector = SelectorStub(mode.key_object)
    with pytest.raises(ValueError, match="image digest mismatch"):
        _admit(capture, mode, tmp_path, selector)
    assert selector.calls == 0

    capture = _capture(poses, tmp_path)
    capture.poses["cam_b"]["pose_world"] = np.eye(4)
    capture.poses["cam_b"]["pose_world"][0, 3] = 0.005
    with pytest.raises(ValueError, match="multiview FoundPose estimates disagree"):
        _admit(capture, mode, tmp_path, selector)
    assert selector.calls == 0


def test_cylinder_identical_end_flip_is_one_physical_pose(tmp_path):
    mode = select_mode("cylinder", 20)
    info = (tmp_path / "object_processing" / mode.key_object /
            "processed_data" / "info")
    info.mkdir(parents=True)
    (info / "symmetry.json").write_text(json.dumps({
        "type": "Dinf", "center": [0, 0, 0.04],
        "axes": [{"fold": "inf", "axis": [0, 0, 1]},
                 {"fold": 2, "axis": [1, 0, 0]}],
    }), encoding="utf-8")
    first = np.eye(4)
    flipped = np.eye(4)
    flipped[:3, :3] = np.diag([1, -1, -1])
    flipped[2, 3] = 0.08
    capture = _capture({"cam_a": first, "cam_b": flipped}, tmp_path)
    result = _admit(capture, mode, tmp_path, SelectorStub(mode.key_object))
    assert result.consistency["cylinder_symmetry_quotient"] is True
    assert result.consistency["max_center_residual_mm"] == pytest.approx(0.0)
    assert result.consistency["max_angle_residual_deg"] == pytest.approx(0.0)


def test_fixed_socket_projection_rejects_key_mask_on_socket(tmp_path):
    mode = select_mode("square", 1.5)
    poses = {serial: np.eye(4) for serial in CAMERAS}
    capture = _capture(poses, tmp_path)
    with pytest.raises(ValueError, match="fixed-socket mask exclusion"):
        _admit(capture, mode, tmp_path, SelectorStub(mode.key_object),
               socket_x=0.0)


def test_held_key_can_overlap_socket_only_with_fresh_wrist_prior_and_iou(tmp_path):
    mode = select_mode("square", 1.5)
    pose = np.eye(4)
    capture = _capture({serial: pose for serial in CAMERAS}, tmp_path)
    common = dict(
        capture=capture, init_orchestrator=SelectorStub(mode.key_object),
        mode=mode, shared_root=tmp_path,
        calibration=_session(tmp_path, mode, socket_x=0.0),
        calibrated_camera_ids=set(CAMERAS),
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        maximum_multiview_center_error_mm=2.0,
        maximum_multiview_angle_error_deg=5.0,
        held_pose_prior_world=pose,
        held_pose_prior_timestamp_s=100.002,
        held_pose_prior_source="measured_wrist_plus_observed_held_relation",
        maximum_held_prior_center_error_mm=2.0,
        maximum_held_prior_angle_error_deg=5.0,
        maximum_held_prior_time_skew_s=0.01,
        minimum_held_refinement_iou=0.5)
    admitted = admit_held_key_capture(**common)
    assert admitted.phase == "held_preinsert"
    assert admitted.consistency["socket_exclusion"] is None
    bundle = write_key_capture_artifacts(
        capture, admitted, tmp_path / "held_evidence")
    assert verify_key_capture_artifacts(bundle)["robot_ready"] is False
    with pytest.raises(ValueError, match="motion prior is stale"):
        admit_held_key_capture(**{
            **common, "held_pose_prior_timestamp_s": 99.0})
    with pytest.raises(ValueError, match="wrist pose-prior gate"):
        far_pose = pose.copy()
        far_pose[0, 3] = 0.01
        admit_held_key_capture(**{
            **common, "held_pose_prior_world": far_pose})
    with pytest.raises(ValueError, match="IoU is below"):
        admit_held_key_capture(**{
            **common, "minimum_held_refinement_iou": 0.9})


def test_first_postlift_prior_is_distinct_from_observed_preinsert_prior(tmp_path):
    mode = select_mode("square", 1.5)
    pose = np.eye(4)
    capture = _capture({serial: pose for serial in CAMERAS}, tmp_path)
    common = dict(
        capture=capture, init_orchestrator=SelectorStub(mode.key_object),
        mode=mode, shared_root=tmp_path,
        calibration=_session(tmp_path, mode, socket_x=0.0),
        calibrated_camera_ids=set(CAMERAS),
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        maximum_multiview_center_error_mm=2.0,
        maximum_multiview_angle_error_deg=5.0,
        candidate_pose_prior_world=pose,
        measured_wrist_timestamp_s=100.002,
        maximum_candidate_prior_center_error_mm=20.0,
        maximum_candidate_prior_angle_error_deg=30.0,
        maximum_prior_time_skew_s=0.01,
        minimum_refinement_iou=0.5)
    admitted = admit_postlift_key_capture(**common)
    assert admitted.phase == "held_postlift"
    assert admitted.consistency["held_pose_prior"]["source"] == (
        "measured_wrist_plus_candidate_grasp")
    bundle = write_key_capture_artifacts(
        capture, admitted, tmp_path / "postlift_evidence")
    assert verify_key_capture_artifacts(bundle)["robot_ready"] is False
    with pytest.raises(ValueError, match="observed held relation"):
        admit_held_key_capture(
            capture=capture, init_orchestrator=SelectorStub(mode.key_object),
            mode=mode, shared_root=tmp_path,
            calibration=_session(tmp_path, mode, socket_x=0.0),
            calibrated_camera_ids=set(CAMERAS),
            view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
            maximum_multiview_center_error_mm=2.0,
            maximum_multiview_angle_error_deg=5.0,
            held_pose_prior_world=pose,
            held_pose_prior_timestamp_s=100.002,
            held_pose_prior_source="measured_wrist_plus_candidate_grasp",
            maximum_held_prior_center_error_mm=20.0,
            maximum_held_prior_angle_error_deg=30.0,
            maximum_held_prior_time_skew_s=0.01,
            minimum_held_refinement_iou=0.5)


def test_key_rejects_camera_recalibration_after_socket_freeze(tmp_path):
    mode = select_mode("square", 1.5)
    capture = _capture({serial: np.eye(4) for serial in CAMERAS}, tmp_path)
    session = _session(tmp_path, mode)
    selector = SelectorStub(mode.key_object)
    selector.intrinsics_undist = dict(selector.intrinsics_undist)
    selector.intrinsics_undist["cam_a"] = (
        selector.intrinsics_undist["cam_a"].copy())
    selector.intrinsics_undist["cam_a"][0, 0] += 1.0
    with pytest.raises(ValueError, match="live camera calibration changed"):
        admit_key_capture(
            capture=capture, init_orchestrator=selector, mode=mode,
            shared_root=tmp_path, calibration=session,
            calibrated_camera_ids=set(CAMERAS),
            view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
            maximum_multiview_center_error_mm=2.0,
            maximum_multiview_angle_error_deg=5.0,
            maximum_socket_mask_overlap_fraction=0.1,
            socket_projection_dilation_px=0,
            minimum_refinement_iou=0.5)
