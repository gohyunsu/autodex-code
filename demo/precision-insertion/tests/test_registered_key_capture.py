"""The optional live ROI path never silently accepts stock poses."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.key_perception import (  # noqa: E402
    KeyPoseObservation, verify_key_capture_artifacts,
    write_key_capture_artifacts,
)
from precision_insertion.live_capture import KeyCaptureInput  # noqa: E402
from precision_insertion.registered_key_capture import (  # noqa: E402
    RegisteredKeyAssistance, derive_registered_key_capture,
    prepare_registered_key_assistance,
    verify_registered_key_diagnostic, write_registered_key_diagnostic,
)
from precision_insertion.registered_key_roi import RegisteredKeyROILimits  # noqa: E402
from precision_insertion.session_key_baseline import SessionKeyBaseline  # noqa: E402
from precision_insertion.live_session_runner import OpenedPrecisionSession  # noqa: E402
from precision_insertion import registered_key_capture as producer  # noqa: E402


CAMERAS = ("cam_a", "cam_b")


def _limits():
    return RegisteredKeyROILimits(
        minimum_shared_corners=8, minimum_homography_inliers=7,
        homography_ransac_px=2., maximum_board_shift_px=5.,
        difference_threshold=40, socket_dilation_px=5,
        closing_kernel_px=3, minimum_component_pixels=100,
        maximum_competitor_fraction=.8, crop_pad_px=8,
        crop_pad_fraction=.15, minimum_sam_pixels=100,
        maximum_sam_pixels=1000,
        minimum_difference_support_fraction=.8,
        maximum_socket_overlap_fraction=.05)


def _board(_image, *, board_ids):
    assert board_ids == ["9"]
    xy = np.asarray([(x, y) for y in (20., 100., 220.)
                     for x in (20., 120., 220.)], dtype=np.float32)
    return {"9": {"checkerIDs": np.arange(len(xy)),
                  "checkerCorner": xy}}


def _inputs(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    baseline = np.full((240, 240, 3), 100, dtype=np.uint8)
    socket_mask_u8 = np.zeros((240, 240), dtype=np.uint8)
    cv2.circle(socket_mask_u8, (100, 100), 12, 1, -1)
    socket_mask = socket_mask_u8 > 0
    current = baseline.copy()
    current[145:165, 170:190] = (0, 0, 255)
    images = {camera: current.copy() for camera in CAMERAS}
    evidence, masks, poses, baseline_source = {}, {}, {}, {}
    for index, camera in enumerate(CAMERAS):
        frame_id = 50 + index
        digest = image_sha256(images[camera])
        evidence[camera] = {
            "frame_id": frame_id, "image_sha256": digest,
            "timestamp_s": 100. + index * .002, "max_error_s": .001,
            "timestamp_method": "hardware_exposure", "clock_domain": "unix_utc",
        }
        source = {"frame_id": frame_id, "image_sha256": digest,
                  "image_space": "autodex_undistorted_full_frame"}
        masks[camera] = {**source, "mask": socket_mask.copy()}
        poses[camera] = {**source, "ok": True, "pose_world": np.eye(4),
                         "quality": .9, "inliers": 20,
                         "mask_pixels": int(socket_mask.sum())}
        baseline_source[camera] = {
            "frame_id": 100 + index,
            "image_sha256": image_sha256(baseline),
            "mask_file_sha256": "a" * 64,
            "mask_pixels_sha256": hashlib.sha256(
                np.ascontiguousarray(socket_mask).tobytes()).hexdigest(),
            "timestamp_s": 90., "capture_id": "socket_001",
            "request_id": 2,
        }
    capture = KeyCaptureInput(
        "key_001", 3, "red tabletop key", images, masks, poses,
        {camera: evidence[camera]["timestamp_s"] for camera in CAMERAS},
        "camera_acquisition", evidence, tmp_path / "raw_key")
    reference = SessionKeyBaseline(
        session_dir, "socket_001", 2, "b" * 64,
        {camera: baseline.copy() for camera in CAMERAS},
        {camera: socket_mask.copy() for camera in CAMERAS}, baseline_source)
    board_path = tmp_path / "board.json"
    board_path.write_text("{}", encoding="utf-8")

    class Segmentor:
        def segment(self, rgb, _prompt):
            return rgb[:, :, 0] > 200

    class FoundPose:
        obj_name = "test_key"
        called_masks = []

        def estimate_one_view(self, *, image_rgb, mask_bool, K, ext_cw):
            self.called_masks.append(mask_bool.copy())
            assert image_rgb.shape == (240, 240, 3)
            assert K.shape == (3, 3) and ext_cw.shape == (4, 4)
            return {"pose_world": np.eye(4), "quality": .8,
                    "inliers": 24, "template_id": 1}

    model = FoundPose()
    assistance = RegisteredKeyAssistance(
        reference, _board, "9", _limits(), Segmentor(), model,
        board_path, hashlib.sha256(board_path.read_bytes()).hexdigest())
    return capture, assistance, model


def _derive(capture, assistance):
    return derive_registered_key_capture(
        capture=capture, assistance=assistance,
        intrinsics_undist={camera: np.eye(3) for camera in CAMERAS},
        extrinsics_full={camera: np.eye(4) for camera in CAMERAS},
        key_object="test_key")


def test_same_full_frame_foundpose_and_distinct_stock_summary(tmp_path):
    capture, assistance, model = _inputs(tmp_path)
    derived, report = _derive(capture, assistance)
    assert len(model.called_masks) == 2
    assert all(int(mask.sum()) == 400 for mask in model.called_masks)
    assert all(int(derived.masks[camera]["mask"].sum()) == 400
               for camera in CAMERAS)
    assert all(derived.poses[camera]["ok"] for camera in CAMERAS)
    assert report["per_view"]["cam_a"]["stock_full_frame"][
        "source"] == "stock_full_frame_sam3_foundpose_not_selected"
    assert report["per_view"]["cam_a"]["status"] == "foundpose_available"
    assert derived.frame_evidence == capture.frame_evidence
    assert derived.request_id == capture.request_id


def test_roi_rejection_does_not_fall_back_to_stock_pose(tmp_path):
    capture, assistance, model = _inputs(tmp_path)

    class BadSegmentor:
        def segment(self, rgb, _prompt):
            return np.ones(rgb.shape[:2], dtype=bool)

    object.__setattr__(assistance, "sam_segmentor", BadSegmentor())
    derived, report = _derive(capture, assistance)
    assert model.called_masks == []
    assert not any(derived.poses[camera]["ok"] for camera in CAMERAS)
    assert all(report["per_view"][camera]["status"] ==
               "registered_roi_view_rejected" for camera in CAMERAS)


def test_changed_baseline_or_current_frame_is_rejected(tmp_path):
    capture, assistance, _model = _inputs(tmp_path)
    assistance.baseline.socket_masks["cam_a"][0, 0] = True
    with pytest.raises(ValueError, match="baseline mask changed"):
        _derive(capture, assistance)
    capture, assistance, _model = _inputs(tmp_path / "second")
    capture.images_bgr["cam_a"][0, 0] = (255, 0, 0)
    with pytest.raises(ValueError, match="source frame metadata"):
        _derive(capture, assistance)


def test_saved_derivation_is_distinct_and_board_config_hashed(
        tmp_path, monkeypatch):
    capture, assistance, _model = _inputs(tmp_path)
    derived, report = _derive(capture, assistance)
    import precision_insertion.session_key_baseline as baseline_module

    def load(**kwargs):
        assert kwargs["expected_session_manifest_sha256"] == "b" * 64
        return assistance.baseline

    monkeypatch.setattr(baseline_module, "load_session_key_baseline", load)
    observation = KeyPoseObservation(
        "key_001", 3, "test_key", "square", np.eye(4), "cam_a",
        100., (99.999, 100.003), dict(capture.frame_evidence),
        {}, {"accepted_views": list(CAMERAS)}, {}, capture.capture_dir)
    output = tmp_path / "key_evidence"
    with pytest.raises(ValueError, match="lacks derivation evidence"):
        write_key_capture_artifacts(derived, observation, output)
    write_key_capture_artifacts(
        derived, observation, output, derivation_report=report)
    assert verify_key_capture_artifacts(output)["robot_ready"] is False
    (output / "board_config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="key evidence changed"):
        verify_key_capture_artifacts(output)


def test_factory_uses_opened_session_and_exact_key_model(tmp_path, monkeypatch):
    _capture, assistance, model = _inputs(tmp_path)
    opened = OpenedPrecisionSession(
        SimpleNamespace(mode=SimpleNamespace(key_object="test_key")),
        assistance.baseline.session_evidence_dir,
        tmp_path / "source_binding.json", None)
    seen = []

    def load(**kwargs):
        seen.append(kwargs)
        return assistance.baseline

    monkeypatch.setattr(producer, "load_bound_session_key_baseline", load)
    monkeypatch.setattr(producer, "detector_from_board_config",
                        lambda **_kwargs: _board)
    selected = prepare_registered_key_assistance(
        opened=opened, socket_capture_id="socket_001",
        calibrated_camera_ids=set(CAMERAS),
        board_config_path=assistance.board_config_path,
        board_id="9", limits=_limits(),
        sam_segmentor=assistance.sam_segmentor, foundpose_model=model)
    assert selected.baseline is assistance.baseline
    assert selected.board_config_sha256 == assistance.board_config_sha256
    assert seen[0]["opened"] is opened
    model.obj_name = "different_key"
    with pytest.raises(ValueError, match="matching key FoundPose"):
        prepare_registered_key_assistance(
            opened=opened, socket_capture_id="socket_001",
            calibrated_camera_ids=set(CAMERAS),
            board_config_path=assistance.board_config_path,
            board_id="9", limits=_limits(),
            sam_segmentor=assistance.sam_segmentor, foundpose_model=model)


def test_failed_roi_is_saved_before_key_pose_admission(tmp_path, monkeypatch):
    capture, assistance, _model = _inputs(tmp_path)
    capture.capture_dir.mkdir()

    class BadSegmentor:
        def segment(self, rgb, _prompt):
            return np.ones(rgb.shape[:2], dtype=bool)

    object.__setattr__(assistance, "sam_segmentor", BadSegmentor())
    derived, report = _derive(capture, assistance)
    import precision_insertion.session_key_baseline as baseline_module
    monkeypatch.setattr(
        baseline_module, "load_session_key_baseline",
        lambda **_kwargs: assistance.baseline)
    diagnostic = write_registered_key_diagnostic(
        capture=derived, derivation_report=report)
    assert verify_registered_key_diagnostic(diagnostic)["robot_ready"] is False
    assert report["per_view"]["cam_a"]["status"] == (
        "registered_roi_view_rejected")
    assert not (tmp_path / "key_evidence").exists()
    with pytest.raises(FileExistsError):
        write_registered_key_diagnostic(
            capture=derived, derivation_report=report)
    (diagnostic / "masks/cam_a.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="diagnostic file changed"):
        verify_registered_key_diagnostic(diagnostic)
