"""Fresh key pose must use bound frames and agreeing AutoDex views."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.key_perception import (  # noqa: E402
    admit_key_capture, verify_key_capture_artifacts,
    write_key_capture_artifacts,
)
from precision_insertion.live_capture import KeyCaptureInput  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402


CAMERAS = ("cam_a", "cam_b")


class SelectorStub:
    intrinsics_undist = {camera: np.eye(3) for camera in CAMERAS}
    extrinsics = {camera: np.eye(4) for camera in CAMERAS}

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


def _admit(capture, mode, root, selector):
    return admit_key_capture(
        capture=capture, init_orchestrator=selector,
        mode=mode, shared_root=root, calibrated_camera_ids=set(CAMERAS),
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        maximum_multiview_center_error_mm=2.0,
        maximum_multiview_angle_error_deg=5.0)


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
