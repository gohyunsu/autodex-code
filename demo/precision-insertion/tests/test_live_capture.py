"""Stock AutoDex camera collectors remain injectable and fail closed on provenance."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.live_capture import (  # noqa: E402
    collect_board_snapshot, collect_socket_capture,
)


CAMERAS = {"cam_a", "cam_b"}
TIMES = {"cam_a": 100.0, "cam_b": 100.005}


def _provider(request_id, *, source="camera_acquisition"):
    return {"request_id": request_id, "source": source,
            "camera_times_s": TIMES}


class SnapshotStub:
    def snap(self, **kwargs):
        assert kwargs["decode"] is True
        assert kwargs["n_expected"] == 2
        return ({serial: {"image": np.zeros((24, 32, 3), dtype=np.uint8)}
                 for serial in CAMERAS},
                {"request_id": kwargs["request_id"]})


class FoundPoseStub:
    obj_name = "socket_model"
    intrinsics_undist = {"cam_a": {}, "cam_b": {}}
    extrinsics = {"cam_a": {}, "cam_b": {}}

    def __init__(self, *, write_images=True):
        self.write_images = write_images

    def collect_payloads(self, **kwargs):
        assert kwargs["prompt"] == "red socket"
        assert kwargs["n_expected_serials"] == 2
        capture_dir = Path(kwargs["save_capture_dir"])
        if self.write_images:
            image_dir = capture_dir / "images"
            image_dir.mkdir(parents=True)
            for serial in CAMERAS:
                assert cv2.imwrite(str(image_dir / f"{serial}.png"),
                                   np.zeros((24, 32, 3), dtype=np.uint8))
        mask = np.zeros((24, 32), dtype=bool)
        mask[6:15, 8:20] = True
        return (
            {serial: {"mask": mask.copy(), "ts": 200.0} for serial in CAMERAS},
            {serial: {"ok": True, "pose_world": np.eye(4),
                      "quality": 0.8, "inliers": 20, "mask_pixels": 108,
                      "ts": 210.0} for serial in CAMERAS},
            {"request_id": kwargs["request_id"]},
        )


def test_board_uses_original_distorted_jpegs_and_independent_times():
    result = collect_board_snapshot(
        snapshot_orchestrator=SnapshotStub(),
        calibrated_camera_ids=CAMERAS,
        acquisition_metadata_for_request=_provider,
        timeout_s=2.0, request_id_factory=lambda: 42)
    assert result.request_id == 42
    assert set(result.images_bgr) == CAMERAS
    assert result.frame_timestamps_s == TIMES
    assert result.frame_timestamp_source == "camera_acquisition"


def test_board_rejects_publication_time_even_with_matching_request():
    with pytest.raises(ValueError, match="acquisition-time metadata"):
        collect_board_snapshot(
            snapshot_orchestrator=SnapshotStub(),
            calibrated_camera_ids=CAMERAS,
            acquisition_metadata_for_request=lambda req: _provider(
                req, source="snapshot_publish"),
            timeout_s=2.0, request_id_factory=lambda: 42)


def test_socket_uses_same_request_saved_frames_and_foundpose(tmp_path):
    result = collect_socket_capture(
        init_orchestrator=FoundPoseStub(), socket_object="socket_model",
        capture_id="socket_001", socket_prompt="red socket",
        capture_root=tmp_path, calibrated_camera_ids=CAMERAS,
        acquisition_metadata_for_request=_provider, timeout_s=2.0,
        request_id_factory=lambda: 43)
    assert result.request_id == 43
    assert result.frame_timestamps_s == TIMES
    assert set(result.images_bgr) == set(result.masks) == set(result.poses)
    assert (tmp_path / "socket_001_request_43/images/cam_a.png").is_file()
    with pytest.raises(FileExistsError):
        collect_socket_capture(
            init_orchestrator=FoundPoseStub(), socket_object="socket_model",
            capture_id="socket_001", socket_prompt="red socket",
            capture_root=tmp_path, calibrated_camera_ids=CAMERAS,
            acquisition_metadata_for_request=_provider, timeout_s=2.0,
            request_id_factory=lambda: 43)


def test_socket_rejects_missing_same_request_image(tmp_path):
    with pytest.raises(TimeoutError, match="same-request socket frames"):
        collect_socket_capture(
            init_orchestrator=FoundPoseStub(write_images=False),
            socket_object="socket_model", capture_id="socket_002",
            socket_prompt="red socket", capture_root=tmp_path,
            calibrated_camera_ids=CAMERAS,
            acquisition_metadata_for_request=_provider, timeout_s=1.0,
            image_write_timeout_s=0.05, request_id_factory=lambda: 44)


def test_socket_rejects_wrong_initialized_foundpose_model(tmp_path):
    with pytest.raises(ValueError, match="not initialized"):
        collect_socket_capture(
            init_orchestrator=FoundPoseStub(), socket_object="other_socket",
            capture_id="socket_001", socket_prompt="red socket",
            capture_root=tmp_path, calibrated_camera_ids=CAMERAS,
            acquisition_metadata_for_request=_provider, timeout_s=1.0,
            request_id_factory=lambda: 43)
