"""Same-session raw VLM frames reuse AutoDex's open key transport."""

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.checkpoint_camera import (  # noqa: E402
    capture_checkpoint_frames, capture_snapshot_checkpoint_frames,
    make_checkpoint_capture,
)
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)
from precision_insertion.tip_frame_window import (  # noqa: E402
    TipFrame, TipFrameWindow, load_tip_frame_window,
)


def test_checkpoint_capture_reuses_session_and_exact_frame_metadata():
    images = {camera: np.zeros((4, 5, 3), dtype=np.uint8)
              for camera in ("cam_a", "cam_b")}
    evidence = {camera: {"frame_id": index}
                for index, camera in enumerate(images, start=11)}
    metadata = {"frames": evidence}
    requests = []

    def collector(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            request_id=17, capture_id=kwargs["capture_id"],
            images_bgr=images, frame_evidence=evidence)

    live = SimpleNamespace(cameras=SimpleNamespace(
        init_orchestrator=object(),
        acquisition_metadata_for_request=lambda request: metadata,
        launch=SimpleNamespace(
            mode=SimpleNamespace(key_object="precision_key"),
            capture_root=Path("/tmp"), camera_ids=("cam_a", "cam_b"))))
    result = capture_checkpoint_frames(
        live, phase="after_lift", key_prompt="white key handle",
        timeout_s=4.0, capture_id="lift_1", collector=collector)
    assert isinstance(result, RawCameraCapture)
    assert result.capture_id == "lift_1"
    assert result.frame_ids == {"cam_a": 11, "cam_b": 12}
    assert requests[0]["init_orchestrator"] is live.cameras.init_orchestrator
    assert requests[0]["key_object"] == "precision_key"


def test_checkpoint_capture_rejects_missing_camera_or_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        make_checkpoint_capture(
            phase="unknown", key_prompt="key", timeout_s=2.0)
    live = SimpleNamespace(cameras=SimpleNamespace(
        init_orchestrator=object(),
        acquisition_metadata_for_request=lambda request: {
            "frames": {"cam_a": {"frame_id": 2}}},
        launch=SimpleNamespace(
            mode=SimpleNamespace(key_object="key"),
            capture_root=Path("/tmp"), camera_ids=("cam_a", "cam_b"))))
    with pytest.raises(ValueError, match="open camera session"):
        capture_checkpoint_frames(
            live, phase="preinsert", key_prompt="key", timeout_s=2.0,
            collector=lambda **kwargs: SimpleNamespace(
                request_id=1, capture_id="test",
                images_bgr={"cam_a": np.zeros((4, 5, 3), dtype=np.uint8)},
                frame_evidence={"cam_a": {"frame_id": 2}}))


def test_raw_snapshot_checkpoint_does_not_depend_on_key_pose(tmp_path):
    original = {camera: np.full((8, 9, 3), index * 30, dtype=np.uint8)
                for index, camera in enumerate(("cam_a", "cam_b"), 1)}
    evidence = {camera: {
        "frame_id": index, "image_sha256": image_sha256(image),
        "timestamp_s": 100.0, "max_error_s": 0.002,
        "timestamp_method": "hardware_exposure", "clock_domain": "unix_utc"}
        for index, (camera, image) in enumerate(original.items(), 1)}
    source = SimpleNamespace(
        request_id=31, images_bgr=original, frame_evidence=evidence)
    snap = object()
    intrinsics = {
        camera: {"K_orig": np.eye(3), "K_undist": np.eye(3),
                 "dist_params": np.zeros(5)} for camera in original}
    live = SimpleNamespace(cameras=SimpleNamespace(
        snapshot_orchestrator=snap,
        acquisition_metadata_for_request=lambda request: None,
        launch=SimpleNamespace(camera_ids=tuple(original),
                               intrinsics_full=intrinsics)))
    requests = []

    def collect(**kwargs):
        requests.append(kwargs)
        return source

    raw = capture_snapshot_checkpoint_frames(
        live, phase="preinsert", timeout_s=2.0,
        capture_id="raw_31", snapshot_collector=collect)
    assert requests[0]["snapshot_orchestrator"] is snap
    assert raw.source_images_bgr is original
    _ok, jpeg = cv2.imencode(".jpg", original["cam_a"])
    window = TipFrameWindow(
        "complete", "cam_a", raw.frame_ids["cam_a"],
        (TipFrame(raw.frame_ids["cam_a"], 100_000_000_000,
                  jpeg.tobytes()),))
    path = write_raw_camera_capture(
        replace(raw, temporal_window=window), tmp_path / "raw",
        phase="preinsert")
    manifest = verify_raw_camera_capture(path, phase="preinsert")
    assert set(manifest["undistort_derivation"]["source_frame_evidence"]) == {
        "cam_a", "cam_b"}
    assert manifest["temporal_window"]["target_frame_id"] == 1
    assert load_tip_frame_window(path) == window
    initial = capture_snapshot_checkpoint_frames(
        live, phase="initial_tabletop", timeout_s=2.0,
        capture_id="initial_31", snapshot_collector=collect,
        retain_original=False)
    assert requests[-1]["verify_pixels"] is False
    assert initial.source_images_bgr is None
    initial_path = write_raw_camera_capture(
        initial, tmp_path / "initial", phase="initial_tabletop")
    initial_manifest = verify_raw_camera_capture(
        initial_path, phase="initial_tabletop")
    assert "undistort_derivation" not in initial_manifest
    assert not (initial_path / "source_original").exists()
    assert make_checkpoint_capture(
        phase="preinsert", timeout_s=2.0).__name__ == \
        "capture"
    with pytest.raises(ValueError, match="key-specific prompt"):
        make_checkpoint_capture(
            phase="preinsert", timeout_s=2.0, transport="init_key")


def test_snapshot_undistortion_matches_opencv_and_offline_derivation(tmp_path):
    cameras = ("cam_a", "cam_b")
    original = {camera: np.random.default_rng(index).integers(
        0, 256, size=(48, 64, 3), dtype=np.uint8)
        for index, camera in enumerate(cameras)}
    evidence = {camera: {
        "frame_id": index + 1, "image_sha256": image_sha256(image),
        "timestamp_s": 100.0, "max_error_s": 0.002,
        "timestamp_method": "hardware_exposure", "clock_domain": "unix_utc"}
        for index, (camera, image) in enumerate(original.items())}
    K = np.array([[55., 0., 32.], [0., 56., 24.], [0., 0., 1.]])
    intrinsics = {camera: {"K_orig": K, "K_undist": K,
                           "dist_params": np.array([.11, -.03, .002, -.001, 0.])}
                  for camera in cameras}
    live = SimpleNamespace(cameras=SimpleNamespace(
        snapshot_orchestrator=object(),
        acquisition_metadata_for_request=lambda _request: None,
        launch=SimpleNamespace(camera_ids=cameras,
                               intrinsics_full=intrinsics)))
    source = SimpleNamespace(request_id=31, images_bgr=original,
                             frame_evidence=evidence)
    raw = capture_snapshot_checkpoint_frames(
        live, phase="preinsert", timeout_s=2., capture_id="raw_31",
        snapshot_collector=lambda **_kwargs: source)
    for camera in cameras:
        expected = cv2.undistort(
            original[camera], K, intrinsics[camera]["dist_params"], None, K)
        np.testing.assert_array_equal(raw.images_bgr[camera], expected)

    # The live writer uses the producer's pixels. Full derivation replay
    # remains available offline and rejects a producer with altered pixels.
    changed = dict(raw.images_bgr)
    changed["cam_a"] = changed["cam_a"].copy()
    changed["cam_a"][0, 0, 0] ^= 1
    metadata = dict(raw.acquisition_metadata)
    metadata["frames"] = {camera: dict(row) for camera, row in
                          metadata["frames"].items()}
    metadata["frames"]["cam_a"]["image_sha256"] = image_sha256(
        changed["cam_a"])
    bundle = write_raw_camera_capture(
        replace(raw, images_bgr=changed, acquisition_metadata=metadata),
        tmp_path / "altered", phase="preinsert")
    with pytest.raises(ValueError, match="original undistortion"):
        verify_raw_camera_capture(bundle, phase="preinsert")
