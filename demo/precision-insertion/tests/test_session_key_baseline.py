"""A registered-key ROI can use only its own frozen socket session."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SessionCalibration, SocketObservation  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.perception_evidence import SocketCaptureEvidence  # noqa: E402
from precision_insertion.session_bootstrap import (  # noqa: E402
    SessionBootstrap, SocketCaptureInput, write_session_bootstrap_artifacts,
)
from precision_insertion.session_key_baseline import load_session_key_baseline  # noqa: E402
from precision_insertion.session_key_baseline import load_bound_session_key_baseline  # noqa: E402
from precision_insertion.live_session_runner import OpenedPrecisionSession  # noqa: E402


CAMERAS = {"cam_a", "cam_b"}


def _frame(image: np.ndarray, stamp: float, frame_id: int) -> dict:
    return {
        "frame_id": frame_id,
        "image_sha256": image_sha256(image),
        "timestamp_s": stamp,
        "max_error_s": .001,
        "timestamp_method": "hardware_exposure",
        "clock_domain": "unix_utc",
    }


def _bundle(tmp_path: Path, *, bound: bool = True) -> Path:
    board_image = np.zeros((32, 48, 3), dtype=np.uint8)
    board_images = {camera: board_image.copy() for camera in CAMERAS}
    board_evidence = {camera: _frame(image, 90. + index * .002,
                                     10 + index)
                      for index, (camera, image) in
                      enumerate(sorted(board_images.items()))}
    captures, admissions = [], []
    for number in (1, 2):
        capture_id = f"socket_{number:03}"
        images = {camera: np.full((32, 48, 3), 10 * number,
                                  dtype=np.uint8) for camera in CAMERAS}
        mask = np.zeros((32, 48), dtype=bool)
        mask[8:24, 12:36] = True
        times = {camera: 100. + number + index * .002
                 for index, camera in enumerate(sorted(CAMERAS))}
        evidence = {camera: _frame(images[camera], times[camera],
                                   number * 100 + index)
                    for index, camera in enumerate(sorted(CAMERAS))}
        masks = {camera: {"mask": mask.copy(),
                          "frame_id": evidence[camera]["frame_id"]}
                 for camera in CAMERAS}
        poses = {camera: {
            "ok": True, "pose_world": np.eye(4), "quality": .9,
            "inliers": 25, "mask_pixels": int(mask.sum()),
            "frame_id": evidence[camera]["frame_id"],
        } for camera in CAMERAS}
        captures.append(SocketCaptureInput(
            capture_id, number, "white fixed socket", images, masks, poses,
            times, "camera_acquisition", evidence if bound else None))
        observations = tuple(SocketObservation(
            capture_id, camera, times[camera], np.eye(4),
            "camera_acquisition") for camera in sorted(CAMERAS))
        admissions.append(SocketCaptureEvidence(
            capture_id, observations,
            {camera: {"accepted": True, "reasons": []}
             for camera in CAMERAS}, "camera_acquisition"))
    calibration = SessionCalibration(
        {}, np.eye(4), {}, {},
        {"socket_pose_source": "per_view_foundpose_legacy"})
    bootstrap = SessionBootstrap(
        calibration, 90, board_images,
        {camera: board_evidence[camera]["timestamp_s"]
         for camera in CAMERAS}, "camera_acquisition",
        tuple(captures), tuple(admissions), board_evidence)
    return write_session_bootstrap_artifacts(bootstrap, tmp_path / "session")


def _load(root: Path, *, capture_id: str = "socket_002",
          cameras: set[str] = CAMERAS, expected_hash: str | None = None):
    digest = hashlib.sha256((root / "evidence_manifest.json").read_bytes()
                            ).hexdigest()
    return load_session_key_baseline(
        session_evidence_dir=root, socket_capture_id=capture_id,
        expected_session_manifest_sha256=(expected_hash or digest),
        calibrated_camera_ids=cameras)


def test_loads_only_explicit_same_session_admitted_socket_views(tmp_path):
    root = _bundle(tmp_path)
    baseline = _load(root)
    assert set(baseline.images_bgr) == CAMERAS
    assert set(baseline.socket_masks) == CAMERAS
    assert baseline.socket_request_id == 2
    assert baseline.source["cam_a"]["frame_id"] == 200
    assert baseline.source["cam_a"]["image_sha256"] == image_sha256(
        baseline.images_bgr["cam_a"])
    assert baseline.to_record()["robot_ready"] is False


def test_rejects_other_capture_or_unbound_session(tmp_path):
    root = _bundle(tmp_path / "bound")
    with pytest.raises(ValueError, match="outside this session"):
        _load(root, capture_id="other_session")
    unbound = _bundle(tmp_path / "unbound", bound=False)
    with pytest.raises(ValueError, match="bound acquisition evidence"):
        _load(unbound)


def test_rejects_tampered_image_or_insufficient_admitted_views(tmp_path):
    root = _bundle(tmp_path / "tampered")
    (root / "socket/socket_002/images/cam_a.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="session evidence changed"):
        _load(root)
    root = _bundle(tmp_path / "views")
    with pytest.raises(ValueError, match="too few calibrated admitted"):
        _load(root, cameras={"cam_a", "cam_c"})


def test_invalid_request_mapping_cannot_be_baseline(tmp_path):
    root = _bundle(tmp_path)
    payload_path = root / "socket/socket_002/payloads.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["request_id"] = 1
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    # A changed hashed payload must be rejected even before identity checks.
    with pytest.raises(ValueError, match="session evidence changed"):
        _load(root)


def test_baseline_is_bound_to_current_opened_run(tmp_path):
    root = _bundle(tmp_path / "current")
    manifest_hash = hashlib.sha256((root / "evidence_manifest.json")
                                   .read_bytes()).hexdigest()
    binding_path = tmp_path / "binding.json"
    binding = {
        "schema": "precision_insertion_session_source_binding_v1",
        "session_evidence_dir": str(root),
        "session_manifest_sha256": manifest_hash,
    }
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    opened = OpenedPrecisionSession(None, root, binding_path, None)
    baseline = load_bound_session_key_baseline(
        opened=opened, socket_capture_id="socket_002",
        calibrated_camera_ids=CAMERAS)
    assert baseline.session_manifest_sha256 == manifest_hash
    with pytest.raises(ValueError, match="opened session binding"):
        _load(root, expected_hash="0" * 64)
    binding["session_evidence_dir"] = str(tmp_path / "different_session")
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ValueError, match="different session evidence binding"):
        load_bound_session_key_baseline(
            opened=opened, socket_capture_id="socket_002",
            calibrated_camera_ids=CAMERAS)
