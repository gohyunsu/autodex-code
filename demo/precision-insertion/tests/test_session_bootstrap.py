"""Session bootstrap joins existing quality and geometry gates without I/O to robots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.calibration import SessionCalibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.perception_evidence import SocketViewLimits  # noqa: E402
from precision_insertion import session_bootstrap as sb  # noqa: E402


def _capture(capture_id: str, timestamp: float) -> sb.SocketCaptureInput:
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:15, 8:20] = True
    images = {serial: np.zeros((24, 32, 3), dtype=np.uint8)
              for serial in ("cam_a", "cam_b")}
    masks = {serial: {"mask": mask.copy(), "ts": timestamp + 10}
             for serial in images}
    poses = {serial: {
        "ok": True, "pose_world": np.eye(4), "quality": 0.8,
        "inliers": 20, "mask_pixels": 108, "ts": timestamp + 12,
    } for serial in images}
    return sb.SocketCaptureInput(
        capture_id, int(timestamp), "red cylindrical socket", images, masks, poses,
        {"cam_a": timestamp, "cam_b": timestamp + 0.005},
        "camera_acquisition")


def _bootstrap(monkeypatch, captures=None, board_source="camera_acquisition",
               extra_camera_ids=()):
    received = {}

    def fake_calibrate_session(**kwargs):
        received.update(kwargs)
        return SessionCalibration(
            board={"test": True}, socket_pose_robot=np.eye(4),
            socket_diagnostics={"accepted": True},
            collision_scene={"mesh": {}, "cuboid": {}},
            record={"schema": "precision_insertion_session_calibration_v1",
                    "board": {"test": True}, "socket_observations": []},
        )

    monkeypatch.setattr(sb, "calibrate_session", fake_calibrate_session)
    camera_ids = {"cam_a", "cam_b", *extra_camera_ids}
    board = {serial: np.zeros((24, 32, 3), dtype=np.uint8)
             for serial in camera_ids}
    board_times = {"cam_a": 90.0, "cam_b": 90.005}
    board_times.update({serial: 90.006 for serial in extra_camera_ids})
    result = sb.bootstrap_session(
        mode=select_mode("cylinder", 20), object_root=Path("/unused"),
        board_request_id=90,
        board_images_bgr=board,
        board_timestamps_s=board_times,
        board_timestamp_source=board_source,
        socket_captures=(_capture("socket_1", 100.0),
                         _capture("socket_2", 101.0)) if captures is None
                        else captures,
        calibrated_camera_ids=camera_ids,
        view_limits=SocketViewLimits(50, 0.5, 10, 2, 0.02),
        intrinsics_full={serial: {} for serial in camera_ids},
        extrinsics_full={serial: {} for serial in camera_ids},
        c2r=np.eye(4), base_scene={"mesh": {}, "cuboid": {}},
        socket_collision_mesh=Path("/unused/socket.obj"),
        max_socket_translation_mm=1.0, max_socket_angle_deg=1.0)
    return result, received


def test_joins_ch_ar_uco_and_repeated_socket_admissions(monkeypatch):
    result, received = _bootstrap(monkeypatch)
    assert [row.capture_id for row in received["socket_observations"]] == [
        "socket_1", "socket_1", "socket_2", "socket_2"]
    assert received["max_capture_skew_s"] == 0.02
    assert received["min_socket_captures"] == 2
    assert result.socket_admissions[0].per_view["cam_b"][
        "frame_capture_timestamp_s"] == 100.005
    assert result.socket_admissions[0].per_view["cam_b"][
        "pose_publish_timestamp_s"] == 112.0


def test_rejects_unverified_time_and_mismatched_raw_frames(monkeypatch):
    with pytest.raises(ValueError, match="board images require camera acquisition"):
        _bootstrap(monkeypatch, board_source="snapshot_publish")
    captures = [_capture("socket_1", 100.0), _capture("socket_2", 101.0)]
    captures[0] = sb.SocketCaptureInput(
        captures[0].capture_id, captures[0].request_id,
        captures[0].prompt, captures[0].images_bgr,
        captures[0].masks, captures[0].poses,
        captures[0].frame_timestamps_s, "foundpose_publish")
    with pytest.raises(ValueError, match="camera acquisition timestamps"):
        _bootstrap(monkeypatch, captures)
    captures[0] = _capture("socket_1", 100.0)
    captures[0].images_bgr["cam_a"] = np.zeros((12, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="dimensions differ"):
        _bootstrap(monkeypatch, captures)


def test_rejects_duplicate_or_unsafe_capture_ids(monkeypatch):
    with pytest.raises(ValueError, match="duplicate socket capture ID"):
        _bootstrap(monkeypatch, [_capture("same", 100.0),
                                 _capture("same", 101.0)])
    with pytest.raises(ValueError, match="safe nonempty file identifier"):
        _bootstrap(monkeypatch, [_capture("../escape", 100.0),
                                 _capture("socket_2", 101.0)])
    with pytest.raises(ValueError, match="duplicate session capture request ID"):
        _bootstrap(monkeypatch, [_capture("socket_1", 100.0),
                                 _capture("socket_2", 100.5)])


def test_preserves_missing_camera_payload_as_rejected_view(monkeypatch):
    captures = [_capture("socket_1", 100.0), _capture("socket_2", 101.0)]
    for capture in captures:
        capture.images_bgr["cam_c"] = np.zeros((24, 32, 3), dtype=np.uint8)
    result, received = _bootstrap(
        monkeypatch, captures, extra_camera_ids=("cam_c",))
    assert len(received["socket_observations"]) == 4
    rejected = result.socket_admissions[0].per_view["cam_c"]
    assert rejected["accepted"] is False
    assert "missing_sam_mask" in rejected["reasons"]
    assert "missing_foundpose_pose" in rejected["reasons"]


def test_writes_non_overwriting_raw_evidence_bundle(monkeypatch, tmp_path):
    result, _ = _bootstrap(monkeypatch)
    target = sb.write_session_bootstrap_artifacts(result, tmp_path / "session")
    manifest = json.loads((target / "evidence_manifest.json").read_text())
    assert manifest["schema"] == "precision_insertion_session_evidence_bundle_v1"
    assert manifest["robot_ready"] is False
    assert manifest["board_request_id"] == 90
    assert manifest["socket_request_ids"] == [100, 101]
    assert len(manifest["files_sha256"]) == 2 + 2 * (2 + 2 + 1) + 1
    for relative, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((target / relative).read_bytes()).hexdigest() == digest
    payload = json.loads((target / "socket/socket_1/payloads.json").read_text())
    assert payload["sam_prompt"] == "red cylindrical socket"
    assert payload["pose_payloads"]["cam_a"]["pose_world"] == np.eye(4).tolist()
    assert sb.verify_session_evidence_bundle(target)["robot_ready"] is False
    with pytest.raises(FileExistsError):
        sb.write_session_bootstrap_artifacts(result, target)
    (target / "socket/socket_1/payloads.json").write_text("{}")
    with pytest.raises(ValueError, match="session evidence changed"):
        sb.verify_session_evidence_bundle(target)
