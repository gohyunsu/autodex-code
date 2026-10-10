"""Exposure-time adapter never treats ParaDex PC receipt as exposure."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.camera_time import (  # noqa: E402
    AcquisitionTimeProvider, CameraTimeCalibration,
)
from precision_insertion.camera_transport import FrameIdentityRegistry  # noqa: E402
from precision_insertion.camera_transport import (  # noqa: E402
    ProvenanceSnapshotAdapter, SnapshotMetadataBuffer,
)
from precision_insertion.frame_provenance import (  # noqa: E402
    image_sha256, verify_frame_provenance,
)
from precision_insertion.live_capture import collect_board_snapshot  # noqa: E402

import cv2


def _calibration_file(tmp_path, *, source_method=
                      "independent_per_camera_trigger_metrology"):
    source = tmp_path / "external_trigger_measurements.bin"
    source.write_bytes(b"independent exposure timestamps")
    origin = 1_800_000_000.
    record = {
        "schema": "precision_insertion_camera_time_calibration_v1",
        "calibration_id": "rig_a_20261010",
        "camera_serial": "cam_a",
        "source_method": source_method,
        "source_files": [{"path": str(source),
                          "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}],
        "fit_samples": [{"frame_id": fid,
                         "exposure_utc_s": origin + .01 * (fid - 100),
                         "max_error_s": .0001} for fid in (100, 102, 104)],
        "validation_samples": [{"frame_id": fid,
                                "exposure_utc_s": origin + .01 * (fid - 100),
                                "max_error_s": .0001}
                               for fid in (101, 103, 105)],
        "clock_offset_error_s": .0002,
        "drift_error_s_per_frame": .00001,
        "max_total_error_s": .001,
        "max_extrapolation_frames": 10,
        "valid_until_utc_s": origin + 1.0,
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path, source, record


def test_independent_fit_and_heldout_bounds_make_frame_id_provider(tmp_path):
    path, _source, _record = _calibration_file(tmp_path)
    calibration = CameraTimeCalibration.load(path)
    timestamp, uncertainty = calibration.exposure(106)
    assert timestamp == pytest.approx(1_800_000_000.06, abs=1e-6)
    assert .00029 <= uncertainty <= .00032
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    registry = FrameIdentityRegistry()
    registry.register(42, {"cam_a": {
        "frame_id": 106, "image_sha256": image_sha256(frame)}})
    provider = AcquisitionTimeProvider(registry, {"cam_a": calibration})
    evidence = provider(42)
    assert evidence["frames"]["cam_a"]["timestamp_method"] == (
        "calibrated_frame_id")
    assert verify_frame_provenance(
        evidence, request_id=42, images_bgr={"cam_a": frame},
        frame_ids={"cam_a": 106})["cam_a"]["calibration_id"] == (
            "rig_a_20261010")


def test_unmeasured_pc_receipt_or_changed_sources_are_rejected(tmp_path):
    path, source, record = _calibration_file(
        tmp_path, source_method="pc_time_after_GetNextImage")
    with pytest.raises(ValueError, match="independent exposure source"):
        CameraTimeCalibration.load(path)
    record["source_method"] = "independent_per_camera_trigger_metrology"
    path.write_text(json.dumps(record), encoding="utf-8")
    source.write_bytes(b"different")
    with pytest.raises(ValueError, match="source file changed"):
        CameraTimeCalibration.load(path)


def test_fit_validation_overlap_and_unbounded_extrapolation_fail(tmp_path):
    path, _source, record = _calibration_file(tmp_path)
    record["validation_samples"][0]["frame_id"] = 100
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        CameraTimeCalibration.load(path)
    record["validation_samples"][0]["frame_id"] = 101
    path.write_text(json.dumps(record), encoding="utf-8")
    calibration = CameraTimeCalibration.load(path)
    with pytest.raises(ValueError, match="outside"):
        calibration.exposure(116)


def test_commissioned_time_error_limit_rejects_poor_fit(tmp_path):
    path, _source, record = _calibration_file(tmp_path)
    record["validation_samples"][1]["exposure_utc_s"] += .003
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="commissioned limit"):
        CameraTimeCalibration.load(path)


def test_registry_refuses_reuse_of_request_with_changed_camera_frame(tmp_path):
    registry = FrameIdentityRegistry()
    registry.register(42, {"cam_a": {"frame_id": 106,
                                      "image_sha256": "a" * 64}})
    with pytest.raises(ValueError, match="changed within request"):
        registry.register(42, {"cam_a": {"frame_id": 107,
                                          "image_sha256": "a" * 64}})
    assert registry.get(42)["cam_a"]["frame_id"] == 106


def test_expired_calibration_refuses_current_use(tmp_path, monkeypatch):
    path, _source, _record = _calibration_file(tmp_path)
    calibration = CameraTimeCalibration.load(path)
    registry = FrameIdentityRegistry()
    registry.register(42, {"cam_a": {"frame_id": 106,
                                      "image_sha256": "a" * 64}})
    provider = AcquisitionTimeProvider(registry, {"cam_a": calibration})
    monkeypatch.setattr("precision_insertion.camera_time.time.time",
                        lambda: calibration.valid_until_utc_s + 1.)
    with pytest.raises(ValueError, match="expired in wall time"):
        provider(42)


def test_board_transport_to_calibrated_capture_is_same_frame(tmp_path):
    path, _source, _record = _calibration_file(tmp_path)
    registry = FrameIdentityRegistry()
    provider = AcquisitionTimeProvider(
        registry, {"cam_a": CameraTimeCalibration.load(path)})
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    image[4:9, 7:13] = (44, 55, 66)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    jpeg = encoded.tobytes()
    decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
    metadata = SnapshotMetadataBuffer()

    class _Snapshot:
        def snap(self, **kwargs):
            metadata.put({"req_id": kwargs["request_id"],
                          "serial": "cam_a", "fid": 106}, jpeg)
            return {"cam_a": {"jpeg": jpeg, "image": decoded}}, {
                "request_id": kwargs["request_id"]}

    adapter = ProvenanceSnapshotAdapter(_Snapshot(), metadata, registry)
    capture = collect_board_snapshot(
        snapshot_orchestrator=adapter,
        calibrated_camera_ids={"cam_a"},
        acquisition_metadata_for_request=provider,
        timeout_s=.2, request_id_factory=lambda: 42)
    assert capture.frame_evidence["cam_a"]["frame_id"] == 106
    assert capture.frame_evidence["cam_a"]["image_sha256"] == (
        image_sha256(decoded))
    assert capture.frame_timestamps_s["cam_a"] == pytest.approx(
        1_800_000_000.06, abs=1e-6)
