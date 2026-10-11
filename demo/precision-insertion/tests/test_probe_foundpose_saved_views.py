"""Saved FoundPose replay must bind camera bytes, calibration and masks."""

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_foundpose_saved_views as probe  # noqa: E402


def _fixture(tmp_path):
    shot = tmp_path / "shot"
    calib = tmp_path / "calib"
    shot.mkdir()
    calib.mkdir()
    k = [[80., 0., 4.], [0., 80., 3.], [0., 0., 1.]]
    intr = {"cam0": {"original_intrinsics": k,
                     "intrinsics_undistort": k,
                     "dist_params": [0., 0., 0., 0., 0.]}}
    extr = {"cam0": [[1., 0., 0., 0.], [0., 1., 0., 0.],
                     [0., 0., 1., 0.]]}
    intr_raw = json.dumps(intr).encode()
    extr_raw = json.dumps(extr).encode()
    (calib / "intrinsics.json").write_bytes(intr_raw)
    (calib / "extrinsics.json").write_bytes(extr_raw)
    bgr = np.full((6, 8, 3), 180, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", bgr)
    assert ok
    jpg = buffer.tobytes()
    (shot / "cam0.jpg").write_bytes(jpg)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[2:5, 3:6] = 255
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), mask)
    (shot / "shot.json").write_text(json.dumps({
        "status": "COMPLETE", "condition_id": "square/socket_only",
        "intrinsics_sha256": hashlib.sha256(intr_raw).hexdigest(),
        "extrinsics_sha256": hashlib.sha256(extr_raw).hexdigest(),
        "images": {"cam0": {"file": "cam0.jpg", "sha256":
                    hashlib.sha256(jpg).hexdigest(), "bytes": len(jpg),
                    "width": 8, "height": 6}},
    }))
    return shot, calib, mask_path


def test_saved_view_binds_exact_camera_and_calibration(tmp_path):
    shot, calib, mask = _fixture(tmp_path)
    result = probe._load_view(shot, calib, "cam0", mask)
    assert result["rgb"].shape == (6, 8, 3)
    assert result["mask"].sum() == 9
    assert result["condition_id"] == "square/socket_only"
    assert result["image_sha256"] == probe._hash_file(shot / "cam0.jpg")


def test_saved_view_rejects_modified_source_image(tmp_path):
    shot, calib, mask = _fixture(tmp_path)
    (shot / "cam0.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="recorded bytes/hash"):
        probe._load_view(shot, calib, "cam0", mask)


def test_saved_view_rejects_modified_calibration(tmp_path):
    shot, calib, mask = _fixture(tmp_path)
    (calib / "intrinsics.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="calibration differs"):
        probe._load_view(shot, calib, "cam0", mask)


def test_saved_view_rejects_nonbinary_or_wrong_size_mask(tmp_path):
    shot, calib, mask = _fixture(tmp_path)
    assert cv2.imwrite(str(mask), np.full((6, 8), 127, np.uint8))
    with pytest.raises(ValueError, match="binary 0/255"):
        probe._load_view(shot, calib, "cam0", mask)
    assert cv2.imwrite(str(mask), np.full((3, 4), 255, np.uint8))
    with pytest.raises(ValueError, match="binary 0/255"):
        probe._load_view(shot, calib, "cam0", mask)


def test_saved_view_condition_cannot_swap_object_or_family():
    probe._validate_condition("square/socket_only", "square",
                              "precision_socket_unified", "socket")
    probe._validate_condition("square/precision_key_1p5mm/pose_01", "square",
                              "precision_key_1p5mm", "key")
    probe._validate_condition(
        "cylinder/precision_socket_cylinder_gap_15mm/with_key/pose_01",
        "cylinder", "precision_key_cylinder_r15_h80", "key",
        "precision_socket_cylinder_gap_15mm")
    with pytest.raises(ValueError, match="requested key"):
        probe._validate_condition("square/precision_key_1p0mm/pose_01",
                                  "square", "precision_key_1p5mm", "key")
    with pytest.raises(ValueError, match="requested family"):
        probe._validate_condition("cylinder/socket_only", "square",
                                  "precision_socket_unified", "socket")
    with pytest.raises(ValueError, match="requested key scene"):
        probe._validate_condition(
            "cylinder/precision_socket_cylinder_gap_20mm/with_key/pose_01",
            "cylinder", "precision_key_cylinder_r15_h80", "key",
            "precision_socket_cylinder_gap_15mm")


def test_end_exchange_uses_geometry_center_and_unsigned_axis():
    first = np.eye(4)
    second = np.eye(4)
    second[:3, :3] = np.diag([1., -1., -1.])
    second[2, 3] = .08
    rows = [{"status": "pose_estimated_not_validated",
             "pose_world_m": pose.tolist()} for pose in (first, second)]
    arguments = {"center_local": np.array([0., 0., .04]),
                 "axis_local": np.array([0., 0., 1.])}
    equivalent = probe._pose_spread(rows, end_exchange=True, **arguments)
    assert equivalent["max_pairwise_frame_origin_mm"] == pytest.approx(80.)
    assert equivalent["max_pairwise_center_mm"] == pytest.approx(0.)
    assert equivalent["max_pairwise_axis_deg"] == pytest.approx(0.)
    assert equivalent["max_pairwise_rotation_deg"] == pytest.approx(180.)
    directed = probe._pose_spread(rows, end_exchange=False, **arguments)
    assert directed["max_pairwise_axis_deg"] == pytest.approx(180.)


def test_mask_directory_selects_only_named_camera_masks(tmp_path):
    for name in ("cam1_mask.png", "cam0_mask.png", "other.png"):
        (tmp_path / name).write_bytes(b"test")
    assert [(camera, path.name) for camera, path in
            probe._view_specs(None, tmp_path)] == [
                ("cam0", "cam0_mask.png"),
                ("cam1", "cam1_mask.png")]
