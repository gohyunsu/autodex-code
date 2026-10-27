"""Saved ZeroDex replay must use the matching image/mask request."""

import json

import cv2
import numpy as np
import pytest

from probe_zerodex_saved_foundpose import _load_inputs


def test_saved_capture_replay_binds_request_and_undistorted_views(tmp_path):
    capture = tmp_path / "key_probe_abc_request_123"
    images = capture / "images"
    masks = tmp_path / "masks"
    calib = tmp_path / "calibration"
    for directory in (images, masks, calib):
        directory.mkdir(parents=True)
    image = np.full((12, 16, 3), 150, dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[2:5, 4:9] = 255
    for serial in ("one", "two"):
        cv2.imwrite(str(images / f"{serial}.png"), image)
        cv2.imwrite(str(masks / f"{serial}_mask.png"), mask)
    camera = {serial: {"intrinsics_undistort": np.eye(3).tolist()}
              for serial in ("one", "two")}
    (calib / "intrinsics.json").write_text(json.dumps(camera))
    (calib / "extrinsics.json").write_text(json.dumps({
        serial: np.eye(4).tolist() for serial in camera}))
    report = tmp_path / "report.json"
    payload = {"socket_request_id": 123, "target_object": "key",
               "views": {serial: {"mask_pixels": 15} for serial in camera}}
    report.write_text(json.dumps(payload))
    views = _load_inputs(capture, masks, calib, report, "key")
    assert [row["serial"] for row in views] == ["one", "two"]
    assert all(row["mask"].sum() == 15 for row in views)
    payload["socket_request_id"] = 124
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different requests"):
        _load_inputs(capture, masks, calib, report, "key")
