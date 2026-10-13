"""The stock daemon writes PNG frames after publishing its mask and pose."""

from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_stock_camera_qa import (  # noqa: E402
    _bound_frame_identity, _read_complete_png,
)
from precision_insertion.frame_provenance import image_sha256  # noqa: E402


def test_waits_for_complete_stock_png_before_reading(tmp_path):
    image = np.full((18, 24, 3), 73, dtype=np.uint8)
    encoded = cv2.imencode(".png", image)[1].tobytes()
    path = tmp_path / "camera.png"
    path.write_bytes(encoded[:-12])
    assert _read_complete_png(path, image_hw=(18, 24), timeout_s=0.1) is None

    def finish_write():
        time.sleep(0.15)
        with path.open("ab") as stream:
            stream.write(encoded[-12:])

    writer = threading.Thread(target=finish_write)
    writer.start()
    try:
        recovered = _read_complete_png(path, image_hw=(18, 24), timeout_s=1.0)
    finally:
        writer.join()
    assert recovered is not None and np.array_equal(recovered, image)


def test_precision_payload_is_bound_to_saved_undistorted_pixels():
    image = np.full((18, 24, 3), 73, dtype=np.uint8)
    common = {"frame_id": 12, "image_sha256": image_sha256(image),
              "image_space": "autodex_undistorted_full_frame"}
    bound = _bound_frame_identity(common, common, image)
    assert bound["sensor_frame_id"] == 12
    assert bound["undistorted_pixel_sha256"] == image_sha256(image)
    with np.testing.assert_raises_regex(ValueError, "do not share"):
        _bound_frame_identity(common, {**common, "frame_id": 13}, image)
    with np.testing.assert_raises_regex(ValueError, "do not share"):
        _bound_frame_identity(common, common, image + 1)
