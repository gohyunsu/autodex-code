"""A capture-PC frame window preserves source IDs and refuses altered JPEGs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_tip_frame_server import FrameWindowBuffer  # noqa: E402
from precision_insertion.tip_frame_window import (  # noqa: E402
    TipFrame, TipFrameWindow, fetch_tip_frame_window, load_tip_frame_window,
    save_tip_frame_window,
)
from precision_insertion import tip_frame_window as frame_window_module  # noqa: E402


def test_capture_pc_window_binds_exact_frame_and_timestamps():
    buffer = FrameWindowBuffer(capacity=9)
    image = np.full((30, 40, 3), 180, dtype=np.uint8)
    for frame_id in range(100, 109):
        buffer.append(frame_id, frame_id * 1000, image)
    status, frames = buffer.window(105, before=2, after=2)
    assert status == "complete"
    assert [frame.source_frame_id for frame in frames] == [103, 104, 105, 106, 107]
    assert [frame.read_unix_ns for frame in frames] == [103000, 104000,
                                                       105000, 106000, 107000]
    assert cv2.imdecode(np.frombuffer(frames[2].jpeg, np.uint8),
                        cv2.IMREAD_COLOR).shape == image.shape
    status, _frames = buffer.window(99, before=2, after=2)
    assert status == "target_frame_missing"


def test_window_round_trip_and_changed_jpeg(tmp_path):
    image = np.full((20, 20, 3), 220, dtype=np.uint8)
    _ok, encoded = cv2.imencode(".jpg", image)
    window = TipFrameWindow("complete", "25322651", 42,
                            (TipFrame(41, 1000, encoded.tobytes()),
                             TipFrame(42, 1033, encoded.tobytes()),
                             TipFrame(43, 1066, encoded.tobytes())))
    save_tip_frame_window(window, tmp_path)
    assert load_tip_frame_window(tmp_path) == window
    manifest = json.loads((tmp_path / "temporal_window/manifest.json").read_text())
    assert [row["source_frame_id"] for row in manifest["frames"]] == [41, 42, 43]
    (tmp_path / "temporal_window/42.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_tip_frame_window(tmp_path)


@pytest.mark.parametrize("mismatch", [False, True])
def test_fetch_window_uses_exact_source_identity_without_camera(
        monkeypatch, mismatch):
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    _ok, encoded = cv2.imencode(".jpg", image)
    frame_id = 99 if mismatch else 42
    header = {"status": "complete", "camera_id": "25322651",
              "target_frame_id": 42,
              "frames": [{"source_frame_id": frame_id,
                          "read_unix_ns": 123456789}]}

    class FakeSocket:
        def setsockopt(self, *_args):
            pass

        def connect(self, address):
            assert address == "tcp://capture2:16951"

        def send_json(self, request):
            assert request == {"op": "window", "camera_id": "25322651",
                               "target_frame_id": 42, "before": 1, "after": 1}

        def recv_multipart(self):
            return [json.dumps(header).encode(), encoded.tobytes()]

        def close(self):
            pass

    class FakeContext:
        def socket(self, _kind):
            return FakeSocket()

        def term(self):
            pass

    monkeypatch.setattr(frame_window_module.zmq, "Context", FakeContext)
    window = fetch_tip_frame_window(
        host="capture2", port=16951, camera_id="25322651",
        target_frame_id=42, before=1, after=1)
    if mismatch:
        assert window.status == "invalid_response"
        assert window.frames == ()
    else:
        assert window.status == "complete"
        assert window.frames[0].source_frame_id == 42
        assert window.frames[0].read_unix_ns == 123456789
