"""Source binding and same-pixel PNG worker replay checks."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from precision_insertion.frame_provenance import image_sha256


SCRIPT = Path(__file__).resolve().parents[1] / "replay_saved_timing.py"
SPEC = importlib.util.spec_from_file_location("replay_saved_timing", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)


def _report(tmp_path: Path) -> Path:
    views = {}
    for index, camera in enumerate(("cam_a", "cam_b")):
        image = np.full((16, 24, 3), index * 60 + 20, dtype=np.uint8)
        path = tmp_path / f"{camera}.png"
        assert cv2.imwrite(str(path), image)
        views[camera] = {
            "raw_image": {"path": str(path),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            "frame_evidence": {"frame_id": index + 100,
                               "image_sha256": image_sha256(image),
                               "timestamp_s": float(index + 1000)},
            "acquisition_timestamp_s": float(index + 1000),
            "sam": {"frame_id": index + 100},
            "foundpose": {"frame_id": index + 100},
        }
    report = tmp_path / "report.json"
    report.write_text(json.dumps({
        "schema": "precision_insertion_camera_only_key_socket_qa_v1",
        "request_id": 123, "expected_camera_ids": ["cam_a", "cam_b"],
        "views": views,
    }))
    return report


def test_key_qa_replay_preserves_identity_and_rejects_changed_bytes(tmp_path):
    report = _report(tmp_path)
    images, source = replay._key_qa_frames(report)
    assert source["request_id"] == 123
    assert source["cameras"]["cam_a"]["frame_id"] == 100
    assert source["cameras"]["cam_b"]["timestamp_s"] == 1001.0
    one = replay._png_batch(images, tmp_path / "workers_1", 1)
    four = replay._png_batch(images, tmp_path / "workers_4", 4)
    assert one["file_sha256"] == four["file_sha256"]
    (tmp_path / "cam_a.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed original key PNG"):
        replay._key_qa_frames(report)


def test_key_qa_replay_rejects_frame_id_mismatch(tmp_path):
    report = _report(tmp_path)
    value = json.loads(report.read_text())
    value["views"]["cam_b"]["sam"]["frame_id"] += 1
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="frame identity"):
        replay._key_qa_frames(report)
