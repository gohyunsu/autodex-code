"""Same-session raw VLM frames reuse AutoDex's open key transport."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from precision_insertion.checkpoint_camera import (  # noqa: E402
    capture_checkpoint_frames, make_checkpoint_capture,
)
from precision_insertion.raw_camera_capture import RawCameraCapture  # noqa: E402


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
        live, phase="after_lift", key_prompt="blue key handle",
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
