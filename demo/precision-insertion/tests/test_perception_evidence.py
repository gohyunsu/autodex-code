"""Socket calibration must not mistake FoundPose publish time for capture time."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.perception_evidence import (  # noqa: E402
    SocketViewLimits, admit_socket_capture, collect_and_admit_socket_capture,
)


def _inputs():
    mask = np.zeros((24, 32), dtype=bool)
    mask[6:15, 8:20] = True
    masks = {
        serial: {"mask": mask.copy(), "ts": published}
        for serial, published in (("camera_a", 110.0), ("camera_b", 117.0))
    }
    poses = {
        serial: {
            "ok": True, "pose_world": np.eye(4), "quality": 0.8,
            "inliers": 20, "mask_pixels": 108, "ts": published,
        }
        for serial, published in (("camera_a", 111.0), ("camera_b", 118.0))
    }
    times = {"camera_a": 100.0, "camera_b": 100.012}
    limits = SocketViewLimits(50, 0.5, 10, 2, 0.02)
    return masks, poses, times, limits


def _admit(masks, poses, times, limits, source="camera_acquisition"):
    return admit_socket_capture(
        capture_id="socket_001", masks=masks, poses=poses,
        frame_timestamps_s=times, frame_timestamp_source=source,
        calibrated_camera_ids={"camera_a", "camera_b"}, limits=limits)


def test_accepts_real_frame_times_without_using_delayed_publication_times():
    masks, poses, times, limits = _inputs()
    result = _admit(masks, poses, times, limits)
    assert [row.camera_id for row in result.observations] == [
        "camera_a", "camera_b"]
    assert [row.timestamp_s for row in result.observations] == [100.0, 100.012]
    assert result.per_view["camera_b"]["pose_publish_timestamp_s"] == 118.0
    assert result.to_record()["robot_ready"] is False


def test_rejects_publish_time_or_missing_acquisition_metadata():
    masks, poses, times, limits = _inputs()
    with pytest.raises(ValueError, match="camera acquisition timestamps"):
        _admit(masks, poses, times, limits, source="foundpose_publish")
    with pytest.raises(ValueError, match="accepted views"):
        _admit(masks, poses, {"camera_a": 100.0}, limits)
    times["camera_b"] = 100.040
    with pytest.raises(ValueError, match="acquisition skew"):
        _admit(masks, poses, times, limits)


def test_masks_quality_and_inliers_are_independent_view_gates():
    masks, poses, times, limits = _inputs()
    masks["camera_b"]["mask"][:, :] = False
    with pytest.raises(ValueError, match="mask_too_small"):
        _admit(masks, poses, times, limits)
    masks, poses, times, limits = _inputs()
    poses["camera_b"]["quality"] = 0.2
    with pytest.raises(ValueError, match="low_foundpose_quality"):
        _admit(masks, poses, times, limits)
    masks, poses, times, limits = _inputs()
    poses["camera_b"]["inliers"] = 3
    with pytest.raises(ValueError, match="too_few_foundpose_inliers"):
        _admit(masks, poses, times, limits)
    masks, poses, times, limits = _inputs()
    masks["camera_b"]["mask"][6, 0] = True
    poses["camera_b"]["mask_pixels"] += 1
    with pytest.raises(ValueError, match="mask_touches_image_border"):
        _admit(masks, poses, times, limits)


def test_rejects_uncalibrated_camera_and_invalid_pose():
    masks, poses, times, limits = _inputs()
    times["unknown"] = 100.0
    with pytest.raises(ValueError, match="uncalibrated camera"):
        _admit(masks, poses, times, limits)
    masks, poses, times, limits = _inputs()
    poses["camera_b"]["pose_world"][0, 0] = 2.0
    with pytest.raises(ValueError, match="invalid_foundpose_pose"):
        _admit(masks, poses, times, limits)


def test_reuses_original_collector_but_requires_matching_frame_metadata():
    masks, poses, times, limits = _inputs()

    class OriginalCollectorStub:
        obj_name = "precision_socket_unified"
        intrinsics_undist = {"camera_a": np.eye(3), "camera_b": np.eye(3)}
        extrinsics = {"camera_a": np.eye(4), "camera_b": np.eye(4)}

        def collect_payloads(self, **kwargs):
            assert kwargs["prompt"] == "red square socket"
            assert kwargs["n_expected_serials"] == 2
            return masks, poses, {"request_id": 42, "n_poses_recv": 2}

    kwargs = {
        "orchestrator": OriginalCollectorStub(),
        "socket_object": "precision_socket_unified",
        "capture_id": "socket_001", "prompt": "red square socket",
        "calibrated_camera_ids": {"camera_a", "camera_b"},
        "limits": limits, "timeout_s": 10.0,
    }
    admitted, timing = collect_and_admit_socket_capture(
        **kwargs, acquisition_metadata_for_request=lambda request_id: {
            "request_id": request_id, "source": "camera_acquisition",
            "camera_times_s": times,
        })
    assert len(admitted.observations) == 2
    assert timing["request_id"] == 42
    with pytest.raises(ValueError, match="does not match FoundPose request"):
        collect_and_admit_socket_capture(
            **kwargs, acquisition_metadata_for_request=lambda _request_id: {
                "request_id": 41, "source": "camera_acquisition",
                "camera_times_s": times,
            })
    with pytest.raises(ValueError, match="camera acquisition timestamps"):
        collect_and_admit_socket_capture(
            **kwargs, acquisition_metadata_for_request=lambda request_id: {
                "request_id": request_id, "source": "foundpose_publish",
                "camera_times_s": times,
            })
    with pytest.raises(ValueError, match="not initialized for this socket"):
        collect_and_admit_socket_capture(
            **{**kwargs, "socket_object": "wrong_socket"},
            acquisition_metadata_for_request=lambda _request_id: {},
        )
