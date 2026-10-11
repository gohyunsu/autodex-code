"""Reuse the already-open AutoDex key camera request for VLM checkpoints.

This acquires raw undistorted frames, not a post-grasp key pose. The existing
FoundPose transport is used only to trigger one provenance-bound request and
save its same-request images; a failed pose estimate is not turned into an
insertion outcome. Robot state must still be sampled at frame exposure by
the robot-PC feedback buffer.
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import uuid4

from .live_capture import collect_key_capture
from .raw_camera_capture import PHASES, RawCameraCapture


def capture_checkpoint_frames(
    live: object, *, phase: str, key_prompt: str,
    timeout_s: float, image_write_timeout_s: float = 5.0,
    capture_id: str | None = None,
    collector: Callable = collect_key_capture,
) -> RawCameraCapture:
    """Acquire one new raw multi-view checkpoint from the live camera rig."""
    if phase not in PHASES:
        raise ValueError("unknown AutoDex checkpoint phase")
    if not isinstance(key_prompt, str) or not key_prompt.strip():
        raise ValueError("checkpoint capture needs the selected key prompt")
    cameras = live.cameras
    launch = cameras.launch
    capture = collector(
        init_orchestrator=cameras.init_orchestrator,
        key_object=launch.mode.key_object,
        capture_id=capture_id or f"{phase}_{uuid4().hex}",
        key_prompt=key_prompt, capture_root=launch.capture_root,
        calibrated_camera_ids=set(launch.camera_ids),
        acquisition_metadata_for_request=(
            cameras.acquisition_metadata_for_request),
        timeout_s=timeout_s,
        image_write_timeout_s=image_write_timeout_s)
    metadata = cameras.acquisition_metadata_for_request(capture.request_id)
    if (set(capture.images_bgr) != set(launch.camera_ids) or
            set(capture.frame_evidence) != set(launch.camera_ids) or
            metadata.get("frames") != capture.frame_evidence):
        raise ValueError("checkpoint frames do not match the open camera session")
    return RawCameraCapture(
        capture_id=capture.capture_id,
        request_id=capture.request_id,
        images_bgr=capture.images_bgr,
        frame_ids={camera: row["frame_id"]
                   for camera, row in capture.frame_evidence.items()},
        acquisition_metadata=metadata)


def make_checkpoint_capture(
    *, phase: str, key_prompt: str, timeout_s: float,
    image_write_timeout_s: float = 5.0,
) -> Callable:
    """Return the `(live, runner, decision)` callback used by live handlers."""
    if phase not in PHASES:
        raise ValueError("unknown AutoDex checkpoint phase")

    def capture(live, _runner, _decision) -> RawCameraCapture:
        return capture_checkpoint_frames(
            live, phase=phase, key_prompt=key_prompt,
            timeout_s=timeout_s,
            image_write_timeout_s=image_write_timeout_s)

    return capture
