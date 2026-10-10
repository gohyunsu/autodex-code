"""Demo-local AutoDex camera adapters for precision session evidence.

The stock snapshot and FoundPose orchestrators perform the actual capture.
They do not expose image-acquisition timestamps. A separate, verified
per-request metadata provider is therefore mandatory; daemon publication
``ts`` and command-dispatch time are never substituted for frame time.
No robot commands are sent here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
import secrets
import time
from typing import Callable, Mapping

import cv2
import numpy as np

from .session_bootstrap import SocketCaptureInput


AcquisitionProvider = Callable[[int], Mapping]
RequestIdFactory = Callable[[], int]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class BoardSnapshotInput:
    request_id: int
    images_bgr: dict[str, np.ndarray]
    frame_timestamps_s: dict[str, float]
    frame_timestamp_source: str


def _request_id(factory: RequestIdFactory) -> int:
    value = factory()
    if type(value) is not int or not 0 < value < 2**31:
        raise ValueError("capture request ID must be a positive int31")
    return value


def _acquisition_metadata(
    provider: AcquisitionProvider, request_id: int,
    calibrated_camera_ids: set[str],
) -> dict[str, float]:
    evidence = provider(request_id)
    if not isinstance(evidence, Mapping) or (
            evidence.get("request_id") != request_id):
        raise ValueError("camera acquisition metadata has wrong request ID")
    if evidence.get("source") != "camera_acquisition":
        raise ValueError("camera acquisition-time metadata is required")
    camera_times = evidence.get("camera_times_s")
    if not isinstance(camera_times, Mapping) or not camera_times:
        raise ValueError("missing per-camera acquisition times")
    if set(camera_times) - calibrated_camera_ids:
        raise ValueError("acquisition times include uncalibrated camera")
    times = {}
    for serial, value in camera_times.items():
        if isinstance(value, bool):
            raise ValueError("invalid camera acquisition time")
        timestamp = float(value)
        if not math.isfinite(timestamp) or timestamp <= 0:
            raise ValueError("invalid camera acquisition time")
        times[serial] = timestamp
    return times


def collect_board_snapshot(
    *, snapshot_orchestrator, calibrated_camera_ids: set[str],
    acquisition_metadata_for_request: AcquisitionProvider,
    timeout_s: float, request_id_factory: RequestIdFactory | None = None,
) -> BoardSnapshotInput:
    """Get the original/distorted ChArUco frames from stock AutoDex.

    The same request ID must identify these JPEGs in the independent
    acquisition metadata side channel. This function does not infer timing
    from the arrival of JPEGs or the daemon's publication timestamps.
    """
    if not calibrated_camera_ids or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("commissioned cameras and positive timeout required")
    if any(not isinstance(serial, str) or
           not _SAFE_ID.fullmatch(serial) for serial in calibrated_camera_ids):
        raise ValueError("unsafe calibrated camera ID")
    request_id = _request_id(request_id_factory or (
        lambda: secrets.randbelow(2**31 - 1) + 1))
    payload, timing = snapshot_orchestrator.snap(
        n_expected=len(calibrated_camera_ids), timeout_s=timeout_s,
        request_id=request_id, decode=True)
    if (not isinstance(timing, Mapping) or
            timing.get("request_id") != request_id):
        raise ValueError("board snapshot request ID mismatch")
    if not isinstance(payload, Mapping) or set(payload) != calibrated_camera_ids:
        raise ValueError("board snapshot has missing or uncalibrated cameras")
    images = {}
    for serial, entry in payload.items():
        image = entry.get("image") if isinstance(entry, Mapping) else None
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3):
            raise ValueError(f"board snapshot camera {serial} has no BGR frame")
        images[serial] = image
    times = _acquisition_metadata(
        acquisition_metadata_for_request, request_id, calibrated_camera_ids)
    if set(times) != calibrated_camera_ids:
        raise ValueError("board snapshot lacks an acquisition time for every frame")
    return BoardSnapshotInput(request_id, images, times, "camera_acquisition")


def _load_saved_frames(
    capture_dir: Path, camera_ids: set[str], timeout_s: float,
) -> dict[str, np.ndarray]:
    """Wait for the stock init daemon's asynchronous same-request PNG writes."""
    if not camera_ids:
        raise ValueError("FoundPose returned no camera payloads")
    deadline = time.monotonic() + timeout_s
    images = {}
    while time.monotonic() < deadline:
        for serial in camera_ids - images.keys():
            image_file = capture_dir / "images" / f"{serial}.png"
            if image_file.is_file():
                image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
                if image is not None and image.size:
                    images[serial] = image
        if set(images) == camera_ids:
            return images
        time.sleep(0.02)
    raise TimeoutError(
        "the original init daemon did not save all same-request socket frames: "
        + repr(sorted(camera_ids - images.keys())))


def collect_socket_capture(
    *, init_orchestrator, socket_object: str, capture_id: str,
    socket_prompt: str, capture_root: Path,
    calibrated_camera_ids: set[str],
    acquisition_metadata_for_request: AcquisitionProvider,
    timeout_s: float, image_write_timeout_s: float = 5.0,
    request_id_factory: RequestIdFactory | None = None,
) -> SocketCaptureInput:
    """Collect same-request undistorted frames, SAM masks and FoundPose poses.

    The stock init daemon writes the frames asynchronously to an exclusive
    request directory on shared storage. This function waits for those exact
    files, rejecting missing images instead of combining a later snapshot
    with the earlier pose. The provider's source tag is checked, but its
    hardware provenance cannot be authenticated by this adapter alone.
    """
    if getattr(init_orchestrator, "obj_name", None) != socket_object:
        raise ValueError("FoundPose is not initialized for the selected socket")
    if (not isinstance(socket_prompt, str) or not socket_prompt.strip() or
            socket_prompt == "object"):
        raise ValueError("a socket-specific SAM prompt is required")
    if (not isinstance(capture_id, str) or
            not _SAFE_ID.fullmatch(capture_id)):
        raise ValueError("capture ID must be a simple path-safe identifier")
    if (not math.isfinite(timeout_s) or timeout_s <= 0 or
            not math.isfinite(image_write_timeout_s) or
            image_write_timeout_s <= 0):
        raise ValueError("positive capture and image-write timeouts required")
    if not calibrated_camera_ids:
        raise ValueError("commissioned cameras are required")
    if (set(getattr(init_orchestrator, "intrinsics_undist", {})) !=
            calibrated_camera_ids or
            set(getattr(init_orchestrator, "extrinsics", {})) !=
            calibrated_camera_ids):
        raise ValueError("FoundPose camera IDs differ from calibrated cameras")
    if any(not isinstance(serial, str) or
           not _SAFE_ID.fullmatch(serial) for serial in calibrated_camera_ids):
        raise ValueError("unsafe calibrated camera ID")
    root = Path(capture_root).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("capture root must be an existing absolute shared directory")
    request_id = _request_id(request_id_factory or (
        lambda: secrets.randbelow(2**31 - 1) + 1))
    capture_dir = root.resolve() / (
        f"{capture_id}_request_{request_id}")
    if capture_dir.exists():
        raise FileExistsError(f"socket capture path already exists: {capture_dir}")
    masks, poses, timing = init_orchestrator.collect_payloads(
        prompt=socket_prompt, request_id=request_id,
        n_expected_serials=len(calibrated_camera_ids), timeout_s=timeout_s,
        save_capture_dir=str(capture_dir))
    if (not isinstance(timing, Mapping) or
            timing.get("request_id") != request_id):
        raise ValueError("FoundPose request ID mismatch")
    if not isinstance(masks, Mapping) or not isinstance(poses, Mapping):
        raise ValueError("FoundPose did not return per-camera payloads")
    payload_ids = set(masks) | set(poses)
    if not payload_ids or payload_ids - calibrated_camera_ids:
        raise ValueError("FoundPose has no calibrated camera payloads")
    images = _load_saved_frames(
        capture_dir, payload_ids, image_write_timeout_s)
    times = _acquisition_metadata(
        acquisition_metadata_for_request, request_id, calibrated_camera_ids)
    if not payload_ids <= set(times):
        raise ValueError("socket payload camera lacks acquisition time")
    return SocketCaptureInput(
        capture_id, request_id, socket_prompt, images, dict(masks),
        dict(poses), times, "camera_acquisition")
