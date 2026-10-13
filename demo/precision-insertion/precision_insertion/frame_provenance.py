"""Bind a reported exposure time to the exact frame used by perception.

This validates evidence supplied by a camera-side producer; it cannot certify
that producer's hardware timestamp or clock synchronization on its own.
Stock AutoDex snapshot/init payloads do not yet carry the required frame IDs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Mapping

import numpy as np


_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_METHODS = {"hardware_exposure", "calibrated_frame_id",
            "bounded_request_interval"}


def image_sha256(image: np.ndarray) -> str:
    """Hash decoded BGR pixels plus shape/dtype, independent of strides."""
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or not image.size):
        raise ValueError("frame must be a nonempty uint8 BGR image")
    header = json.dumps({"dtype": str(image.dtype), "shape": image.shape},
                        sort_keys=True).encode("ascii")
    return hashlib.sha256(
        header + b"\0" + np.ascontiguousarray(image).tobytes()).hexdigest()


def verify_frame_provenance(
    metadata: Mapping, *, request_id: int,
    images_bgr: Mapping[str, np.ndarray],
    frame_ids: Mapping[str, int],
) -> dict[str, dict]:
    """Require matching request, serial, frame ID, decoded pixels and time.

    ``max_error_s`` bounds exposure time in the robot-host Unix clock. A
    bounded request interval is admissible only because stock live daemons
    wait for *new* frames after dispatch; its midpoint is not an exposure
    measurement. Receipt/processing/publication times alone are not admitted.
    """
    if not isinstance(metadata, Mapping) or metadata.get("request_id") != request_id:
        raise ValueError("camera acquisition metadata has wrong request ID")
    if metadata.get("source") != "camera_acquisition":
        raise ValueError("camera acquisition-time metadata is required")
    frames = metadata.get("frames")
    if (not isinstance(frames, Mapping) or not frames or
            set(frames) != set(images_bgr) or set(frame_ids) != set(images_bgr)):
        raise ValueError("every image needs matching per-camera frame provenance")
    verified = {}
    for serial, image in images_bgr.items():
        row = frames[serial]
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid frame provenance for {serial}")
        fid = row.get("frame_id")
        observed_fid = frame_ids[serial]
        if (type(fid) is not int or fid <= 0 or
                type(observed_fid) is not int or fid != observed_fid):
            raise ValueError(f"frame ID mismatch for {serial}")
        digest = row.get("image_sha256")
        if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest) or
                digest != image_sha256(image)):
            raise ValueError(f"image digest mismatch for {serial}")
        stamp = row.get("timestamp_s")
        error = row.get("max_error_s")
        if (type(stamp) not in (int, float) or not math.isfinite(stamp) or
                stamp <= 0 or type(error) not in (int, float) or
                not math.isfinite(error) or error < 0):
            raise ValueError(f"invalid acquisition time or uncertainty for {serial}")
        method = row.get("timestamp_method")
        if (not isinstance(method, str) or method not in _METHODS or
                row.get("clock_domain") != "unix_utc"):
            raise ValueError(f"unverified acquisition timestamp method for {serial}")
        calibration_id = row.get("calibration_id")
        if calibration_id is not None and (
                not isinstance(calibration_id, str) or
                not _SAFE_ID.fullmatch(calibration_id)):
            raise ValueError(f"invalid timing calibration ID for {serial}")
        if method == "calibrated_frame_id" and calibration_id is None:
            raise ValueError(f"missing timing calibration ID for {serial}")
        window = None
        if method == "bounded_request_interval":
            start = row.get("request_window_start_s")
            end = row.get("request_window_end_s")
            if (type(start) not in (int, float) or
                    type(end) not in (int, float) or
                    not math.isfinite(start) or not math.isfinite(end) or
                    start <= 0 or end <= start or
                    abs(float(stamp) - (start + end) / 2) > 1e-6 or
                    error + 1e-9 < (end - start) / 2 or
                    calibration_id is not None):
                raise ValueError(f"request interval does not bound exposure for {serial}")
            window = (float(start), float(end))
        elif method == "calibrated_frame_id" and (
                "request_window_start_s" in row or
                "request_window_end_s" in row):
            start = row.get("request_window_start_s")
            end = row.get("request_window_end_s")
            if (type(start) not in (int, float) or
                    type(end) not in (int, float) or
                    not math.isfinite(start) or not math.isfinite(end) or
                    start <= 0 or end <= start or
                    stamp + error < start or stamp - error > end):
                raise ValueError(
                    f"calibrated exposure contradicts live request for {serial}")
            window = (float(start), float(end))
        verified[serial] = {
            "frame_id": fid,
            "image_sha256": digest,
            "timestamp_s": float(stamp),
            "max_error_s": float(error),
            "timestamp_method": method,
            "clock_domain": "unix_utc",
        }
        if calibration_id is not None:
            verified[serial]["calibration_id"] = calibration_id
        if window is not None:
            verified[serial]["request_window_start_s"] = window[0]
            verified[serial]["request_window_end_s"] = window[1]
    return verified


def bounded_capture_skew_s(frames: Mapping[str, Mapping]) -> float:
    """Worst-case cross-camera exposure skew, including stated time errors."""
    if not frames:
        raise ValueError("no frame provenance")
    upper = max(row["timestamp_s"] + row["max_error_s"]
                for row in frames.values())
    lower = min(row["timestamp_s"] - row["max_error_s"]
                for row in frames.values())
    return float(upper - lower)
