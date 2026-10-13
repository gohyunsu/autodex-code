"""Commissioned frame-ID to Unix-UTC exposure-time evidence adapter.

The existing ParaDex `pc_time` is recorded after GetNextImage and is *not*
an exposure timestamp. This module will not infer exposure from it. It only
accepts per-imaging-camera exposure references with retained source files,
held-out validation samples, bounded extrapolation and a declared clock error.
Physical validity of those references must be commissioned on the camera rig.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from typing import Mapping

import numpy as np

from .camera_transport import FrameIdentityRegistry


_SCHEMA = "precision_insertion_camera_time_calibration_v1"
_METHODS = frozenset({"same_imaging_camera_exposure_chunk_utc",
                      "independent_per_camera_trigger_metrology"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _samples(value, name: str) -> list[tuple[int, float, float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name} needs at least three exposure references")
    rows = []
    for row in value:
        if not isinstance(row, dict) or set(row) != {
                "frame_id", "exposure_utc_s", "max_error_s"}:
            raise ValueError(f"{name} has invalid exposure-reference fields")
        fid = row["frame_id"]
        timestamp = row["exposure_utc_s"]
        error = row["max_error_s"]
        if (type(fid) is not int or fid <= 0 or
                type(timestamp) not in (int, float) or
                not math.isfinite(timestamp) or timestamp <= 0 or
                type(error) not in (int, float) or
                not math.isfinite(error) or error < 0):
            raise ValueError(f"{name} has invalid frame ID or timestamp")
        rows.append((fid, float(timestamp), float(error)))
    if len({fid for fid, _t, _e in rows}) != len(rows):
        raise ValueError(f"{name} repeats a sensor frame ID")
    return sorted(rows)


@dataclass(frozen=True)
class CameraTimeCalibration:
    calibration_id: str
    camera_serial: str
    source_method: str
    frame_id_origin: float
    exposure_origin_utc_s: float
    period_s_per_frame: float
    base_error_s: float
    clock_offset_error_s: float
    drift_error_s_per_frame: float
    max_total_error_s: float
    latest_validated_frame_id: int
    earliest_valid_frame_id: int
    latest_valid_frame_id: int
    valid_until_utc_s: float
    calibration_sha256: str

    @classmethod
    def load(cls, path: Path) -> "CameraTimeCalibration":
        source = Path(path).expanduser().resolve()
        raw = source.read_bytes()
        record = json.loads(raw)
        if not isinstance(record, dict) or record.get("schema") != _SCHEMA:
            raise ValueError("unknown camera-time calibration schema")
        calibration_id = record.get("calibration_id")
        serial = record.get("camera_serial")
        if (not isinstance(calibration_id, str) or not calibration_id or
                not isinstance(serial, str) or not serial or
                record.get("source_method") not in _METHODS):
            raise ValueError("calibration needs per-camera independent exposure source")
        source_files = record.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError("calibration needs retained exposure source files")
        for row in source_files:
            if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
                raise ValueError("invalid camera-time source file reference")
            file_path = Path(row["path"]).expanduser()
            if (not file_path.is_absolute() or not file_path.is_file() or
                    _sha(file_path) != row["sha256"]):
                raise ValueError("camera-time source file changed or is missing")
        training = _samples(record.get("fit_samples"), "fit samples")
        validation = _samples(record.get("validation_samples"),
                              "held-out validation samples")
        if set(row[0] for row in training) & set(row[0] for row in validation):
            raise ValueError("time fit and held-out validation frame IDs overlap")
        fid = np.asarray([row[0] for row in training], dtype=float)
        timestamps = np.asarray([row[1] for row in training], dtype=float)
        origin = float(fid.mean())
        time_origin = float(timestamps.mean())
        centered = fid - origin
        denominator = float(centered @ centered)
        if denominator <= 0:
            raise ValueError("camera-time fit lacks frame-ID span")
        period = float(centered @ (timestamps - time_origin) / denominator)
        if not 0.001 <= period <= 1.0:
            raise ValueError("frame-ID exposure period is physically implausible")
        all_rows = training + validation
        base_error = max(abs(time_origin + period * (f - origin) - t) + e
                         for f, t, e in all_rows)
        clock_error = record.get("clock_offset_error_s")
        drift = record.get("drift_error_s_per_frame")
        max_total_error = record.get("max_total_error_s")
        max_extrapolation = record.get("max_extrapolation_frames")
        valid_until = record.get("valid_until_utc_s")
        if (type(clock_error) not in (int, float) or
                not math.isfinite(clock_error) or clock_error < 0 or
                type(drift) not in (int, float) or not math.isfinite(drift) or
                drift < 0 or type(max_total_error) not in (int, float) or
                not math.isfinite(max_total_error) or max_total_error <= 0 or
                type(max_extrapolation) is not int or
                max_extrapolation < 0 or type(valid_until) not in (int, float) or
                not math.isfinite(valid_until) or valid_until <= 0):
            raise ValueError("camera-time clock, drift and validity bounds are required")
        latest_validated = max(row[0] for row in all_rows)
        earliest = min(row[0] for row in all_rows)
        latest = latest_validated + max_extrapolation
        predicted_end = time_origin + period * (latest - origin)
        if valid_until < predicted_end:
            latest = min(latest, math.floor(
                origin + (valid_until - time_origin) / period))
        if latest < latest_validated:
            raise ValueError("camera-time calibration expires before validation")
        if (base_error + clock_error +
                (latest - latest_validated) * drift > max_total_error):
            raise ValueError("camera-time error exceeds commissioned limit")
        return cls(
            calibration_id, serial, record["source_method"], origin,
            time_origin, period, float(base_error), float(clock_error),
            float(drift), float(max_total_error), latest_validated, earliest, latest,
            float(valid_until), hashlib.sha256(raw).hexdigest())

    def exposure(self, frame_id: int) -> tuple[float, float]:
        if (type(frame_id) is not int or frame_id < self.earliest_valid_frame_id
                or frame_id > self.latest_valid_frame_id):
            raise ValueError("frame ID lies outside validated/extrapolation range")
        timestamp = self.exposure_origin_utc_s + self.period_s_per_frame * (
            frame_id - self.frame_id_origin)
        if timestamp > self.valid_until_utc_s:
            raise ValueError("camera-time calibration has expired")
        future_frames = max(0, frame_id - self.latest_validated_frame_id)
        max_error = (self.base_error_s + self.clock_offset_error_s +
                     future_frames * self.drift_error_s_per_frame)
        return timestamp, max_error


class AcquisitionTimeProvider:
    """Merge request-bound frame identities with independently calibrated time."""

    def __init__(self, registry: FrameIdentityRegistry,
                 calibrations: Mapping[str, CameraTimeCalibration],
                 request_windows: RequestWindowRegistry | None = None):
        if not calibrations or any(
                not isinstance(calibration, CameraTimeCalibration) or
                calibration.camera_serial != serial
                for serial, calibration in calibrations.items()):
            raise ValueError(
                "live frame-ID provider needs CameraTimeCalibration "
                "records with matching camera IDs")
        self.registry = registry
        self.calibrations = dict(calibrations)
        self.request_windows = request_windows

    def __call__(self, request_id: int) -> dict:
        identities = self.registry.get(request_id)
        if not identities or set(identities) - set(self.calibrations):
            raise ValueError("request lacks calibrated camera frame identities")
        if any(time.time() > self.calibrations[serial].valid_until_utc_s
               for serial in identities):
            raise ValueError("camera-time calibration has expired in wall time")
        window = (None if self.request_windows is None else
                  self.request_windows.get(request_id))
        frames = {}
        for serial, row in identities.items():
            calibration = self.calibrations[serial]
            timestamp, error = calibration.exposure(row["frame_id"])
            if window is not None and (
                    timestamp + error < window[0] or
                    timestamp - error > window[1]):
                raise ValueError(
                    f"camera {serial} calibrated exposure contradicts its "
                    "live request window")
            frames[serial] = {
                **row, "timestamp_s": timestamp, "max_error_s": error,
                "timestamp_method": "calibrated_frame_id",
                "calibration_id": calibration.calibration_id,
                "clock_domain": "unix_utc",
            }
            if window is not None:
                frames[serial]["request_window_start_s"] = window[0]
                frames[serial]["request_window_end_s"] = window[1]
        return {"request_id": request_id,
                "source": "camera_acquisition", "frames": frames}


class RequestWindowRegistry:
    """Robot-host UTC interval enclosing a new-frame AutoDex request.

    This deliberately gives no per-camera exposure estimate. Stock live
    daemons wait for a frame newer than the one present *after* receiving the
    request; their response is received after that frame was captured. The
    entire robot-host request/response interval therefore encloses exposure,
    even without synchronized capture-PC clocks.
    """

    def __init__(self, *, max_clock_step_s: float = .05):
        if (not math.isfinite(max_clock_step_s) or max_clock_step_s <= 0):
            raise ValueError("request-window clock-step limit must be positive")
        self.max_clock_step_s = float(max_clock_step_s)
        self._lock = threading.Lock()
        self._windows: dict[int, tuple[float, float]] = {}

    def put(self, request_id: int, start_s: float, end_s: float,
            start_monotonic_s: float, end_monotonic_s: float) -> None:
        wall_span = end_s - start_s
        mono_span = end_monotonic_s - start_monotonic_s
        if (type(request_id) is not int or request_id <= 0 or
                not all(math.isfinite(float(value)) for value in (
                    start_s, end_s, start_monotonic_s, end_monotonic_s)) or
                start_s <= 0 or wall_span <= 0 or mono_span <= 0 or
                abs(wall_span - mono_span) > self.max_clock_step_s):
            raise ValueError("request interval or robot-host clock is invalid")
        with self._lock:
            if request_id in self._windows:
                raise ValueError("request interval was already recorded")
            self._windows[request_id] = (float(start_s), float(end_s))

    def get(self, request_id: int) -> tuple[float, float]:
        with self._lock:
            if request_id not in self._windows:
                raise ValueError("request has no bounded acquisition interval")
            return self._windows[request_id]


class BoundedRequestTimeProvider:
    """Conservative live bring-up timing without a camera clock calibration.

    The midpoint is a serialization convention only. ``max_error_s`` spans
    the *entire* request/response interval, so consumers with tight timing or
    feedback gates reject a slow capture instead of mistaking the midpoint
    for a measured exposure timestamp.
    """

    def __init__(self, identities: FrameIdentityRegistry,
                 windows: RequestWindowRegistry):
        self.identities = identities
        self.windows = windows

    def __call__(self, request_id: int) -> dict:
        rows = self.identities.get(request_id)
        if not rows:
            raise ValueError("request lacks matched camera frame identities")
        start, end = self.windows.get(request_id)
        middle, radius = (start + end) / 2, (end - start) / 2
        return {
            "request_id": request_id, "source": "camera_acquisition",
            "frames": {serial: {
                **row, "timestamp_s": middle, "max_error_s": radius,
                "timestamp_method": "bounded_request_interval",
                "request_window_start_s": start,
                "request_window_end_s": end,
                "clock_domain": "unix_utc",
            } for serial, row in rows.items()},
        }
