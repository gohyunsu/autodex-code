"""Opt-in same-frame FLIR chunk-timestamp journal for ParaDex capture PCs.

This captures raw camera clock ticks beside the frame ID *before* ParaDex
releases the PySpin image. Ticks are not Unix UTC, and host receipt time is
only a diagnostic. Neither value is accepted as session acquisition time
without a separately measured camera-clock conversion and error bound.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from queue import Queue
import threading
import time
from typing import Callable


_SCHEMA = "precision_insertion_camera_chunk_journal_v1"
_END = object()


class ChunkTimestampJournal:
    """Write an exclusive per-camera JSONL journal off the grab hot path."""

    def __init__(self, path: Path, *, camera_serial: str,
                 queue_capacity: int = 4096):
        target = Path(path).expanduser().resolve()
        if (not camera_serial or not camera_serial.isalnum() or
                type(queue_capacity) is not int or queue_capacity < 1):
            raise ValueError("chunk journal needs a camera and bounded queue")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.path = target
        self.camera_serial = camera_serial
        self._queue: Queue = Queue(maxsize=queue_capacity)
        self._last_frame_id = 0
        self._last_ticks = 0
        self._error: BaseException | None = None
        self._closed = False
        self._lock = threading.Lock()
        # Refuse to replace a previous journal, even after a crash.
        self._stream = target.open("x", encoding="utf-8")
        self._stream.write(json.dumps({
            "schema": _SCHEMA, "camera_serial": camera_serial,
            "timestamp_unit": "camera_ticks_unknown_frequency",
            "host_time_scope": "after_GetNextImage_not_exposure",
        }, sort_keys=True) + "\n")
        self._stream.flush()
        self._thread = threading.Thread(
            target=self._writer, name=f"chunk_journal_{camera_serial}",
            daemon=True)
        self._thread.start()

    def _writer(self) -> None:
        try:
            while True:
                row = self._queue.get()
                try:
                    if row is _END:
                        break
                    self._stream.write(json.dumps(
                        row, sort_keys=True, allow_nan=False) + "\n")
                finally:
                    self._queue.task_done()
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException as exc:
            self._error = exc
        finally:
            self._stream.close()

    def record(self, image) -> None:
        """Record chunk metadata from the exact unreleased GetNextImage image."""
        frame_id = image.GetFrameID()
        chunk_ticks = image.GetChunkData().GetTimestamp()
        host_time = time.time()
        if (type(frame_id) is not int or frame_id <= 0 or
                type(chunk_ticks) is not int or chunk_ticks <= 0 or
                not math.isfinite(host_time)):
            raise ValueError("same-frame chunk ID/timestamp is unavailable")
        with self._lock:
            if self._closed or self._error is not None:
                raise RuntimeError("chunk journal is closed or failed")
            if frame_id <= self._last_frame_id or chunk_ticks <= self._last_ticks:
                raise ValueError("camera chunk frame ID or clock did not advance")
            self._queue.put_nowait({
                "frame_id": frame_id, "chunk_timestamp_ticks": chunk_ticks,
                "host_received_utc_s": host_time,
            })
            self._last_frame_id = frame_id
            self._last_ticks = chunk_ticks

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            failed = self._error
        if failed is not None:
            raise RuntimeError("chunk journal writer failed") from failed
        # The camera has stopped, so waiting here cannot stall its grab loop.
        self._queue.put(_END, timeout=5.0)
        self._thread.join(timeout=10.0)
        if self._thread.is_alive() or self._error is not None:
            raise RuntimeError("chunk journal did not durably close") from self._error


class TimestampedCameraPointer:
    """Delegate all PySpin operations, intercepting only GetNextImage."""

    def __init__(
        self, pointer, journal: ChunkTimestampJournal | None = None, *,
        journal_factory: Callable[[], ChunkTimestampJournal] | None = None,
    ):
        if (journal is None) == (journal_factory is None):
            raise ValueError("provide exactly one journal or journal factory")
        self._pointer = pointer
        self._journal = journal
        self._journal_factory = journal_factory
        self._acquiring = False

    def __getattr__(self, name):
        return getattr(self._pointer, name)

    def BeginAcquisition(self, *args, **kwargs):
        if self._acquiring:
            raise ValueError("camera acquisition already active")
        if self._journal_factory is not None:
            self._journal = self._journal_factory()
        try:
            result = self._pointer.BeginAcquisition(*args, **kwargs)
        except BaseException:
            if self._journal_factory is not None:
                self.close_journal()
            raise
        self._acquiring = True
        return result

    def EndAcquisition(self, *args, **kwargs):
        try:
            return self._pointer.EndAcquisition(*args, **kwargs)
        finally:
            # ParaDex drains the camera buffer after EndAcquisition. Those
            # discarded frames are not used by the image stream or inference.
            self._acquiring = False
            if self._journal_factory is not None:
                self.close_journal()

    def close_journal(self) -> None:
        if self._journal is not None:
            self._journal.close()
            if self._journal_factory is not None:
                self._journal = None

    def GetNextImage(self, *args, **kwargs):
        image = self._pointer.GetNextImage(*args, **kwargs)
        if not self._acquiring:
            return image
        try:
            if self._journal is None:
                raise RuntimeError("camera acquisition has no chunk journal")
            self._journal.record(image)
        except BaseException:
            image.Release()
            raise
        return image


def verify_chunk_journal(path: Path, *, camera_serial: str) -> dict:
    """Replay a closed journal; never convert ticks or host time to exposure."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (header != {
                "schema": _SCHEMA, "camera_serial": camera_serial,
                "timestamp_unit": "camera_ticks_unknown_frequency",
                "host_time_scope": "after_GetNextImage_not_exposure"}):
            raise ValueError("invalid camera chunk journal header")
        count, last_id, last_ticks = 0, 0, 0
        first_id = first_ticks = None
        for line in stream:
            row = json.loads(line)
            if (not isinstance(row, dict) or set(row) != {
                    "frame_id", "chunk_timestamp_ticks",
                    "host_received_utc_s"} or
                    type(row["frame_id"]) is not int or
                    type(row["chunk_timestamp_ticks"]) is not int or
                    type(row["host_received_utc_s"]) not in (int, float) or
                    not math.isfinite(row["host_received_utc_s"]) or
                    row["frame_id"] <= last_id or
                    row["chunk_timestamp_ticks"] <= last_ticks):
                raise ValueError("invalid or nonmonotonic camera chunk row")
            if first_id is None:
                first_id, first_ticks = (
                    row["frame_id"], row["chunk_timestamp_ticks"])
            count += 1
            last_id, last_ticks = row["frame_id"], row["chunk_timestamp_ticks"]
    return {
        "schema": "precision_insertion_camera_chunk_journal_audit_v1",
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "camera_serial": camera_serial, "frame_count": count,
        "first_frame_id": first_id, "last_frame_id": last_id,
        "first_chunk_ticks": first_ticks, "last_chunk_ticks": last_ticks,
        "exposure_utc_admissible": False,
        "scope": "same_frame_camera_ticks_not_utc_calibration",
    }
