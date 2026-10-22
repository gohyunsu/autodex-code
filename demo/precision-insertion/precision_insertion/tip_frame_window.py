"""Fetch and persist a capture-PC frame window for temporal key-tip tracking."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import zmq


@dataclass(frozen=True)
class TipFrame:
    source_frame_id: int
    read_unix_ns: int
    jpeg: bytes


@dataclass(frozen=True)
class TipFrameWindow:
    status: str
    camera_id: str
    target_frame_id: int
    frames: tuple[TipFrame, ...]


_WINDOW_STATUSES = frozenset({
    "complete", "waiting", "target_frame_missing", "service_unavailable",
    "invalid_response",
})


def fetch_tip_frame_window(
    *, host: str, port: int, camera_id: str, target_frame_id: int,
    before: int = 3, after: int = 3, timeout_s: float = 4.,
) -> TipFrameWindow:
    """Request JPEG copies of source frames around the snapshot frame ID."""
    if (not host or not camera_id or not 0 < port < 65536 or
            type(target_frame_id) is not int or target_frame_id < 1 or
            not 0 <= before <= 8 or not 0 <= after <= 8):
        raise ValueError("invalid tracked camera window request")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
    try:
        socket.connect(f"tcp://{host}:{port}")
        socket.send_json({"op": "window", "camera_id": camera_id,
                          "target_frame_id": target_frame_id,
                          "before": before, "after": after})
        parts = socket.recv_multipart()
    except zmq.ZMQError:
        return TipFrameWindow(
            "service_unavailable", camera_id, target_frame_id, ())
    finally:
        socket.close()
        context.term()
    try:
        header = json.loads(parts[0])
        if (header.get("status") not in _WINDOW_STATUSES or
                header.get("camera_id") != camera_id or
                header.get("target_frame_id") != target_frame_id or
                len(header.get("frames", ())) != len(parts) - 1):
            raise ValueError("capture-PC tracker response changed frame identity")
        frames = tuple(TipFrame(int(row["source_frame_id"]),
                                int(row["read_unix_ns"]), bytes(jpeg))
                       for row, jpeg in zip(header["frames"], parts[1:]))
        ids = [frame.source_frame_id for frame in frames]
        if (len(set(ids)) != len(ids) or ids != sorted(ids) or
                any(fid < target_frame_id - before or
                    fid > target_frame_id + after for fid in ids) or
                (header["status"] == "complete" and target_frame_id not in ids)):
            raise ValueError("capture-PC tracker returned unrelated source frames")
        return TipFrameWindow(header["status"], camera_id,
                              target_frame_id, frames)
    except (IndexError, KeyError, TypeError, ValueError):
        # This optional temporal estimator must not interrupt insertion.
        return TipFrameWindow("invalid_response", camera_id, target_frame_id, ())


def save_tip_frame_window(window: TipFrameWindow, bundle: Path) -> None:
    folder = bundle / "temporal_window"
    folder.mkdir(exist_ok=False)
    rows = []
    for frame in window.frames:
        name = f"{frame.source_frame_id}.jpg"
        with (folder / name).open("xb") as stream:
            stream.write(frame.jpeg)
        rows.append({"source_frame_id": frame.source_frame_id,
                     "read_unix_ns": frame.read_unix_ns,
                     "file": name,
                     "sha256": hashlib.sha256(frame.jpeg).hexdigest()})
    manifest = {"schema": "precision_tip_frame_window_v1",
                "status": window.status, "camera_id": window.camera_id,
                "target_frame_id": window.target_frame_id,
                "timestamp_kind": "capture_pc_read_unix_ns_not_exposure_time",
                "frames": rows}
    with (folder / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def load_tip_frame_window(bundle: Path) -> TipFrameWindow | None:
    folder = bundle / "temporal_window"
    if not folder.is_dir():
        return None
    manifest = json.loads((folder / "manifest.json").read_text(
        encoding="utf-8"))
    if manifest.get("schema") != "precision_tip_frame_window_v1":
        raise ValueError("invalid temporal frame window manifest")
    frames = []
    for row in manifest["frames"]:
        frame_id = int(row["source_frame_id"])
        name = f"{frame_id}.jpg"
        if row["file"] != name:
            raise ValueError("temporal frame path differs from source ID")
        jpeg = (folder / name).read_bytes()
        if hashlib.sha256(jpeg).hexdigest() != row["sha256"]:
            raise ValueError("saved temporal frame changed")
        frames.append(TipFrame(frame_id, int(row["read_unix_ns"]), jpeg))
    return TipFrameWindow(manifest["status"], manifest["camera_id"],
                          int(manifest["target_frame_id"]), tuple(frames))
