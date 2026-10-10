"""Demo-only transport adapters preserving AutoDex camera frame provenance.

The stock capture daemons perform imaging/SAM/FoundPose. This module changes
only metadata propagation. Neither a sensor frame ID nor a publication time
is an exposure timestamp: a separate commissioned acquisition-time provider
must still supply the latter before live_capture admits a session.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Mapping

import cv2
import numpy as np

from .frame_provenance import image_sha256


class RecordingReader:
    """Preserve the exact SHM frame ID used by stock init inference."""

    def __init__(self, reader, owner):
        self._reader = reader
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._reader, name)

    def wait_for_new_frames(self, *args, **kwargs):
        request_id = getattr(self._owner, "_precision_active_request_id", None)
        if type(request_id) is not int or request_id <= 0:
            raise ValueError("precision capture needs an active request ID")
        frames = self._reader.wait_for_new_frames(*args, **kwargs)
        if not isinstance(frames, Mapping) or not frames:
            raise ValueError("precision capture received no camera frames")
        for serial, value in frames.items():
            image, frame_id = value
            if (type(frame_id) is not int or frame_id <= 0 or
                    frame_id <= self._owner._precision_last_frame_ids.get(serial, 0)):
                raise ValueError(f"camera {serial} frame ID did not advance")
            if serial not in self._owner.undistort_maps:
                raise ValueError(f"camera {serial} has no undistortion map")
            maps = self._owner.undistort_maps[serial]
            if maps is None:
                raise ValueError("precision inference must use live camera frames")
            undistorted = cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)
            self._owner._precision_frames[(request_id, serial)] = {
                "frame_id": frame_id,
                "image_sha256": image_sha256(undistorted),
                "image_space": "autodex_undistorted_full_frame",
            }
            self._owner._precision_last_frame_ids[serial] = frame_id
        return frames


class ProvenancePublisher:
    """Augment existing SAM/FoundPose PUB metadata without changing blobs."""

    def __init__(self, publisher, owner):
        self._publisher = publisher
        self._owner = owner

    def __getattr__(self, name):
        return getattr(self._publisher, name)

    def send_data(self, metadata, blobs):
        if (not isinstance(metadata, list) or not metadata or
                not all(isinstance(item, dict) for item in metadata)):
            raise ValueError("precision publisher needs per-camera metadata")
        enriched = []
        for item in metadata:
            req_id = item.get("req_id")
            serial = item.get("serial")
            row = self._owner._precision_frames.get((req_id, serial))
            if row is None:
                raise ValueError("SAM/FoundPose has no bound source camera frame")
            if any(name in item for name in (
                    "fid", "image_sha256", "image_space")):
                raise ValueError("stock publisher unexpectedly set precision fields")
            enriched.append({**item, "fid": row["frame_id"],
                             "image_sha256": row["image_sha256"],
                             "image_space": row["image_space"]})
        return self._publisher.send_data(enriched, blobs)


def install_init_provenance(daemon) -> None:
    """Wrap stock publishers once; call before serving any precision run."""
    if isinstance(daemon.pub_mask, ProvenancePublisher) or isinstance(
            daemon.pub_pose, ProvenancePublisher):
        raise ValueError("precision provenance is already installed")
    daemon._precision_frames = {}
    daemon._precision_last_frame_ids = {}
    daemon._precision_active_request_id = None
    daemon.pub_mask = ProvenancePublisher(daemon.pub_mask, daemon)
    daemon.pub_pose = ProvenancePublisher(daemon.pub_pose, daemon)


def decode_precision_mask(item: Mapping, blob: bytes) -> tuple[int, str, dict]:
    """Robot-side replacement for the stock callback that drops `fid`."""
    request_id, serial, fid, digest = _metadata_identity(item)
    image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8),
                         cv2.IMREAD_GRAYSCALE)
    if (image is None or image.shape != (item.get("h"), item.get("w"))):
        raise ValueError("precision SAM mask dimensions or PNG are invalid")
    return request_id, serial, {
        "mask": image > 127, "h": int(item["h"]), "w": int(item["w"]),
        "t_sam3": float(item.get("t_sam3", 0.0)),
        "ts": float(item.get("ts", 0.0)),
        "frame_id": fid, "image_sha256": digest,
        "image_space": "autodex_undistorted_full_frame",
    }


def decode_precision_pose(item: Mapping, blob: bytes) -> tuple[int, str, dict]:
    """Robot-side FoundPose callback retaining the same frame ID and hash."""
    request_id, serial, fid, digest = _metadata_identity(item)
    ok = bool(item.get("ok", False))
    entry = {
        "ok": ok, "t_fp": float(item.get("t_fp", 0.0)),
        "ts": float(item.get("ts", 0.0)), "frame_id": fid,
        "image_sha256": digest,
        "image_space": "autodex_undistorted_full_frame",
    }
    if ok:
        if len(blob) != 16 * 8:
            raise ValueError("precision FoundPose SE3 payload is malformed")
        entry.update({
            "pose_world": np.frombuffer(blob, dtype=np.float64).reshape(4, 4).copy(),
            "quality": float(item.get("quality", 0.0)),
            "inliers": int(item.get("inliers", 0)),
            "mask_pixels": int(item.get("mask_pixels", 0)),
        })
    return request_id, serial, entry


def _metadata_identity(item: Mapping) -> tuple[int, str, int, str]:
    if not isinstance(item, Mapping):
        raise ValueError("precision payload lacks metadata")
    req_id, serial, fid = item.get("req_id"), item.get("serial"), item.get("fid")
    digest = item.get("image_sha256")
    if (type(req_id) is not int or req_id <= 0 or
            not isinstance(serial, str) or not serial or
            type(fid) is not int or fid <= 0 or
            not isinstance(digest, str) or len(digest) != 64 or
            any(ch not in "0123456789abcdef" for ch in digest) or
            item.get("image_space") != "autodex_undistorted_full_frame"):
        raise ValueError("precision payload lacks bound camera frame metadata")
    return req_id, serial, fid, digest


class SnapshotMetadataBuffer:
    """Second SUB subscriber records the stock snapshot JPEG envelope."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records: dict[int, dict[str, dict]] = {}

    def put(self, item: Mapping, blob: bytes) -> None:
        req_id, serial, fid = (item.get("req_id"), item.get("serial"),
                               item.get("fid"))
        if (type(req_id) is not int or req_id <= 0 or
                not isinstance(serial, str) or not serial or
                type(fid) is not int or fid <= 0 or not blob):
            raise ValueError("snapshot metadata lacks request/frame/JPEG")
        record = {"frame_id": fid,
                  "jpeg_sha256": hashlib.sha256(blob).hexdigest()}
        with self._lock:
            previous = self._records.setdefault(req_id, {}).get(serial)
            if previous is not None and previous != record:
                raise ValueError("two different JPEGs share one request/camera")
            self._records[req_id][serial] = record

    def get(self, request_id: int) -> dict[str, dict]:
        with self._lock:
            return dict(self._records.get(request_id, {}))

    def drop(self, request_id: int) -> None:
        with self._lock:
            self._records.pop(request_id, None)


class FrameIdentityRegistry:
    """Request-bound sensor frame IDs and hashes, without any time claims."""

    def __init__(self):
        self._lock = threading.Lock()
        self._rows: dict[int, dict[str, dict]] = {}

    def register(self, request_id: int, identities: Mapping[str, Mapping]) -> None:
        if type(request_id) is not int or request_id <= 0 or not identities:
            raise ValueError("frame identity needs a positive request and cameras")
        checked = {}
        for serial, row in identities.items():
            fid, digest = row.get("frame_id"), row.get("image_sha256")
            if (not isinstance(serial, str) or not serial or
                    type(fid) is not int or fid <= 0 or
                    not isinstance(digest, str) or len(digest) != 64 or
                    any(ch not in "0123456789abcdef" for ch in digest)):
                raise ValueError("frame identity needs sensor ID and pixel hash")
            checked[serial] = {"frame_id": fid, "image_sha256": digest}
        with self._lock:
            previous = self._rows.get(request_id)
            if previous is not None and previous != checked:
                raise ValueError("camera identities changed within request")
            self._rows[request_id] = checked

    def get(self, request_id: int) -> dict[str, dict]:
        with self._lock:
            return {serial: dict(row) for serial, row in
                    self._rows.get(request_id, {}).items()}

    def drop(self, request_id: int) -> None:
        with self._lock:
            self._rows.pop(request_id, None)


class ProvenanceSnapshotAdapter:
    """Use unchanged AutoDex snap dispatch plus a matching metadata tap."""

    def __init__(self, snapshot_orchestrator, metadata_buffer,
                 identity_registry: FrameIdentityRegistry | None = None,
                 *, poll_s: float = 0.01):
        if not math.isfinite(poll_s) or poll_s <= 0:
            raise ValueError("snapshot metadata poll interval must be positive")
        self.snapshot = snapshot_orchestrator
        self.metadata = metadata_buffer
        self.identity_registry = identity_registry
        self.poll_s = poll_s

    def snap(self, **kwargs):
        if kwargs.get("decode") is not True:
            raise ValueError("precision board capture requires decoded JPEG pixels")
        request_id = kwargs.get("request_id")
        if type(request_id) is not int or request_id <= 0:
            raise ValueError("precision board capture requires request ID")
        payload, timing = self.snapshot.snap(**kwargs)
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("stock snapshot returned no camera images")
        deadline = time.monotonic() + float(kwargs.get("timeout_s", 0.0))
        while time.monotonic() < deadline:
            metadata = self.metadata.get(request_id)
            if set(payload) <= set(metadata):
                break
            time.sleep(self.poll_s)
        metadata = self.metadata.get(request_id)
        if set(payload) != set(metadata):
            raise ValueError("snapshot metadata tap missed exact camera JPEGs")
        enriched = {}
        for serial, entry in payload.items():
            blob = entry.get("jpeg") if isinstance(entry, Mapping) else None
            image = entry.get("image") if isinstance(entry, Mapping) else None
            if (not isinstance(blob, bytes) or not isinstance(image, np.ndarray)
                    or hashlib.sha256(blob).hexdigest() !=
                    metadata[serial]["jpeg_sha256"]):
                raise ValueError("snapshot JPEG differs from frame-ID metadata")
            decoded = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8),
                                   cv2.IMREAD_COLOR)
            if decoded is None or not np.array_equal(decoded, image):
                raise ValueError("snapshot decoded pixels differ from matched JPEG")
            enriched[serial] = {**entry, "frame_id": metadata[serial]["frame_id"]}
        if self.identity_registry is not None:
            self.identity_registry.register(request_id, {
                serial: {"frame_id": entry["frame_id"],
                         "image_sha256": image_sha256(entry["image"])}
                for serial, entry in enriched.items()})
        self.metadata.drop(request_id)
        return enriched, timing


class SnapshotMetadataTap:
    """Observe the same stock snapshot PUB stream without re-triggering cameras.

    The subscriber uses the existing AutoDex envelope parser. A missed PUB
    message causes the board adapter to fail closed; it is never replaced by
    a later frame or by the publication timestamp.
    """

    def __init__(self, capture_ips: list[str], port_snap: int = 5009):
        from autodex.perception.init_orchestrator import _SubThread

        if not capture_ips or not all(isinstance(ip, str) and ip for ip in capture_ips):
            raise ValueError("snapshot tap needs capture PC IPs")
        self.buffer = SnapshotMetadataBuffer()
        self._thread = _SubThread(
            "precision_snapshot_meta", capture_ips, port_snap,
            self.buffer, self.buffer.put)
        self._thread.start()

    def close(self) -> None:
        self._thread.stop()
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            raise RuntimeError("snapshot metadata subscriber did not stop")


class PrecisionInitOrchestrator:
    """Wrap a stock InitOrchestrator, retaining extra fid/hash PUB fields.

    The stock instance still owns command dispatch and FoundPose selection.
    This replaces only its two per-message callbacks, before any run. Stock
    daemons lack the fields and will be rejected by the strict decoders.
    """

    def __init__(self, stock_orchestrator,
                 identity_registry: FrameIdentityRegistry | None = None):
        self.stock = stock_orchestrator
        self.identity_registry = identity_registry
        mask_thread = getattr(stock_orchestrator, "_mask_thread", None)
        pose_thread = getattr(stock_orchestrator, "_pose_thread", None)
        if (mask_thread is None or pose_thread is None or
                not hasattr(mask_thread, "on_message") or
                not hasattr(pose_thread, "on_message")):
            raise ValueError("stock InitOrchestrator has no mask/pose subscribers")
        mask_thread.on_message = self._on_mask
        pose_thread.on_message = self._on_pose

    def __getattr__(self, name):
        return getattr(self.stock, name)

    def _on_mask(self, item, blob):
        request_id, serial, row = decode_precision_mask(item, blob)
        self.stock.mask_buf.put(request_id, serial, row)

    def _on_pose(self, item, blob):
        request_id, serial, row = decode_precision_pose(item, blob)
        self.stock.pose_buf.put(request_id, serial, row)

    def collect_payloads(self, *args, **kwargs):
        masks, poses, timing = self.stock.collect_payloads(*args, **kwargs)
        if not isinstance(timing, Mapping) or type(timing.get("request_id")) is not int:
            raise ValueError("FoundPose collection has no request identity")
        request_id = timing["request_id"]
        if (set(masks) != set(poses) or not masks or
                any(masks[s].get("frame_id") != poses[s].get("frame_id") or
                    masks[s].get("image_sha256") != poses[s].get("image_sha256")
                    for s in masks)):
            raise ValueError("FoundPose mask and pose do not share camera frames")
        if self.identity_registry is not None:
            self.identity_registry.register(request_id, {
                s: {"frame_id": masks[s]["frame_id"],
                    "image_sha256": masks[s]["image_sha256"]}
                for s in masks})
        return masks, poses, timing
