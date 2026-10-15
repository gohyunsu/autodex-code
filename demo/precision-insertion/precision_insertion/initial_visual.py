"""Bind a separate initial tabletop snapshot to a FoundPose key observation."""

from __future__ import annotations

import json
from concurrent.futures import Future
from pathlib import Path
from threading import Lock

from .key_perception import verify_key_capture_artifacts
from .raw_camera_capture import (
    RawCameraCapture, verify_raw_camera_capture, write_raw_camera_capture,
)


_PENDING: dict[Path, Future] = {}
_PENDING_LOCK = Lock()


def register_pending_initial_visual(key_bundle: Path, future: Future) -> None:
    """Track a save that may finish after candidate planning and pickup."""
    root = Path(key_bundle).expanduser().resolve()
    with _PENDING_LOCK:
        if root in _PENDING:
            raise ValueError("initial visual save is already pending")
        _PENDING[root] = future


def _wait_for_initial_visual(key_bundle: Path) -> None:
    root = Path(key_bundle).expanduser().resolve()
    with _PENDING_LOCK:
        pending = _PENDING.get(root)
    if pending is not None:
        pending.result()
        with _PENDING_LOCK:
            if _PENDING.get(root) is pending:
                _PENDING.pop(root, None)


def drain_pending_initial_visuals() -> None:
    """Finish any remaining evidence saves before the live trace closes."""
    with _PENDING_LOCK:
        roots = list(_PENDING)
    for root in roots:
        _wait_for_initial_visual(root)


def write_initial_visual_capture(
    key_bundle: Path, capture: RawCameraCapture,
) -> Path:
    """Save undistorted raw VLM pixels while preserving both frame identities."""
    root = Path(key_bundle).expanduser().resolve()
    manifest = verify_key_capture_artifacts(root)
    if manifest.get("schema") != "precision_insertion_key_pose_evidence_v2":
        raise ValueError("separate initial visual needs a pose-only key bundle")
    if (set(capture.frame_ids) != set(manifest["camera_ids"]) or
            capture.request_id == manifest["request_id"]):
        raise ValueError("initial visual snapshot must use the same cameras and a new request")
    visual_root = root / "initial_tabletop"
    write_raw_camera_capture(capture, visual_root, phase="initial_tabletop")
    visual = verify_raw_camera_capture(visual_root, phase="initial_tabletop")
    binding = {
        "schema": "precision_insertion_key_initial_visual_v1",
        "pose_capture_id": manifest["capture_id"],
        "pose_request_id": manifest["request_id"],
        "visual_capture_id": visual["capture_id"],
        "visual_request_id": visual["request_id"],
        "camera_ids": sorted(visual["frame_evidence"]),
        "visual_bundle": "initial_tabletop",
        "scope": "separate_raw_vlm_frame_not_foundpose_source_frame",
    }
    with (root / "initial_visual_binding.json").open("x", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return visual_root


def initial_visual_source(key_bundle: Path) -> tuple[Path, dict]:
    """Return the actual raw VLM frame root and its independently timed rows."""
    root = Path(key_bundle).expanduser().resolve()
    _wait_for_initial_visual(root)
    manifest = verify_key_capture_artifacts(root)
    if manifest["schema"] == "precision_insertion_key_capture_evidence_v1":
        record = json.loads((root / "key_observation.json").read_text(
            encoding="utf-8"))
        return root, record
    if manifest["schema"] != "precision_insertion_key_pose_evidence_v2":
        raise ValueError("unknown initial key evidence schema")
    binding = json.loads((root / "initial_visual_binding.json").read_text(
        encoding="utf-8"))
    visual_root = root / "initial_tabletop"
    visual = verify_raw_camera_capture(visual_root, phase="initial_tabletop")
    if (binding.get("schema") != "precision_insertion_key_initial_visual_v1" or
            binding.get("pose_capture_id") != manifest["capture_id"] or
            binding.get("pose_request_id") != manifest["request_id"] or
            binding.get("visual_capture_id") != visual["capture_id"] or
            binding.get("visual_request_id") != visual["request_id"] or
            binding.get("visual_bundle") != "initial_tabletop" or
            binding.get("camera_ids") != sorted(visual["frame_evidence"]) or
            set(visual["frame_evidence"]) != set(manifest["camera_ids"])):
        raise ValueError("initial tabletop snapshot differs from key pose session")
    record = json.loads((root / "key_observation.json").read_text(
        encoding="utf-8"))
    if min(row["timestamp_s"] for row in visual["frame_evidence"].values()) <= (
            max(row["timestamp_s"]
                for row in record["frame_evidence"].values())):
        raise ValueError("initial visual snapshot does not follow key pose capture")
    record["pose_frame_evidence"] = record["frame_evidence"]
    record["frame_evidence"] = visual["frame_evidence"]
    record["visual_capture_id"] = visual["capture_id"]
    return visual_root, record
