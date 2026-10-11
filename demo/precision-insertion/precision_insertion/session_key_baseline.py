"""Load a socket-only key-ROI baseline from the frozen session evidence.

The baseline is a *segmentation aid*, not another socket calibration or a
replacement for the fixed collision world. It must come from the same saved
session that admitted the repeated socket captures. Its provenance is checked
before any key image or model is processed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .frame_provenance import image_sha256
from .live_session_runner import OpenedPrecisionSession
from .session_bootstrap import _safe_id, verify_session_evidence_bundle


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SessionKeyBaseline:
    session_evidence_dir: Path
    socket_capture_id: str
    socket_request_id: int
    session_manifest_sha256: str
    images_bgr: dict[str, np.ndarray]
    socket_masks: dict[str, np.ndarray]
    source: dict[str, dict]

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_session_key_baseline_v1",
            "session_evidence_dir": str(self.session_evidence_dir),
            "socket_capture_id": self.socket_capture_id,
            "socket_request_id": self.socket_request_id,
            "session_manifest_sha256": self.session_manifest_sha256,
            "source": self.source,
            "scope": "same_session_socket_only_roi_reference_not_key_pose",
            "robot_ready": False,
        }


def load_session_key_baseline(
    *, session_evidence_dir: Path, socket_capture_id: str,
    expected_session_manifest_sha256: str,
    calibrated_camera_ids: set[str], minimum_views: int = 2,
) -> SessionKeyBaseline:
    """Read admitted socket-only views without trusting unbound PNGs.

    ``socket_capture_id`` is explicit so a retry cannot silently switch its
    reference image. The session writer records *what it was told* about
    camera timing; commissioning still has to establish that the camera-side
    provider really reports exposure metadata and that the board was clear
    of keys during socket capture.
    """
    root = Path(session_evidence_dir).expanduser().resolve()
    capture_id = _safe_id(socket_capture_id, "socket capture ID")
    manifest_path = root / "evidence_manifest.json"
    if (not isinstance(expected_session_manifest_sha256, str) or
            not _DIGEST.fullmatch(expected_session_manifest_sha256) or
            hashlib.sha256(manifest_path.read_bytes()).hexdigest() !=
            expected_session_manifest_sha256):
        raise ValueError("socket baseline differs from opened session binding")
    if (not isinstance(calibrated_camera_ids, set) or
            any(_safe_id(serial, "camera ID") != serial
                for serial in calibrated_camera_ids) or
            type(minimum_views) is not int or minimum_views < 2 or
            len(calibrated_camera_ids) < minimum_views):
        raise ValueError("baseline needs commissioned camera IDs and view count")
    manifest = verify_session_evidence_bundle(root)
    if manifest.get("all_frames_bound_to_acquisition_evidence") is not True:
        raise ValueError("socket baseline lacks bound acquisition evidence")
    capture_ids = manifest.get("socket_capture_ids", [])
    request_ids = manifest.get("socket_request_ids", [])
    if capture_id not in capture_ids:
        raise ValueError("socket baseline capture is outside this session")
    folder = root / "socket" / capture_id
    payload = json.loads((folder / "payloads.json").read_text(
        encoding="utf-8"))
    if (payload.get("capture_id") != capture_id or
            len(request_ids) != len(capture_ids) or
            payload.get("request_id") != request_ids[capture_ids.index(capture_id)]):
        raise ValueError("socket baseline identity differs from session")
    admitted = payload.get("admission", {}).get("accepted_camera_ids")
    if (not isinstance(admitted, list) or len(admitted) != len(set(admitted)) or
            len(admitted) < minimum_views or
            not set(admitted) <= calibrated_camera_ids):
        raise ValueError("too few calibrated admitted socket baseline views")
    frame_evidence = payload.get("frame_evidence")
    if not isinstance(frame_evidence, dict):
        raise ValueError("socket baseline has no camera frame evidence")
    images: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    sources: dict[str, dict] = {}
    for serial in sorted(admitted):
        safe = _safe_id(serial, "camera ID")
        row = frame_evidence.get(serial)
        mask_metadata = payload.get("mask_payload_metadata", {}).get(serial)
        pose_metadata = payload.get("pose_payloads", {}).get(serial)
        if (not isinstance(row, dict) or
                not isinstance(mask_metadata, dict) or
                not isinstance(pose_metadata, dict) or
                not pose_metadata.get("ok") or
                row.get("frame_id") != mask_metadata.get("frame_id") or
                row.get("frame_id") != pose_metadata.get("frame_id")):
            raise ValueError(f"socket baseline frame/payload mismatch: {serial}")
        image_path = folder / "images" / f"{safe}.png"
        mask_path = folder / "masks" / f"{safe}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if (image is None or mask is None or
                mask.shape != image.shape[:2] or not np.any(mask) or
                image_sha256(image) != row.get("image_sha256")):
            raise ValueError(f"socket baseline pixels do not match: {serial}")
        images[serial] = image
        masks[serial] = mask > 0
        sources[serial] = {
            "frame_id": row["frame_id"],
            "image_sha256": row["image_sha256"],
            "mask_file_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "mask_pixels_sha256": hashlib.sha256(
                np.ascontiguousarray(masks[serial]).tobytes()).hexdigest(),
            "timestamp_s": row["timestamp_s"],
            "capture_id": capture_id,
            "request_id": payload["request_id"],
        }
    return SessionKeyBaseline(
        root, capture_id, payload["request_id"],
        expected_session_manifest_sha256,
        images, masks, sources)


def load_bound_session_key_baseline(
    *, opened: OpenedPrecisionSession, socket_capture_id: str,
    calibrated_camera_ids: set[str], minimum_views: int = 2,
) -> SessionKeyBaseline:
    """Bind the ROI reference to the *currently opened* precision run."""
    if not isinstance(opened, OpenedPrecisionSession):
        raise TypeError("baseline needs an opened precision session")
    binding = json.loads(opened.source_binding_path.read_text(encoding="utf-8"))
    if (binding.get("schema") !=
            "precision_insertion_session_source_binding_v1" or
            binding.get("session_evidence_dir") !=
            str(opened.evidence_dir.resolve())):
        raise ValueError("opened run has a different session evidence binding")
    return load_session_key_baseline(
        session_evidence_dir=opened.evidence_dir,
        socket_capture_id=socket_capture_id,
        expected_session_manifest_sha256=binding.get(
            "session_manifest_sha256"),
        calibrated_camera_ids=calibrated_camera_ids,
        minimum_views=minimum_views)
