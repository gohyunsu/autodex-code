"""Join existing AutoDex evidence gates into one auditable session bootstrap.

The caller supplies frames and *independently verified* acquisition times.
This module does no camera I/O or robot motion: it admits repeated socket
captures, reuses the existing ChArUco/session calibration, and preserves the
source images, masks, poses and provenance in a new directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

import cv2
import numpy as np

from .calibration import SessionCalibration, calibrate_session
from .config import TaskMode
from .frame_provenance import bounded_capture_skew_s, verify_frame_provenance
from .perception_evidence import (
    SocketCaptureEvidence, SocketViewLimits, admit_socket_capture,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SocketCaptureInput:
    capture_id: str
    request_id: int
    prompt: str
    images_bgr: Mapping[str, np.ndarray]
    masks: Mapping[str, dict]
    poses: Mapping[str, dict]
    frame_timestamps_s: Mapping[str, float]
    frame_timestamp_source: str
    frame_evidence: Mapping[str, dict] | None = None


@dataclass(frozen=True)
class SessionBootstrap:
    calibration: SessionCalibration
    board_request_id: int
    board_images_bgr: Mapping[str, np.ndarray]
    board_timestamps_s: Mapping[str, float]
    board_timestamp_source: str
    socket_captures: tuple[SocketCaptureInput, ...]
    socket_admissions: tuple[SocketCaptureEvidence, ...]
    board_frame_evidence: Mapping[str, dict] | None = None


def _safe_id(value: str, name: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{name} must be a safe nonempty file identifier")
    return value


def _request_id(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer request ID")
    return value


def _bgr_image(image: np.ndarray, name: str) -> np.ndarray:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
            image.ndim != 3 or image.shape[2] != 3 or
            any(size <= 0 for size in image.shape)):
        raise ValueError(f"{name} must be a nonempty uint8 BGR image")
    return image


def bootstrap_session(
    *, mode: TaskMode, object_root: Path, board_request_id: int,
    board_images_bgr: Mapping[str, np.ndarray],
    board_timestamps_s: Mapping[str, float],
    board_timestamp_source: str,
    socket_captures: Sequence[SocketCaptureInput],
    calibrated_camera_ids: set[str], view_limits: SocketViewLimits,
    intrinsics_full: Mapping, extrinsics_full: Mapping,
    c2r: np.ndarray, base_scene: dict, socket_collision_mesh: Path,
    max_socket_translation_mm: float, max_socket_angle_deg: float,
    board_frame_evidence: Mapping[str, dict] | None = None,
) -> SessionBootstrap:
    """Admit repeated socket observations, then freeze one AutoDex world.

    The socket-specific SAM prompt and raw images are retained as evidence;
    the key must not be on the board during these captures.  A source labeled
    ``camera_acquisition`` is an assertion by the capture adapter, not a value
    inferred from AutoDex's publication timestamps.
    """
    if not isinstance(board_images_bgr, Mapping) or not board_images_bgr:
        raise ValueError("board camera images are required")
    _request_id(board_request_id, "board")
    if board_timestamp_source != "camera_acquisition":
        raise ValueError("board images require camera acquisition timestamps")
    if set(board_images_bgr) != set(board_timestamps_s):
        raise ValueError("board image and acquisition-time camera IDs must match")
    if set(intrinsics_full) != set(extrinsics_full) or (
            set(intrinsics_full) != calibrated_camera_ids):
        raise ValueError("calibrated camera IDs must match AutoDex calibrations")
    for camera_id, image in board_images_bgr.items():
        _safe_id(camera_id, "board camera ID")
        _bgr_image(image, f"board image {camera_id}")
    if board_frame_evidence is not None:
        board_verified = verify_frame_provenance(
            {"request_id": board_request_id, "source": "camera_acquisition",
             "frames": board_frame_evidence},
            request_id=board_request_id, images_bgr=board_images_bgr,
            frame_ids={s: row.get("frame_id") for s, row in
                       board_frame_evidence.items()})
        if (any(board_timestamps_s[s] != row["timestamp_s"] for s, row in
                board_verified.items()) or
                bounded_capture_skew_s(board_verified) >
                view_limits.maximum_capture_skew_s):
            raise ValueError("board frame timing conflicts with verified evidence")
    if not isinstance(socket_captures, Sequence) or len(socket_captures) < 2:
        raise ValueError("at least two socket captures are required")
    admissions: list[SocketCaptureEvidence] = []
    seen_ids: set[str] = set()
    seen_requests: set[int] = {board_request_id}
    for capture in socket_captures:
        if not isinstance(capture, SocketCaptureInput):
            raise TypeError("socket captures must be SocketCaptureInput")
        capture_id = _safe_id(capture.capture_id, "socket capture ID")
        if capture_id in seen_ids:
            raise ValueError(f"duplicate socket capture ID: {capture_id}")
        seen_ids.add(capture_id)
        request_id = _request_id(capture.request_id, "socket")
        if request_id in seen_requests:
            raise ValueError(f"duplicate session capture request ID: {request_id}")
        seen_requests.add(request_id)
        if (not isinstance(capture.prompt, str) or
                not capture.prompt.strip() or capture.prompt == "object"):
            raise ValueError("each socket capture needs a specific SAM prompt")
        if (not isinstance(capture.images_bgr, Mapping) or
                not isinstance(capture.masks, Mapping) or
                not isinstance(capture.poses, Mapping) or
                not (set(capture.masks) | set(capture.poses)) <=
                set(capture.images_bgr)):
            raise ValueError("socket payloads need their same-request raw images")
        if not set(capture.images_bgr) <= calibrated_camera_ids:
            raise ValueError("socket capture includes uncalibrated camera")
        for camera_id, image in capture.images_bgr.items():
            _safe_id(camera_id, "socket camera ID")
            _bgr_image(image, f"socket image {capture_id}/{camera_id}")
            if camera_id in capture.masks:
                mask_entry = capture.masks[camera_id]
                if not isinstance(mask_entry, Mapping):
                    raise ValueError("socket mask payload must be a mapping")
                mask = mask_entry.get("mask")
                if not isinstance(mask, np.ndarray) or mask.shape != image.shape[:2]:
                    raise ValueError("socket mask and raw image dimensions differ")
        if capture.frame_evidence is not None:
            frame_ids = {}
            for camera_id in capture.images_bgr:
                mask_fid = capture.masks.get(camera_id, {}).get("frame_id")
                pose_fid = capture.poses.get(camera_id, {}).get("frame_id")
                if mask_fid != pose_fid:
                    raise ValueError("socket SAM/FoundPose frame IDs differ")
                frame_ids[camera_id] = mask_fid
            verified = verify_frame_provenance(
                {"request_id": capture.request_id,
                 "source": "camera_acquisition",
                 "frames": capture.frame_evidence},
                request_id=capture.request_id, images_bgr=capture.images_bgr,
                frame_ids=frame_ids)
            if (any(capture.frame_timestamps_s[s] != row["timestamp_s"]
                    for s, row in verified.items()) or
                    bounded_capture_skew_s(verified) >
                    view_limits.maximum_capture_skew_s):
                raise ValueError("socket frame timing conflicts with verified evidence")
        admissions.append(admit_socket_capture(
            capture_id=capture_id, masks=capture.masks, poses=capture.poses,
            frame_timestamps_s=capture.frame_timestamps_s,
            frame_timestamp_source=capture.frame_timestamp_source,
            calibrated_camera_ids=calibrated_camera_ids, limits=view_limits))
    calibration = calibrate_session(
        mode=mode, object_root=object_root,
        board_images_bgr=board_images_bgr,
        board_timestamps_s=board_timestamps_s,
        board_timestamp_source=board_timestamp_source,
        socket_observations=tuple(
            observation for admission in admissions
            for observation in admission.observations),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        c2r=c2r, base_scene=base_scene,
        socket_collision_mesh=socket_collision_mesh,
        max_capture_skew_s=view_limits.maximum_capture_skew_s,
        max_socket_translation_mm=max_socket_translation_mm,
        max_socket_angle_deg=max_socket_angle_deg,
        min_socket_captures=2,
        min_views_per_capture=view_limits.minimum_accepted_views,
    )
    return SessionBootstrap(
        calibration, board_request_id, board_images_bgr, board_timestamps_s,
        board_timestamp_source, tuple(socket_captures), tuple(admissions),
        board_frame_evidence)


def _write_png(path: Path, image: np.ndarray) -> str:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def write_session_bootstrap_artifacts(
    bootstrap: SessionBootstrap, output_dir: Path,
) -> Path:
    """Save exact input evidence and the frozen scene without overwriting.

    This records the supplied timestamp source but cannot independently prove
    it.  Live hardware use still needs a commissioned acquisition-time adapter.
    """
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    files: dict[str, str] = {}
    board_dir = target / "board"
    board_dir.mkdir()
    for camera_id, image in sorted(bootstrap.board_images_bgr.items()):
        path = board_dir / f"{_safe_id(camera_id, 'camera ID')}.png"
        files[str(path.relative_to(target))] = _write_png(path, image)
    socket_root = target / "socket"
    socket_root.mkdir()
    for capture, admission in zip(
            bootstrap.socket_captures, bootstrap.socket_admissions,
            strict=True):
        capture_dir = socket_root / _safe_id(capture.capture_id, "capture ID")
        capture_dir.mkdir()
        image_dir = capture_dir / "images"
        mask_dir = capture_dir / "masks"
        image_dir.mkdir()
        mask_dir.mkdir()
        payload = {
            "capture_id": capture.capture_id,
            "request_id": capture.request_id,
            "sam_prompt": capture.prompt,
            "admission": admission.to_record(),
            "frame_timestamps_s": dict(capture.frame_timestamps_s),
            "frame_evidence": (dict(capture.frame_evidence)
                               if capture.frame_evidence is not None else None),
            "mask_payload_metadata": {},
            "pose_payloads": {},
        }
        for camera_id, image in sorted(capture.images_bgr.items()):
            image_file = image_dir / f"{camera_id}.png"
            files[str(image_file.relative_to(target))] = _write_png(image_file, image)
            if camera_id in capture.masks:
                mask_entry = capture.masks[camera_id]
                mask = np.asarray(mask_entry["mask"], dtype=np.uint8) * 255
                mask_file = mask_dir / f"{camera_id}.png"
                files[str(mask_file.relative_to(target))] = _write_png(mask_file, mask)
                payload["mask_payload_metadata"][camera_id] = {
                    key: value for key, value in mask_entry.items() if key != "mask"}
            if camera_id in capture.poses:
                pose_entry = dict(capture.poses[camera_id])
                if "pose_world" in pose_entry:
                    pose_entry["pose_world"] = np.asarray(
                        pose_entry["pose_world"], dtype=float).tolist()
                payload["pose_payloads"][camera_id] = pose_entry
        payload_file = capture_dir / "payloads.json"
        files[str(payload_file.relative_to(target))] = _write_json(
            payload_file, payload)
    calibration_file = target / "session_calibration.json"
    files[str(calibration_file.relative_to(target))] = _write_json(
        calibration_file, bootstrap.calibration.record)
    _write_json(target / "evidence_manifest.json", {
        "schema": "precision_insertion_session_evidence_bundle_v1",
        "board_timestamp_source": bootstrap.board_timestamp_source,
        "board_request_id": bootstrap.board_request_id,
        "board_timestamps_s": dict(bootstrap.board_timestamps_s),
        "board_frame_evidence": (dict(bootstrap.board_frame_evidence)
                                 if bootstrap.board_frame_evidence is not None
                                 else None),
        "all_frames_bound_to_acquisition_evidence": (
            bootstrap.board_frame_evidence is not None and
            all(capture.frame_evidence is not None
                for capture in bootstrap.socket_captures)),
        "socket_capture_ids": [capture.capture_id for capture in
                               bootstrap.socket_captures],
        "socket_request_ids": [capture.request_id for capture in
                               bootstrap.socket_captures],
        "files_sha256": files,
        "scope": "saved_calibration_evidence_not_live_robot_authorization",
        "robot_ready": False,
    })
    return target


def verify_session_evidence_bundle(output_dir: Path) -> dict:
    """Check saved evidence bytes before replay or NAS handoff.

    SHA-256 here guards against accidental changes to the files listed by the
    manifest. It is not a cryptographic signature of who captured the frames.
    """
    root = Path(output_dir).expanduser().resolve()
    manifest = json.loads((root / "evidence_manifest.json").read_text(
        encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema") !=
            "precision_insertion_session_evidence_bundle_v1"):
        raise ValueError("unknown session evidence bundle schema")
    hashes = manifest.get("files_sha256")
    if (not isinstance(hashes, dict) or
            "session_calibration.json" not in hashes or not hashes):
        raise ValueError("session evidence bundle has no calibration digest")
    for relative, expected in hashes.items():
        if (not isinstance(relative, str) or not relative or
                not isinstance(expected, str) or len(expected) != 64 or
                any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError("invalid session evidence digest entry")
        path = root / relative
        if (not path.resolve().is_relative_to(root) or
                not path.is_file()):
            raise ValueError(f"missing or out-of-root session evidence: {relative}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"session evidence changed: {relative}")
    board_evidence = manifest.get("board_frame_evidence")
    if board_evidence is not None:
        board_images = {
            serial: cv2.imread(str(root / "board" / f"{_safe_id(serial, 'camera ID')}.png"),
                               cv2.IMREAD_COLOR)
            for serial in board_evidence}
        verified = verify_frame_provenance(
            {"request_id": manifest["board_request_id"],
             "source": "camera_acquisition", "frames": board_evidence},
            request_id=manifest["board_request_id"], images_bgr=board_images,
            frame_ids={serial: row.get("frame_id")
                       for serial, row in board_evidence.items()})
        if {serial: row["timestamp_s"] for serial, row in verified.items()} != (
                manifest.get("board_timestamps_s")):
            raise ValueError("saved board timing differs from frame evidence")
    bound_sockets = True
    for capture_id in manifest.get("socket_capture_ids", []):
        folder = root / "socket" / _safe_id(capture_id, "capture ID")
        payload = json.loads((folder / "payloads.json").read_text(
            encoding="utf-8"))
        evidence = payload.get("frame_evidence")
        if evidence is None:
            bound_sockets = False
            continue
        images = {
            serial: cv2.imread(str(folder / "images" /
                                  f"{_safe_id(serial, 'camera ID')}.png"),
                               cv2.IMREAD_COLOR)
            for serial in evidence}
        frame_ids = {}
        for serial in evidence:
            mask_fid = payload["mask_payload_metadata"].get(serial, {}).get(
                "frame_id")
            pose_fid = payload["pose_payloads"].get(serial, {}).get("frame_id")
            if mask_fid != pose_fid:
                raise ValueError("saved SAM/FoundPose frame IDs differ")
            frame_ids[serial] = mask_fid
        verified = verify_frame_provenance(
            {"request_id": payload["request_id"],
             "source": "camera_acquisition", "frames": evidence},
            request_id=payload["request_id"], images_bgr=images,
            frame_ids=frame_ids)
        if {serial: row["timestamp_s"] for serial, row in verified.items()} != (
                payload.get("frame_timestamps_s")):
            raise ValueError("saved socket timing differs from frame evidence")
    if manifest.get("all_frames_bound_to_acquisition_evidence") != (
            board_evidence is not None and bound_sockets):
        raise ValueError("saved frame-binding flag conflicts with evidence")
    return manifest
