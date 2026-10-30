"""Capture key and fixed socket with AutoDex cameras, without robot control.

This is a diagnostic of raw camera, SAM and FoundPose outputs. It deliberately
does not admit a key pose for robot planning or invent exposure-aligned robot
feedback. The socket is frozen by the normal board/socket camera session
before the operator places the key.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np

from .assets import AssetPaths
from .frame_provenance import image_sha256
from .live_capture import KeyCaptureInput, collect_key_capture
from .prompt_contract import reject_incorrect_object_color
from .session_launch import SessionLaunch, open_session_cameras
from .silhouette_compat import prepare_key_silhouette


SCHEMA = "precision_insertion_camera_only_key_socket_qa_v1"
DEFAULT_KEY_PROMPT = "small white solid peg"


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _pose_summary(payload: dict) -> dict:
    result = {"ok": payload.get("ok") is True,
              "frame_id": payload.get("frame_id"),
              "image_sha256": payload.get("image_sha256")}
    if result["ok"]:
        try:
            pose = np.asarray(payload.get("pose_world"), dtype=float)
        except (TypeError, ValueError):
            pose = np.empty((0, 0))
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            result["ok"] = False
            result["error"] = "FoundPose returned an invalid 4x4 pose"
        else:
            result["pose_world"] = pose.tolist()
    for name in ("quality", "inliers", "mask_pixels"):
        value = payload.get(name)
        if isinstance(value, (int, float, np.number)) and np.isfinite(value):
            result[name] = float(value)
    return result


def write_camera_only_key_qa(
    *, capture: KeyCaptureInput, evidence_dir: Path,
    session_calibration_path: Path,
    expected_camera_ids: set[str],
) -> Path:
    """Write every view, including empty masks and failed pose estimates."""
    folder = Path(evidence_dir).expanduser().resolve() / "key_socket_qa"
    if folder.exists():
        raise FileExistsError("key/socket QA already exists in this session")
    folder.mkdir(parents=False)
    masks_dir = folder / "masks"
    masks_dir.mkdir()
    views = {}
    for camera_id in sorted(capture.images_bgr):
        image = capture.images_bgr[camera_id]
        mask_payload = capture.masks.get(camera_id, {})
        pose_payload = capture.poses.get(camera_id, {})
        raw_mask = mask_payload.get("mask")
        mask = np.asarray((np.zeros(image.shape[:2]) if raw_mask is None else
                           raw_mask),
                          dtype=bool)
        if mask.shape != image.shape[:2]:
            raise ValueError(f"{camera_id}: SAM mask does not match raw frame")
        mask_file = masks_dir / f"{camera_id}.png"
        if not cv2.imwrite(str(mask_file), mask.astype(np.uint8) * 255):
            raise OSError(f"{camera_id}: failed to save SAM mask")
        image_file = capture.capture_dir / "images" / f"{camera_id}.png"
        if not image_file.is_file():
            raise FileNotFoundError(f"{camera_id}: raw FoundPose frame is missing")
        views[camera_id] = {
            "frame_evidence": capture.frame_evidence[camera_id],
            "acquisition_timestamp_s": capture.frame_timestamps_s[camera_id],
            "raw_image": {"path": str(image_file), "sha256": _sha(image_file)},
            "sam": {
                "frame_id": mask_payload.get("frame_id"),
                "image_sha256": mask_payload.get("image_sha256"),
                "mask_pixels": int(np.count_nonzero(mask)),
                "mask_png": {"path": str(mask_file), "sha256": _sha(mask_file)},
            },
            "foundpose": _pose_summary(pose_payload),
        }
    calibration_path = Path(session_calibration_path).expanduser().resolve()
    report = {
        "schema": SCHEMA,
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "key_prompt": capture.prompt,
        "camera_count": len(views),
        "expected_camera_ids": sorted(expected_camera_ids),
        "missing_camera_ids": sorted(expected_camera_ids - views.keys()),
        "session_calibration": {"path": str(calibration_path),
                                "sha256": _sha(calibration_path)},
        "source_capture_dir": str(capture.capture_dir),
        "views": views,
        "scope": "camera_only_sam_foundpose_diagnostic_no_key_pose_admission_or_robot_state",
        "robot_ready": False,
    }
    report_path = folder / "report.json"
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    verify_camera_only_key_qa(report_path)
    return report_path


def verify_camera_only_key_qa(report_path: Path) -> dict:
    """Replay saved frame/mask hashes and same-request provenance."""
    source = Path(report_path).expanduser().resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    views = report.get("views")
    if (report.get("schema") != SCHEMA or
            report.get("robot_ready") is not False or
            report.get("scope") !=
            "camera_only_sam_foundpose_diagnostic_no_key_pose_admission_or_robot_state" or
            not isinstance(views, dict) or
            report.get("camera_count") != len(views) or len(views) < 2):
        raise ValueError("invalid key/socket camera-only QA report")
    expected = report.get("expected_camera_ids")
    if (not isinstance(expected, list) or
            len(set(expected)) != len(expected) or
            not set(views) <= set(expected) or
            report.get("missing_camera_ids") != sorted(set(expected) - views.keys())):
        raise ValueError("camera QA view inventory differs from the launch")
    calibration = report["session_calibration"]
    calibration_path = Path(calibration["path"]).expanduser().resolve()
    if _sha(calibration_path) != calibration["sha256"]:
        raise ValueError("camera QA session calibration changed")
    capture_dir = Path(report["source_capture_dir"]).expanduser().resolve()
    for camera_id, row in views.items():
        frame = row["frame_evidence"]
        image_ref = row["raw_image"]
        mask_ref = row["sam"]["mask_png"]
        image_file = Path(image_ref["path"]).expanduser().resolve()
        mask_file = Path(mask_ref["path"]).expanduser().resolve()
        if (image_file != capture_dir / "images" / f"{camera_id}.png" or
                mask_file != source.parent / "masks" / f"{camera_id}.png" or
                _sha(image_file) != image_ref["sha256"] or
                _sha(mask_file) != mask_ref["sha256"]):
            raise ValueError(f"{camera_id}: camera QA saved pixels changed")
        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if (image is None or mask is None or
                image.shape[:2] != mask.shape or
                image_sha256(image) != frame["image_sha256"] or
                row["sam"]["image_sha256"] != frame["image_sha256"] or
                row["foundpose"]["image_sha256"] != frame["image_sha256"] or
                row["sam"]["frame_id"] != frame["frame_id"] or
                row["foundpose"]["frame_id"] != frame["frame_id"] or
                row["acquisition_timestamp_s"] != frame["timestamp_s"] or
                int(np.count_nonzero(mask)) != row["sam"]["mask_pixels"]):
            raise ValueError(f"{camera_id}: camera QA provenance differs")
    return report


def run_camera_only_key_socket_qa(
    *, launch: SessionLaunch, confirm_key_placement: Callable[[], bool],
    confirm_socket_placement: Callable[[], bool] | None = None,
    key_prompt: str = DEFAULT_KEY_PROMPT,
    capture_id: str | None = None,
    init_ports: tuple[int, int, int] | None = None,
) -> Path:
    """Freeze socket, then collect one same-session key/SAM/FoundPose request."""
    if (not isinstance(launch, SessionLaunch) or
            launch.execution_scope not in {"live", "camera_only_diagnostic"}):
        raise PermissionError("key/socket QA needs a camera-enabled launch")
    if (not callable(confirm_key_placement) or
            not isinstance(key_prompt, str) or not key_prompt.strip()):
        raise ValueError("camera QA needs an operator placement callback and prompt")
    reject_incorrect_object_color(key_prompt)
    if launch.mode.family != "cylinder":
        raise ValueError("key/socket camera QA currently supports the cylinder")
    identifier = capture_id or time.strftime("key_socket_%Y%m%d_%H%M%S")
    with open_session_cameras(launch, init_ports=init_ports) as cameras:
        started = cameras.start(
            confirm_socket_placement=confirm_socket_placement)
        if confirm_key_placement() is not True:
            raise RuntimeError("operator canceled key/socket camera QA")
        paths = AssetPaths(launch.shared_root, launch.mode)
        key = launch.mode.key_object
        raw_mesh = paths.raw_mesh(key)
        if not raw_mesh.is_file() or not paths.foundpose_repre(key).is_file():
            raise FileNotFoundError("key mesh or current FoundPose representation is missing")
        prepare_key_silhouette(
            init_orchestrator=cameras.init_orchestrator,
            object_name=key, raw_mesh=raw_mesh)
        cameras.init_orchestrator.init_object(
            obj_name=key, mesh_path=str(raw_mesh),
            assets_root=str(paths.foundpose_assets_root(key)),
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            image_hw=launch.image_hw, mode="live", load_silhouette=False)
        capture = collect_key_capture(
            init_orchestrator=cameras.init_orchestrator,
            key_object=key, capture_id=identifier,
            key_prompt=key_prompt, capture_root=launch.capture_root,
            calibrated_camera_ids=launch.camera_ids,
            acquisition_metadata_for_request=(
                cameras.acquisition_metadata_for_request),
            timeout_s=launch.socket_timeout_s,
            image_write_timeout_s=launch.image_write_timeout_s)
        return write_camera_only_key_qa(
            capture=capture, evidence_dir=started.evidence_dir,
            session_calibration_path=(
                started.evidence_dir / "session_calibration.json"),
            expected_camera_ids=launch.camera_ids)
