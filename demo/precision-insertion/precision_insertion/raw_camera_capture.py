"""Persist provenance-bound AutoDex camera frames without requiring FoundPose.

The camera-side producer must supply trustworthy exposure timestamps and frame
IDs. Saving/verifying these bytes does not establish object pose or motion.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .frame_provenance import verify_frame_provenance
from .runtime_evidence_cache import recalled, remember
from .session_bootstrap import _safe_id
from .timing_profile import timed
from .tip_frame_window import TipFrameWindow, save_tip_frame_window


PHASES = frozenset({"initial_tabletop", "after_lift", "preinsert", "post_lateral_hold",
                    "final_or_abort", "final_clear_view"})


@dataclass(frozen=True)
class RawCameraCapture:
    capture_id: str
    request_id: int
    images_bgr: Mapping[str, np.ndarray]
    frame_ids: Mapping[str, int]
    acquisition_metadata: Mapping
    source_images_bgr: Mapping[str, np.ndarray] | None = None
    source_acquisition_metadata: Mapping | None = None
    undistort_calibration: Mapping[str, Mapping] | None = None
    temporal_window: TipFrameWindow | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_png_hash(path: Path, image: np.ndarray) -> str:
    """Hash the encoded bytes once, without reading the new NAS file again."""
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise OSError(f"could not encode raw camera frame: {path}")
    data = payload.tobytes()
    with path.open("xb") as stream:
        stream.write(data)
    return hashlib.sha256(data).hexdigest()


def configured_capture_io_workers() -> int:
    """Bound local camera-file reads and writes by the same operator setting."""
    try:
        configured_workers = int(os.environ.get("PRECISION_CAPTURE_IO_WORKERS", "8"))
    except ValueError as exc:
        raise ValueError("PRECISION_CAPTURE_IO_WORKERS must be an integer") from exc
    if not 1 <= configured_workers <= 8:
        raise ValueError("PRECISION_CAPTURE_IO_WORKERS must be in 1..8")
    return configured_workers


def write_png_jobs(jobs: list[tuple[str, Path, np.ndarray, str]], *,
                   phase: str) -> tuple[list[tuple[str, str, str]], int]:
    """Use the production PNG worker setting for a batch of identified frames."""
    if not jobs:
        raise ValueError("PNG batch needs images")
    def write_one(job):
        camera, path, image, source = job
        with timed("evidence.png_write_hash", phase="evidence", kind="io",
                   camera_id=camera, capture_phase=phase, source=source):
            return camera, source, _write_png_hash(path, image)

    configured_workers = configured_capture_io_workers()
    workers = min(configured_workers, len(jobs))
    with timed("evidence.png_batch_write", phase="evidence", kind="io",
               capture_phase=phase, image_count=len(jobs), workers=workers):
        if workers == 1:
            results = [write_one(job) for job in jobs]
        else:
            with ThreadPoolExecutor(max_workers=workers,
                                    thread_name_prefix="precision-png") as pool:
                futures = [pool.submit(copy_context().run, write_one, job)
                           for job in jobs]
                results = [future.result() for future in futures]
    return results, workers


def write_raw_camera_capture(
    capture: RawCameraCapture, output_dir: Path, *, phase: str,
) -> Path:
    """Save one same-request set of full-frame, undistorted camera images."""
    if phase not in PHASES:
        raise ValueError("unknown raw camera capture phase")
    if not isinstance(capture, RawCameraCapture):
        raise TypeError("capture must be a RawCameraCapture")
    _safe_id(capture.capture_id, "raw capture ID")
    if type(capture.request_id) is not int or capture.request_id <= 0:
        raise ValueError("raw capture needs a positive request ID")
    if len(capture.images_bgr) < 2:
        raise ValueError("raw capture needs at least two calibrated views")
    for camera in capture.images_bgr:
        _safe_id(camera, "camera ID")
    evidence = verify_frame_provenance(
        capture.acquisition_metadata, request_id=capture.request_id,
        images_bgr=capture.images_bgr, frame_ids=capture.frame_ids)
    derived = (capture.source_images_bgr is not None or
               capture.source_acquisition_metadata is not None or
               capture.undistort_calibration is not None)
    source_evidence = None
    calibrations = None
    if derived:
        if (capture.source_images_bgr is None or
                capture.source_acquisition_metadata is None or
                capture.undistort_calibration is None or
                set(capture.source_images_bgr) != set(capture.images_bgr) or
                set(capture.undistort_calibration) != set(capture.images_bgr)):
            raise ValueError("undistorted capture needs all original frames and calibration")
        source_evidence = verify_frame_provenance(
            capture.source_acquisition_metadata,
            request_id=capture.request_id,
            images_bgr=capture.source_images_bgr,
            frame_ids=capture.frame_ids)
        calibrations = {}
        for camera, image in capture.images_bgr.items():
            if any(evidence[camera][key] != source_evidence[camera][key]
                   for key in ("frame_id", "timestamp_s", "max_error_s",
                               "timestamp_method", "clock_domain")):
                raise ValueError("derived frame changed source exposure identity")
            row = capture.undistort_calibration[camera]
            original = np.asarray(row["K_orig"], dtype=float)
            target = np.asarray(row["K_undist"], dtype=float)
            distortion = np.asarray(row["dist_params"], dtype=float).reshape(-1)
            if (original.shape != (3, 3) or target.shape != (3, 3) or
                    not np.all(np.isfinite(original)) or
                    not np.all(np.isfinite(target)) or
                    not np.all(np.isfinite(distortion))):
                raise ValueError("invalid undistortion calibration")
            # The snapshot producer has already performed this transform.
            # Offline verify_raw_camera_capture replays it from the saved
            # original PNG, so the live writer need not repeat it here.
            calibrations[camera] = {
                "K_orig": original.tolist(), "K_undist": target.tolist(),
                "dist_params": distortion.tolist()}
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    image_dir = target / "images"
    image_dir.mkdir()
    jobs = [(camera, image_dir / f"{camera}.png", image, "undistorted")
            for camera, image in sorted(capture.images_bgr.items())]
    if derived:
        source_dir = target / "source_original"
        source_dir.mkdir()
        jobs.extend((camera, source_dir / f"{camera}.png", image, "original")
                    for camera, image in sorted(capture.source_images_bgr.items()))

    results, _workers = write_png_jobs(jobs, phase=phase)
    hashes = {camera: digest for camera, source, digest in results
              if source == "undistorted"}
    source_hashes = ({camera: digest for camera, source, digest in results
                      if source == "original"} if derived else None)
    manifest = {
        "schema": "precision_insertion_raw_capture_v1",
        "phase": phase,
        "capture_id": capture.capture_id,
        "request_id": capture.request_id,
        "image_space": "autodex_undistorted_full_frame",
        "frame_evidence": evidence,
        "image_file_sha256": hashes,
        "scope": "raw_camera_pixels_not_object_pose_or_motion",
        "robot_ready": False,
    }
    if derived:
        manifest["undistort_derivation"] = {
            "source_frame_evidence": source_evidence,
            "source_image_file_sha256": source_hashes,
            "calibration": calibrations,
        }
    if capture.temporal_window is not None:
        window = capture.temporal_window
        if (window.camera_id not in capture.frame_ids or
                window.target_frame_id != capture.frame_ids[window.camera_id]):
            raise ValueError("temporal window differs from checkpoint frame")
        with timed("evidence.temporal_window_write", phase="evidence",
                   kind="io", camera_id=window.camera_id,
                   capture_phase=phase, frame_count=len(window.frames)):
            save_tip_frame_window(window, target)
        manifest["temporal_window"] = {
            "camera_id": window.camera_id,
            "target_frame_id": window.target_frame_id,
            "status": window.status,
            "manifest": "temporal_window/manifest.json",
        }
    with (target / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    remember("raw_camera", target, phase, manifest)
    return target


def verify_raw_camera_capture(output_dir: Path, *, phase: str) -> dict:
    """Recheck source PNG bytes, decoded pixels, frame IDs and exposure data."""
    if phase not in PHASES:
        raise ValueError("unknown raw camera capture phase")
    root = Path(output_dir).expanduser().resolve()
    cached = recalled("raw_camera", root, phase)
    if cached is not None:
        with timed("evidence.live_manifest_reuse", phase="evidence",
                   kind="io", capture_phase=phase):
            return cached
    report = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(report, dict) or
            report.get("schema") != "precision_insertion_raw_capture_v1" or
            report.get("phase") != phase or
            report.get("image_space") != "autodex_undistorted_full_frame" or
            not isinstance(report.get("frame_evidence"), dict) or
            len(report["frame_evidence"]) < 2 or
            set(report["frame_evidence"]) !=
            set(report.get("image_file_sha256", {}))):
        raise ValueError("invalid raw capture manifest")
    images, frame_ids = {}, {}
    for camera, source in report["frame_evidence"].items():
        path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        with timed("evidence.png_read_verify", phase="evidence", kind="io",
                   camera_id=camera, capture_phase=phase, source="undistorted"):
            if not path.is_file() or _sha(path) != report["image_file_sha256"][camera]:
                raise ValueError(("final " if phase == "final_or_abort" else "") +
                                 "raw camera PNG changed")
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("raw camera PNG cannot be decoded")
        images[camera] = image
        frame_ids[camera] = source["frame_id"]
    expected = verify_frame_provenance(
        {"request_id": report["request_id"],
         "source": "camera_acquisition", "frames": report["frame_evidence"]},
        request_id=report["request_id"], images_bgr=images,
        frame_ids=frame_ids)
    if expected != report["frame_evidence"]:
        raise ValueError("raw frame evidence changed")
    derivation = report.get("undistort_derivation")
    if derivation is not None:
        if (not isinstance(derivation, dict) or
                set(derivation) != {"source_frame_evidence",
                                    "source_image_file_sha256", "calibration"} or
                any(set(derivation[key]) != set(images) for key in derivation)):
            raise ValueError("invalid original snapshot derivation")
        originals = {}
        for camera, image in images.items():
            source = root / "source_original" / f"{camera}.png"
            with timed("evidence.png_read_verify", phase="evidence", kind="io",
                       camera_id=camera, capture_phase=phase, source="original"):
                if (not source.is_file() or _sha(source) !=
                        derivation["source_image_file_sha256"][camera]):
                    raise ValueError("original snapshot PNG changed")
                original = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if original is None:
                raise ValueError("original snapshot PNG cannot be decoded")
            originals[camera] = original
            row = derivation["calibration"][camera]
            with timed("evidence.undistort_replay", phase="evidence",
                       kind="image", camera_id=camera, capture_phase=phase):
                rebuilt = cv2.undistort(
                    original, np.asarray(row["K_orig"], dtype=float),
                    np.asarray(row["dist_params"], dtype=float), None,
                    np.asarray(row["K_undist"], dtype=float))
            if not np.array_equal(rebuilt, image):
                raise ValueError("raw frame differs from original undistortion")
        source_evidence = verify_frame_provenance(
            {"request_id": report["request_id"],
             "source": "camera_acquisition",
             "frames": derivation["source_frame_evidence"]},
            request_id=report["request_id"], images_bgr=originals,
            frame_ids=frame_ids)
        if source_evidence != derivation["source_frame_evidence"]:
            raise ValueError("original snapshot exposure identity changed")
        for camera in images:
            if any(source_evidence[camera][key] != expected[camera][key]
                   for key in ("frame_id", "timestamp_s", "max_error_s",
                               "timestamp_method", "clock_domain")):
                raise ValueError("raw exposure differs from source snapshot")
    return report
