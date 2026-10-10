#!/usr/bin/env python3
"""Read-only local-VLM metric-alignment probe on saved, calibrated views.

This is an offline diagnostic, not a live capture, robot controller or
calibration certificate. Manifest timestamps and transforms are claims made by
the operator; only a commissioned acquisition-bound session can use them for
robot decisions.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.geometry import validate_se3  # noqa: E402
from precision_insertion.grounded_alignment import (  # noqa: E402
    AlignmentLimits, estimate_grounded_line_alignment,
    observe_grounded_cylinder_axis,
)
from precision_insertion.observer import load_vlm_backend  # noqa: E402
from precision_insertion.xy_overlay import CalibratedXYFrame  # noqa: E402


SCHEMA = "precision_insertion_saved_grounding_probe_v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CAMERA = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _source(path_value: str, root: Path, expected_sha: str) -> tuple[Path, bytes]:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError("each image needs a path")
    if not isinstance(expected_sha, str) or not _DIGEST.fullmatch(expected_sha):
        raise ValueError("each image needs a lowercase SHA-256")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError(f"saved image changed: {path}")
    return path, raw


def _load_manifest(
    path: Path, raw_manifest: bytes,
) -> tuple[dict, list[CalibratedXYFrame], list[dict]]:
    record = json.loads(raw_manifest)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        raise ValueError("unknown saved grounding manifest schema")
    rows = record.get("views")
    if not isinstance(rows, list) or len(rows) < 2:
        raise ValueError("saved grounding needs at least two camera views")
    skew_limit = record.get("max_camera_skew_s")
    if (type(skew_limit) not in (int, float) or
            not math.isfinite(skew_limit) or skew_limit <= 0):
        raise ValueError("positive camera-exposure skew limit is required")
    limit_record = record.get("alignment_limits")
    if (not isinstance(limit_record, dict) or
            set(limit_record) != {field.name for field in fields(AlignmentLimits)}):
        raise ValueError("explicit commissioned alignment limits are required")
    limits = AlignmentLimits(**limit_record)
    limits.validate()
    for field_name in ("socket_rim_z_m", "verification_depth_m"):
        value = record.get(field_name)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"finite {field_name} is required")
    if record["verification_depth_m"] <= 0:
        raise ValueError("positive verification depth is required")

    frames, sources, camera_ids, times = [], [], set(), []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid camera view")
        camera_id = row.get("camera_id")
        if (not isinstance(camera_id, str) or
                not _CAMERA.fullmatch(camera_id) or camera_id in camera_ids):
            raise ValueError("unique path-safe camera IDs are required")
        camera_ids.add(camera_id)
        image_record = row.get("image")
        if not isinstance(image_record, dict):
            raise ValueError("camera view lacks image provenance")
        timestamp = image_record.get("timestamp_s")
        if (type(timestamp) not in (int, float) or
                not math.isfinite(timestamp) or timestamp <= 0):
            raise ValueError("each view needs a claimed acquisition timestamp")
        times.append(float(timestamp))
        image_path, raw = _source(
            image_record.get("path"), path.parent,
            image_record.get("sha256"))
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert("RGB")
        transform = validate_se3(
            row.get("T_camera_socket"), name=f"{camera_id} T_camera_socket")
        intrinsics = np.asarray(row.get("intrinsics"), dtype=float)
        if (intrinsics.shape != (3, 3) or
                not np.all(np.isfinite(intrinsics)) or
                intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or
                not np.allclose(intrinsics[2], [0, 0, 1], atol=1e-9)):
            raise ValueError(f"invalid intrinsics for {camera_id}")
        frames.append(CalibratedXYFrame(
            camera_id, float(timestamp), rgb, transform, intrinsics))
        sources.append({
            "camera_id": camera_id, "path": str(image_path),
            "sha256": image_record["sha256"],
            "timestamp_s_claimed": float(timestamp),
            "size_px": list(rgb.size),
        })
    if len(frames) < limits.minimum_views:
        raise ValueError("fewer views than the alignment limit requires")
    if max(times) - min(times) > skew_limit:
        raise ValueError("claimed camera exposures exceed skew limit")
    return record, frames, sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="slow offline smoke test only")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON report; refuses overwrite")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    source = args.manifest.expanduser().resolve()
    target = args.output.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"report already exists: {target}")
    raw_manifest = source.read_bytes()
    manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    record, frames, sources = _load_manifest(source, raw_manifest)
    backend = load_vlm_backend(
        mode="local", model_id=args.model_id,
        max_input_size=(max(frame.raw.width for frame in frames),
                        max(frame.raw.height for frame in frames)),
        max_new_tokens=args.max_new_tokens,
        require_cuda=not args.allow_cpu, require_native_pixels=True)
    grounded, observations = observe_grounded_cylinder_axis(backend, frames)
    limits = AlignmentLimits(**record["alignment_limits"])
    alignment = estimate_grounded_line_alignment(
        grounded, socket_rim_z_m=float(record["socket_rim_z_m"]),
        verification_depth_m=float(record["verification_depth_m"]),
        limits=limits)
    report = {
        "schema": "precision_insertion_saved_grounding_report_v1",
        "backend": "local", "model_id": args.model_id,
        "manifest_path": str(source), "manifest_sha256": manifest_sha,
        "source_images": sources,
        "observations": [observation.to_record() for observation in observations],
        "alignment": alignment,
        "timestamp_source": "manifest_claim_not_verified_acquisition_evidence",
        "calibration_source": "manifest_claim_not_commissioned_live_session",
        "scope": "saved_image_metric_diagnostic_not_robot_motion",
        "robot_ready": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
