#!/usr/bin/env python3
"""Replay the demo-local key ROI on saved AutoDex frames without robot motion.

An optional saved evaluator mask is *only* audited against the new ROI. This
script does not rerun SAM3, FoundPose, or a live camera request. Its JSON is
an exploratory diagnostic and never a key-pose or grasp-success label.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.registered_key_roi import (  # noqa: E402
    RegisteredKeyROILimits, detector_from_board_config,
    propose_registered_key_roi,
    segment_registered_key_roi,
)


def _read_image(path: Path, flag: int) -> tuple[np.ndarray, dict]:
    source = path.expanduser().resolve()
    raw = source.read_bytes()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flag)
    if image is None:
        raise ValueError(f"cannot decode source image: {source}")
    return image, {"path": str(source),
                   "sha256": hashlib.sha256(raw).hexdigest()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-jpeg", type=Path, required=True)
    parser.add_argument("--socket-only-jpeg", type=Path, required=True)
    parser.add_argument("--socket-only-mask", type=Path, required=True,
                        help="same-camera undistorted mask from baseline")
    parser.add_argument("--saved-evaluator-mask", type=Path,
                        help="optional same-frame full undistorted SAM mask QA")
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--board-id", default="9")
    parser.add_argument("--board-config-json", type=Path, required=True,
                        help="explicit AutoDex rig ChArUco config; never uses system/current")
    parser.add_argument("--limits-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"diagnostic already exists: {output}")
    current, current_ref = _read_image(args.current_jpeg, cv2.IMREAD_COLOR)
    baseline, baseline_ref = _read_image(
        args.socket_only_jpeg, cv2.IMREAD_COLOR)
    socket_mask, socket_ref = _read_image(
        args.socket_only_mask, cv2.IMREAD_GRAYSCALE)
    calib = args.calib_dir.expanduser().resolve() / "intrinsics.json"
    calib_bytes = calib.read_bytes()
    row = json.loads(calib_bytes).get(args.camera_id)
    if not isinstance(row, dict):
        raise ValueError("camera is absent from stock intrinsics")
    h, w = row.get("height"), row.get("width")
    if (type(h) is not int or type(w) is not int or
            current.shape != (h, w, 3) or baseline.shape != current.shape or
            socket_mask.shape != (h, w)):
        raise ValueError("saved frames and mask differ from calibration geometry")
    original = np.asarray(row["original_intrinsics"], dtype=np.float64).reshape(3, 3)
    undistorted = np.asarray(row["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(row["dist_params"], dtype=np.float64).reshape(-1)
    mapx, mapy = cv2.initUndistortRectifyMap(
        original, distortion, None, undistorted, (w, h), cv2.CV_16SC2)
    current = cv2.remap(current, mapx, mapy, cv2.INTER_LINEAR)
    baseline = cv2.remap(baseline, mapx, mapy, cv2.INTER_LINEAR)
    limits_path = args.limits_json.expanduser().resolve()
    limits_bytes = limits_path.read_bytes()
    limits_record = json.loads(limits_bytes)
    if (not isinstance(limits_record, dict) or
            limits_record.get("schema") !=
            "precision_insertion_registered_key_roi_limits_v1" or
            not isinstance(limits_record.get("limits"), dict) or
            set(limits_record["limits"]) !=
            {field.name for field in fields(RegisteredKeyROILimits)}):
        raise ValueError("unknown or incomplete ROI limits")
    limits = RegisteredKeyROILimits(**limits_record["limits"])
    limits.validate()
    board_config = args.board_config_json.expanduser().resolve()
    board_config_bytes = board_config.read_bytes()
    detect_charuco = detector_from_board_config(
        board_config_path=board_config, board_id=args.board_id)

    proposal = propose_registered_key_roi(
        current_bgr=current, socket_only_bgr=baseline,
        socket_only_mask=socket_mask, board_detector=detect_charuco,
        board_id=args.board_id, limits=limits)
    qa = None
    sources = {"current_jpeg": current_ref,
               "socket_only_jpeg": baseline_ref,
               "socket_only_mask": socket_ref,
               "intrinsics": {"path": str(calib),
                              "sha256": hashlib.sha256(calib_bytes).hexdigest()},
               "limits": {"path": str(limits_path),
                          "sha256": hashlib.sha256(limits_bytes).hexdigest()},
               "board_config": {
                   "path": str(board_config),
                   "sha256": hashlib.sha256(board_config_bytes).hexdigest()}}
    if args.saved_evaluator_mask is not None:
        saved, saved_ref = _read_image(
            args.saved_evaluator_mask, cv2.IMREAD_GRAYSCALE)
        if saved.shape != (h, w):
            raise ValueError("saved evaluator mask has wrong full-frame geometry")
        sources["saved_evaluator_mask"] = saved_ref

        class SavedMask:
            def segment(self, crop, _prompt):
                x0, y0, x1, y1 = proposal.crop_xyxy
                if crop.shape[:2] != (y1 - y0, x1 - x0):
                    raise ValueError("saved-mask crop differs from ROI")
                return saved[y0:y1, x0:x1] > 0

        qa = segment_registered_key_roi(
            current_rgb=cv2.cvtColor(current, cv2.COLOR_BGR2RGB),
            proposal=proposal, sam_segmentor=SavedMask(),
            prompt="saved evaluator mask QA only", limits=limits).to_record()
    report = {
        "schema": "precision_insertion_saved_registered_key_roi_probe_v1",
        "camera_id": args.camera_id, "board_id": args.board_id,
        "sources": sources, "proposal": proposal.to_record(),
        "saved_mask_qa": qa,
        "source_relationship": "caller_selected_saved_files_not_session_authenticated",
        "scope": "read_only_saved_image_roi_diagnostic_not_live_perception",
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
