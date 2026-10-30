#!/usr/bin/env python3
"""Measure capture4's same-request inference path without board or robot.

The default checks lossless source PNG transport; ``--pose-only`` measures the
normal per-trial mask/FoundPose request without shipping source PNGs. This is
not a key/socket-pose or task-success validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.live_capture import collect_socket_capture
from precision_insertion.session_launch import load_session_launch, open_session_cameras


def _frame_brightness(image_bgr: np.ndarray) -> dict[str, float]:
    """Report illumination evidence; never turn it into a capture gate."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return {"gray_mean": float(gray.mean()),
            "gray_p99": float(np.percentile(gray, 99))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("key", "socket"), default="key")
    parser.add_argument("--prompt", help="override the SAM3 text prompt for camera-only comparison")
    parser.add_argument("--pose-only", action="store_true",
                        help="request masks/FoundPose without source PNGs")
    parser.add_argument("--with-pose-images", action="store_true",
                        help="save same-request source frames and masks with FoundPose")
    parser.add_argument("--board-fit-session", type=Path,
                        help="also fit an upright key to the saved session board")
    args = parser.parse_args(argv)
    if args.pose_only and args.with_pose_images:
        parser.error("choose either --pose-only or --with-pose-images")
    output = args.evidence_dir.expanduser().resolve()
    if output.exists():
        parser.error("evidence directory already exists")
    launch = load_session_launch(args.launch_config, evidence_dir_override=output)
    if launch.init_frame_transport != "inline_png":
        parser.error("the launch does not select inline PNG transport")
    output.mkdir(parents=True)
    object_name = (launch.mode.key_object if args.target == "key"
                   else launch.mode.socket_object)
    prompt = args.prompt or ("white plastic cylindrical key" if args.target == "key"
                             else launch.socket_prompt)
    paths = AssetPaths(launch.shared_root, launch.mode)
    with open_autodex_camera_stream(
            launch.pc_list, capture_ips=launch.capture_ips,
            name="precision_inline_transport_probe", warmup_s=1,
            acquisition_mode=launch.camera_acquisition_mode):
        with open_session_cameras(
                launch, init_ports=(15006, 15007, 16893)) as cameras:
            start = time.perf_counter()
            cameras.init_orchestrator.init_object(
                obj_name=object_name, mesh_path=str(paths.raw_mesh(object_name)),
                assets_root=str(paths.foundpose_assets_root(object_name)),
                intrinsics_full=launch.intrinsics_full,
                extrinsics_full=launch.extrinsics_full,
                image_hw=launch.image_hw, mode="live",
                load_silhouette=False,
                mask_only=not (args.pose_only or args.with_pose_images))
            init_s = time.perf_counter() - start
            start = time.perf_counter()
            capture = collect_socket_capture(
                init_orchestrator=cameras.init_orchestrator,
                socket_object=object_name,
                capture_id="inline_probe_" + uuid4().hex[:12],
                socket_prompt=prompt, capture_root=launch.capture_root,
                calibrated_camera_ids=launch.camera_ids,
                acquisition_metadata_for_request=(
                    cameras.acquisition_metadata_for_request),
                timeout_s=launch.socket_timeout_s,
                image_write_timeout_s=launch.image_write_timeout_s,
                pose_only=args.pose_only,
                mask_only=not (args.pose_only or args.with_pose_images))
            capture_s = time.perf_counter() - start
            board_fit = None
            if args.board_fit_session is not None:
                from precision_insertion.board_socket_fit import fit_board_socket_body
                calibration = json.loads((args.board_fit_session /
                                          "session_calibration.json").read_text())
                fitted = {serial: row["mask"] for serial, row in
                          capture.masks.items() if np.asarray(row["mask"]).any()}
                pose, diagnostic = fit_board_socket_body(
                    masks=fitted,
                    intrinsics_undist={serial: cameras.init_orchestrator.intrinsics_undist[serial]
                                       for serial in fitted},
                    extrinsics_camera_world={serial: cameras.init_orchestrator.extrinsics[serial]
                                             for serial in fitted},
                    c2r=np.asarray(calibration["c2r"]),
                    board_pose_robot=np.asarray(calibration["board"]["T_robot_board"]),
                    raw_mesh=paths.raw_mesh(object_name))
                board_fit = {"pose_world": pose.tolist(), "diagnostic": diagnostic}
    if args.with_pose_images:
        for serial, image in capture.images_bgr.items():
            mask = np.asarray(capture.masks[serial]["mask"], dtype=bool)
            overlay = image.copy()
            overlay[mask] = (0.65 * image[mask] +
                             0.35 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.imwrite(str(output / f"{serial}_mask.png"),
                        (mask.astype(np.uint8) * 255))
            cv2.imwrite(str(output / f"{serial}_overlay.png"), overlay)
    report = {
        "scope": ("pose_only_camera_request_no_board_no_robot" if args.pose_only
                  else "pose_and_frames_camera_request_no_robot" if args.with_pose_images
                  else "camera_transport_only_no_board_no_robot"),
        "request_id": capture.request_id,
        "target": args.target,
        "init_s": init_s,
        "capture_request_s": capture_s,
        "capture_and_local_png_s": None if args.pose_only else capture_s,
        "camera_count": len(capture.masks),
        "board_fit": board_fit,
        "raw_png_dir": (None if args.pose_only else str(
            launch.capture_root /
            f"{capture.capture_id}_request_{capture.request_id}" / "images")),
        "frames": {serial: {
            "frame_id": capture.masks[serial]["frame_id"],
            "image_sha256": capture.masks[serial]["image_sha256"],
            "mask_pixels": int(capture.masks[serial]["mask"].sum()),
            "pose_ok": bool(capture.poses[serial].get("ok", False)),
            "pose_error": capture.poses[serial].get("error"),
            **({} if args.pose_only else
               _frame_brightness(capture.images_bgr[serial])),
        } for serial in sorted(capture.masks)},
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
