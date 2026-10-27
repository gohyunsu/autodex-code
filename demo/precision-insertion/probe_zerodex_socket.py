#!/usr/bin/env python3
"""Save four-view SAM3 overlays for the fixed socket or loose key.

Camera-only diagnostic: no Franka or Inspire connection. A low body IoU is
reported with the images, not hidden by changing the configured threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from precision_insertion.assets import AssetPaths
from precision_insertion.board_socket_fit import fit_board_socket_body
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.live_capture import (
    collect_board_snapshot, collect_socket_capture,
)
from precision_insertion.session_launch import (
    load_session_launch, open_session_cameras,
)
from src.execution.charuco_tabletop import measure_tabletop_from_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("socket", "key"), default="socket")
    parser.add_argument("--prompt")
    parser.add_argument("--foundpose", action="store_true",
                        help="also run FoundPose; default is fast SAM3 mask-only")
    args = parser.parse_args(argv)
    output = args.evidence_dir.expanduser().resolve()
    if output.exists():
        parser.error("evidence directory already exists")
    launch = load_session_launch(args.launch_config, evidence_dir_override=output)
    output.mkdir(parents=True)
    if launch.board_id != "9" or launch.mode.family != "cylinder":
        parser.error("this probe is for the board-9 cylindrical socket")
    paths = AssetPaths(launch.shared_root, launch.mode)
    target_object = (launch.mode.socket_object if args.target == "socket"
                     else launch.mode.key_object)
    prompt = args.prompt or (launch.socket_prompt if args.target == "socket"
                             else "white plastic cylindrical key")
    with open_autodex_camera_stream(
            launch.pc_list, capture_ips=launch.capture_ips,
            name="precision_zerodex_socket_probe", warmup_s=1,
            acquisition_mode=launch.camera_acquisition_mode):
        with open_session_cameras(
                launch, init_ports=(15006, 15007, 16893)) as cameras:
            board = collect_board_snapshot(
                snapshot_orchestrator=cameras.snapshot_orchestrator,
                calibrated_camera_ids=launch.camera_ids,
                acquisition_metadata_for_request=(
                    cameras.acquisition_metadata_for_request),
                timeout_s=launch.board_timeout_s)
            for serial, frame in board.images_bgr.items():
                cv2.imwrite(str(output / f"{serial}_board.jpg"), frame)
            measured = measure_tabletop_from_images(
                board.images_bgr, launch.intrinsics_full,
                launch.extrinsics_full, launch.c2r, board_id="9",
                minimum_board_corners=launch.minimum_board_corners)
            init = cameras.init_orchestrator
            init.init_object(
                obj_name=target_object,
                mesh_path=str(paths.raw_mesh(target_object)),
                assets_root=str(paths.foundpose_assets_root(
                    target_object)),
                intrinsics_full=launch.intrinsics_full,
                extrinsics_full=launch.extrinsics_full,
                image_hw=launch.image_hw, mode="live",
                load_silhouette=False, mask_only=not args.foundpose)
            capture = collect_socket_capture(
                init_orchestrator=init,
                socket_object=target_object,
                capture_id=args.target + "_probe_" + uuid4().hex[:12],
                socket_prompt=prompt, capture_root=launch.capture_root,
                calibrated_camera_ids=launch.camera_ids,
                acquisition_metadata_for_request=(
                    cameras.acquisition_metadata_for_request),
                timeout_s=launch.socket_timeout_s,
                image_write_timeout_s=launch.image_write_timeout_s,
                mask_only=not args.foundpose)
            stock = init.stock
            usable = {serial: row["mask"] for serial, row in
                      capture.masks.items() if np.any(row["mask"])}
            fit = None
            if args.target == "socket" and len(usable) >= 2:
                _, fit = fit_board_socket_body(
                    masks=usable,
                    intrinsics_undist={serial: stock.intrinsics_undist[serial]
                                       for serial in usable},
                    extrinsics_camera_world={serial: stock.extrinsics[serial]
                                             for serial in usable},
                    c2r=launch.c2r,
                    board_pose_robot=np.asarray(measured["T_robot_board"]),
                    raw_mesh=paths.raw_mesh(launch.mode.socket_object))
            views = {}
            overlay_tiles = []
            for serial, frame in capture.images_bgr.items():
                mask = np.asarray(capture.masks[serial]["mask"], dtype=bool)
                overlay = frame.copy()
                overlay[mask] = (0.55 * overlay[mask] +
                                 0.45 * np.array([0, 255, 0])).astype(np.uint8)
                cv2.imwrite(str(output / f"{serial}_mask.png"),
                            mask.astype(np.uint8) * 255)
                cv2.imwrite(str(output / f"{serial}_overlay.jpg"), overlay)
                views[serial] = {"mask_pixels": int(mask.sum()),
                                 "body_iou": (None if fit is None else
                                              fit["per_view_iou"].get(serial)),
                                 "foundpose_ok": bool(capture.poses[serial].get("ok")),
                                 "foundpose_quality": capture.poses[serial].get("quality")}
                overlay_tiles.append((serial, cv2.resize(overlay, (1024, 768))))
            if len(overlay_tiles) == 4:
                tiles = dict(overlay_tiles)
                order = ("26053248", "25322639", "25305462", "25322642")
                grid = np.vstack((np.hstack((tiles[order[0]], tiles[order[1]])),
                                  np.hstack((tiles[order[2]], tiles[order[3]]))))
                cv2.imwrite(str(output / "overlays_4view.jpg"), grid)
            report = {
                "scope": "camera_only_" + args.target + "_diagnostic",
                "target_object": target_object,
                "prompt": prompt, "board_request_id": board.request_id,
                "board_corner_counts": measured.get("camera_corner_counts"),
                "socket_request_id": capture.request_id,
                "body_fit": fit, "views": views,
                "configured_minimum_body_iou": launch.minimum_socket_refined_iou,
            }
            (output / "report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            print(json.dumps({"report": str(output / "report.json"),
                              "mean_body_iou": (None if fit is None else
                                                fit["mean_body_iou"]),
                              "views": views}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
