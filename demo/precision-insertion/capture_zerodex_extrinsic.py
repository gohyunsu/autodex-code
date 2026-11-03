#!/usr/bin/env python3
"""Capture project-owned four-view ChArUco extrinsic observations.

The shared ParaDex ``system/current`` and calibration directories are read by
neither this tool nor its board detector. Camera transport uses the already
namespaced precision-insertion capture4 services.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from calibration.charuco import detect, draw_detection, load_board
from calibration.monitor import (
    append_capture_centres,
    load_capture_centres,
    preview_grid,
    TkMonitor,
)
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.live_capture import collect_board_snapshot
from precision_insertion.session_launch import load_session_launch, open_session_cameras


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _next_index(session_dir: Path) -> int:
    indices = [int(path.name) for path in session_dir.iterdir()
               if path.is_dir() and path.name.isdigit()]
    return 0 if not indices else max(indices) + 1


def _save_capture(session_dir: Path, index: int, snapshot, detector) -> dict:
    capture_dir = session_dir / f"{index:04d}"
    image_dir = capture_dir / "images"
    marker_dir = capture_dir / "markers_2d"
    overlay_dir = capture_dir / "overlays"
    for path in (image_dir, marker_dir, overlay_dir):
        path.mkdir(parents=True, exist_ok=False)
    views = {}
    for serial, image in sorted(snapshot.images_bgr.items()):
        corners, ids = detect(detector, image)
        cv2.imwrite(str(image_dir / f"{serial}.png"), image)
        cv2.imwrite(str(overlay_dir / f"{serial}.png"),
                    draw_detection(image, corners, ids))
        np.save(marker_dir / f"{serial}_corner.npy", corners)
        np.save(marker_dir / f"{serial}_id.npy", ids)
        views[serial] = {
            "corner_count": int(len(ids)),
            "corner_ids": ids.tolist(),
            "frame_evidence": snapshot.frame_evidence[serial],
        }
    record = {
        "capture_index": index,
        "request_id": snapshot.request_id,
        "views": views,
    }
    _write_json(capture_dir / "capture.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--board-config", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true",
                        help="capture one observation without an interactive loop")
    parser.add_argument("--monitor", action="store_true",
                        help="show a refreshing 4-view overlay; c saves and q exits")
    parser.add_argument("--preview-tile-width", type=int, default=640,
                        help="display width per camera; source images stay full resolution")
    args = parser.parse_args(argv)
    if args.once and args.monitor:
        parser.error("--once and --monitor are mutually exclusive")
    if args.preview_tile_width < 240:
        parser.error("--preview-tile-width must be at least 240 pixels")
    session_dir = args.session_dir.expanduser().resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    board_info, _, detector = load_board(args.board_config)
    manifest = session_dir / "session.json"
    if not manifest.exists():
        _write_json(manifest, {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "launch_config": str(args.launch_config.expanduser().resolve()),
            "board_config": str(args.board_config.expanduser().resolve()),
            "board": board_info,
        })
    # SessionLaunch requires a path that has not already been used as runtime
    # evidence. This token is connection metadata only and is never created.
    evidence_token = session_dir / ("_connection_" + uuid4().hex)
    launch = load_session_launch(
        args.launch_config, evidence_dir_override=evidence_token)
    history = load_capture_centres(session_dir, launch.camera_ids)
    monitor = (TkMonitor(
        "Precision insertion extrinsic | c: capture | q: quit")
        if args.monitor else None)
    with open_autodex_camera_stream(
            launch.pc_list, capture_ips=launch.capture_ips,
            name="precision_extrinsic_capture", warmup_s=1,
            acquisition_mode=launch.camera_acquisition_mode):
        with open_session_cameras(launch) as cameras:
            while True:
                if not args.once and not args.monitor:
                    answer = input(
                        "Move the board to a new shared view, then press Enter "
                        "to capture (q to finish): ").strip().lower()
                    if answer == "q":
                        break
                snapshot = collect_board_snapshot(
                    snapshot_orchestrator=cameras.snapshot_orchestrator,
                    calibrated_camera_ids=launch.camera_ids,
                    acquisition_metadata_for_request=(
                        cameras.acquisition_metadata_for_request),
                    timeout_s=launch.board_timeout_s)
                if args.monitor:
                    grid, _counts = preview_grid(
                        snapshot, detector,
                        tile_width=args.preview_tile_width,
                        history=history)
                    command = monitor.show(grid)
                    if command == "quit":
                        break
                    if command != "capture":
                        continue
                record = _save_capture(
                    session_dir, _next_index(session_dir), snapshot, detector)
                append_capture_centres(
                    history, record["capture_index"], snapshot, detector)
                counts = {serial: row["corner_count"]
                          for serial, row in record["views"].items()}
                print(json.dumps({"capture": record["capture_index"],
                                  "corners": counts}, sort_keys=True))
                if args.once:
                    break
    if monitor is not None:
        monitor.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
