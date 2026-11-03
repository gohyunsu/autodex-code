#!/usr/bin/env python3
"""Capture stationary FR3/ChArUco observations for project hand-eye solve.

This tool never commands robot motion. The operator places the robot at each
pose, waits for it to stop, and presses ``c`` in the four-view monitor. Camera
frames plus Franka state immediately before and after the snapshot are saved
under the project NAS namespace.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import cv2
import numpy as np

from calibration.charuco import detect, draw_detection, load_board
from calibration.monitor import (
    append_capture_points,
    load_capture_points,
    preview_grid,
    TkMonitor,
)
from paradex.io.robot_controller.franka_controller import FrankaController
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.live_capture import collect_board_snapshot
from precision_insertion.session_launch import load_session_launch, open_session_cameras


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def _state(controller: FrankaController) -> dict:
    state = controller.get_data()
    if state is None:
        raise RuntimeError("Franka daemon returned no state")
    return {
        "sample_wall_time_s": time.time(),
        "robot_time_s": float(state["time"]),
        "qpos": np.asarray(state["qpos"], dtype=float).tolist(),
        "qvel": np.asarray(state["qvel"], dtype=float).tolist(),
        "T_robot_ee": np.asarray(state["position"], dtype=float).tolist(),
        "wrench": np.asarray(state["wrench"], dtype=float).tolist(),
    }


def _next(session: Path) -> int:
    values = [int(path.name) for path in session.iterdir()
              if path.is_dir() and path.name.isdigit()]
    return 0 if not values else max(values) + 1


def _save(session: Path, index: int, snapshot, detector,
          before: dict, after: dict) -> dict:
    root = session / f"{index:04d}"
    images = root / "images"
    markers = root / "markers_2d"
    overlays = root / "overlays"
    for path in (images, markers, overlays):
        path.mkdir(parents=True, exist_ok=False)
    views = {}
    for serial, image in sorted(snapshot.images_bgr.items()):
        corners, ids = detect(detector, image)
        cv2.imwrite(str(images / f"{serial}.png"), image)
        cv2.imwrite(str(overlays / f"{serial}.png"),
                    draw_detection(image, corners, ids))
        np.save(markers / f"{serial}_corner.npy", corners)
        np.save(markers / f"{serial}_id.npy", ids)
        views[serial] = {
            "corner_count": int(len(ids)),
            "corner_ids": ids.tolist(),
            "frame_evidence": snapshot.frame_evidence[serial],
        }
    record = {
        "capture_index": index,
        "request_id": snapshot.request_id,
        "robot_before": before,
        "robot_after": after,
        "views": views,
    }
    _json(root / "capture.json", record)
    _export_trajectory(session)
    return record


def _export_trajectory(session: Path) -> None:
    """Refresh a replayable waypoint file after every successful capture."""
    rows = []
    for capture_dir in sorted(
            path for path in session.iterdir()
            if path.is_dir() and path.name.isdigit()):
        record_path = capture_dir / "capture.json"
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        state = record["robot_after"]
        rows.append({
            "capture_index": int(record["capture_index"]),
            "qpos": state["qpos"],
            "T_robot_ee": state["T_robot_ee"],
        })
    _json(session / "trajectory.json", {
        "schema_version": 1,
        "source": "stationary_handeye_captures",
        "waypoint_count": len(rows),
        "waypoints": rows,
    })
    if rows:
        np.save(session / "trajectory_qpos.npy", np.asarray(
            [row["qpos"] for row in rows], dtype=np.float64))
        np.save(session / "trajectory_T_robot_ee.npy", np.asarray(
            [row["T_robot_ee"] for row in rows], dtype=np.float64))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--board-config", type=Path, required=True)
    parser.add_argument("--camera-calibration-dir", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--daemon-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5555)
    parser.add_argument("--state-port", type=int, default=5556)
    parser.add_argument("--preview-tile-width", type=int, default=640)
    args = parser.parse_args(argv)
    if args.preview_tile_width < 240:
        parser.error("--preview-tile-width must be at least 240 pixels")
    session = args.session_dir.expanduser().resolve()
    session.mkdir(parents=True, exist_ok=True)
    board_info, _, detector = load_board(args.board_config)
    manifest = session / "session.json"
    if not manifest.exists():
        _json(manifest, {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "launch_config": str(args.launch_config.expanduser().resolve()),
            "board_config": str(args.board_config.expanduser().resolve()),
            "camera_calibration_dir": str(
                args.camera_calibration_dir.expanduser().resolve()),
            "board": board_info,
            "robot": "fr3",
            "target_mount": "rigid_on_end_effector",
        })

    evidence_token = session / ("_connection_" + uuid4().hex)
    launch = load_session_launch(
        args.launch_config, evidence_dir_override=evidence_token)
    history = load_capture_points(session, launch.camera_ids)
    robot = FrankaController(
        args.daemon_host, command_port=args.command_port,
        state_port=args.state_port)
    if robot.get_data() is None:
        robot.end()
        raise RuntimeError(
            f"Franka daemon has no state at {args.daemon_host}:{args.state_port}")
    monitor = TkMonitor("FR3 hand-eye | c: capture stationary pose | q: quit")
    try:
        with open_autodex_camera_stream(
                launch.pc_list, capture_ips=launch.capture_ips,
                name="precision_handeye_capture", warmup_s=1,
                acquisition_mode=launch.camera_acquisition_mode):
            with open_session_cameras(launch) as cameras:
                while True:
                    before = _state(robot)
                    snapshot = collect_board_snapshot(
                        snapshot_orchestrator=cameras.snapshot_orchestrator,
                        calibrated_camera_ids=launch.camera_ids,
                        acquisition_metadata_for_request=(
                            cameras.acquisition_metadata_for_request),
                        timeout_s=launch.board_timeout_s)
                    after = _state(robot)
                    grid, _counts = preview_grid(
                        snapshot, detector,
                        tile_width=args.preview_tile_width,
                        history=history)
                    command = monitor.show(grid)
                    if command == "quit":
                        break
                    if command != "capture":
                        continue
                    record = _save(
                        session, _next(session), snapshot, detector,
                        before, after)
                    append_capture_points(
                        history, record["capture_index"], snapshot, detector)
                    counts = {serial: row["corner_count"]
                              for serial, row in record["views"].items()}
                    print(json.dumps({"capture": record["capture_index"],
                                      "corners": counts}, sort_keys=True))
    finally:
        monitor.close()
        robot.end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
