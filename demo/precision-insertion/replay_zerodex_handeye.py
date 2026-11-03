#!/usr/bin/env python3
"""Replay a taught FR3 hand-eye waypoint sequence and recapture four views.

Unlike ``capture_zerodex_handeye.py``, this program commands the robot. Each
saved joint waypoint is executed in order; once the daemon's blocking move
returns and the configured settle interval elapses, one synchronized four-view
observation and the measured robot state are stored in a new session.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from calibration.charuco import load_board
from calibration.monitor import (
    append_capture_points,
    load_capture_points,
    preview_grid,
    TkMonitor,
)
from capture_zerodex_handeye import _json, _next, _save, _state
from paradex.io.robot_controller.franka_controller import FrankaController
from precision_insertion.camera_stream import open_autodex_camera_stream
from precision_insertion.live_capture import collect_board_snapshot
from precision_insertion.session_launch import load_session_launch, open_session_cameras


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-config", type=Path, required=True)
    parser.add_argument("--board-config", type=Path, required=True)
    parser.add_argument("--camera-calibration-dir", type=Path, required=True)
    parser.add_argument("--trajectory-session", type=Path, required=True)
    parser.add_argument("--session-dir", type=Path, required=True)
    parser.add_argument("--daemon-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5555)
    parser.add_argument("--state-port", type=int, default=5556)
    parser.add_argument("--speed-scale", type=float, default=0.1)
    parser.add_argument("--settle-s", type=float, default=0.5)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--preview-tile-width", type=int, default=640)
    args = parser.parse_args(argv)

    trajectory_path = args.trajectory_session.expanduser().resolve() / "trajectory_qpos.npy"
    qposes = np.load(trajectory_path).astype(np.float64)
    if qposes.ndim != 2 or qposes.shape[1] != 7 or len(qposes) == 0:
        raise ValueError(f"expected non-empty (N, 7) trajectory: {trajectory_path}")
    session = args.session_dir.expanduser().resolve()
    session.mkdir(parents=True, exist_ok=True)
    board_info, _, detector = load_board(args.board_config)
    manifest = session / "session.json"
    if not manifest.exists():
        _json(manifest, {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "capture_mode": "automatic_taught_trajectory_replay",
            "source_trajectory_session": str(
                args.trajectory_session.expanduser().resolve()),
            "launch_config": str(args.launch_config.expanduser().resolve()),
            "board_config": str(args.board_config.expanduser().resolve()),
            "camera_calibration_dir": str(
                args.camera_calibration_dir.expanduser().resolve()),
            "board": board_info,
            "robot": "fr3",
            "target_mount": "rigid_on_end_effector",
            "speed_scale": args.speed_scale,
            "settle_s": args.settle_s,
        })

    launch = load_session_launch(
        args.launch_config,
        evidence_dir_override=session / ("_connection_" + uuid4().hex))
    history = load_capture_points(session, launch.camera_ids)
    robot = FrankaController(
        args.daemon_host, command_port=args.command_port,
        state_port=args.state_port)
    if robot.get_data() is None:
        robot.end()
        raise RuntimeError(
            f"Franka daemon has no state at {args.daemon_host}:{args.state_port}")
    monitor = (TkMonitor("FR3 hand-eye replay | q: stop after current move")
               if args.monitor else None)
    try:
        with open_autodex_camera_stream(
                launch.pc_list, capture_ips=launch.capture_ips,
                name="precision_handeye_replay", warmup_s=1,
                acquisition_mode=launch.camera_acquisition_mode):
            with open_session_cameras(launch) as cameras:
                for waypoint_index, qpos in enumerate(qposes):
                    response = robot.move(
                        qpos, is_servo=False, speed_scale=args.speed_scale)
                    if response.get("type") == "error":
                        raise RuntimeError(
                            f"waypoint {waypoint_index} move failed: {response}")
                    time.sleep(args.settle_s)
                    before = _state(robot)
                    snapshot = collect_board_snapshot(
                        snapshot_orchestrator=cameras.snapshot_orchestrator,
                        calibrated_camera_ids=launch.camera_ids,
                        acquisition_metadata_for_request=(
                            cameras.acquisition_metadata_for_request),
                        timeout_s=launch.board_timeout_s)
                    after = _state(robot)
                    index = _next(session)
                    record = _save(
                        session, index, snapshot, detector, before, after)
                    append_capture_points(history, index, snapshot, detector)
                    counts = {serial: row["corner_count"]
                              for serial, row in record["views"].items()}
                    print(json.dumps({
                        "capture": index,
                        "source_waypoint": waypoint_index,
                        "corners": counts,
                    }, sort_keys=True))
                    if monitor is not None:
                        grid, _ = preview_grid(
                            snapshot, detector,
                            tile_width=args.preview_tile_width,
                            history=history)
                        if monitor.show(grid) == "quit":
                            break
    finally:
        if monitor is not None:
            monitor.close()
        robot.end()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
