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


def _move_joint_velocity_stream(
        robot: FrankaController, target: np.ndarray, *, max_speed: float,
        control_dt: float, position_gain: float, converge_s: float) -> float:
    """Follow the straight joint-space segment with a smooth velocity stream."""
    start = np.asarray(robot.get_data()["qpos"], dtype=np.float64)
    delta = np.asarray(target, dtype=np.float64) - start
    extent = float(np.max(np.abs(delta)))
    # A minimum-jerk scalar has peak derivative 1.875. Five controller ticks
    # keep even a tiny move continuous rather than degenerating to one sample.
    duration = max(5.0 * control_dt, 1.875 * extent / max_speed)
    command_validity_ms = max(100, int(np.ceil(4.0 * control_dt * 1000.0)))
    started = time.monotonic()
    try:
        while True:
            tick = time.monotonic()
            u = min(1.0, (tick - started) / duration)
            blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
            blend_rate = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration
            reference = start + blend * delta
            feedforward = blend_rate * delta
            measured = np.asarray(robot.get_data()["qpos"], dtype=np.float64)
            velocity = feedforward + position_gain * (reference - measured)
            peak = float(np.max(np.abs(velocity)))
            if peak > max_speed:
                velocity *= max_speed / peak
            response = robot.set_joint_velocity(
                velocity, duration_ms=command_validity_ms)
            if response.get("type") == "error":
                raise RuntimeError(f"joint velocity stream failed: {response}")
            if u >= 1.0:
                break
            time.sleep(max(0.0, control_dt - (time.monotonic() - tick)))

        converge_until = time.monotonic() + converge_s
        while time.monotonic() < converge_until:
            tick = time.monotonic()
            measured = np.asarray(robot.get_data()["qpos"], dtype=np.float64)
            velocity = position_gain * (target - measured)
            peak = float(np.max(np.abs(velocity)))
            if peak > max_speed:
                velocity *= max_speed / peak
            response = robot.set_joint_velocity(
                velocity, duration_ms=command_validity_ms)
            if response.get("type") == "error":
                raise RuntimeError(f"joint velocity convergence failed: {response}")
            time.sleep(max(0.0, control_dt - (time.monotonic() - tick)))
    finally:
        robot.set_joint_velocity(np.zeros(7), duration_ms=100)
        time.sleep(0.08)
        response = robot.stop_streaming()
        if response.get("type") == "error":
            raise RuntimeError(f"could not stop joint velocity stream: {response}")
    return time.monotonic() - started


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
    parser.add_argument("--motion-mode", choices=("velocity_stream", "blocking"),
                        default="velocity_stream")
    parser.add_argument("--joint-speed-rad-s", type=float, default=0.2)
    parser.add_argument("--control-dt-s", type=float, default=0.02)
    parser.add_argument("--stream-position-gain", type=float, default=2.0)
    parser.add_argument("--stream-converge-s", type=float, default=0.5)
    parser.add_argument("--start-waypoint", type=int, default=0,
                        help="source trajectory index to replay first")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--preview-tile-width", type=int, default=640)
    args = parser.parse_args(argv)

    trajectory_path = args.trajectory_session.expanduser().resolve() / "trajectory_qpos.npy"
    qposes = np.load(trajectory_path).astype(np.float64)
    if qposes.ndim != 2 or qposes.shape[1] != 7 or len(qposes) == 0:
        raise ValueError(f"expected non-empty (N, 7) trajectory: {trajectory_path}")
    if not 0 <= args.start_waypoint < len(qposes):
        parser.error(
            f"--start-waypoint must be in [0, {len(qposes) - 1}]")
    if (args.joint_speed_rad_s <= 0 or args.control_dt_s <= 0 or
            args.stream_position_gain <= 0 or args.stream_converge_s < 0):
        parser.error("joint stream parameters must be positive (converge may be zero)")
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
            "replay_invocations": [],
        })
    manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_record.setdefault("replay_invocations", []).append({
        "started_at": datetime.now().astimezone().isoformat(),
        "start_waypoint": args.start_waypoint,
        "speed_scale": args.speed_scale,
        "settle_s": args.settle_s,
        "motion_mode": args.motion_mode,
        "joint_speed_rad_s": args.joint_speed_rad_s,
        "control_dt_s": args.control_dt_s,
        "stream_position_gain": args.stream_position_gain,
        "stream_converge_s": args.stream_converge_s,
    })
    _json(manifest, manifest_record)

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
                for waypoint_index in range(args.start_waypoint, len(qposes)):
                    qpos = qposes[waypoint_index]
                    if args.motion_mode == "velocity_stream":
                        motion_duration_s = _move_joint_velocity_stream(
                            robot, qpos,
                            max_speed=args.joint_speed_rad_s,
                            control_dt=args.control_dt_s,
                            position_gain=args.stream_position_gain,
                            converge_s=args.stream_converge_s)
                    else:
                        motion_started = time.monotonic()
                        response = robot.move(
                            qpos, is_servo=False, speed_scale=args.speed_scale)
                        if response.get("type") == "error":
                            raise RuntimeError(
                                f"waypoint {waypoint_index} move failed: {response}")
                        motion_duration_s = time.monotonic() - motion_started
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
                    record["source_waypoint"] = waypoint_index
                    _json(session / f"{index:04d}" / "capture.json", record)
                    append_capture_points(history, index, snapshot, detector)
                    counts = {serial: row["corner_count"]
                              for serial, row in record["views"].items()}
                    print(json.dumps({
                        "capture": index,
                        "source_waypoint": waypoint_index,
                        "motion_mode": args.motion_mode,
                        "motion_duration_s": motion_duration_s,
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
