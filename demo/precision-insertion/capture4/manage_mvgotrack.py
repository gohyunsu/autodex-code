#!/usr/bin/env python3
"""Manage an isolated capture4 MV-GoTrack process for precision insertion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(os.environ.get(
    "PRECISION_MVGOTRACK_ROOT",
    "/home/capture4/precision_insertion_mvgotrack"))
PYTHON = Path(os.environ.get(
    "PRECISION_MVGOTRACK_PYTHON",
    "/home/capture4/anaconda3/envs/gotrack_cu128/bin/python"))
SOURCE = ROOT / "MV-GoTrack"
STATE = ROOT / "state"
CAMERAS = ("26053248", "25322639", "25305462", "25322642")
ACTIVE_TRACKER = "tracker.active.json"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_subset_sha256(path: Path) -> str:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"camera calibration is not a mapping: {path}")
    missing = [camera for camera in CAMERAS if camera not in values]
    if missing:
        raise ValueError(f"camera calibration lacks {missing}: {path}")
    payload = json.dumps(
        {camera: values[camera] for camera in CAMERAS},
        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def calibration_identity(cam_param: Path | None = None) -> str:
    """Identify the exact camera world used by tracker input and output."""
    root = ROOT / "cam_param" if cam_param is None else Path(cam_param)
    intrinsic = _camera_subset_sha256(root / "intrinsics.json")
    extrinsic = _camera_subset_sha256(root / "extrinsics.json")
    cameras = ",".join(CAMERAS)
    return f"precision_calib_v1:{cameras}:{intrinsic}:{extrinsic}"


def _pid(name: str) -> int | None:
    path = STATE / f"{name}.pid"
    if not path.is_file():
        return None
    value = int(path.read_text().strip())
    try:
        os.kill(value, 0)
    except ProcessLookupError:
        path.unlink(missing_ok=True)
        return None
    return value


def _spawn(name: str, command: list[str]) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    log = (STATE / f"{name}.log").open("ab", buffering=0)
    environment = dict(os.environ)
    environment["MVGOTRACK_ANY_GPU"] = "1"
    # The isolated capture4 deployment owns its only physical CUDA device.
    # Expose that GPU as logical cuda:0 instead of inheriting MV-GoTrack's
    # original two-GPU workstation assignment (physical GPU 1).
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["EGL_DEVICE_ID"] = "0"
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True, env=environment)
    (STATE / f"{name}.pid").write_text(f"{process.pid}\n")
    return process.pid


def _stop(name: str) -> None:
    pid = _pid(name)
    if pid is None:
        if name == "tracker":
            (STATE / ACTIVE_TRACKER).unlink(missing_ok=True)
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(50):
        if _pid(name) is None:
            break
        time.sleep(.1)
    else:
        os.killpg(pid, signal.SIGKILL)
    (STATE / f"{name}.pid").unlink(missing_ok=True)
    if name == "tracker":
        (STATE / ACTIVE_TRACKER).unlink(missing_ok=True)


def _active_tracker() -> dict | None:
    path = STATE / ACTIVE_TRACKER
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    required = {"object", "mesh", "mesh_sha256"}
    return row if isinstance(row, dict) and required <= row.keys() else None


def _write_active_tracker(args) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    record = {
        "object": args.object,
        "mesh": str(Path(args.mesh).resolve()),
        "mesh_sha256": _file_sha256(args.mesh),
        "session": str(Path(args.session).resolve()),
        "started_at_s": time.time(),
    }
    path = STATE / ACTIVE_TRACKER
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def _reinit_mesh(args) -> Path | None:
    """Return the active mesh when a new session carries identical geometry."""
    active = _active_tracker()
    if (active is None or active["object"] != args.object or
            active["mesh_sha256"] != _file_sha256(args.mesh)):
        return None
    path = Path(active["mesh"])
    return path if path.is_file() else None


def _ensure_stream() -> None:
    if _pid("stream") is None:
        _spawn("stream", [
            str(PYTHON), str(ROOT / "mvgotrack_stream.py"),
            "--cameras", *CAMERAS, "--port", "5484"])


def _tracker_command(args) -> list[str]:
    endpoint = args.endpoint.replace("147.46.219.136", "0.0.0.0")
    return [
        str(PYTHON), str(SOURCE / "deploy/live_tracker.py"),
        "--source", "stream", "--endpoint", endpoint,
        "--mesh", str(args.mesh), "--object", args.object,
        "--init-pose", str(args.init_pose),
        "--cameras", *CAMERAS,
        "--cam-param", str(ROOT / "cam_param"),
        "--calib-session", calibration_identity(),
        "--pc-json", str(ROOT / "pc.json"),
        "--control-file", str(STATE / "tracker.control.json"),
        "--output", str(Path(args.session) / "tracker_output"),
        "--engine", "auto", "--min-score", "0",
        "--max-workspace-m", "0", "--max-jump-m", "0",
    ]


def start_or_reinit(args) -> None:
    _ensure_stream()
    if _pid("tracker") is not None:
        mesh = _reinit_mesh(args)
        if mesh is not None:
            subprocess.run([
                str(PYTHON), str(SOURCE / "deploy/live_tracker.py"),
                "--reinit", str(args.init_pose),
                "--control-file", str(STATE / "tracker.control.json"),
                "--object", args.object, "--mesh", str(mesh),
            ], check=True)
            return
        # A different object/geometry (or a legacy tracker without provenance)
        # cannot be reinitialized in place. Restart this experiment-owned
        # process; shared capture4 camera services remain untouched.
        _stop("tracker")
    _spawn("tracker", _tracker_command(args))
    _write_active_tracker(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    start = sub.add_parser("start-or-reinit")
    start.add_argument("--session", type=Path, required=True)
    start.add_argument("--object", required=True)
    start.add_argument("--mesh", type=Path, required=True)
    start.add_argument("--init-pose", type=Path, required=True)
    start.add_argument("--endpoint", required=True)
    sub.add_parser("stop")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "start-or-reinit":
        start_or_reinit(args)
    elif args.action == "stop":
        _stop("tracker")
        _stop("stream")
    else:
        print(json.dumps({"tracker_pid": _pid("tracker"),
                          "stream_pid": _pid("stream")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
