#!/usr/bin/env python3
"""Run the ZeroDex pipeline while capture4 records one view to its own SSD.

This wrapper never writes to NAS. Use upload_zerodex_recording.py only after
the pipeline exits. The recorder reads the existing camera shared memory and
does not claim the camera or change capture4's installed ParaDex source.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent


def _ssh(host: str, port: str, command: str, *, check: bool = True):
    return subprocess.run(
        ["ssh", "-p", port, "-o", "BatchMode=yes", host,
         "bash -lc " + shlex.quote(command)],
        check=check, capture_output=True, text=True)


def _start_recorder(*, host: str, port: str, adapter: Path,
                    remote_home: Path, remote_dir: Path, camera_id: str) -> None:
    script = adapter / "demo/precision-insertion/record_selected_capture_view.py"
    _ssh(host, port, f"mkdir -p {shlex.quote(str(remote_dir))}")
    python = remote_home / "anaconda3/envs/flir_env/bin/python"
    paradex = remote_home / "paradex"
    video = remote_dir / "video"
    log = remote_dir / "recorder.log"
    pid = remote_dir / "recorder.pid"
    command = (
        f"nohup env PYTHONPATH={shlex.quote(str(paradex))} "
        f"{shlex.quote(str(python))} "
        f"{shlex.quote(str(script))} "
        f"--camera-id {shlex.quote(camera_id)} "
        f"--output-dir {shlex.quote(str(video))} --fps 10 "
        f"--max-seconds 21600 > {shlex.quote(str(log))} 2>&1 < /dev/null "
        f"& echo $! > {shlex.quote(str(pid))}")
    _ssh(host, port, command)


def _stop_recorder(*, host: str, port: str, remote_dir: Path) -> dict | None:
    pid = remote_dir / "recorder.pid"
    manifest = remote_dir / "video/recording_manifest.json"
    command = (
        f"if test -f {shlex.quote(str(pid))}; then "
        f"kill -TERM \"$(cat {shlex.quote(str(pid))})\" 2>/dev/null || true; fi; "
        f"for i in $(seq 1 100); do test -f {shlex.quote(str(manifest))} "
        f"&& break; sleep 0.1; done; "
        f"test -f {shlex.quote(str(manifest))} && "
        f"cat {shlex.quote(str(manifest))}")
    result = _ssh(host, port, command, check=False)
    return json.loads(result.stdout) if result.returncode == 0 else None


def run(mode: str) -> int:
    if mode not in ("--run", "--plan-only"):
        raise ValueError("expected --run or --plan-only")
    stage = Path(os.environ["PRECISION_STAGE_ROOT"]).expanduser().resolve()
    host = os.environ.get("PRECISION_CAPTURE4_SSH_HOST", "capture4@147.46.219.136")
    port = os.environ.get("PRECISION_CAPTURE4_SSH_PORT", "77")
    remote_home = Path(os.environ.get("PRECISION_CAPTURE4_HOME", "/home/capture4"))
    adapter = Path(os.environ.get(
        "PRECISION_CAPTURE4_ADAPTER_DIR",
        str(remote_home / "precision_insertion_capture4_20261027")))
    camera_id = os.environ.get("PRECISION_RECORD_CAMERA_ID", "26053248")
    run_tag = os.environ.get("PRECISION_RUN_TAG") or (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_tag):
        raise ValueError("PRECISION_RUN_TAG must be a path-safe run ID")
    day = datetime.now().strftime("%Y%m%d")
    remote_dir = adapter / "recordings" / day / run_tag
    receipt_dir = stage / "runs" / "recordings" / run_tag
    receipt_dir.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PRECISION_RUN_TAG": run_tag}
    start_ns = time.time_ns()
    recorder_started = False
    recording_error = None
    recording_manifest = None
    exit_code = 1
    try:
        try:
            _start_recorder(host=host, port=port, adapter=adapter,
                            remote_home=remote_home, remote_dir=remote_dir,
                            camera_id=camera_id)
            recorder_started = True
            print(f"[recording] capture4 local: {remote_dir}", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            recording_error = str(exc)
            print(f"[recording] could not start: {exc}", file=sys.stderr,
                  flush=True)
        exit_code = subprocess.run(
            ["bash", str(HERE / "run_robot_pc.sh"), mode], env=env,
            check=False).returncode
    finally:
        if recorder_started:
            try:
                recording_manifest = _stop_recorder(
                    host=host, port=port, remote_dir=remote_dir)
                if recording_manifest is None:
                    recording_error = "recorder did not write a manifest"
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                recording_error = str(exc)
        receipt = {
            "schema": "precision_insertion_zerodex_recording_receipt_v1",
            "rig": "zerodex_capture4", "mode": mode, "run_id": run_tag,
            "camera_id": camera_id, "remote_host": host,
            "remote_directory": str(remote_dir),
            "started_unix_ns": start_ns, "ended_unix_ns": time.time_ns(),
            "pipeline_exit_code": exit_code,
            "recording": recording_manifest,
            "recording_error": recording_error,
            "nas_uploaded": False,
        }
        (receipt_dir / "recording_ref.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"[recording] receipt: {receipt_dir / 'recording_ref.json'}",
              flush=True)
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_zerodex_recorded.py [--run|--plan-only]")
    raise SystemExit(run(sys.argv[1]))
