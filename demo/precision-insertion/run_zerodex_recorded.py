#!/usr/bin/env python3
"""Record one view from confirmed transfer through hand recovery.

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


def recording_receipt_directory(root: Path, run_id: str) -> Path:
    """Keep lightweight recorder references beside runs, never in assets."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run ID must be path-safe")
    return Path(root).expanduser().resolve() / run_id


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
    receipt_root = Path(os.environ.get(
        "PRECISION_RECORDING_RECEIPT_ROOT",
        str(Path(os.environ["PRECISION_RUN_OUTPUT_ROOT"]) /
            "_recording_receipts")))
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
    receipt_dir = recording_receipt_directory(receipt_root, run_tag)
    receipt_dir.mkdir(parents=True, exist_ok=False)
    trigger_root = Path(os.environ.get(
        "PRECISION_RECORDING_TRIGGER_ROOT", "/tmp/precision-insertion-recording"))
    trigger_root.mkdir(parents=True, exist_ok=True)
    trigger = trigger_root / f"{run_tag}.preinsert.json"
    stop_trigger = trigger_root / f"{run_tag}.release_clear.json"
    for marker in (trigger, stop_trigger):
        marker.unlink(missing_ok=True)
    env = {**os.environ, "PRECISION_RUN_TAG": run_tag,
           "PRECISION_PREINSERT_CONFIRMED_MARKER": str(trigger),
           "PRECISION_RELEASE_CLEAR_COMPLETED_MARKER": str(stop_trigger)}
    code_commit = os.environ.get("PRECISION_CODE_COMMIT")
    if code_commit is None:
        result = subprocess.run(
            ["git", "-C", str(HERE.parents[1]), "rev-parse", "HEAD"],
            check=False, capture_output=True, text=True)
        code_commit = result.stdout.strip() if result.returncode == 0 else None
    start_ns = time.time_ns()
    recorder_started = False
    recorder_was_started = False
    recording_error = None
    recording_manifest = None
    recording_started_ns = None
    recording_stopped_ns = None
    recording_trigger = None
    recording_stop_trigger = None
    recording_stop_reason = None
    exit_code = 1
    try:
        pipeline = subprocess.Popen(
            ["bash", str(HERE / "run_robot_pc.sh"), mode], env=env)
        while True:
            status = pipeline.poll()
            if not recorder_was_started and trigger.is_file():
                recording_trigger = json.loads(
                    trigger.read_text(encoding="utf-8"))
                try:
                    _start_recorder(
                        host=host, port=port, adapter=adapter,
                        remote_home=remote_home, remote_dir=remote_dir,
                        camera_id=camera_id)
                    recorder_started = True
                    recorder_was_started = True
                    recording_started_ns = time.time_ns()
                    print(
                        "[recording] capture4 local after confirmed transfer: "
                        f"{remote_dir}", flush=True)
                except (OSError, subprocess.CalledProcessError) as exc:
                    recording_error = str(exc)
                    print(f"[recording] could not start: {exc}",
                          file=sys.stderr, flush=True)
                    # Do not retry on every 20 ms poll. Recording is evidence,
                    # not a motion gate; the physical pipeline keeps running.
                    recorder_started = False
                    trigger.unlink(missing_ok=True)
            if recorder_started and stop_trigger.is_file():
                recording_stop_trigger = json.loads(
                    stop_trigger.read_text(encoding="utf-8"))
                try:
                    recording_manifest = _stop_recorder(
                        host=host, port=port, remote_dir=remote_dir)
                    if recording_manifest is None:
                        recording_error = "recorder did not write a manifest"
                except (OSError, subprocess.CalledProcessError,
                        ValueError) as exc:
                    recording_error = str(exc)
                recorder_started = False
                recording_stopped_ns = time.time_ns()
                recording_stop_reason = "release_and_clear_completed"
                print("[recording] stopped after hand recovery", flush=True)
            if status is not None:
                exit_code = status
                break
            time.sleep(.02)
    finally:
        if recorder_started:
            try:
                recording_manifest = _stop_recorder(
                    host=host, port=port, remote_dir=remote_dir)
                if recording_manifest is None:
                    recording_error = "recorder did not write a manifest"
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                recording_error = str(exc)
            recorder_started = False
            recording_stopped_ns = time.time_ns()
            recording_stop_reason = "pipeline_exit"
        receipt = {
            "schema": "precision_insertion_zerodex_recording_receipt_v1",
            "rig": "zerodex_capture4", "mode": mode, "run_id": run_tag,
            "camera_id": camera_id, "remote_host": host,
            "remote_directory": (str(remote_dir)
                                 if recorder_was_started else None),
            "started_unix_ns": start_ns, "ended_unix_ns": time.time_ns(),
            "recording_policy": (
                "after_preinsert_until_release_and_clear_completed"),
            "recording_trigger": recording_trigger,
            "recording_started_unix_ns": recording_started_ns,
            "recording_stop_trigger": recording_stop_trigger,
            "recording_stopped_unix_ns": recording_stopped_ns,
            "recording_stop_reason": recording_stop_reason,
            "pipeline_exit_code": exit_code,
            "code_commit": code_commit,
            "recording": recording_manifest,
            "recording_error": recording_error,
            "nas_uploaded": False,
        }
        (receipt_dir / "recording_ref.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        print(f"[recording] receipt: {receipt_dir / 'recording_ref.json'}",
              flush=True)
        for marker in (trigger, stop_trigger):
            marker.unlink(missing_ok=True)
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_zerodex_recorded.py [--run|--plan-only]")
    raise SystemExit(run(sys.argv[1]))
