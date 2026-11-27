#!/usr/bin/env python3
"""Record all capture4 views from first motion through final clear-view capture.

This wrapper never writes to NAS. Use upload_zerodex_recording.py only after
the pipeline exits. The recorder reads the existing camera shared memory and
does not claim the camera or change capture4's installed ParaDex source.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time


HERE = Path(__file__).resolve().parent


def recording_receipt_directory(root: Path, run_id: str) -> Path:
    """Keep lightweight recorder references beside runs, never in assets."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run ID must be path-safe")
    return Path(root).expanduser().resolve() / run_id


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tee_pipeline_output(stream, log_path: Path) -> None:
    """Mirror child output to the terminal and a local evidence file."""
    with log_path.open("w", encoding="utf-8") as output:
        for line in stream:
            output.write(line)
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()


def _ssh(host: str, port: str, command: str, *, check: bool = True):
    return subprocess.run(
        ["ssh", "-p", port, "-o", "BatchMode=yes", host,
         "bash -lc " + shlex.quote(command)],
        check=check, capture_output=True, text=True)


def _start_recorder(*, host: str, port: str, adapter: Path,
                    remote_home: Path, remote_dir: Path,
                    camera_ids: tuple[str, ...]) -> None:
    script = adapter / "demo/precision-insertion/record_selected_capture_view.py"
    _ssh(host, port, f"mkdir -p {shlex.quote(str(remote_dir))}")
    python = remote_home / "anaconda3/envs/flir_env/bin/python"
    paradex = remote_home / "paradex"
    launches = [f"mkdir -p {shlex.quote(str(remote_dir / 'video'))}"]
    for camera_id in camera_ids:
        video = remote_dir / "video" / camera_id
        log = remote_dir / f"recorder.{camera_id}.log"
        pid = remote_dir / f"recorder.{camera_id}.pid"
        ready = remote_dir / f"recorder.{camera_id}.ready.json"
        launches.append(
            f"nohup env PYTHONPATH={shlex.quote(str(paradex))} "
            f"{shlex.quote(str(python))} "
            f"{shlex.quote(str(script))} "
            f"--camera-id {shlex.quote(camera_id)} "
            f"--output-dir {shlex.quote(str(video))} "
            f"--ready-file {shlex.quote(str(ready))} --fps 10 "
            f"--max-seconds 21600 > {shlex.quote(str(log))} 2>&1 "
            f"< /dev/null & echo $! > {shlex.quote(str(pid))}")
    _ssh(host, port, "; ".join(launches))
    ready_checks = " && ".join(
        f"test -f {shlex.quote(str(remote_dir / f'recorder.{camera}.ready.json'))}"
        for camera in camera_ids)
    _ssh(host, port, (
        "for i in $(seq 1 200); do "
        f"if {ready_checks}; then exit 0; fi; sleep 0.05; done; exit 1"))


def _stop_recorder(*, host: str, port: str, remote_dir: Path,
                   camera_ids: tuple[str, ...]) -> dict:
    commands = []
    for camera_id in camera_ids:
        pid = remote_dir / f"recorder.{camera_id}.pid"
        commands.append(
            f"if test -f {shlex.quote(str(pid))}; then "
            f"kill -TERM \"$(cat {shlex.quote(str(pid))})\" "
            "2>/dev/null || true; fi")
    manifests = [remote_dir / "video" / camera / "recording_manifest.json"
                 for camera in camera_ids]
    ready = " && ".join(f"test -f {shlex.quote(str(path))}"
                         for path in manifests)
    commands.append(
        "for i in $(seq 1 200); do "
        f"if {ready}; then break; fi; sleep 0.05; done")
    _ssh(host, port, "; ".join(commands), check=False)
    views, missing = {}, []
    for camera_id, manifest in zip(camera_ids, manifests):
        result = _ssh(
            host, port,
            f"test -f {shlex.quote(str(manifest))} && "
            f"cat {shlex.quote(str(manifest))}", check=False)
        if result.returncode == 0 and result.stdout.strip():
            views[camera_id] = json.loads(result.stdout)
        else:
            missing.append(camera_id)
    return {
        "schema": "precision_insertion_multiview_recording_manifest_v1",
        "camera_ids": list(camera_ids), "views": views,
        "missing_camera_ids": missing,
    }


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
    raw_camera_ids = os.environ.get(
        "PRECISION_RECORD_CAMERA_IDS",
        os.environ.get("PRECISION_CAPTURE4_CAMERA_IDS",
                       "26053248,25322639,25305462,25322642"))
    camera_ids = tuple(value.strip() for value in raw_camera_ids.split(",")
                       if value.strip())
    if (not camera_ids or len(set(camera_ids)) != len(camera_ids) or
            any(not re.fullmatch(r"[0-9]+", value) for value in camera_ids)):
        raise ValueError("recording camera IDs must be unique numeric serials")
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
    trigger = trigger_root / f"{run_tag}.execution_ready.json"
    ready_ack = trigger_root / f"{run_tag}.recording_ready.json"
    stop_trigger = trigger_root / f"{run_tag}.final_capture.json"
    local_pipeline_log = trigger_root / f"{run_tag}.pipeline.log"
    for marker in (trigger, ready_ack, stop_trigger):
        marker.unlink(missing_ok=True)
    local_pipeline_log.unlink(missing_ok=True)
    env = {**os.environ, "PRECISION_RUN_TAG": run_tag,
           "PRECISION_EXECUTION_READY_MARKER": str(trigger),
           "PRECISION_RECORDING_READY_ACK": str(ready_ack),
           "PRECISION_FINAL_CAPTURE_COMPLETED_MARKER": str(stop_trigger)}
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
    output_thread = None
    exit_code = 1
    try:
        pipeline = subprocess.Popen(
            ["bash", str(HERE / "run_robot_pc.sh"), mode], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        if pipeline.stdout is None:
            raise RuntimeError("pipeline stdout capture was not created")
        output_thread = threading.Thread(
            target=_tee_pipeline_output,
            args=(pipeline.stdout, local_pipeline_log),
            name=f"precision-pipeline-log-{run_tag}",
            daemon=True)
        output_thread.start()
        while True:
            status = pipeline.poll()
            if not recorder_was_started and trigger.is_file():
                recording_trigger = json.loads(
                    trigger.read_text(encoding="utf-8"))
                try:
                    _start_recorder(
                        host=host, port=port, adapter=adapter,
                        remote_home=remote_home, remote_dir=remote_dir,
                        camera_ids=camera_ids)
                    recorder_started = True
                    recorder_was_started = True
                    recording_started_ns = time.time_ns()
                    ready_ack.write_text(json.dumps({
                        "schema": "precision_insertion_recording_ready_v1",
                        "ready": True, "camera_ids": list(camera_ids),
                        "ready_at_unix_ns": recording_started_ns,
                    }, sort_keys=True) + "\n", encoding="utf-8")
                    print(
                        "[recording] capture4 all views before first motion: "
                        f"{remote_dir}", flush=True)
                except (OSError, subprocess.CalledProcessError) as exc:
                    recording_error = str(exc)
                    ready_ack.write_text(json.dumps({
                        "schema": "precision_insertion_recording_ready_v1",
                        "ready": False, "error": recording_error,
                        "ready_at_unix_ns": time.time_ns(),
                    }, sort_keys=True) + "\n", encoding="utf-8")
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
                        host=host, port=port, remote_dir=remote_dir,
                        camera_ids=camera_ids)
                    if recording_manifest["missing_camera_ids"]:
                        recording_error = (
                            "recorders missing manifests: " + ",".join(
                                recording_manifest["missing_camera_ids"]))
                except (OSError, subprocess.CalledProcessError,
                        ValueError) as exc:
                    recording_error = str(exc)
                recorder_started = False
                recording_stopped_ns = time.time_ns()
                recording_stop_reason = "final_capture_completed_before_vlm_label"
                print("[recording] stopped after final capture, before VLM label",
                      flush=True)
            if status is not None:
                exit_code = status
                break
            time.sleep(.02)
    finally:
        if output_thread is not None:
            output_thread.join()
        if recorder_started:
            try:
                recording_manifest = _stop_recorder(
                    host=host, port=port, remote_dir=remote_dir,
                    camera_ids=camera_ids)
                if recording_manifest["missing_camera_ids"]:
                    recording_error = (
                        "recorders missing manifests: " + ",".join(
                            recording_manifest["missing_camera_ids"]))
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                recording_error = str(exc)
            recorder_started = False
            recording_stopped_ns = time.time_ns()
            recording_stop_reason = "pipeline_exit"
        pipeline_log = None
        if local_pipeline_log.is_file():
            preserved_log = receipt_dir / "pipeline.log"
            shutil.copyfile(local_pipeline_log, preserved_log)
            pipeline_log = {
                "path": preserved_log.name,
                "size_bytes": preserved_log.stat().st_size,
                "sha256": _sha256(preserved_log),
            }
        receipt = {
            "schema": "precision_insertion_zerodex_recording_receipt_v1",
            "rig": "zerodex_capture4", "mode": mode, "run_id": run_tag,
            "camera_ids": list(camera_ids), "remote_host": host,
            "remote_directory": (str(remote_dir)
                                 if recorder_was_started else None),
            "started_unix_ns": start_ns, "ended_unix_ns": time.time_ns(),
            "recording_policy": (
                "after_planning_before_first_motion_until_final_capture_before_vlm"),
            "recording_trigger": recording_trigger,
            "recording_started_unix_ns": recording_started_ns,
            "recording_stop_trigger": recording_stop_trigger,
            "recording_stopped_unix_ns": recording_stopped_ns,
            "recording_stop_reason": recording_stop_reason,
            "pipeline_exit_code": exit_code,
            "pipeline_log": pipeline_log,
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
        for marker in (trigger, ready_ack, stop_trigger):
            marker.unlink(missing_ok=True)
        local_pipeline_log.unlink(missing_ok=True)
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_zerodex_recorded.py [--run|--plan-only]")
    raise SystemExit(run(sys.argv[1]))
