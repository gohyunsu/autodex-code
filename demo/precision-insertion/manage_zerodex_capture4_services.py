#!/usr/bin/env python3
"""Deploy and manage one capture4 camera/SAM3/FoundPose stack.

Only this namespaced adapter is copied. The capture PC's AutoDex checkout and
camera calibration remain untouched. Start is camera-only; no robot process is
opened. A persistent perception process avoids cold SAM3 load on every trial.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time


HOST = os.environ.get("PRECISION_CAPTURE4_SSH_HOST", "capture4@147.46.219.136")
SSH_PORT = os.environ.get("PRECISION_CAPTURE4_SSH_PORT", "77")
REMOTE_HOME = Path(os.environ.get("PRECISION_CAPTURE4_HOME", "/home/capture4"))
REMOTE = Path(os.environ.get(
    "PRECISION_CAPTURE4_ADAPTER_DIR",
    str(REMOTE_HOME / "precision_insertion_capture4_20261027")))
ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "src/execution/__init__.py",
    "src/execution/daemon/__init__.py",
    "src/execution/daemon/init_daemon.py",
    "src/execution/daemon/snapshot_daemon.py",
    "demo/precision-insertion/precision_init_daemon.py",
    "demo/precision-insertion/capture4_exposure_daemon.py",
    "demo/precision-insertion/precision_insertion/__init__.py",
    "demo/precision-insertion/precision_insertion/camera_transport.py",
    "demo/precision-insertion/precision_insertion/sam3_batch.py",
    "demo/precision-insertion/precision_insertion/frame_provenance.py",
)
PORTS = {"camera": 5480, "snapshot": 6894, "precision": 16893}
READY_PORTS = frozenset((5480, 5481, 5482, 5009, 6894, 15006, 15007, 16893))
_PORT_RE = re.compile(r":(\d+)$")


def _ssh(script: str) -> None:
    subprocess.run(
        ["ssh", "-p", SSH_PORT, "-o", "BatchMode=yes", HOST,
         "bash -lc " + shlex.quote(script)],
        check=True,
    )


def _listening_ports() -> set[int]:
    result = subprocess.run(
        ["ssh", "-p", SSH_PORT, "-o", "BatchMode=yes", HOST,
         "bash -lc " + shlex.quote("ss -lntH")],
        check=True, capture_output=True, text=True,
    )
    ports = set()
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) < 4:
            continue
        match = _PORT_RE.search(columns[3])
        if match:
            ports.add(int(match.group(1)))
    return ports


def deploy() -> None:
    _ssh(f"mkdir -p {shlex.quote(str(REMOTE))}")
    subprocess.run(
        ["rsync", "-a", "-e", f"ssh -p {SSH_PORT} -o BatchMode=yes", "--relative",
         *("./" + item for item in FILES), f"{HOST}:{REMOTE}/"],
        cwd=ROOT, check=True,
    )


def _command(kind: str, *, exposure_us: float | None = None) -> str:
    if kind == "camera":
        root = REMOTE_HOME / "paradex"
        python = REMOTE_HOME / "anaconda3/envs/flir_env/bin/python"
        setting = ("" if exposure_us is None else
                   f"PRECISION_CAPTURE_EXPOSURE_US={exposure_us:g} ")
        return (f"cd {root} && exec env PYTHONPATH={root} {setting}"
                f"{python} {REMOTE}/demo/precision-insertion/capture4_exposure_daemon.py")
    python = REMOTE_HOME / "anaconda3/envs/gotrack_cu128/bin/python"
    module_root = REMOTE / "demo/precision-insertion"
    pythonpath = ":".join((str(module_root), str(REMOTE),
                           str(REMOTE_HOME / "AutoDex"),
                           str(REMOTE_HOME / "paradex")))
    if kind == "snapshot":
        return (f"exec env PYTHONPATH={pythonpath} {python} "
                f"{REMOTE}/src/execution/daemon/snapshot_daemon.py")
    stage = (REMOTE_HOME / "shared_data/hyunsu/tmp/20261023_prev/"
             "precision_insertion_recipient_stage_20261013_b7e2392")
    sam3_batch = os.environ.get("PRECISION_SAM3_BATCH", "auto")
    if sam3_batch not in ("auto", "0", "1"):
        raise ValueError("PRECISION_SAM3_BATCH must be auto, 0, or 1")
    # This capture4 GPU cannot execute FoundPose's DINOv2 float32 xFormers
    # attention kernel. The vendored DINOv2 uses PyTorch SDPA when disabled;
    # keep this setting local to the ZeroDex precision service, not stock.
    return (f"exec env HF_HUB_OFFLINE=1 XFORMERS_DISABLED=1 "
            f"PRECISION_SAM3_BATCH={sam3_batch} AUTODEX_STOCK_ROOT={REMOTE} "
            f"PRECISION_CAPTURE_ASSET_ROOT={stage} PYTHONPATH={pythonpath} "
            f"{python} {module_root}/precision_init_daemon.py "
            "--port-mask 15006 --port-pose 15007 --port-cmd 16893")


def start(*, exposure_us: float | None = None) -> None:
    if exposure_us is not None and (not math.isfinite(exposure_us) or
                                    exposure_us <= 0):
        raise ValueError("exposure override must be finite positive microseconds")
    # A warm session needs the running services, not another rsync into their
    # namespaced checkout. Reuse them without touching shared capture4 files.
    listening = _listening_ports()
    if READY_PORTS <= listening:
        if exposure_us is not None:
            raise RuntimeError("camera services are already running; stop our stack before changing exposure")
        print("capture4 services ready: " + ", ".join(map(str, sorted(READY_PORTS))))
        return
    if exposure_us is not None and PORTS["camera"] in listening:
        raise RuntimeError("camera daemon is already running; exposure override cannot be applied")
    deploy()
    for kind, port in PORTS.items():
        log = REMOTE / f"{kind}.log"
        pid = REMOTE / f"{kind}.pid"
        command = _command(kind, exposure_us=exposure_us if kind == "camera" else None)
        # Reuse an already listening service; never replace another owner.
        _ssh(
            f"if ss -lnt | grep -q ':{port} '; then "
            f"echo '{kind}: already listening on {port}'; "
            "else "
            f"nohup bash -lc {shlex.quote(command)} > {shlex.quote(str(log))} "
            f"2>&1 < /dev/null & echo $! > {shlex.quote(str(pid))}; "
            f"echo '{kind}: started'; fi"
        )
    # The camera and perception processes bind their sockets after Python and
    # model startup. Return only once the immediate caller can connect; polling
    # avoids a fixed sleep on warm starts.
    deadline = time.monotonic() + 120.0
    while True:
        missing = READY_PORTS - _listening_ports()
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"capture4 services did not bind ports: {sorted(missing)}")
        time.sleep(0.25)
    check()


def check() -> None:
    missing = READY_PORTS - _listening_ports()
    if missing:
        raise RuntimeError(f"capture4 services are not ready: {sorted(missing)}")
    print("capture4 services ready: " + ", ".join(map(str, sorted(READY_PORTS))))


def stop() -> None:
    for kind in reversed(tuple(PORTS)):
        pid = REMOTE / f"{kind}.pid"
        expected = {
            "camera": str(REMOTE / "demo/precision-insertion/capture4_exposure_daemon.py"),
            "snapshot": str(REMOTE / "src/execution/daemon/snapshot_daemon.py"),
            "precision": str(REMOTE / "demo/precision-insertion/precision_init_daemon.py"),
        }[kind]
        # Older namespaced runs started the stock camera script directly.
        # Their PID file remains ours; accept that one previous command too.
        command_pattern = ("*capture4_exposure_daemon.py*|*src/camera/server_daemon.py*"
                           if kind == "camera" else
                           f"*{shlex.quote(expected)}*")
        _ssh(
            f"if test -f {shlex.quote(str(pid))}; then "
            f"p=$(cat {shlex.quote(str(pid))}); "
            "case $p in ''|*[!0-9]*) exit 1;; esac; "
            "if test ! -r /proc/$p/cmdline; then "
            f"echo '{kind}: already stopped'; "
            "else "
            "cmd=$(tr '\\000' ' ' < /proc/$p/cmdline); "
            f"case \"$cmd\" in {command_pattern}) kill -TERM \"$p\";; "
            "*) echo 'PID no longer belongs to our service' >&2; exit 1;; "
            "esac; fi; "
            f"rm {shlex.quote(str(pid))}; echo '{kind}: stopped'; fi"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "start", "check", "stop"))
    parser.add_argument("--exposure-us", type=float,
                        help="capture4-only process-local exposure override; start action only")
    args = parser.parse_args()
    if args.exposure_us is not None and args.action != "start":
        parser.error("--exposure-us is valid only with start")
    if args.action == "start":
        start(exposure_us=args.exposure_us)
    else:
        {"deploy": deploy, "check": check, "stop": stop}[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
