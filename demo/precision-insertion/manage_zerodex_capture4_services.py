#!/usr/bin/env python3
"""Deploy and manage one capture4 camera/SAM3/FoundPose stack.

Only this namespaced adapter is copied. The capture PC's AutoDex checkout and
camera calibration remain untouched. Start is camera-only; no robot process is
opened. A persistent perception process avoids cold SAM3 load on every trial.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import time


HOST = "capture4@147.46.219.136"
REMOTE = Path("/home/capture4/precision_insertion_capture4_20261027")
REMOTE_HOME = Path("/home/capture4")
ROOT = Path(__file__).resolve().parents[2]
FILES = (
    "src/execution/__init__.py",
    "src/execution/daemon/__init__.py",
    "src/execution/daemon/init_daemon.py",
    "src/execution/daemon/snapshot_daemon.py",
    "demo/precision-insertion/precision_init_daemon.py",
    "demo/precision-insertion/precision_insertion/__init__.py",
    "demo/precision-insertion/precision_insertion/camera_transport.py",
    "demo/precision-insertion/precision_insertion/frame_provenance.py",
)
PORTS = {"camera": 5480, "snapshot": 6894, "precision": 16893}
READY_PORTS = frozenset((5480, 5481, 5482, 5009, 6894, 15006, 15007, 16893))
_PORT_RE = re.compile(r":(\d+)$")


def _ssh(script: str) -> None:
    subprocess.run(
        ["ssh", "-p", "77", "-o", "BatchMode=yes", HOST,
         "bash -lc " + shlex.quote(script)],
        check=True,
    )


def _listening_ports() -> set[int]:
    result = subprocess.run(
        ["ssh", "-p", "77", "-o", "BatchMode=yes", HOST,
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
        ["rsync", "-a", "-e", "ssh -p 77 -o BatchMode=yes", "--relative",
         *("./" + item for item in FILES), f"{HOST}:{REMOTE}/"],
        cwd=ROOT, check=True,
    )


def _command(kind: str) -> str:
    if kind == "camera":
        root = REMOTE_HOME / "paradex"
        python = REMOTE_HOME / "anaconda3/envs/flir_env/bin/python"
        return (f"cd {root} && exec env PYTHONPATH={root} "
                f"{python} src/camera/server_daemon.py")
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
    # This capture4 GPU cannot execute FoundPose's DINOv2 float32 xFormers
    # attention kernel. The vendored DINOv2 uses PyTorch SDPA when disabled;
    # keep this setting local to the ZeroDex precision service, not stock.
    return (f"exec env XFORMERS_DISABLED=1 AUTODEX_STOCK_ROOT={REMOTE} "
            f"PRECISION_CAPTURE_ASSET_ROOT={stage} PYTHONPATH={pythonpath} "
            f"{python} {module_root}/precision_init_daemon.py "
            "--port-mask 15006 --port-pose 15007 --port-cmd 16893")


def start() -> None:
    deploy()
    for kind, port in PORTS.items():
        log = REMOTE / f"{kind}.log"
        pid = REMOTE / f"{kind}.pid"
        command = _command(kind)
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
            "camera": "src/camera/server_daemon.py",
            "snapshot": str(REMOTE / "src/execution/daemon/snapshot_daemon.py"),
            "precision": str(REMOTE / "demo/precision-insertion/precision_init_daemon.py"),
        }[kind]
        _ssh(
            f"if test -f {shlex.quote(str(pid))}; then "
            f"p=$(cat {shlex.quote(str(pid))}); "
            "case $p in ''|*[!0-9]*) exit 1;; esac; "
            "if test ! -r /proc/$p/cmdline; then "
            f"echo '{kind}: already stopped'; "
            "else "
            "cmd=$(tr '\\000' ' ' < /proc/$p/cmdline); "
            f"case \"$cmd\" in *{shlex.quote(expected)}*) kill -TERM \"$p\";; "
            "*) echo 'PID no longer belongs to our service' >&2; exit 1;; "
            "esac; fi; "
            f"rm {shlex.quote(str(pid))}; echo '{kind}: stopped'; fi"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("deploy", "start", "check", "stop"))
    args = parser.parse_args()
    {"deploy": deploy, "start": start, "check": check, "stop": stop}[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
