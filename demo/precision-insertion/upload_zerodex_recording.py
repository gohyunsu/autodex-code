#!/usr/bin/env python3
"""Upload one finished capture4 recording to the independent NAS project.

Run this after run_zerodex_capture4.sh exits. No runtime path calls this
uploader, so NAS transfer cannot delay camera, perception or robot control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upload(receipt_path: Path, nas_root: Path, *, ssh_port: int = 77) -> Path:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != "precision_insertion_zerodex_recording_receipt_v1":
        raise ValueError("not a capture4 recording receipt")
    run_id = receipt["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or len(run_id) < 8:
        raise ValueError("invalid run ID")
    recording = receipt.get("recording")
    if not isinstance(recording, dict) or not recording.get("video"):
        raise ValueError("finished run has no recorded video")
    day = run_id[:8]
    if not re.fullmatch(r"\d{8}", day):
        raise ValueError("run ID does not begin with YYYYMMDD")
    destination = (nas_root.expanduser().resolve() / "runs" /
                   "zerodex_capture4" / day[:4] / day[4:6] /
                   day[6:8] / run_id)
    destination.mkdir(parents=True, exist_ok=True)
    source = (f"{receipt['remote_host']}:"
              f"{receipt['remote_directory'].rstrip('/')}/")
    subprocess.run(
        ["rsync", "-a", "--no-owner", "--no-group", "--no-perms",
         "-e", f"ssh -p {ssh_port} -o BatchMode=yes",
         source, str(destination) + "/"], check=True)
    video = destination / "video" / recording["video"]
    if _sha256(video) != recording["video_sha256"]:
        raise ValueError(f"uploaded video hash differs from recorder: {video}")
    shutil.copy2(receipt_path, destination / "recording_ref.json")
    summary = {
        "schema": "precision_insertion_zerodex_nas_run_v1",
        "run_id": run_id, "rig": "zerodex_capture4",
        "recording": str(video.relative_to(nas_root.resolve())),
        "video_sha256": recording["video_sha256"],
        "frame_count": recording["frame_count"],
        "pipeline_exit_code": receipt["pipeline_exit_code"],
        "code_commit": receipt.get("code_commit"),
    }
    (destination / "run_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--nas-root", type=Path,
                        default=Path("/mnt/paradex2/hyunsu/precision_insertion"))
    parser.add_argument("--ssh-port", type=int, default=77)
    args = parser.parse_args()
    print(upload(args.receipt, args.nas_root, ssh_port=args.ssh_port))


if __name__ == "__main__":
    main()
