#!/usr/bin/env python3
"""Record one already-running capture-PC camera without changing its owner.

The stock camera daemon owns acquisition and publishes a double-buffered SHM
stream. This reader copies only the selected serial and writes an AVI plus a
line-per-frame index carrying the original frame IDs and read timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import cv2

from paradex.io.camera_system.camera_reader import CameraReader


def record(camera_id: str, output_dir: Path, *, fps: int, max_seconds: float,
           poll_seconds: float = 0.005) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    video_path = output_dir / f"{camera_id}.avi"
    index_path = output_dir / f"{camera_id}.frames.jsonl"
    stop = False

    def request_stop(_signal, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    reader = CameraReader(camera_id, timeout=5.0)
    writer = None
    count = 0
    skipped = 0
    first_id = None
    last_id = reader.get_frame_id()
    frame_period_ns = int(1_000_000_000 / fps)
    next_write_ns = time.monotonic_ns()
    started_ns = time.time_ns()
    status = "completed"
    try:
        with index_path.open("x", encoding="utf-8") as index:
            deadline = time.monotonic() + max_seconds
            while not stop and time.monotonic() < deadline:
                now_ns = time.monotonic_ns()
                if now_ns < next_write_ns:
                    time.sleep(min(poll_seconds,
                                   (next_write_ns - now_ns) / 1_000_000_000))
                    continue
                fid = reader.get_frame_id()
                if fid < last_id:
                    # The camera stream starts a new frame-ID sequence for
                    # this run; keep its first frames instead of waiting for
                    # the previous run's last ID.
                    last_id = 0
                if fid <= last_id:
                    time.sleep(poll_seconds)
                    continue
                image, fid = reader.get_image(copy=True)
                if fid <= last_id:
                    continue
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(video_path), cv2.VideoWriter_fourcc(*"MJPG"),
                        float(fps), (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"could not open video writer: {video_path}")
                read_ns = time.time_ns()
                skipped += max(0, fid - last_id - 1) if count else 0
                writer.write(image)
                index.write(json.dumps({"video_frame": count, "camera_id": camera_id,
                                        "source_frame_id": fid,
                                        "read_unix_ns": read_ns}) + "\n")
                if first_id is None:
                    first_id = fid
                last_id = fid
                count += 1
                next_write_ns = now_ns + frame_period_ns
    except Exception:
        status = "error"
        raise
    finally:
        if writer is not None:
            writer.release()
        reader.close()
        ended_ns = time.time_ns()
        manifest = {"status": status, "camera_id": camera_id, "fps": fps,
                    "frame_count": count, "first_source_frame_id": first_id,
                    "last_source_frame_id": last_id if count else None,
                    "source_frames_skipped": skipped,
                    "started_unix_ns": started_ns, "ended_unix_ns": ended_ns,
                    "video": video_path.name if count else None,
                    "frame_index": index_path.name}
        if count:
            digest = hashlib.sha256()
            with video_path.open("rb") as video:
                for chunk in iter(lambda: video.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            manifest["video_sha256"] = digest.hexdigest()
        (output_dir / "recording_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30,
                        help="AVI playback rate; selected FLIR view is 30 Hz")
    parser.add_argument("--max-seconds", type=float, default=900.0)
    args = parser.parse_args()
    if args.fps < 1 or args.max_seconds <= 0:
        parser.error("fps and max-seconds must be positive")
    print(json.dumps(record(args.camera_id, args.output_dir, fps=args.fps,
                            max_seconds=args.max_seconds)), flush=True)


if __name__ == "__main__":
    main()
