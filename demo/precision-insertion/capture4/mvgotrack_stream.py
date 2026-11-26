#!/usr/bin/env python3
"""Read capture4 ParaDex SHM and expose MV-GoTrack's persistent HTTP API.

This is an experiment-owned read-only consumer.  It neither starts cameras
nor changes the shared capture4 camera daemon.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import struct
import threading
import time
from urllib.parse import urlparse

import cv2

from paradex.io.camera_system.camera_reader import MultiCameraReader


HEADER = struct.Struct("!QQI")


class LatestFrames:
    def __init__(self, cameras: list[str], quality: int):
        self.cameras = cameras
        self.quality = quality
        self.reader = MultiCameraReader(camera_names=cameras)
        self.condition = threading.Condition()
        self.latest: dict[str, tuple[int, int, bytes]] = {}
        self.stop = threading.Event()
        self.threads = [threading.Thread(
            target=self._loop, args=(camera, reader), daemon=True,
            name=f"mvgotrack-shm-{camera}")
            for camera, reader in zip(self.reader.camera_names,
                                      self.reader.readers)]

    def start(self):
        for thread in self.threads:
            thread.start()

    def _loop(self, camera, reader):
        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        while not self.stop.is_set():
            try:
                frame, frame_id = reader.wait_for_new_frame(timeout=1.)
            except TimeoutError:
                continue
            ok, encoded = cv2.imencode(".jpg", frame, parameters)
            if not ok:
                continue
            row = (int(frame_id), time.monotonic_ns(), encoded.tobytes())
            with self.condition:
                self.latest[camera] = row
                self.condition.notify_all()

    def after(self, camera: str, frame_id: int):
        with self.condition:
            while not self.stop.is_set():
                row = self.latest.get(camera)
                if row is not None and row[0] > frame_id:
                    return row
                self.condition.wait(.2)
        return None

    def close(self):
        self.stop.set()
        with self.condition:
            self.condition.notify_all()
        for thread in self.threads:
            thread.join(timeout=2.)
        self.reader.close()


def handler(frames: LatestFrames):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/cameras":
                body = json.dumps({
                    "backend": "paradex_shm_read_only",
                    "cameras": [{"name": camera}
                                for camera in frames.cameras],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if not path.startswith("/stream/"):
                self.send_error(404)
                return
            camera = path.split("/", 2)[2]
            if camera not in frames.cameras:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            previous = -1
            try:
                while True:
                    row = frames.after(camera, previous)
                    if row is None:
                        return
                    previous, stamp_ns, jpeg = row
                    self.wfile.write(HEADER.pack(previous, stamp_ns, len(jpeg)))
                    self.wfile.write(jpeg)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
    return Handler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", required=True)
    parser.add_argument("--port", type=int, default=5484)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    frames = LatestFrames(list(args.cameras), args.jpeg_quality)
    frames.start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler(frames))
    try:
        server.serve_forever()
    finally:
        server.server_close()
        frames.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
