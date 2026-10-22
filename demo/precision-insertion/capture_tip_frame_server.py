#!/usr/bin/env python3
"""Serve a short, frame-ID-bound camera window from one capture PC.

This is a read-only CameraReader consumer. The stock camera daemon continues
to own acquisition. No robot or hand controller is opened here.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import time

import cv2
import zmq


@dataclass(frozen=True)
class BufferedFrame:
    source_frame_id: int
    read_unix_ns: int
    jpeg: bytes


class FrameWindowBuffer:
    def __init__(self, *, capacity: int = 180, jpeg_quality: int = 95):
        if capacity < 7 or not 50 <= jpeg_quality <= 100:
            raise ValueError("invalid camera window settings")
        self.frames: deque[BufferedFrame] = deque(maxlen=capacity)
        self.jpeg_quality = jpeg_quality

    def append(self, frame_id: int, read_unix_ns: int, image_bgr) -> None:
        if self.frames and frame_id <= self.frames[-1].source_frame_id:
            return
        success, encoded = cv2.imencode(
            ".jpg", image_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            raise OSError("could not encode camera history frame")
        self.frames.append(BufferedFrame(
            int(frame_id), int(read_unix_ns), encoded.tobytes()))

    def window(self, target_frame_id: int, *, before: int, after: int):
        if not self.frames:
            return "waiting", ()
        if self.frames[-1].source_frame_id < target_frame_id + after:
            return "waiting", ()
        chosen = tuple(frame for frame in self.frames
                       if target_frame_id - before <= frame.source_frame_id <=
                       target_frame_id + after)
        if not any(frame.source_frame_id == target_frame_id for frame in chosen):
            return "target_frame_missing", chosen
        return "complete", chosen


def serve(*, camera_id: str, port: int, buffer_frames: int,
          jpeg_quality: int) -> None:
    from paradex.io.camera_system.camera_reader import CameraReader

    reader = CameraReader(camera_id, timeout=5.0)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://0.0.0.0:{port}")
    buffer = FrameWindowBuffer(
        capacity=buffer_frames, jpeg_quality=jpeg_quality)
    last_id = reader.get_frame_id()
    pending = None
    try:
        while True:
            frame_id = reader.get_frame_id()
            if frame_id > last_id:
                image, frame_id = reader.get_image(copy=True)
                if frame_id > last_id:
                    buffer.append(frame_id, time.time_ns(), image)
                    last_id = frame_id
            if pending is None and socket.poll(timeout=0):
                request = socket.recv_json()
                if (request.get("op") == "ping" and
                        request.get("camera_id") == camera_id):
                    socket.send_json({"status": "ready", "camera_id": camera_id,
                                      "latest_source_frame_id": last_id})
                elif (request.get("op") == "window" and
                      request.get("camera_id") == camera_id and
                      type(request.get("target_frame_id")) is int and
                      0 <= request.get("before", -1) <= 8 and
                      0 <= request.get("after", -1) <= 8):
                    pending = (request, time.monotonic() + 2.0)
                else:
                    socket.send_json({"status": "invalid_request",
                                      "camera_id": camera_id})
            if pending is not None:
                request, deadline = pending
                target = request["target_frame_id"]
                status, chosen = buffer.window(
                    target, before=request["before"], after=request["after"])
                if status != "waiting" or time.monotonic() >= deadline:
                    header = {"status": status, "camera_id": camera_id,
                              "target_frame_id": target,
                              "frames": [{"source_frame_id": f.source_frame_id,
                                          "read_unix_ns": f.read_unix_ns}
                                         for f in chosen]}
                    socket.send_multipart(
                        [json.dumps(header).encode("utf-8"),
                         *(f.jpeg for f in chosen)])
                    pending = None
            time.sleep(.002)
    finally:
        socket.close()
        context.term()
        reader.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-id", default="25322651")
    parser.add_argument("--port", type=int, default=16951)
    parser.add_argument("--buffer-frames", type=int, default=180)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()
    serve(camera_id=args.camera_id, port=args.port,
          buffer_frames=args.buffer_frames,
          jpeg_quality=args.jpeg_quality)


if __name__ == "__main__":
    main()
