#!/usr/bin/env python3
"""Compose nine rendered pipeline videos into a borderless 3x3 showcase.

The center video begins full-screen.  During the transition it shrinks into
the center cell, revealing the eight surrounding videos already playing at
their final grid positions.  Surrounding videos start from centered temporal
windows so the montage shows representative middle portions rather than nine
identical introductions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional


AUTODEX_PYTHON = Path("/home/robot/anaconda3/envs/autodex/bin/python")


def _ensure_runtime() -> None:
    if importlib.util.find_spec("cv2") and importlib.util.find_spec("numpy"):
        return
    current = Path(sys.executable).resolve()
    if AUTODEX_PYTHON.is_file() and current != AUTODEX_PYTHON.resolve():
        os.execv(
            str(AUTODEX_PYTHON),
            [str(AUTODEX_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
        )
    raise RuntimeError("OpenCV and NumPy are required to render the grid")


_ensure_runtime()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


GRID_POSITIONS = (
    (0, 0), (1, 0), (2, 0),
    (0, 1),         (2, 1),
    (0, 2), (1, 2), (2, 2),
)


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    duration_s: float
    frame_count: int


def probe_video(path: Path) -> VideoInfo:
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    video = next(
        stream for stream in payload.get("streams", [])
        if stream.get("codec_type") == "video"
    )
    numerator, denominator = str(video.get("avg_frame_rate", "0/1")).split("/")
    fps = float(numerator) / max(float(denominator), 1.0)
    duration_s = float(payload.get("format", {}).get("duration") or 0.0)
    return VideoInfo(
        path=str(path),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        duration_s=duration_s,
        frame_count=int(video.get("nb_frames") or round(duration_s * fps)),
    )


class FrameCursor:
    """Sequential decoder that serves frames at requested source timestamps."""

    def __init__(self, info: VideoInfo, start_s: float = 0.0) -> None:
        self.info = info
        self.capture = cv2.VideoCapture(info.path)
        if not self.capture.isOpened():
            raise RuntimeError(f"could not open video: {info.path}")
        self.frame: Optional[np.ndarray] = None
        self.frame_index = -1
        self.seek(start_s)

    def seek(self, time_s: float) -> None:
        target = max(0, min(round(time_s * self.info.fps), self.info.frame_count - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not seek to {time_s:.3f}s: {self.info.path}")
        self.frame = frame
        self.frame_index = target

    def at(self, time_s: float) -> np.ndarray:
        target = max(0, min(round(time_s * self.info.fps), self.info.frame_count - 1))
        if target < self.frame_index:
            self.seek(time_s)
        while self.frame_index < target:
            ok, frame = self.capture.read()
            if not ok or frame is None:
                break
            self.frame = frame
            self.frame_index += 1
        if self.frame is None:
            raise RuntimeError(f"no decoded frame: {self.info.path}")
        return self.frame

    def close(self) -> None:
        self.capture.release()


def _resize(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    interpolation = (
        cv2.INTER_AREA
        if frame.shape[1] > width or frame.shape[0] > height
        else cv2.INTER_LINEAR
    )
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _paste_clipped(
    canvas: np.ndarray, image: np.ndarray, x0: int, y0: int,
) -> None:
    """Paste an image whose virtual-grid bounds may extend past the canvas."""
    canvas_height, canvas_width = canvas.shape[:2]
    image_height, image_width = image.shape[:2]
    dst_x0, dst_y0 = max(0, x0), max(0, y0)
    dst_x1 = min(canvas_width, x0 + image_width)
    dst_y1 = min(canvas_height, y0 + image_height)
    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return
    src_x0, src_y0 = dst_x0 - x0, dst_y0 - y0
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    canvas[dst_y0:dst_y1, dst_x0:dst_x1] = image[
        src_y0:src_y1, src_x0:src_x1
    ]


def centered_window_start(duration_s: float, visible_s: float) -> float:
    """Choose a source offset whose visible window is centered in the video."""
    if visible_s >= duration_s:
        return 0.0
    return max(0.0, (duration_s - visible_s) / 2.0)


def _validate_geometry(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0 or width % 3 or height % 3:
        raise ValueError("output width and height must be positive multiples of 3")
    cell_width, cell_height = width // 3, height // 3
    if abs((width / height) - (cell_width / cell_height)) > 1e-9:
        raise ValueError("canvas and grid cells must have matching aspect ratios")
    return cell_width, cell_height


def render(args: argparse.Namespace) -> dict:
    center_path = args.center.expanduser().resolve()
    tile_paths = [path.expanduser().resolve() for path in args.tiles]
    missing = [str(path) for path in [center_path, *tile_paths] if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input video(s): " + ", ".join(missing))
    if args.transition_start < 0 or args.transition_duration <= 0:
        raise ValueError("transition times must be positive")
    transition_end = args.transition_start + args.transition_duration
    if transition_end >= args.duration:
        raise ValueError("transition must finish before the output ends")

    cell_width, cell_height = _validate_geometry(args.width, args.height)
    center_info = probe_video(center_path)
    tile_infos = [probe_video(path) for path in tile_paths]
    if args.duration > center_info.duration_s + (0.5 / args.fps):
        raise ValueError(
            f"requested {args.duration:.3f}s but center video is only "
            f"{center_info.duration_s:.3f}s"
        )

    visible_s = args.duration - args.transition_start
    automatic_offsets = [
        centered_window_start(info.duration_s, visible_s) for info in tile_infos
    ]
    if args.tile_starts:
        tile_offsets = []
        for value, automatic, info in zip(
            args.tile_starts, automatic_offsets, tile_infos
        ):
            offset = automatic if value == "auto" else float(value)
            if offset < 0 or offset + visible_s > info.duration_s + (0.5 / args.fps):
                raise ValueError(
                    f"tile start {offset:.3f}s cannot provide {visible_s:.3f}s "
                    f"from {info.path} ({info.duration_s:.3f}s)"
                )
            tile_offsets.append(offset)
    else:
        tile_offsets = automatic_offsets
    center_cursor = FrameCursor(center_info)
    tile_cursors = [
        FrameCursor(info, offset) for info, offset in zip(tile_infos, tile_offsets)
    ]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = round(args.duration * args.fps)
    ffmpeg = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{args.width}x{args.height}",
        "-r", f"{args.fps:g}", "-i", "-", "-an",
        "-c:v", "libx264", "-preset", args.preset,
        "-crf", str(args.crf), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ]
    encoder = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    next_progress = 0.05
    try:
        if encoder.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        for frame_number in range(frame_count):
            time_s = frame_number / args.fps
            center = center_cursor.at(time_s)
            if time_s < args.transition_start:
                canvas = _resize(center, args.width, args.height)
            else:
                canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                tile_elapsed = time_s - args.transition_start
                progress = _smoothstep(
                    (time_s - args.transition_start) / args.transition_duration
                )
                tile_frames = []
                for cursor, offset in zip(tile_cursors, tile_offsets):
                    source_s = min(offset + tile_elapsed, cursor.info.duration_s)
                    tile_frames.append(cursor.at(source_s))

                if args.transition_mode == "zoom-grid":
                    # View a pre-existing virtual 3x3 grid from its center.
                    # Each cell initially fills the viewport; the entire grid
                    # then zooms out until all nine cells exactly fill it.
                    zoom_width = round(
                        args.width + (cell_width - args.width) * progress
                    )
                    zoom_height = round(
                        args.height + (cell_height - args.height) * progress
                    )
                    center_x0 = (args.width - zoom_width) // 2
                    center_y0 = (args.height - zoom_height) // 2
                    frames_and_positions = [
                        *zip(tile_frames, GRID_POSITIONS),
                        (center, (1, 1)),
                    ]
                    for frame, (column, row) in frames_and_positions:
                        image = _resize(frame, zoom_width, zoom_height)
                        x0 = center_x0 + (column - 1) * zoom_width
                        y0 = center_y0 + (row - 1) * zoom_height
                        _paste_clipped(canvas, image, x0, y0)
                else:
                    for frame, (column, row) in zip(tile_frames, GRID_POSITIONS):
                        tile = _resize(frame, cell_width, cell_height)
                        x0, y0 = column * cell_width, row * cell_height
                        canvas[y0:y0 + cell_height, x0:x0 + cell_width] = tile
                    overlay_width = round(
                        args.width + (cell_width - args.width) * progress
                    )
                    overlay_height = round(
                        args.height + (cell_height - args.height) * progress
                    )
                    overlay = _resize(center, overlay_width, overlay_height)
                    x0 = (args.width - overlay_width) // 2
                    y0 = (args.height - overlay_height) // 2
                    canvas[y0:y0 + overlay_height, x0:x0 + overlay_width] = overlay

            encoder.stdin.write(canvas.tobytes())
            fraction = (frame_number + 1) / frame_count
            if fraction >= next_progress or frame_number + 1 == frame_count:
                print(f"[grid] {fraction * 100:5.1f}%", flush=True)
                next_progress += 0.05
        encoder.stdin.close()
        encoder.stdin = None
        return_code = encoder.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, ffmpeg)
    except BaseException:
        if encoder.stdin is not None:
            encoder.stdin.close()
        encoder.terminate()
        encoder.wait()
        raise
    finally:
        center_cursor.close()
        for cursor in tile_cursors:
            cursor.close()

    verification_info = probe_video(output)
    decoder = cv2.VideoCapture(str(output))
    decoded_frames = 0
    while True:
        ok, _ = decoder.read()
        if not ok:
            break
        decoded_frames += 1
    decoder.release()
    decode_passed = decoded_frames == frame_count
    if not decode_passed:
        raise RuntimeError(
            f"output decode stopped at {decoded_frames}/{frame_count} frames"
        )

    tile_records = []
    for info, offset, position in zip(tile_infos, tile_offsets, GRID_POSITIONS):
        tile_records.append({
            "video": asdict(info),
            "source_start_s": offset,
            "visible_duration_s": visible_s,
            "grid_column": position[0],
            "grid_row": position[1],
        })
    metadata = {
        "schema_version": 1,
        "rendered_utc": datetime.now(timezone.utc).isoformat(),
        "output_video": str(output),
        "output": {
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "duration_s": args.duration,
            "frame_count": frame_count,
        },
        "center": {
            "video": asdict(center_info),
            "source_start_s": 0.0,
            "grid_column": 1,
            "grid_row": 1,
        },
        "tiles": tile_records,
        "transition": {
            "start_s": args.transition_start,
            "duration_s": args.transition_duration,
            "end_s": transition_end,
            "easing": "smoothstep",
            "reveal": args.transition_mode,
            "borders": False,
            "gutters_px": 0,
        },
        "verification": {
            "decoded_frames": decoded_frames,
            "expected_frames": frame_count,
            "full_video_decode": "passed",
            "probed_duration_s": verification_info.duration_s,
        },
    }
    metadata_path = args.metadata or output.with_suffix(".json")
    metadata_path = metadata_path.expanduser().resolve()
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", type=Path, required=True)
    parser.add_argument(
        "--tiles", type=Path, nargs=8, required=True,
        metavar=("TL", "TC", "TR", "ML", "MR", "BL", "BC", "BR"),
        help="eight videos in row-major order, excluding the center cell",
    )
    parser.add_argument(
        "--tile-starts", nargs=8,
        metavar=("TL", "TC", "TR", "ML", "MR", "BL", "BC", "BR"),
        help="source start seconds for each tile; use 'auto' for centered windows",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--duration", type=float, default=50.0)
    parser.add_argument("--transition-start", type=float, default=33.0)
    parser.add_argument("--transition-duration", type=float, default=1.5)
    parser.add_argument(
        "--transition-mode", choices=("overlay", "zoom-grid"),
        default="overlay",
        help="overlay shrinks only the center; zoom-grid zooms out all cells",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = render(args)
    print(json.dumps({
        "output": metadata["output_video"],
        "duration_s": metadata["output"]["duration_s"],
        "transition": metadata["transition"],
        "full_video_decode": metadata["verification"]["full_video_decode"],
        "metadata": str((args.metadata or args.output.with_suffix(".json")).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
