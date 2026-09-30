#!/usr/bin/env python3
"""Align a separately recorded video to an AutoDex pipeline timeline.

For robust automatic alignment, connect the UTGE900 TTL output to a small LED
visible inside ``--roi`` and run the pipeline with ``--external-sync-cue``.
The script detects the start/end bursts and fits ``video_s = a*pipeline_s+b``.
Manual cue timestamps may be supplied when detection is inconvenient.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Iterable, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autodex.pipeline_edit import build_edit_package, load_events


def fit_time_transform(
    pipeline_seconds: Iterable[float], video_seconds: Iterable[float],
) -> dict:
    """Fit a clock offset and, with >=2 anchors, clock-rate drift."""
    xs = [float(value) for value in pipeline_seconds]
    ys = [float(value) for value in video_seconds]
    if len(xs) != len(ys) or not xs:
        raise ValueError("pipeline/video cue counts must match and be non-zero")
    if len(xs) == 1:
        slope = 1.0
        offset = ys[0] - xs[0]
    else:
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((value - x_mean) ** 2 for value in xs)
        if denom <= 0:
            raise ValueError("pipeline cue timestamps are not distinct")
        slope = sum((x - x_mean) * (y - y_mean)
                    for x, y in zip(xs, ys)) / denom
        offset = y_mean - slope * x_mean
    residuals = [y - (slope * x + offset) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    return {
        "equation": "video_time_s = slope * pipeline_time_s + offset_s",
        "slope": slope,
        "offset_s": offset,
        "drift_ppm": (slope - 1.0) * 1_000_000,
        "anchor_residual_rms_s": rms,
    }


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, width, height = (int(part) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height") from exc
    if min(x, y) < 0 or min(width, height) <= 0:
        raise argparse.ArgumentTypeError("ROI coordinates must be non-negative and size > 0")
    return x, y, width, height


def detect_led_bursts(
    video_path: Path, roi: tuple[int, int, int, int], *,
    threshold: Optional[float] = None, burst_gap_s: float = 1.5,
) -> tuple[list[float], dict]:
    """Return LED-burst onset timestamps using ROI mean brightness."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("automatic cue detection requires opencv-python and numpy") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    x, y, width, height = roi
    times: list[float] = []
    brightness: list[float] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        crop = frame[y:y + height, x:x + width]
        if crop.size == 0:
            capture.release()
            raise ValueError(f"ROI {roi} is outside video frame {frame.shape[1]}x{frame.shape[0]}")
        timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        timestamp_s = (timestamp_ms / 1000.0 if timestamp_ms > 0
                       else frame_index / fps)
        times.append(timestamp_s)
        brightness.append(float(np.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))))
        frame_index += 1
    capture.release()
    if not brightness or fps <= 0:
        raise RuntimeError("video has no decodable frames or valid FPS")

    values = np.asarray(brightness, dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    auto_threshold = median + max(8.0, 6.0 * mad)
    if auto_threshold >= float(np.max(values)):
        auto_threshold = median + 0.65 * (float(np.max(values)) - median)
    used_threshold = float(auto_threshold if threshold is None else threshold)
    active_times = [times[index] for index in np.flatnonzero(values >= used_threshold)]
    bursts: list[list[float]] = []
    for timestamp in active_times:
        if not bursts or timestamp - bursts[-1][-1] > burst_gap_s:
            bursts.append([timestamp])
        else:
            bursts[-1].append(timestamp)
    # A cue spans repeated flashes. Single bright frames are usually glare.
    bursts = [group for group in bursts if len(group) >= 2]
    return [group[0] for group in bursts], {
        "fps": fps,
        "frame_count": len(times),
        "roi": list(roi),
        "threshold": used_threshold,
        "brightness_median": median,
        "brightness_max": float(np.max(values)),
        "burst_frame_counts": [len(group) for group in bursts],
    }


def _cue_events(events: list[dict]) -> list[dict]:
    return [
        event for event in events
        if event.get("name") == "sync.visual_cue"
        and event.get("edge") == "instant"
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="Pipeline run directory")
    parser.add_argument("--video", required=True, help="External-camera video")
    parser.add_argument("--roi", type=_parse_roi,
                        help="Visible sync LED ROI as x,y,width,height")
    parser.add_argument("--cue-video-seconds", type=float, nargs="+",
                        help="Manual video times corresponding to logged cues")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Optional fixed LED ROI brightness threshold")
    parser.add_argument("--burst-gap-s", type=float, default=1.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run).expanduser().resolve()
    video_path = Path(args.video).expanduser().resolve()
    events = load_events(run_dir)
    cues = _cue_events(events)
    if not cues:
        parser.error("run contains no sync.visual_cue events; use --external-sync-cue")
    pipeline_cues = [float(event["pipeline_time_s"]) for event in cues]

    detection = None
    if args.cue_video_seconds is not None:
        video_cues = list(args.cue_video_seconds)
        source = "manual"
    else:
        if args.roi is None:
            parser.error("--roi is required unless --cue-video-seconds is supplied")
        video_cues, detection = detect_led_bursts(
            video_path, args.roi, threshold=args.threshold,
            burst_gap_s=args.burst_gap_s)
        source = "led_roi"
    if len(video_cues) != len(pipeline_cues):
        raise SystemExit(
            f"cue mismatch: pipeline logged {len(pipeline_cues)}, video has "
            f"{len(video_cues)}; inspect ROI or pass --cue-video-seconds")

    transform = fit_time_transform(pipeline_cues, video_cues)
    anchors = [
        {
            "label": event.get("attributes", {}).get("label"),
            "pipeline_time_s": pipeline_time,
            "video_time_s": video_time,
            "residual_s": video_time - (
                transform["slope"] * pipeline_time + transform["offset_s"]),
        }
        for event, pipeline_time, video_time in zip(
            cues, pipeline_cues, video_cues)
    ]
    output = (Path(args.output).expanduser().resolve() if args.output else
              run_dir / "sync" / "external_video_sync.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": cues[0].get("run_id"),
        "external_video": str(video_path),
        "source": source,
        "transform": transform,
        "anchors": anchors,
        "detection": detection,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    build_edit_package(run_dir, sync_path=output)
    print(f"sync: {output}")
    print(f"video_s = {transform['slope']:.9f} * pipeline_s "
          f"+ {transform['offset_s']:.6f} "
          f"(drift {transform['drift_ppm']:.1f} ppm)")


if __name__ == "__main__":
    main()
