#!/usr/bin/env python3
"""Opt-in, saved-image VLM smoke test; never connects to cameras or robot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

from PIL import Image


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.observer import (  # noqa: E402
    LabeledFrame, load_vlm_backend, observe_insertion_visual, observe_lift,
)


_CAMERA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "gemini"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--task", choices=("lift", "insertion_visual"),
                        required=True)
    parser.add_argument("--view", nargs=3, action="append", required=True,
                        metavar=("CAMERA", "BEFORE_PNG", "AFTER_PNG"),
                        help="repeat for calibrated camera views; order is preserved")
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON report; refuses overwrite")
    parser.add_argument("--allow-external-images", action="store_true",
                        help="required to send saved frames to Gemini API")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="slow local-model smoke test only")
    parser.add_argument("--max-input-width", type=int, default=640,
                        help="semantic-label resize ceiling only; not metric grounding")
    parser.add_argument("--max-input-height", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.backend == "gemini" and not args.allow_external_images:
        parser.error("Gemini needs explicit --allow-external-images consent")
    if (args.max_input_width <= 0 or args.max_input_height <= 0 or
            args.max_new_tokens <= 0):
        parser.error("positive local image/token limits required")
    camera_ids = [row[0] for row in args.view]
    if (len(set(camera_ids)) != len(camera_ids) or
            any(not _CAMERA.fullmatch(camera) for camera in camera_ids)):
        parser.error("unique path-safe camera IDs are required")
    before_phase, after_phase = (
        ("before_grasp", "after_lift") if args.task == "lift" else
        ("preinsert", "final_or_abort"))
    frames, sources = [], []
    for camera, before_name, after_name in args.view:
        for phase, stamp, name in ((before_phase, 0., before_name),
                                   (after_phase, 1., after_name)):
            path = Path(name).expanduser().resolve()
            raw = path.read_bytes()
            with Image.open(path) as image:
                rgb = image.convert("RGB")
            frames.append(LabeledFrame(camera, phase, stamp, rgb))
            sources.append({
                "camera_id": camera, "phase": phase, "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_px": list(rgb.size),
            })
    backend = load_vlm_backend(
        mode=args.backend, model_id=args.model_id,
        max_input_size=(args.max_input_width, args.max_input_height),
        max_new_tokens=args.max_new_tokens,
        require_cuda=not args.allow_cpu,
        require_native_pixels=False,  # semantic smoke test only
    )
    observed = (observe_lift(backend, frames) if args.task == "lift" else
                observe_insertion_visual(backend, frames))
    report = {
        "schema": "precision_insertion_saved_image_vlm_probe_v1",
        "backend": args.backend, "model_id": args.model_id,
        "task": args.task, "source_images": sources,
        "observation": observed.to_record(),
        "timestamp_source": "synthetic_order_only_not_camera_acquisition",
        "scope": "saved_image_semantic_probe_not_metric_grounding_or_task_label",
        "robot_ready": False,
    }
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(str(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
