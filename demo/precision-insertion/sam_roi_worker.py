#!/usr/bin/env python3
"""Persistent, local SAM3 worker for exact saved-image key ROIs.

The robot-PC planner process uses a different Python environment. This
worker keeps the stock AutoDex SAM3 model warm in the installed autodex
environment and accepts only PNG image bytes over stdin. It opens no camera
or robot transport.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def _reply(value: dict) -> None:
    print(json.dumps(value, allow_nan=False), flush=True)


def _error_response(request: dict, exc: Exception) -> dict:
    return {"type": "error", "request_id": request.get("request_id"),
            "camera_id": request.get("camera_id"),
            "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    sam3_source = (Path(__file__).resolve().parents[2] /
                   "autodex/perception/thirdparty/sam3")
    if not sam3_source.is_dir():
        raise FileNotFoundError("the current AutoDex SAM3 source is missing")
    sys.path.insert(0, str(sam3_source))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 ROI worker needs CUDA")
    with torch.inference_mode():
        model = build_sam3_image_model(
            device="cuda", checkpoint_path=str(checkpoint),
            load_from_HF=False)
        processor = Sam3Processor(
            model, device="cuda", confidence_threshold=0.5)
    _reply({"type": "ready", "model": str(checkpoint)})
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("type") == "close":
            return 0
        request_id = request.get("request_id")
        try:
            if (request.get("type") != "segment" or
                    not isinstance(request_id, str) or
                    not isinstance(request.get("prompt"), str) or
                    not request["prompt"].strip()):
                raise ValueError("SAM3 worker needs an identified image and prompt")
            image = Image.open(io.BytesIO(base64.b64decode(
                request["image_png_base64"], validate=True))).convert("RGB")
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(), torch.autocast(
                    "cuda", dtype=torch.bfloat16):
                state = processor.set_image(image)
                state = processor.set_text_prompt(
                    state=state, prompt=request["prompt"])
                scores = state.get("scores")
                if scores is None or len(scores) == 0:
                    mask_png = None
                    score = None
                else:
                    best = int(scores.argmax().item())
                    mask = state["masks"][best].squeeze().cpu().numpy()
                    mask = np.asarray(mask, dtype=np.uint8) * 255
                    if mask.shape != (image.height, image.width):
                        raise ValueError("SAM3 ROI mask changed image geometry")
                    buffer = io.BytesIO()
                    Image.fromarray(mask, mode="L").save(buffer, format="PNG")
                    mask_png = base64.b64encode(buffer.getvalue()).decode("ascii")
                    score = float(scores[best].item())
            _reply({
                "type": "result", "request_id": request_id,
                "camera_id": request.get("camera_id"),
                "image_size_wh": [image.width, image.height],
                "mask_png_base64": mask_png, "score": score,
                "inference_s": time.perf_counter() - started,
                "cuda_peak_bytes": int(torch.cuda.max_memory_allocated()),
            })
        except Exception as exc:
            _reply(_error_response(request, exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
