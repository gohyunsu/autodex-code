"""One-camera, persistent Qwen worker reached over the capture PC's SSH stdin.

The worker reads only image bytes sent by the robot PC. It does not open a
camera, start ParaDex, or control the robot. All protocol messages are JSON
lines on stdout; model progress and errors stay on stderr.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

from PIL import Image

from .local_qwen import LocalQwenModel


def _isolate_protocol_stdout():
    """Reserve the JSON pipe before model loading or inference can print."""
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w",
                         encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return protocol


def _native_images(model: LocalQwenModel, images: list[Image.Image]) -> list[Image.Image]:
    processor = model.processor.image_processor
    factor = processor.patch_size * processor.merge_size
    minimum = processor.size["shortest_edge"]
    maximum = processor.size["longest_edge"]
    prepared = []
    for image in images:
        width = ((image.width + factor - 1) // factor) * factor
        height = ((image.height + factor - 1) // factor) * factor
        if (width > model.max_input_size[0] or
                height > model.max_input_size[1] or
                not minimum <= width * height <= maximum):
            raise ValueError("Qwen processor would resize native input pixels")
        if (width, height) == image.size:
            prepared.append(image)
        else:
            padded = Image.new("RGB", (width, height), (127, 127, 127))
            padded.paste(image, (0, 0))
            prepared.append(padded)
    return prepared


def _images(payload: dict) -> list[Image.Image]:
    encoded = payload.get("images_png_base64")
    if not isinstance(encoded, list) or not 1 <= len(encoded) <= 2:
        raise ValueError("worker needs one or two ordered PNG images")
    images = []
    for item in encoded:
        if not isinstance(item, str):
            raise ValueError("invalid PNG payload")
        data = base64.b64decode(item, validate=True)
        with Image.open(io.BytesIO(data)) as source:
            if source.format != "PNG" or source.mode != "RGB":
                raise ValueError("worker needs lossless RGB PNG input")
            images.append(source.copy())
    return images


def serve(*, camera_id: str, model_id: str,
          max_input_size: tuple[int, int], protocol=None) -> None:
    if protocol is None:
        protocol = _isolate_protocol_stdout()
    import torch

    model = LocalQwenModel(model_id=model_id, max_input_size=max_input_size)
    print(json.dumps({"type": "ready", "camera_id": camera_id,
                      "model_id": model_id,
                      "gpu_name": torch.cuda.get_device_name(0),
                      "cuda_allocated_bytes": torch.cuda.memory_allocated()}),
          file=protocol, flush=True)
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            if (request.get("type") == "shutdown" and
                    request.get("camera_id") == camera_id):
                print(json.dumps({"type": "stopped", "camera_id": camera_id}),
                      file=protocol, flush=True)
                break
            if request.get("type") != "infer" or request.get("camera_id") != camera_id:
                raise ValueError("request is for another camera")
            images = _images(request)
            if request.get("native_pixels") is True:
                images = _native_images(model, images)
            prompt = request.get("prompt")
            tokens = request.get("max_new_tokens")
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            answer = model.infer_images_prompt(
                images, prompt, max_new_tokens=tokens).answer
            reply = {"type": "result", "request_id": request.get("request_id"),
                     "camera_id": camera_id, "answer": answer,
                     "inference_s": time.perf_counter() - started,
                     "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                     "image_sizes": [list(image.size) for image in images]}
        except Exception as exc:
            reply = {"type": "error", "request_id": request.get("request_id")
                     if isinstance(request, dict) else None,
                     "camera_id": camera_id, "error": repr(exc)}
        print(json.dumps(reply, ensure_ascii=False), file=protocol, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--max-width", required=True, type=int)
    parser.add_argument("--max-height", required=True, type=int)
    args = parser.parse_args()
    serve(camera_id=args.camera_id, model_id=args.model_id,
          max_input_size=(args.max_width, args.max_height))


if __name__ == "__main__":
    main()
