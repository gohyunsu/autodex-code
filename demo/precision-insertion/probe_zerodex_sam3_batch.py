#!/usr/bin/env python3
"""Offline, camera-free SAM3 serial/batch comparison for the capture4 GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autodex.perception.mask import Sam3ImageSegmentor  # noqa: E402
from precision_insertion.sam3_batch import segment_batch  # noqa: E402


def _measure(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    masks = call()
    torch.cuda.synchronize()
    return masks, time.perf_counter() - start


def _iou(left, right):
    if left is None or right is None:
        return 1.0 if left is right else 0.0
    a, b = left > 0, right > 0
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    rgbs = []
    for path in args.images:
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise FileNotFoundError(path)
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    segmentor = Sam3ImageSegmentor(gpu=0)
    serial_call = lambda: [segmentor.segment(rgb, args.prompt) for rgb in rgbs]
    _, serial_cold_s = _measure(serial_call)
    serial, serial_s = _measure(serial_call)
    result = {"serial_cold_s": serial_cold_s, "serial_s": serial_s,
              "serial_pixels": [
        0 if mask is None else int(np.count_nonzero(mask)) for mask in serial]}
    for batch_size in dict.fromkeys((2, len(rgbs))):
        masks = []
        try:
            def batch_call():
                return [mask for offset in range(0, len(rgbs), batch_size)
                        for mask in segment_batch(
                            segmentor, rgbs[offset:offset + batch_size], args.prompt)]

            _, cold_s = _measure(batch_call)
            masks, warm_s = _measure(batch_call)
            result[f"batch{batch_size}_cold_s"] = cold_s
            result[f"batch{batch_size}_s"] = warm_s
            result[f"batch{batch_size}_pixels"] = [
                0 if mask is None else int(np.count_nonzero(mask)) for mask in masks]
            result[f"batch{batch_size}_iou_vs_serial"] = [
                _iou(left, right) for left, right in zip(serial, masks)]
            result[f"batch{batch_size}_peak_allocated_gib"] = round(
                torch.cuda.max_memory_allocated() / 2**30, 3)
        except (RuntimeError, ValueError) as exc:
            result[f"batch{batch_size}_error"] = repr(exc)
            torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
