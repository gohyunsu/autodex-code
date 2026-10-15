#!/usr/bin/env python3
"""Compare exact old serial PNG loop and current worker loop on saved pixels."""

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2

root = Path("/home/robot/autodex-precision-insertion")
sys.path.insert(0, str(root / "demo/precision-insertion"))
sys.path.insert(1, str(root))
from replay_saved_timing import _key_qa_frames  # noqa: E402
from precision_insertion.raw_camera_capture import write_png_jobs  # noqa: E402

report = Path("/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical/strict_camera_only_static_interval_20261013_143002/key_socket_qa/report.json")
images, provenance = _key_qa_frames(report)
output = Path("/tmp/precision-pull-png-comparison-20261015")
if output.exists():
    raise FileExistsError(output)
output.mkdir()


def old_write(directory):
    directory.mkdir()
    result = {}
    for camera, image in sorted(images.items()):
        path = directory / f"{camera}.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(path)
        result[camera] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def new_write(directory, workers):
    directory.mkdir()
    jobs = [(camera, directory / f"{camera}.png", image, "saved_camera")
            for camera, image in sorted(images.items())]
    before = os.environ.get("PRECISION_CAPTURE_IO_WORKERS")
    os.environ["PRECISION_CAPTURE_IO_WORKERS"] = str(workers)
    try:
        rows, actual = write_png_jobs(jobs, phase="saved_replay")
    finally:
        if before is None:
            os.environ.pop("PRECISION_CAPTURE_IO_WORKERS", None)
        else:
            os.environ["PRECISION_CAPTURE_IO_WORKERS"] = before
    if actual != workers:
        raise ValueError(actual)
    return {camera: digest for camera, _source, digest in rows}


runs = []
for iteration in range(4):
    methods = ("old_serial", "new_1", "new_4")
    if iteration % 2:
        methods = tuple(reversed(methods))
    row = {"iteration": iteration, "seconds": {}}
    hashes = []
    for method in methods:
        directory = output / f"{method}_{iteration:02d}"
        started = time.perf_counter()
        digest = (old_write(directory) if method == "old_serial" else
                  new_write(directory, int(method.split("_")[-1])))
        row["seconds"][method] = time.perf_counter() - started
        hashes.append(digest)
    if hashes[0] != hashes[1] or hashes[1] != hashes[2]:
        raise ValueError("PNG bytes changed between methods")
    runs.append(row)
    print(json.dumps(row), flush=True)

summary = {"scope": "same_saved_20_png_pixels_local_tmp_encode_write_hash_only",
           "source_report": str(report),
           "source_report_sha256": provenance["report_sha256"],
           "camera_count": len(images),
           "image_size_px": [2048, 1536],
           "runs": runs,
           "warm_median_s": {method: statistics.median(
               row["seconds"][method] for row in runs[1:])
               for method in ("old_serial", "new_1", "new_4")}}
(output / "comparison.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n")
