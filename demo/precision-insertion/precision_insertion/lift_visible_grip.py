"""Judge the post-lift key in two close camera views, regardless of its angle.

The initial key mask selects the same region in each saved post-lift frame.
The mask is a crop anchor; neither its pose nor the key's orientation is a
grasp label. Both camera workers answer independently and concurrently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

from PIL import Image

from .observer import (ImageVLM, LabeledFrame, VLMObservation,
                       resize_classification_images)
from .timing_profile import timed
from .view_parallel import map_view_calls


QUESTIONS = {
    "24122734": (
        "Is the small white cylindrical key visibly between the black robot "
        "fingers in this image, rather than loose on the checkerboard? "
        "Reply only YES or NO."),
    "25322638": (
        "Is the small white cylindrical key held between the black robot "
        "fingers and lifted clear of the checkerboard? "
        "Answer only YES or NO."),
}
IMAGE_SIZE = {"24122734": 384, "25322638": 512}
SCHEMA = "precision_insertion_visible_grip_v1"


def _vote(answer: str) -> bool | None:
    value = answer.strip().upper()
    return True if value == "YES" else False if value == "NO" else None


def _mask_roi(mask_path: Path, image_size: tuple[int, int]) -> tuple[int, ...]:
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        if mask.size != image_size:
            raise ValueError("key mask and after-lift image have different sizes")
        box = mask.getbbox()
    if box is None:
        raise ValueError("initial key mask is empty")
    width, height = image_size
    side = min(900, width, height)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    left = max(0, min(width - side, cx - side // 2))
    top = max(0, min(height - side, cy - 2 * side // 3))
    return left, top, left + side, top + side


def _crop(image: Image.Image, mask_path: Path) -> tuple[Image.Image, dict]:
    roi = _mask_roi(mask_path, image.size)
    return image.crop(roi), {
        "source_roi_xyxy": list(roi),
        "mask_path": str(mask_path),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
    }


def _answer(backend: ImageVLM, camera: str, frame: LabeledFrame,
            before_bundle: Path) -> dict:
    mask_path = before_bundle / "masks" / f"{camera}.png"
    cropped, source = _crop(frame.image, mask_path)
    size = IMAGE_SIZE[camera]
    prepared, transform = resize_classification_images([cropped], (size, size))
    prompt = QUESTIONS[camera]
    infer_for_camera = getattr(backend, "infer_for_camera", None)
    start = time.perf_counter()
    with timed("vlm.infer", phase="vlm", kind="inference",
               stage="post_lift", camera_id=camera, image_count=1,
               image_sizes=[list(prepared[0].size)],
               model=getattr(backend, "model", None)):
        raw = (infer_for_camera(camera, prepared, prompt)
               if callable(infer_for_camera) else backend.infer(prepared, prompt))
    if not isinstance(raw, str):
        raise TypeError("lift VLM response must be text")
    return {
        "camera_id": camera,
        "image_order": [f"after_lift/{camera}@{frame.timestamp_s:.6f}"],
        "prompt": prompt, "raw_answer": raw,
        "yes": _vote(raw),
        "latency_s": time.perf_counter() - start,
        "input_transform": transform,
        "crop": source,
    }


def _result(rows: list[dict]) -> dict:
    votes = [row["yes"] for row in rows]
    label = ("held" if votes == [True, True] else
             "miss" if votes == [False, False] else "unobservable")
    return {
        "class": label,
        "evidence": "Visible key between fingers in two views" if label == "held"
                    else "No visible held key in two views" if label == "miss"
                    else "Close views disagree or do not resolve the key",
        "evidence_views": list(QUESTIONS) if label != "unobservable" else [],
    }


def observe_visible_grip(
    backend: ImageVLM, frames: Sequence[LabeledFrame], before_bundle: Path,
) -> VLMObservation:
    paired = {(frame.phase, frame.camera_id): frame for frame in frames}
    cameras = {frame.camera_id for frame in frames}
    if (len(paired) != len(frames) or
            len(frames) != 2 * len(cameras) or
            any((phase, camera) not in paired
                for phase in ("before_grasp", "after_lift")
                for camera in cameras) or
            not set(QUESTIONS) <= cameras):
        raise ValueError("visible grip needs both saved phases and close cameras")
    source = Path(before_bundle).expanduser().resolve()
    rows = map_view_calls(
        backend, list(QUESTIONS),
        lambda camera: _answer(backend, camera, paired["after_lift", camera], source))
    parsed = _result(rows)
    order = tuple(
        f"{frame.phase}/{frame.camera_id}@{frame.timestamp_s:.6f}"
        for frame in frames)
    prompt = ("Images are supplied in this exact order:\n" +
              "\n".join(f"{i + 1}: {label}" for i, label in enumerate(order)) +
              "\n\nPost-lift close-view questions are recorded per camera in aggregation.")
    return VLMObservation(
        "post_lift", parsed, json.dumps(parsed, sort_keys=True), prompt,
        order, None, str(getattr(backend, "model", "local_qwen")),
        sum(row["latency_s"] for row in rows),
        aggregation={"schema": SCHEMA, "per_view": rows})


def verify_visible_grip(visual: dict, cameras: set[str],
                        before_bundle: Path) -> None:
    aggregation = visual.get("aggregation")
    rows = aggregation.get("per_view") if isinstance(aggregation, dict) else None
    if (not isinstance(aggregation, dict) or
            aggregation.get("schema") != SCHEMA or
            not set(QUESTIONS) <= cameras or
            not isinstance(rows, list) or
            [row.get("camera_id") for row in rows] != list(QUESTIONS)):
        raise ValueError("visible grip camera evidence changed")
    source = Path(before_bundle).expanduser().resolve()
    for row in rows:
        camera = row["camera_id"]
        mask_path = source / "masks" / f"{camera}.png"
        crop = row.get("crop")
        if not mask_path.is_file():
            raise ValueError("visible grip key mask is missing")
        with Image.open(mask_path) as mask:
            mask_size = mask.size
        order = next((item for item in visual.get("image_order", [])
                      if item.startswith(f"after_lift/{camera}@")), None)
        if (row.get("prompt") != QUESTIONS[camera] or
                not isinstance(row.get("raw_answer"), str) or
                row.get("yes") is not _vote(row["raw_answer"]) or
                row.get("input_transform", {}).get("max_size") !=
                [IMAGE_SIZE[camera], IMAGE_SIZE[camera]] or
                order is None or row.get("image_order") != [order] or
                not isinstance(crop, dict) or
                crop.get("mask_path") != str(mask_path) or
                crop.get("mask_sha256") !=
                hashlib.sha256(mask_path.read_bytes()).hexdigest() or
                crop.get("source_roi_xyxy") != list(_mask_roi(
                    mask_path, mask_size))):
            raise ValueError("visible grip prompt, crop or answer changed")
    expected = _result(rows)
    if (visual.get("parsed") != expected or
            visual.get("raw_answer") != json.dumps(expected, sort_keys=True) or
            visual.get("parse_error") is not None):
        raise ValueError("visible grip consensus changed")
