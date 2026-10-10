#!/usr/bin/env python3
"""Batch-replay saved camera pairs against independent visual annotations.

This is a read-only VLM evaluation tool. It never acquires images, contacts a
robot or promotes an appearance label to insertion success. All image bytes
and human-annotation bytes are hash-bound before they are used.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
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


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LABELS = {
    "lift": ("class", frozenset({"held", "miss", "slip", "unobservable"}),
             "before_grasp", "after_lift"),
    "insertion_visual": (
        "visual_class", frozenset({"normal_appearance", "partial", "rim_jam",
                                  "slip", "unobservable"}),
        "preinsert", "final_or_abort"),
}


def _source_path(name: object, manifest_dir: Path) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("dataset source path must be a nonempty string")
    source = Path(name).expanduser()
    return (source if source.is_absolute() else manifest_dir / source).resolve()


def _hashed_bytes(source: Path, expected: object) -> bytes:
    if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
        raise ValueError("dataset source needs a lowercase SHA-256")
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"dataset source bytes changed: {source}")
    return payload


def _positive_time(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} needs a positive finite timestamp")
    return float(value)


def _load_cases(manifest_path: Path) -> tuple[list[dict], str]:
    """Validate immutable case metadata before loading a potentially large VLM."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if (not isinstance(manifest, dict) or manifest.get("schema") !=
            "precision_insertion_vlm_visual_benchmark_v1" or
            not isinstance(manifest.get("cases"), list) or
            not manifest["cases"]):
        raise ValueError("expected a nonempty visual benchmark manifest")
    max_skew = _positive_time(manifest.get("max_phase_skew_s"),
                              "maximum phase camera skew")
    seen_ids, seen_annotations = set(), set()
    cases = []
    for case in manifest["cases"]:
        if not isinstance(case, dict):
            raise ValueError("benchmark case must be an object")
        case_id, task = case.get("case_id"), case.get("task")
        if (not isinstance(case_id, str) or not _SAFE_ID.fullmatch(case_id) or
                case_id in seen_ids or task not in _LABELS):
            raise ValueError("duplicate/unsafe case ID or unknown benchmark task")
        seen_ids.add(case_id)
        annotation = case.get("annotation")
        if not isinstance(annotation, dict):
            raise ValueError("case requires a separate annotation file")
        annotation_path = _source_path(annotation.get("path"),
                                       manifest_path.parent)
        if annotation_path in seen_annotations:
            raise ValueError("cases cannot reuse one annotation file")
        seen_annotations.add(annotation_path)
        annotation_bytes = _hashed_bytes(annotation_path,
                                         annotation.get("sha256"))
        reviewed = json.loads(annotation_bytes)
        label_field, allowed, before, after = _LABELS[task]
        if (not isinstance(reviewed, dict) or
                reviewed.get("schema") !=
                "precision_insertion_visual_annotation_v1" or
                reviewed.get("case_id") != case_id or
                reviewed.get("task") != task or
                reviewed.get("label") not in allowed or
                reviewed.get("source") != "independent_human_review" or
                not isinstance(reviewed.get("reviewer_id"), str) or
                not reviewed["reviewer_id"].strip() or
                not isinstance(reviewed.get("evidence"), str) or
                not reviewed["evidence"].strip()):
            raise ValueError("case annotation is incomplete or mismatched")
        views = case.get("views")
        if not isinstance(views, list) or len(views) < 2:
            raise ValueError("benchmark needs at least two paired cameras")
        seen_cameras, phase_times = set(), {before: [], after: []}
        for view in views:
            camera = view.get("camera_id") if isinstance(view, dict) else None
            if (not isinstance(camera, str) or
                    not _SAFE_ID.fullmatch(camera) or camera in seen_cameras):
                raise ValueError("paired views need unique safe camera IDs")
            seen_cameras.add(camera)
            for phase in (before, after):
                row = view.get(phase)
                if (not isinstance(row, dict) or
                        not isinstance(row.get("sha256"), str) or
                        not _SHA256.fullmatch(row["sha256"])):
                    raise ValueError("every view phase needs a path and SHA-256")
                _hashed_bytes(_source_path(row.get("path"), manifest_path.parent),
                              row["sha256"])
                phase_times[phase].append(_positive_time(
                    row.get("timestamp_s"), f"{case_id}/{camera}/{phase}"))
            if phase_times[before][-1] >= phase_times[after][-1]:
                raise ValueError("paired camera phases are not time ordered")
        if any(max(times) - min(times) > max_skew
               for times in phase_times.values()):
            raise ValueError("benchmark camera phase exceeds stated skew")
        cases.append({"case_id": case_id, "task": task,
                      "truth": reviewed["label"],
                      "annotation_path": str(annotation_path),
                      "annotation_sha256": annotation["sha256"],
                      "views": views})
    return cases, hashlib.sha256(manifest_bytes).hexdigest()


def _frames_for_case(case: dict, manifest_dir: Path) -> tuple[list[LabeledFrame], list[dict]]:
    _, _, before, after = _LABELS[case["task"]]
    frames, sources = [], []
    for view in case["views"]:
        for phase in (before, after):
            row = view[phase]
            source = _source_path(row["path"], manifest_dir)
            raw = _hashed_bytes(source, row["sha256"])
            with Image.open(io.BytesIO(raw)) as image:
                rgb = image.convert("RGB")
            frames.append(LabeledFrame(view["camera_id"], phase,
                                       float(row["timestamp_s"]), rgb))
            sources.append({"camera_id": view["camera_id"], "phase": phase,
                            "path": str(source), "sha256": row["sha256"],
                            "size_px": list(rgb.size),
                            "timestamp_s": float(row["timestamp_s"])})
    return frames, sources


def _summarize(rows: list[dict]) -> dict:
    by_task = {}
    for task in _LABELS:
        subset = [row for row in rows if row["task"] == task]
        confusion = Counter((row["truth"], row["prediction"])
                            for row in subset)
        positive = "held" if task == "lift" else "normal_appearance"
        by_task[task] = {
            "count": len(subset),
            "correct": sum(row["correct"] for row in subset),
            "parse_failures": sum(row["parse_error"] is not None
                                  for row in subset),
            "target_class": positive,
            "target_class_false_positives": sum(
                row["prediction"] == positive and row["truth"] != positive
                for row in subset),
            "confusion": [
                {"truth": truth, "prediction": prediction, "count": count}
                for (truth, prediction), count in sorted(confusion.items())
            ],
        }
    return by_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=("local", "gemini"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-external-images", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-input-width", type=int, default=640)
    parser.add_argument("--max-input-height", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args(argv)
    if args.backend == "gemini" and not args.allow_external_images:
        parser.error("Gemini needs explicit --allow-external-images consent")
    if (args.max_input_width <= 0 or args.max_input_height <= 0 or
            args.max_new_tokens <= 0):
        parser.error("positive image/token limits required")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"benchmark report already exists: {output}")
    manifest_path = args.manifest.expanduser().resolve()
    cases, manifest_sha = _load_cases(manifest_path)
    backend = load_vlm_backend(
        mode=args.backend, model_id=args.model_id,
        max_input_size=(args.max_input_width, args.max_input_height),
        max_new_tokens=args.max_new_tokens, require_cuda=not args.allow_cpu,
        require_native_pixels=False)  # semantic classification only
    rows = []
    for case in cases:
        frames, sources = _frames_for_case(case, manifest_path.parent)
        observed = (observe_lift(backend, frames) if case["task"] == "lift"
                    else observe_insertion_visual(backend, frames))
        field = _LABELS[case["task"]][0]
        prediction = observed.parsed[field]
        rows.append({
            "case_id": case["case_id"], "task": case["task"],
            "truth": case["truth"], "prediction": prediction,
            "correct": prediction == case["truth"] and
                       observed.parse_error is None,
            "parse_error": observed.parse_error,
            "annotation_path": case["annotation_path"],
            "annotation_sha256": case["annotation_sha256"],
            "source_images": sources, "observation": observed.to_record(),
        })
    report = {
        "schema": "precision_insertion_vlm_visual_benchmark_report_v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "backend": args.backend, "model_id": args.model_id,
        "cases": rows, "summary": _summarize(rows),
        "scope": "visual_label_dataset_replay_not_physical_task_success",
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
