#!/usr/bin/env python3
"""Read-only replay of a saved live inside_socket checkpoint and optional XY.

This accepts only an already completed, source-verified released checkpoint.
An optional XY input names the same failed trial's saved preinsert bundle,
frozen calibration, postlift report and commissioned alignment limits. It
calls the production checkpoint and grounded-alignment functions; no camera,
SAM, FoundPose, planner, or robot connection is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from PIL import Image

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
for root in (str(REPO_ROOT), str(DEMO_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from precision_insertion.calibration import load_session_calibration  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.grounded_alignment import AlignmentLimits  # noqa: E402
from precision_insertion.observer import load_vlm_backend  # noqa: E402
from precision_insertion.raw_camera_capture import (  # noqa: E402
    verify_raw_camera_capture,
)
from precision_insertion.released_insertion_checkpoint import (  # noqa: E402
    _image_frames, assess_released_insertion_checkpoint,
    verify_released_insertion_checkpoint,
)
from precision_insertion.retry_session import _compute_grounded_alignment  # noqa: E402
from precision_insertion.timing_profile import profile_run, timed  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(ref: dict) -> Path:
    if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"} or
            not isinstance(ref["path"], str) or
            not isinstance(ref["sha256"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", ref["sha256"])):
        raise ValueError("source needs an absolute path and SHA-256")
    path = Path(ref["path"])
    if not path.is_absolute() or not path.is_file() or _sha(path) != ref["sha256"]:
        raise ValueError("saved VLM replay source changed or is missing")
    return path.resolve()


def _xy_input(path: Path, released: dict) -> dict:
    """Validate a same-attempt failed-trial XY context before model loading."""
    context = json.loads(path.read_text(encoding="utf-8"))
    if (context.get("schema") != "precision_insertion_saved_xy_replay_input_v1" or
            released["outcome"]["insertion_success"] is not False or
            context.get("attempt_id") != released["attempt_id"]):
        raise ValueError("XY replay needs the same observed failed attempt")
    calibration_path = _source(context["session_calibration"])
    postlift_path = _source(context["postlift_preflight"])
    bundle_manifest = _source(context["preinsert_manifest"])
    bundle = bundle_manifest.parent
    if (bundle != Path(released["preinsert_bundle"]).resolve() or
            bundle_manifest.name != "manifest.json"):
        raise ValueError("XY preinsert source differs from failed attempt")
    preinsert = verify_raw_camera_capture(bundle, phase="preinsert")
    if not set(preinsert["frame_evidence"]) & {
            row["camera_id"] for row in released["frames"]}:
        raise ValueError("XY snapshot has no cameras in the failed trial")
    limits = AlignmentLimits(**context["alignment_limits"])
    limits.validate()
    mode = select_mode(context["mode"]["family"],
                       context["mode"]["gap_mm"])
    shared_root = Path(context["shared_root"]).resolve()
    calibration = load_session_calibration(
        calibration_path, mode=mode, shared_root=shared_root)
    cameras = sorted(preinsert["frame_evidence"])
    frames, sources = _image_frames(bundle, preinsert, cameras, "preinsert_hold")
    roi = context.get("roi_xyxy_by_camera")
    if roi is not None:
        if not isinstance(roi, dict) or set(roi) != set(cameras):
            raise ValueError("XY crop must cover every saved camera")
        roi = {camera: tuple(value) for camera, value in roi.items()}
    return {"mode": mode, "shared_root": shared_root,
            "calibration": calibration, "postlift_file": postlift_path,
            "frames": frames, "source_images": sources,
            "intrinsics_full": calibration.record["camera_calibration"]
            ["intrinsics_full"],
            "extrinsics_full": calibration.record["camera_calibration"]
            ["extrinsics_full"],
            "alignment_limits": limits,
            "axis_reference_key_z_m": context.get("axis_reference_key_z_m"),
            "roi_xyxy_by_camera": roi,
            "square_translation_only": context.get("square_translation_only", False),
            "search_index": context.get("search_index", 0)}


class MeasuredBackend:
    """Wrap the exact local backend without changing prompts or image pixels."""

    preferred_image_batch_size = 2
    native_pixel_coordinates = True
    max_parallel_views = 1

    def __init__(self, backend, iteration: int):
        self.backend = backend
        self.vlm = backend.vlm
        self.iteration = iteration
        self.calls: list[dict] = []

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        import torch
        match = re.search(r"(?:initial_tabletop|preinsert_hold)/([^@\s]+)@", prompt)
        if match is None:
            match = re.search(r"Camera ([A-Za-z0-9_.-]+)\.", prompt)
        camera = match.group(1) if match else None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        started = time.perf_counter()
        answer = self.backend.infer(images, prompt)
        torch.cuda.synchronize()
        self.calls.append({
            "iteration": self.iteration,
            "camera_id": camera,
            "camera_count": 1,
            "stage": ("grounded_axis" if len(images) == 1 else
                      "released_insertion_visual"),
            "image_count": len(images),
            "image_sizes_px": [list(image.size) for image in images],
            "model": self.vlm.model_id,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "inference_s": time.perf_counter() - started,
            "cuda_allocated_before_bytes": before,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })
        return answer


def run(args: argparse.Namespace) -> dict:
    output = args.output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError("replay output already exists")
    released_path = args.released_report.expanduser().resolve()
    released = verify_released_insertion_checkpoint(released_path)
    xy = _xy_input(args.xy_input.resolve(), released) if args.xy_input else None
    protected = (released_path.parent, Path(released["initial_key_bundle"]),
                 Path(released["final_bundle"]),
                 Path(released["preinsert_bundle"]))
    if args.xy_input:
        protected += (args.xy_input.resolve().parent,)
    for source in protected:
        if output == source or source in output.parents:
            raise ValueError("replay output cannot be inside original evidence")
    images = [Image.open(row["path"]) for row in released["frames"]]
    if xy is not None:
        images.extend(frame.image for frame in xy["frames"])
    ceiling = (max((image.width + 31) // 32 * 32 for image in images),
               max((image.height + 31) // 32 * 32 for image in images))
    for image in images[:len(released["frames"])]:
        image.close()
    output.mkdir(parents=True)
    rows = []
    backend = None
    for iteration in range(args.repeats):
        row = {"iteration": iteration, "calls": [],
               "classification": "cold_model_and_first_inference" if iteration == 0
               else "warm_inference"}
        with profile_run(output / f"replay_{iteration:02d}.json"):
            if backend is None:
                with timed("replay.model_load", phase="startup"):
                    started = time.perf_counter()
                    backend = load_vlm_backend(
                        mode="local", model_id=str(args.model_id.resolve()),
                        max_input_size=ceiling,
                        max_new_tokens=args.max_new_tokens,
                        require_cuda=True, require_native_pixels=True)
                    row["model_load_s"] = time.perf_counter() - started
            measured = MeasuredBackend(backend, iteration)
            with timed("replay.released_inside_socket", phase="vlm"):
                result = assess_released_insertion_checkpoint(
                    attempt_id=released["attempt_id"],
                    candidate_id=released["candidate_id"],
                    target_depth_m=released["target_depth_m"],
                    session_calibration_sha256=released[
                        "session_calibration_sha256"],
                    initial_key_bundle=Path(released["initial_key_bundle"]),
                    preinsert_bundle=Path(released["preinsert_bundle"]),
                    final_bundle=Path(released["final_bundle"]),
                    metric_record_path=Path(released["metric_record_path"]),
                    release_record_path=Path(released["release_record_path"]),
                    preinsert_reached_at_s=released["preinsert_reached_at_s"],
                    decision_timestamp_s=released["decision_timestamp_s"],
                    backend=measured,
                    max_phase_skew_s=released["max_phase_skew_s"],
                    max_preinsert_age_s=released["max_preinsert_age_s"],
                    max_final_observation_gap_s=released[
                        "max_final_observation_gap_s"],
                    minimum_visual_views=released["minimum_visual_views"],
                    label_mode=released["label_mode"],
                    manual_annotation=released["manual_annotation"])
            row["released_visual_class"] = result["effective_vlm_class"]
            if xy is not None:
                with timed("replay.saved_preinsert_xy_core", phase="xy"):
                    _geometry_hash, alignment, _observations = (
                        _compute_grounded_alignment(
                            mode=xy["mode"], root=xy["shared_root"],
                            calibration=xy["calibration"],
                            postlift_file=xy["postlift_file"],
                            frames=xy["frames"],
                            intrinsics_full=xy["intrinsics_full"],
                            extrinsics_full=xy["extrinsics_full"],
                            backend=measured,
                            alignment_limits=xy["alignment_limits"],
                            axis_reference_key_z_m=xy[
                                "axis_reference_key_z_m"],
                            roi_xyxy_by_camera=xy["roi_xyxy_by_camera"],
                            square_translation_only=xy[
                                "square_translation_only"],
                            search_index=xy["search_index"]))
                row["xy_status"] = alignment["status"]
            row["calls"] = measured.calls
        (output / f"replay_{iteration:02d}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n")
        rows.append(row)
    report = {"schema": "precision_insertion_saved_live_vlm_replay_v1",
              "released_report": str(released_path),
              "released_report_sha256": _sha(released_path),
              "source_images": released["frames"] + (
                  xy["source_images"] if xy is not None else []),
              "model": str(args.model_id.resolve()),
              "max_new_tokens": args.max_new_tokens,
              "runs": rows,
              "scope": "read_only_saved_checkpoint_inference_no_live_success_claim",
              "robot_ready": False}
    (output / "replay_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--released-report", type=Path, required=True)
    parser.add_argument("--xy-input", type=Path)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args(argv)
    if args.repeats < 4 or args.max_new_tokens < 1:
        parser.error("need first cold inference, three warm repeats and tokens")
    run(args)
    print(args.output_root.expanduser().resolve() / "replay_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
