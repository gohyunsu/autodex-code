#!/usr/bin/env python3
"""Profile immutable, saved precision-insertion evidence without live adapters.

The only writes are to --output-root. In particular, this does not instantiate
camera, SAM, FoundPose, planner, or robot controllers. It times the production
session verifier, catalog preflight, key QA byte/pixel verification, and the
production PNG encoder/hash function on the same saved camera pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
for root in (str(REPO_ROOT), str(DEMO_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.frame_provenance import image_sha256  # noqa: E402
from precision_insertion.live_session_runner import (  # noqa: E402
    _sha as stock_file_sha256, preflight_endpoint_catalog_file,
)
from precision_insertion.raw_camera_capture import write_png_jobs  # noqa: E402
from precision_insertion.session_bootstrap import (  # noqa: E402
    verify_session_evidence_bundle,
)
from precision_insertion.timing_profile import profile_run, timed  # noqa: E402


def _key_qa_frames(report_path: Path) -> tuple[dict, dict]:
    """Verify the saved QA request, retaining its sensor identity verbatim."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("schema") !=
            "precision_insertion_camera_only_key_socket_qa_v1" or
            not isinstance(report.get("request_id"), int) or
            not isinstance(report.get("views"), dict) or
            len(report["views"]) < 2):
        raise ValueError("expected an identified real key QA capture")
    images, sources = {}, {}
    for camera, row in sorted(report["views"].items()):
        source = row["raw_image"]
        path = Path(source["path"]).resolve()
        if not path.is_file() or path.suffix.lower() != ".png":
            raise ValueError(f"missing original camera PNG: {path}")
        with timed("replay.key_png_read_verify", phase="key_replay", kind="io",
                   camera_id=camera):
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"changed original key PNG: {path}")
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR)
            if image is None or image_sha256(image) != row[
                    "frame_evidence"]["image_sha256"]:
                raise ValueError(f"changed key frame pixels: {path}")
        frame = row["frame_evidence"]
        if (row["foundpose"]["frame_id"] != frame["frame_id"] or
                row["sam"]["frame_id"] != frame["frame_id"] or
                frame["timestamp_s"] != row["acquisition_timestamp_s"]):
            raise ValueError(f"key QA frame identity differs: {camera}")
        images[camera] = image
        sources[camera] = {
            "path": str(path), "file_sha256": source["sha256"],
            "pixel_sha256": frame["image_sha256"],
            "frame_id": frame["frame_id"],
            "timestamp_s": frame["timestamp_s"],
            "frame_evidence": frame,
            "size_px": [image.shape[1], image.shape[0]],
        }
    if set(images) != set(report["expected_camera_ids"]):
        raise ValueError("key QA camera set is incomplete")
    return images, {"request_id": report["request_id"], "cameras": sources,
                    "report_path": str(report_path),
                    "report_sha256": hashlib.sha256(
                        report_path.read_bytes()).hexdigest()}


def _png_batch(images: dict, target: Path, workers: int) -> dict:
    target.mkdir(parents=True, exist_ok=False)
    jobs = [(camera, target / f"{camera}.png", image, "saved_camera")
            for camera, image in sorted(images.items())]
    previous = os.environ.get("PRECISION_CAPTURE_IO_WORKERS")
    os.environ["PRECISION_CAPTURE_IO_WORKERS"] = str(workers)
    started = time.perf_counter()
    try:
        rows, actual_workers = write_png_jobs(jobs, phase="saved_replay")
    finally:
        if previous is None:
            os.environ.pop("PRECISION_CAPTURE_IO_WORKERS", None)
        else:
            os.environ["PRECISION_CAPTURE_IO_WORKERS"] = previous
    if actual_workers != min(workers, len(jobs)):
        raise ValueError("PNG worker setting was not applied")
    return {"elapsed_s": time.perf_counter() - started,
            "file_sha256": {camera: digest for camera, _source, digest in rows},
            "workers": workers,
            "image_count": len(jobs), "output_dir": str(target)}


def run(args: argparse.Namespace) -> dict:
    output = args.output_root.expanduser().resolve()
    key_report = args.key_report.expanduser().resolve()
    key_header = json.loads(key_report.read_text(encoding="utf-8"))
    raw_dirs = {Path(row["raw_image"]["path"]).resolve().parent
                for row in key_header["views"].values()}
    for source_root in (args.session_evidence, key_report.parent,
                        args.catalog.parent, *raw_dirs):
        if output == source_root.resolve() or source_root.resolve() in output.parents:
            raise ValueError("replay output cannot be inside source evidence")
    if output.exists():
        raise FileExistsError(f"replay output exists: {output}")
    output.mkdir(parents=True)
    evidence = args.session_evidence.expanduser().resolve()
    catalog_path = args.catalog.expanduser().resolve()
    catalog_header = json.loads(catalog_path.read_text(encoding="utf-8"))
    mode_data = catalog_header["mode"]
    mode = select_mode(mode_data["family"], mode_data["gap_mm"])
    shared_root = Path(catalog_header["shared_root"]).resolve()
    rows = []
    for iteration in range(args.repeats):
        row = {"iteration": iteration,
               "classification": "cold_first_replay" if iteration == 0
               else "warm_replay", "scope": "saved_data_only_no_live_io"}
        run_output = output / f"replay_{iteration:02d}.json"
        with profile_run(run_output):
            with timed("replay.total", phase="replay"):
                with timed("replay.session_evidence_verify", phase="evidence",
                           kind="io"):
                    verify_session_evidence_bundle(evidence)
                with timed("replay.catalog_preflight", phase="catalog",
                           kind="io"):
                    catalog = preflight_endpoint_catalog_file(
                        mode=mode, shared_root=shared_root,
                        catalog_path=catalog_path)
                with timed("replay.live_small_binding_hashes", phase="ledger",
                           kind="io"):
                    row["small_binding_hashes"] = {
                        name: stock_file_sha256(path) for name, path in (
                            ("session_manifest", evidence / "evidence_manifest.json"),
                            ("session_calibration", evidence /
                             "session_calibration.json"),
                            ("endpoint_catalog", catalog_path))}
                with timed("replay.key_qa_verify", phase="key_replay",
                           kind="io"):
                    images, provenance = _key_qa_frames(key_report)
                row["candidate_count"] = len(catalog["candidates"])
                row["eligible_count"] = catalog["eligible_count"]
                row["key_source"] = provenance
                row["png"] = []
                for workers in ((1, 4) if iteration % 2 == 0 else (4, 1)):
                    row["png"].append(_png_batch(
                        images, output / f"png_workers_{workers}" /
                        f"iteration_{iteration:02d}", workers))
                if row["png"][0]["file_sha256"] != row["png"][1]["file_sha256"]:
                    raise ValueError("PNG worker counts encoded different bytes")
        rows.append(row)
        run_output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    result = {"schema": "precision_insertion_saved_timing_replay_v1",
              "session_evidence": str(evidence), "key_report": str(key_report),
              "endpoint_catalog": str(catalog_path), "runs": rows,
              "warm_png_median_s": {str(workers): statistics.median(
                  next(p["elapsed_s"] for p in row["png"]
                       if p["workers"] == workers) for row in rows[1:])
                  for workers in (1, 4)},
              "unavailable": ["key_perception_model_inference",
                              "candidate_motion_planning", "lift_vlm",
                              "preinsert_vlm", "final_inside_socket_vlm",
                              "failed_trial_xy", "motor_motion",
                              "camera_network_round_trip", "contact_control"],
              "robot_ready": False}
    (output / "replay_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-evidence", type=Path, required=True)
    parser.add_argument("--key-report", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args(argv)
    if args.repeats < 4:
        parser.error("need first cold replay and at least three warm replays")
    result = run(args)
    print(args.output_root.expanduser().resolve() / "replay_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
