#!/usr/bin/env python3
"""Replay the current capture4 lift policy on saved attempts; never open hardware.

The input is the ordinary ZeroDex attempt root and the runtime JSON that will
be used for the next experiment.  Every historical lift report is verified
against its PNG bytes, frame metadata, calibration and measured joint state
before a model is loaded.  ``--verify-only`` performs that complete source
audit without CUDA, SSH, cameras or a robot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
for root in (str(REPO_ROOT), str(DEMO_DIR)):
    if root not in sys.path:
        sys.path.insert(0, root)

from precision_insertion.distributed_qwen import (  # noqa: E402
    DistributedQwenBackend,
)
from precision_insertion.lift_checkpoint import (  # noqa: E402
    verify_lift_checkpoint,
)
from precision_insertion.lift_visible_grip import (  # noqa: E402
    CAPTURE4_SCHEMA, observe_visible_grip, verify_visible_grip,
)
from precision_insertion.observer import (  # noqa: E402
    LabeledFrame, load_vlm_backend,
)
from precision_insertion.timing_profile import profile_run, timed  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _runtime_vlm(path: Path) -> tuple[dict, str]:
    payload = path.read_bytes()
    record = json.loads(payload)
    vlm = record.get("vlm") if isinstance(record, dict) else None
    if (not isinstance(vlm, dict) or vlm.get("mode") not in {
            "local", "distributed"} or
            type(vlm.get("max_new_tokens")) is not int or
            vlm["max_new_tokens"] <= 0 or
            not isinstance(vlm.get("model_id"), str) or
            not vlm["model_id"]):
        raise ValueError("runtime JSON needs a local or distributed VLM")
    return vlm, hashlib.sha256(payload).hexdigest()


def _latest_lift_reports(attempt_root: Path) -> list[Path]:
    reports = []
    for attempt in sorted(attempt_root.glob("attempt_*")):
        candidates = sorted((attempt / "lift_assessments").glob("*/report.json"))
        if candidates:
            reports.append(candidates[-1])
    if not reports:
        raise ValueError("attempt root contains no saved lift reports")
    return reports


def _verified_cases(attempt_root: Path) -> tuple[list[dict], tuple[int, int]]:
    root = attempt_root.expanduser().resolve()
    cases, maximum_width, maximum_height = [], 0, 0
    for report_path in _latest_lift_reports(root):
        report = verify_lift_checkpoint(report_path)
        annotation = report.get("manual_annotation")
        if (not isinstance(annotation, dict) or
                type(annotation.get("value")) is not bool):
            continue
        result_path = report_path.parents[2] / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        session_run = Path(result.get("session_run", "")).expanduser().resolve()
        calibration_path = session_run.parent / "frozen_session_calibration.json"
        if (not _within(report_path, root) or not _within(session_run, root) or
                not calibration_path.is_file()):
            raise ValueError("lift case references data outside its attempt root")
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        aggregation = report.get("visual", {}).get("aggregation", {})
        camera_ids = aggregation.get("camera_ids")
        if (not isinstance(camera_ids, list) or len(camera_ids) < 2 or
                len(set(camera_ids)) != len(camera_ids)):
            raise ValueError("capture4 lift report lacks its camera selection")
        frames = []
        for row in report["frames"]:
            source = Path(row["path"]).expanduser().resolve()
            if not _within(source, root):
                raise ValueError("lift image is outside its attempt root")
            with Image.open(source) as image:
                rgb = image.convert("RGB")
            maximum_width = max(maximum_width, rgb.width)
            maximum_height = max(maximum_height, rgb.height)
            frames.append(LabeledFrame(
                row["camera_id"], row["phase"], row["timestamp_s"], rgb))
        cases.append({
            "attempt_id": report["attempt_id"],
            "candidate_id": report["candidate_id"],
            "truth": "held" if annotation["value"] else "miss",
            "manual_note": annotation.get("note", ""),
            "report": report,
            "report_path": report_path,
            "report_sha256": _sha256(report_path),
            "calibration": calibration,
            "calibration_path": calibration_path,
            "calibration_sha256": _sha256(calibration_path),
            "camera_ids": tuple(camera_ids),
            "frames": frames,
        })
    if not cases:
        raise ValueError("attempt root has no boolean human lift annotations")
    ceiling = (((maximum_width + 31) // 32) * 32,
               ((maximum_height + 31) // 32) * 32)
    return cases, ceiling


def _open_backend(vlm: dict, *, ceiling: tuple[int, int], output: Path,
                  model_override: Path | None, allow_cpu: bool = False):
    model_id = str(model_override.expanduser().resolve()) if model_override else (
        vlm["model_id"])
    image_size = (tuple(vlm["classification_image_size"])
                  if vlm.get("classification_image_size") is not None else None)
    if vlm["mode"] == "distributed":
        return DistributedQwenBackend(
            model_id=model_id, worker_root=Path(vlm["worker_root"]),
            workers=vlm["workers"], max_input_size=ceiling,
            max_new_tokens=vlm["max_new_tokens"],
            classification_image_size=image_size,
            log_path=output.with_suffix(".distributed.jsonl"))
    return load_vlm_backend(
        mode="local", model_id=model_id, max_input_size=ceiling,
        max_new_tokens=vlm["max_new_tokens"], require_cuda=not allow_cpu,
        require_native_pixels=bool(vlm.get("require_native_pixels", True)),
        **({"classification_image_size": image_size}
           if image_size is not None else {}))


def _summary(rows: list[dict]) -> dict:
    confusion = Counter((row["truth"], row["prediction"]) for row in rows)
    return {
        "count": len(rows),
        "correct": sum(row["truth"] == row["prediction"] for row in rows),
        "unobservable": sum(row["prediction"] == "unobservable" for row in rows),
        "false_held": sum(row["truth"] == "miss" and
                          row["prediction"] == "held" for row in rows),
        "false_miss": sum(row["truth"] == "held" and
                          row["prediction"] == "miss" for row in rows),
        "confusion": [
            {"truth": truth, "prediction": prediction, "count": count}
            for (truth, prediction), count in sorted(confusion.items())],
    }


def run(args: argparse.Namespace) -> dict:
    attempt_root = args.attempt_root.expanduser().resolve()
    runtime_config = args.runtime_config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"replay output already exists: {output}")
    vlm, config_sha256 = _runtime_vlm(runtime_config)
    with timed("replay.lift_sources_verify", phase="replay", kind="io"):
        cases, ceiling = _verified_cases(attempt_root)
    common = {
        "schema": "precision_insertion_saved_lift_current_replay_v1",
        "attempt_root": str(attempt_root),
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": config_sha256,
        "backend_mode": vlm["mode"],
        "source_case_count": len(cases),
        "source_reports": [{
            "attempt_id": case["attempt_id"],
            "candidate_id": case["candidate_id"],
            "truth": case["truth"],
            "manual_note": case["manual_note"],
            "path": str(case["report_path"]),
            "sha256": case["report_sha256"],
            "calibration_path": str(case["calibration_path"]),
            "calibration_sha256": case["calibration_sha256"],
            "camera_ids": list(case["camera_ids"]),
        } for case in cases],
        "scope": "saved_images_only_no_camera_robot_or_physical_success_claim",
        "robot_ready": False,
    }
    if args.verify_only:
        report = {**common, "status": "sources_verified", "results": []}
    else:
        backend = _open_backend(
            vlm, ceiling=ceiling, output=output,
            model_override=args.model_id,
            allow_cpu=getattr(args, "allow_cpu", False))
        backend.supports_visible_grip_lift = True
        backend.supports_finger_focus_retry = True
        rows = []
        try:
            for case in cases:
                backend.visible_grip_camera_ids = case["camera_ids"]
                with timed("replay.lift_case", phase="replay",
                           attempt_id=case["attempt_id"]):
                    visual = observe_visible_grip(
                        backend, case["frames"],
                        Path(case["report"]["before_bundle"]),
                        calibration=case["calibration"],
                        joint_q=case["report"]["joint_feedback"]["full_q"],
                        robot_urdf=Path(
                            case["report"]["visual"]["aggregation"][
                                "robot_urdf"]))
                    saved = visual.to_record()
                    verify_visible_grip(
                        saved, set(case["camera_ids"]),
                        Path(case["report"]["before_bundle"]),
                        calibration=case["calibration"],
                        joint_q=case["report"]["joint_feedback"]["full_q"])
                rows.append({
                    "attempt_id": case["attempt_id"],
                    "candidate_id": case["candidate_id"],
                    "truth": case["truth"],
                    "prediction": visual.parsed["class"],
                    "correct": visual.parsed["class"] == case["truth"],
                    "visual": saved,
                })
        finally:
            close = getattr(backend, "close", None)
            if callable(close):
                close()
        report = {**common, "status": "inference_complete",
                  "summary": _summary(rows), "results": rows,
                  "current_visual_schema": CAPTURE4_SCHEMA}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", type=Path,
                        help="optional local model path override")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="allow a slow offline local-model replay without CUDA; live "
             "runtime requirements are unchanged")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    context = profile_run(args.output) if args.profile else None
    if context is None:
        report = run(args)
    else:
        with context:
            report = run(args)
    print(json.dumps({"output": str(args.output.expanduser().resolve()),
                      "status": report["status"],
                      "source_case_count": report["source_case_count"],
                      **({"summary": report["summary"]}
                         if "summary" in report else {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
