#!/usr/bin/env python3
"""Read-only audit of real AutoDex images and pending FoundPose onboarding.

Numerical silhouette loss is a diagnostic, not a real-image pose label. This
script never copies a candidate PTH into the canonical runtime asset tree.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safe_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("capture image has no relative file name")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("capture image path escapes its shot directory")
    return path


def audit_handoff(
    capture_index: Path, candidate_index: Path, evaluation_root: Path,
    *, verify_candidate_hashes: bool = False,
) -> dict:
    """Summarize byte integrity, coverage and *unmet* live-perception gates."""
    index = _read_json(Path(capture_index).expanduser().resolve())
    candidates = _read_json(Path(candidate_index).expanduser().resolve())
    root = Path(evaluation_root).expanduser().resolve()
    errors: list[str] = []
    expected_camera_ids: set[str] | None = None
    shot_rows: dict[str, dict] = {}
    calibration_sessions: set[str] = set()
    missing_timing = []
    selected = index.get("shots")
    if not isinstance(selected, list) or not selected:
        raise ValueError("capture index has no selected shots")
    for entry in selected:
        condition = entry.get("condition_id")
        if not isinstance(condition, str) or not condition:
            errors.append("capture index contains an unnamed condition")
            continue
        if condition in shot_rows:
            errors.append(f"duplicate selected condition: {condition}")
            continue
        shot_dir = Path(entry.get("shot_dir", "")).expanduser().resolve()
        shot_file = shot_dir / "shot.json"
        row = {"shot_dir": str(shot_dir), "camera_count": 0,
               "image_hashes_verified": False,
               "per_camera_acquisition_time": False}
        shot_rows[condition] = row
        if not shot_file.is_file():
            errors.append(f"{condition}: selected shot.json is missing")
            continue
        if _sha256(shot_file) != entry.get("shot_json_sha256"):
            errors.append(f"{condition}: selected shot.json hash changed")
            continue
        try:
            shot = _read_json(shot_file)
        except (OSError, ValueError) as exc:
            errors.append(f"{condition}: shot JSON is invalid: {exc}")
            continue
        images = shot.get("images")
        serials = shot.get("expected_camera_serials")
        if (shot.get("condition_id") != condition or
                shot.get("status") != "COMPLETE" or
                not isinstance(images, dict) or not images or
                not isinstance(serials, list) or
                len(set(serials)) != len(serials) or
                set(images) != set(serials) or
                entry.get("camera_count") != len(images)):
            errors.append(f"{condition}: shot condition/camera contract failed")
            continue
        camera_ids = set(images)
        if expected_camera_ids is None:
            expected_camera_ids = camera_ids
        elif camera_ids != expected_camera_ids:
            errors.append(f"{condition}: camera serial set changed")
        calibration_sessions.add(str(shot.get("calibration_session")))
        row["camera_count"] = len(images)
        image_errors = 0
        time_bound = True
        for serial, image in images.items():
            try:
                image_file = _safe_child(shot_dir, image.get("file"))
                if (not image_file.is_file() or
                        image_file.stat().st_size != image.get("bytes") or
                        _sha256(image_file) != image.get("sha256")):
                    image_errors += 1
            except (OSError, TypeError, ValueError):
                image_errors += 1
            if (type(image.get("frame_id")) is not int or
                    not isinstance(image.get("exposure_utc_s"), (int, float)) or
                    not math.isfinite(image["exposure_utc_s"]) or
                    not isinstance(image.get("max_error_s"), (int, float)) or
                    not math.isfinite(image["max_error_s"]) or
                    image["max_error_s"] <= 0):
                time_bound = False
        if image_errors:
            errors.append(f"{condition}: {image_errors} camera image(s) failed hash check")
        row["image_hashes_verified"] = image_errors == 0
        row["per_camera_acquisition_time"] = time_bound
        if not time_bound:
            missing_timing.append(condition)

    if index.get("condition_count") != len(selected):
        errors.append("capture index condition_count differs from shot entries")
    if len(calibration_sessions) != 1:
        errors.append("selected captures use different or missing calibrations")
    missing_planned = index.get("missing_planned_conditions", [])
    if not isinstance(missing_planned, list):
        errors.append("missing_planned_conditions is not a list")
        missing_planned = []
    candidate_rows = []
    for entry in candidates.get("candidates", []):
        path = Path(entry.get("path", "")).expanduser().resolve()
        present = path.is_file() and path.stat().st_size == entry.get("bytes")
        hash_ok = None
        if verify_candidate_hashes and present:
            hash_ok = _sha256(path) == entry.get("sha256")
        if not present or hash_ok is False:
            errors.append(f"{entry.get('object')}: candidate PTH missing or changed")
        candidate_rows.append({
            "object": entry.get("object"), "path": str(path),
            "present_expected_size": present, "hash_verified": hash_ok,
            "source_status": entry.get("status"),
            "canonical_runtime_asset": False,
        })

    evaluations = []
    by_variant: dict[str, dict[str, int]] = {}
    for report_file in sorted((root / "results").rglob("report.json")):
        try:
            report = _read_json(report_file)
        except (OSError, ValueError):
            # A producer may still be writing into a .partial directory.
            continue
        condition = report.get("condition_id")
        shot = shot_rows.get(condition)
        report_shot = Path(report.get("shot_dir", "")).expanduser().resolve()
        shot_match = shot is not None and report_shot == Path(shot["shot_dir"])
        if not shot_match:
            errors.append(f"{condition}: evaluation is not bound to selected shot")
        n_valid = report.get("n_valid_mask_pose")
        n_expected = report.get("n_expected")
        numeric = report.get("passes_numeric_sil_threshold") is True
        variant = report_file.relative_to(root / "results").parts[0]
        tally = by_variant.setdefault(variant, {"reports": 0,
                                                 "numeric_pass": 0})
        tally["reports"] += 1
        tally["numeric_pass"] += int(numeric)
        per_view_iou = report.get("per_view_iou", {})
        valid_iou = ([float(value) for value in per_view_iou.values()
                      if isinstance(value, (int, float)) and
                      math.isfinite(value)]
                     if isinstance(per_view_iou, dict) else [])
        if (type(n_valid) is not int or type(n_expected) is not int or
                not 0 <= n_valid <= n_expected or
                (shot is not None and n_expected != shot["camera_count"])):
            errors.append(f"{condition}: evaluation view counts are invalid")
        evaluations.append({
            "condition_id": condition, "report": str(report_file),
            "report_sha256": _sha256(report_file),
            "evaluation_variant": variant,
            "selected_shot_match": shot_match,
            "n_valid_mask_pose": n_valid,
            "n_expected": n_expected,
            "numeric_silhouette_pass": numeric,
            "final_mean_iou": report.get("final_mean_iou_all_valid_masks"),
            "minimum_per_view_iou": min(valid_iou) if valid_iou else None,
            "zero_iou_view_count": sum(value <= 0 for value in valid_iou),
            "manual_pose_mask_rim_review": "REQUIRED_NOT_PROVEN",
        })
    evaluated = {row["condition_id"] for row in evaluations}
    not_evaluated = sorted(set(shot_rows) - {"charuco_only"} - evaluated)
    return {
        "schema": "precision_insertion_nas_perception_audit_v1",
        "sampled_at_utc": datetime.now(timezone.utc).isoformat(),
        "capture_index": str(Path(capture_index).resolve()),
        "candidate_index": str(Path(candidate_index).resolve()),
        "evaluation_root": str(root),
        "selected_condition_count": len(shot_rows),
        "missing_planned_conditions": missing_planned,
        "camera_ids": sorted(expected_camera_ids or ()),
        "calibration_sessions": sorted(calibration_sessions),
        "shots": shot_rows,
        "candidate_representations": candidate_rows,
        "evaluations": evaluations,
        "evaluation_variant_summary": by_variant,
        "evaluation_count": len(evaluations),
        "numeric_silhouette_pass_count": sum(
            row["numeric_silhouette_pass"] for row in evaluations),
        "selected_conditions_without_evaluation": not_evaluated,
        "conditions_without_per_camera_acquisition_time": missing_timing,
        "integrity_errors": errors,
        "capture_integrity_pass": not errors and all(
            row["image_hashes_verified"] for row in shot_rows.values()),
        "perception_promotion_ready": False,
        "session_start_eligible": False,
        "reason": (
            "Real images and numeric silhouette fits are not a commissioned "
            "pose/clock validation. Candidate PTHs remain noncanonical; repeated "
            "same-session socket poses and manual rim/orientation review are required."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--candidate-index", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--verify-candidate-hashes", action="store_true")
    parser.add_argument("--output", type=Path,
                        help="optional new JSON report; refuses to overwrite")
    args = parser.parse_args(argv)
    result = audit_handoff(
        args.capture_index, args.candidate_index, args.evaluation_root,
        verify_candidate_hashes=args.verify_candidate_hashes)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        target = args.output.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")
    return 0 if not result["integrity_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
