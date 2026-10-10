"""Offline comparison of VLM-derived XY advice with independent held-key pose.

This evaluates *saved* read-only diagnostics. The reference pose must be
measured separately from the VLM/nominal wrist hypothesis; file hashes and
capture times bind evidence but cannot certify a measurement method's accuracy.
No metric here authorizes robot motion.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np


_SCHEMA = "precision_insertion_grounded_eval_manifest_v1"
_TRUTH_SCHEMA = "precision_insertion_independent_key_pose_v1"
_REPORT_SCHEMA = "precision_insertion_grounded_xy_diagnostic_v1"
_METHODS = frozenset({"external_optical_metrology", "independent_fiducial_pose"})
_CARDINAL_STEPS = ((.001, 0.), (-.001, 0.), (0., .001), (0., -.001))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} needs {size} finite components")
    return vector


def _true_residuals(truth: Mapping, alignment: Mapping
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tip = _finite_vector(truth.get("tip_socket_m"), 3, "independent tip")
    axis = _finite_vector(truth.get("insertion_axis_socket"), 3,
                          "independent insertion axis")
    norm = float(np.linalg.norm(axis))
    if abs(norm - 1.0) > 0.01 or axis[2] >= -1e-6:
        raise ValueError("independent insertion axis must be unit and downward")
    rim = float(alignment["socket_entry_plane_z_m"])
    depth = float(alignment["verification_depth_m"])
    if not all(math.isfinite(v) for v in (rim, depth)) or depth <= 0:
        raise ValueError("diagnostic has invalid socket rim/depth")
    entry = tip[:2] + (rim - tip[2]) * axis[:2] / axis[2]
    at_depth = tip[:2] + (rim - depth - tip[2]) * axis[:2] / axis[2]
    return (entry + at_depth) / 2, entry, at_depth


def _verify_diagnostic(path: Path) -> dict:
    report = _load(path)
    if report.get("schema") != _REPORT_SCHEMA or report.get("robot_ready") is not False:
        raise ValueError("not a read-only grounded XY diagnostic")
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("grounded diagnostic lacks saved source image hashes")
    for relative, expected in hashes.items():
        relative_path = Path(relative)
        if (relative_path.is_absolute() or ".." in relative_path.parts or
                len(expected) != 64):
            raise ValueError("unsafe or invalid diagnostic artifact path/hash")
        source = path.parent / relative_path
        if not source.is_file() or _sha(source) != expected:
            raise ValueError(f"diagnostic source image changed: {source}")
    frame_binding = report.get("frame_binding")
    if (not isinstance(frame_binding, dict) or not frame_binding or
            not isinstance(report.get("frame_request_id"), int) or
            not isinstance(report.get("alignment"), dict)):
        raise ValueError("diagnostic is missing camera/request binding")
    return report


def _verify_truth(path: Path, diagnostic: Mapping,
                  max_capture_skew_s: float,
                  max_translation_uncertainty_95_m: float,
                  max_axis_uncertainty_95_deg: float) -> dict:
    truth = _load(path)
    if (truth.get("schema") != _TRUTH_SCHEMA or
            truth.get("measurement_method") not in _METHODS):
        raise ValueError("reference pose is not a named independent measurement")
    for field in ("attempt_id", "candidate_id", "frame_request_id",
                  "session_calibration_sha256", "task_geometry_sha256"):
        if truth.get(field) != diagnostic.get(field):
            raise ValueError(f"independent reference differs on {field}")
    references = truth.get("source_files")
    if not isinstance(references, list) or not references:
        raise ValueError("independent reference needs source file hashes")
    for row in references:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("invalid independent source reference")
        source = Path(row["path"]).expanduser()
        if not source.is_absolute():
            raise ValueError("independent source path must be absolute")
        source = source.resolve()
        if not source.is_file() or _sha(source) != row["sha256"]:
            raise ValueError("independent reference source changed")
    capture_time = float(truth["capture_time_s"])
    if not math.isfinite(capture_time):
        raise ValueError("independent reference timestamp is invalid")
    for row in diagnostic["frame_binding"].values():
        timestamp = float(row["timestamp_s"])
        error = float(row["max_error_s"])
        if (not all(math.isfinite(x) for x in (timestamp, error)) or
                error < 0 or
                abs(capture_time - timestamp) > max_capture_skew_s + error):
            raise ValueError("independent pose is not synchronized with VLM frames")
    translation_uncertainty = float(truth["translation_uncertainty_95_m"])
    axis_uncertainty = float(truth["axis_uncertainty_95_deg"])
    if (not all(math.isfinite(x) and x >= 0 for x in (
            translation_uncertainty, axis_uncertainty)) or
            translation_uncertainty > max_translation_uncertainty_95_m or
            axis_uncertainty > max_axis_uncertainty_95_deg):
        raise ValueError("independent pose uncertainty exceeds evaluation budget")
    _true_residuals(truth, diagnostic["alignment"])
    return truth


def evaluate_grounding_manifest(manifest_path: Path) -> dict:
    """Evaluate saved proposals; never infer a threshold or pass robot mode."""
    manifest_file = Path(manifest_path).expanduser().resolve()
    manifest = _load(manifest_file)
    if manifest.get("schema") != _SCHEMA:
        raise ValueError("unknown grounded-alignment evaluation manifest")
    max_skew = float(manifest.get("max_reference_capture_skew_s", -1))
    max_translation_uncertainty = float(manifest.get(
        "max_truth_translation_uncertainty_95_m", -1))
    max_axis_uncertainty = float(manifest.get(
        "max_truth_axis_uncertainty_95_deg", -1))
    if (not all(math.isfinite(v) and v > 0 for v in (
            max_skew, max_translation_uncertainty, max_axis_uncertainty))):
        raise ValueError("commissioned reference/camera skew bound is required")
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("evaluation needs saved diagnostic/reference pairs")
    rows, seen = [], set()
    for item in samples:
        if not isinstance(item, dict) or set(item) != {
                "diagnostic_report", "independent_pose"}:
            raise ValueError("each evaluation sample needs exact file paths")
        report_path = (manifest_file.parent / item["diagnostic_report"]).resolve()
        truth_path = (manifest_file.parent / item["independent_pose"]).resolve()
        diagnostic = _verify_diagnostic(report_path)
        truth = _verify_truth(
            truth_path, diagnostic, max_skew,
            max_translation_uncertainty, max_axis_uncertainty)
        identity = (diagnostic["attempt_id"], diagnostic["frame_request_id"])
        if identity in seen:
            raise ValueError("duplicate attempt/camera request in evaluation")
        seen.add(identity)
        alignment = diagnostic["alignment"]
        true_mean, entry, depth = _true_residuals(truth, alignment)
        estimate = alignment.get("mean_error_xy_m")
        estimate_xy = (None if estimate is None else
                       _finite_vector(estimate, 2, "estimated lateral error"))
        estimated_axis = alignment.get("insertion_axis_socket")
        if estimated_axis is not None:
            estimated_axis = _finite_vector(estimated_axis, 3,
                                            "estimated insertion axis")
            axis_norm = float(np.linalg.norm(estimated_axis))
            if abs(axis_norm - 1.0) > .01:
                raise ValueError("reported insertion axis is not unit length")
            true_axis = _finite_vector(truth["insertion_axis_socket"], 3,
                                       "independent insertion axis")
            axis_error_deg = math.degrees(math.acos(float(np.clip(
                estimated_axis @ true_axis / axis_norm, -1, 1))))
        else:
            axis_error_deg = None
        suggested = alignment.get("step_socket_m")
        if (suggested is not None and
                (alignment.get("status") != "diagnostic_1mm_step" or
                 tuple(suggested) not in _CARDINAL_STEPS)):
            raise ValueError("diagnostic step is not a single cardinal millimetre")
        if alignment.get("status") == "diagnostic_1mm_step" and suggested is None:
            raise ValueError("advice status lacks its 1 mm step")
        truth_improvement = None
        best_improvement = max(-2 * float(true_mean @ np.asarray(step)) - 1e-6
                               for step in _CARDINAL_STEPS)
        if suggested is not None:
            delta = np.asarray(suggested, dtype=float)
            truth_improvement = -2 * float(true_mean @ delta) - 1e-6
        error_m = (None if estimate_xy is None else
                   float(np.linalg.norm(estimate_xy - true_mean)))
        radius = alignment.get("lateral_uncertainty_95_m")
        if radius is not None and (not math.isfinite(float(radius)) or radius <= 0):
            raise ValueError("invalid reported 95% lateral uncertainty")
        rows.append({
            "attempt_id": identity[0], "frame_request_id": identity[1],
            "diagnostic_report": str(report_path),
            "diagnostic_sha256": _sha(report_path),
            "independent_pose": str(truth_path),
            "independent_pose_sha256": _sha(truth_path),
            "measurement_method": truth["measurement_method"],
            "status": alignment.get("status"),
            "reason": alignment.get("reason"),
            "true_mean_lateral_xy_m": true_mean.tolist(),
            "true_entry_lateral_xy_m": entry.tolist(),
            "true_20mm_lateral_xy_m": depth.tolist(),
            "estimated_lateral_error_m": error_m,
            "axis_angle_error_deg": axis_error_deg,
            "inside_reported_95_radius": (None if error_m is None or radius is None
                                          else error_m <= radius),
            "advised_step_socket_m": suggested,
            "advised_step_true_improvement_m2": truth_improvement,
            "advised_step_reduces_true_error": (None if truth_improvement is None
                                               else truth_improvement > 0),
            "advised_step_is_best_cardinal": (
                None if truth_improvement is None else
                truth_improvement >= best_improvement - 1e-12),
        })
    advised = [row for row in rows if row["advised_step_socket_m"] is not None]
    errors = [row["estimated_lateral_error_m"] for row in rows
              if row["estimated_lateral_error_m"] is not None]
    axis_errors = [row["axis_angle_error_deg"] for row in rows
                   if row["axis_angle_error_deg"] is not None]
    covered = [row for row in rows
               if row["inside_reported_95_radius"] is not None]
    return {
        "schema": "precision_insertion_grounded_eval_report_v1",
        "manifest": str(manifest_file), "manifest_sha256": _sha(manifest_file),
        "sample_count": len(rows), "estimate_count": len(errors),
        "advice_count": len(advised),
        "abstention_count": len(rows) - len(advised),
        "false_advice_count": sum(row["advised_step_reduces_true_error"] is False
                                  for row in advised),
        "best_cardinal_advice_count": sum(
            row["advised_step_is_best_cardinal"] is True for row in advised),
        "empirical_95_radius_coverage": (
            None if not covered else sum(
                row["inside_reported_95_radius"] is True for row in covered
            ) / len(covered)),
        "median_lateral_error_m": (
            None if not errors else float(np.median(errors))),
        "p95_lateral_error_m": (
            None if not errors else float(np.percentile(errors, 95))),
        "median_axis_error_deg": (
            None if not axis_errors else float(np.median(axis_errors))),
        "p95_axis_error_deg": (
            None if not axis_errors else float(np.percentile(axis_errors, 95))),
        "samples": rows,
        "scope": "offline_external_pose_comparison_not_robot_motion_permission",
        "robot_ready": False,
    }
