"""Read-only evaluation of visible-rear key-depth estimates.

Independent depth references are supplied by the experimenter. Hash and time
checks make a saved comparison reproducible; they cannot establish that an
external instrument was calibrated or independent. Evaluation never promotes
the visual interval to a physical insertion-success source.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .calibration import SessionCalibration, _canonical_sha256
from .config import TaskMode
from .saved_exposed_depth import verify_saved_exposed_depth


_MANIFEST = "precision_insertion_exposed_depth_eval_manifest_v1"
_TRUTH = "precision_insertion_independent_depth_v1"
_METHODS = frozenset({"external_optical_metrology", "calibrated_depth_gauge"})
_TARGET_M = .020


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _relative(base: Path, value: str) -> Path:
    if not isinstance(value, str):
        raise ValueError("evaluation sample needs a relative file path")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("evaluation sample path must stay under its manifest")
    result = (base / path).resolve()
    if not result.is_relative_to(base):
        raise ValueError("evaluation sample path escapes its manifest")
    return result


def _file_reference(value: dict, *, name: str) -> Path:
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"} or
            not isinstance(value["path"], str) or
            not isinstance(value["sha256"], str)):
        raise ValueError(f"invalid {name} file reference")
    path = Path(value["path"])
    if (not path.is_absolute() or not path.is_file() or
            _sha(path) != value["sha256"]):
        raise ValueError(f"{name} file changed or is missing")
    return path.resolve()


def _truth_interval(truth: dict, report: dict, max_skew_s: float) -> tuple[float, float]:
    source = report["source"]
    binding = {
        "attempt_id": report["attempt_id"],
        "candidate_id": report["candidate_id"],
        "mode": report["mode"],
        "session_calibration_sha256": source["session_calibration_sha256"],
        "camera_calibration_sha256": source["camera_calibration_sha256"],
        "task_geometry_sha256": report["task_geometry_sha256"],
        "final_manifest_sha256": source["final_manifest_sha256"],
        "final_capture_id": source["capture_id"],
        "final_request_id": source["request_id"],
    }
    if (truth.get("schema") != _TRUTH or
            truth.get("measurement_method") not in _METHODS or
            any(truth.get(key) != value for key, value in binding.items())):
        raise ValueError("independent depth reference differs from final capture")
    evidence = truth.get("raw_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("independent depth needs raw instrument evidence")
    refs = [_file_reference(row, name="raw metrology") for row in evidence]
    refs.append(_file_reference(truth.get("instrument_calibration"),
                                name="instrument calibration"))
    if len(refs) != len(set(refs)):
        raise ValueError("independent depth repeats a source file")
    capture_hashes = set(source["image_file_sha256"].values())
    if any(_sha(ref) in capture_hashes for ref in refs):
        raise ValueError("independent depth reused a VLM camera image")
    interval = truth.get("depth_interval_m")
    if (not isinstance(interval, list) or len(interval) != 2 or
            any(type(value) not in (int, float) or not math.isfinite(value)
                for value in interval) or interval[0] > interval[1]):
        raise ValueError("independent depth needs a finite bounded interval")
    timestamp = truth.get("measurement_time_s")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise ValueError("independent depth needs a measurement timestamp")
    for frame in source["frame_evidence"].values():
        if (not isinstance(truth.get("clock_domain"), str) or
                truth["clock_domain"] != frame["clock_domain"]):
            raise ValueError("independent depth uses another timestamp clock")
        if abs(timestamp - frame["timestamp_s"]) > (
                max_skew_s + frame["max_error_s"]):
            raise ValueError("independent depth is not synchronized to final images")
    return float(interval[0]), float(interval[1])


def evaluate_exposed_depth_manifest(
    manifest_path: Path, *, mode: TaskMode, shared_root: Path,
    calibration: SessionCalibration,
) -> dict:
    """Score saved visual depth against separate measurements; never admit it.

    A sampled maximum error is descriptive, not a future worst-case guarantee.
    In particular, even zero observed false successes does not commission the
    endpoint or alter the insertion checkpoint's fail-closed depth gate.
    """
    path = Path(manifest_path).expanduser().resolve()
    manifest = _load(path)
    max_skew = manifest.get("max_reference_capture_skew_s")
    if (manifest.get("schema") != _MANIFEST or
            type(max_skew) not in (int, float) or
            not math.isfinite(max_skew) or max_skew <= 0 or
            not isinstance(manifest.get("samples"), list) or
            not manifest["samples"]):
        raise ValueError("invalid independent depth evaluation manifest")
    rows, seen = [], set()
    for sample in manifest["samples"]:
        if (not isinstance(sample, dict) or
                set(sample) != {"depth_report", "independent_depth"}):
            raise ValueError("depth sample needs report and independent reference")
        estimate_path = _relative(path.parent, sample["depth_report"])
        truth_path = _relative(path.parent, sample["independent_depth"])
        report = verify_saved_exposed_depth(
            estimate_path, mode=mode, shared_root=shared_root,
            calibration=calibration)
        truth = _load(truth_path)
        lower, upper = _truth_interval(truth, report, float(max_skew))
        identity = (report["attempt_id"], report["candidate_id"],
                    report["source"]["request_id"])
        if identity in seen:
            raise ValueError("duplicate depth trial/camera request")
        seen.add(identity)
        estimate = report["diagnostic"]
        result = {
            "attempt_id": identity[0], "candidate_id": identity[1],
            "final_request_id": identity[2],
            "depth_report": str(estimate_path),
            "depth_report_sha256": _sha(estimate_path),
            "independent_depth": str(truth_path),
            "independent_depth_sha256": _sha(truth_path),
            "model_id": report["vlm_observations"][0]["backend_model"],
            "status": estimate["status"],
            "truth_interval_m": [lower, upper],
            "truth_definitely_short": upper < _TARGET_M,
            "truth_definitely_reached_20mm": lower >= _TARGET_M,
        }
        if estimate["status"] == "bounded_visual_depth":
            estimated_lower, estimated_upper = estimate["key_depth_interval_m"]
            nominal = estimate["nominal_depth_m"]
            bound = estimate["worst_case_depth_error_bound_m"]
            result.update({
                "estimated_interval_m": [estimated_lower, estimated_upper],
                "nominal_depth_m": nominal,
                "declared_error_bound_m": bound,
                "required_overestimate_bound_m": max(0., nominal - lower),
                "truth_interval_contained": (
                    estimated_lower <= lower and upper <= estimated_upper),
                "visual_interval_claims_success": estimated_lower >= _TARGET_M,
                "definite_false_success": (
                    estimated_lower >= _TARGET_M and upper < _TARGET_M),
                "possibly_false_success": (
                    estimated_lower >= _TARGET_M and lower < _TARGET_M),
            })
        elif estimate["status"] == "abstain":
            result.update({
                "estimated_interval_m": None,
                "visual_interval_claims_success": False,
                "definite_false_success": False,
                "possibly_false_success": False,
                "truth_interval_contained": None,
            })
        else:
            raise ValueError("unsupported exposed-depth result status")
        rows.append(result)
    bounded = [row for row in rows if row["status"] == "bounded_visual_depth"]
    return {
        "schema": "precision_insertion_exposed_depth_eval_v1",
        "manifest_path": str(path), "manifest_sha256": _sha(path),
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm},
        "camera_calibration_sha256": calibration.record.get(
            "camera_calibration_sha256"),
        "session_calibration_sha256": _canonical_sha256(calibration.record),
        "target_depth_m": _TARGET_M,
        "max_reference_capture_skew_s": float(max_skew),
        "total": len(rows), "bounded": len(bounded),
        "abstained": len(rows) - len(bounded),
        "truth_interval_not_contained": sum(
            row["truth_interval_contained"] is False for row in rows),
        "definite_false_successes": sum(
            row["definite_false_success"] for row in rows),
        "possibly_false_successes": sum(
            row["possibly_false_success"] for row in rows),
        "max_observed_required_overestimate_bound_m": max(
            (row["required_overestimate_bound_m"] for row in bounded),
            default=None),
        "rows": rows,
        "depth_source_admissible_for_task_label": False,
        "scope": "heldout_depth_error_audit_not_commissioning_or_success_label",
        "robot_ready": False,
    }
