"""Opt-in admission of a replayable visual/CAD key-depth interval.

The image geometry is replayed from saved pixels. A separate reviewed
held-out error audit supplies the physical error-bound claim; without it,
the same depth report remains diagnostic and cannot label 20 mm success.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .calibration import load_session_calibration
from .config import select_mode
from .saved_exposed_depth import verify_saved_exposed_depth


_SCHEMA = "precision_insertion_exposed_depth_commissioning_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit_exposed_depth(
    *, depth_report_path: Path, commissioning_path: Path,
    attempt_id: str, candidate_id: str,
    session_calibration_sha256: str, final_bundle: Path,
    guarded_completed_at_s: float, decision_timestamp_s: float,
) -> tuple[tuple[float, float], dict]:
    """Return an independently grounded depth only for matching live data.

    This verifies the report's saved pixels and VLM text, *not* future VLM
    accuracy. The reviewed audit must therefore cover the same camera model,
    VLM model, key/socket geometry and declared deterministic error limits.
    """
    source = Path(depth_report_path).expanduser().resolve()
    commissioned = Path(commissioning_path).expanduser().resolve()
    saved = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or not isinstance(saved.get("mode"), dict):
        raise ValueError("depth report has no task mode")
    mode = select_mode(saved["mode"]["family"], saved["mode"]["gap_mm"])
    root = Path(saved["shared_root"]).expanduser().resolve()
    calibration = load_session_calibration(
        Path(saved["source"]["session_calibration_path"]),
        mode=mode, shared_root=root)
    report = verify_saved_exposed_depth(
        source, mode=mode, shared_root=root, calibration=calibration)
    binding = report["source"]
    if (report["attempt_id"] != attempt_id or
            report["candidate_id"] != candidate_id or
            binding["session_calibration_sha256"] !=
                session_calibration_sha256 or
            Path(binding["final_bundle"]).resolve() !=
                Path(final_bundle).expanduser().resolve() or
            binding["completed_at_s"] !=
                float(guarded_completed_at_s) or
            binding["decision_timestamp_s"] !=
                float(decision_timestamp_s)):
        raise ValueError("depth report belongs to another guarded stroke")
    diagnostic = report["diagnostic"]
    if diagnostic["status"] != "bounded_visual_depth":
        raise ValueError("visible rear/axis depth estimate abstained")
    review = json.loads(commissioned.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("invalid depth commissioning record")
    mode_record = {
        "family": mode.family, "gap_mm": mode.gap_mm,
        "key_object": mode.key_object, "socket_object": mode.socket_object,
    }
    model = report["vlm_observations"][0]["backend_model"]
    if (review.get("schema") != _SCHEMA or
            review.get("mode") != mode_record or
            review.get("camera_calibration_sha256") !=
                binding["camera_calibration_sha256"] or
            review.get("task_geometry_sha256") !=
                report["task_geometry_sha256"] or
            review.get("vlm_model_id") != model or
            review.get("limits") != diagnostic["limits"] or
            not isinstance(review.get("reviewed_by"), str) or
            not review["reviewed_by"].strip()):
        raise ValueError("depth commissioning does not cover this estimator")
    minimum = review.get("minimum_heldout_trials")
    if type(minimum) is not int or minimum < 1:
        raise ValueError("depth commissioning needs an explicit trial requirement")
    reference = review.get("heldout_evaluation")
    if (not isinstance(reference, dict) or
            set(reference) != {"path", "sha256"} or
            not isinstance(reference["path"], str) or
            not isinstance(reference["sha256"], str)):
        raise ValueError("depth commissioning lacks a held-out audit reference")
    evaluation_path = Path(reference["path"]).expanduser().resolve()
    if _sha(evaluation_path) != reference["sha256"]:
        raise ValueError("held-out depth audit changed")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    bound = float(diagnostic["worst_case_depth_error_bound_m"])
    observed = evaluation.get("max_observed_required_overestimate_bound_m")
    if (evaluation.get("schema") != "precision_insertion_exposed_depth_eval_v1" or
            evaluation.get("mode") != {"family": mode.family,
                                       "gap_mm": mode.gap_mm} or
            evaluation.get("camera_calibration_sha256") !=
                binding["camera_calibration_sha256"] or
            evaluation.get("target_depth_m") != mode.target_depth_m or
            type(evaluation.get("total")) is not int or
            evaluation["total"] < minimum or
            evaluation.get("truth_interval_not_contained") != 0 or
            evaluation.get("definite_false_successes") != 0 or
            evaluation.get("possibly_false_successes") != 0 or
            type(observed) not in (float, int) or
            not math.isfinite(observed) or observed > bound):
        raise ValueError("held-out audit does not support the depth bound")
    interval = tuple(float(v) for v in diagnostic["key_depth_interval_m"])
    return interval, {
        "schema": "precision_insertion_key_depth_admissibility_v1",
        "status": "admitted_exposed_length_cad",
        "depth_report_path": str(source),
        "depth_report_sha256": _sha(source),
        "commissioning_path": str(commissioned),
        "commissioning_sha256": _sha(commissioned),
        "heldout_evaluation_path": str(evaluation_path),
        "heldout_evaluation_sha256": _sha(evaluation_path),
        "key_depth_source": "exposed_length_cad",
        "key_depth_interval_m": list(interval),
        "scope": "reviewed_visual_depth_bound_for_this_final_capture",
    }
