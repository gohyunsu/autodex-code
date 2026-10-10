"""Read-only, grasp-specific key/hand calibration from *physical* trials.

This is a commissioning artifact, not an online key pose estimate or a robot
motion permit. A MuJoCo achieved pose is a useful prior but is not accepted as
a physical sample. In particular, empirical scatter is not a worst-case bound
on the next pickup; a later path gate must account for that distinction.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import TaskMode
from .geometry import pose_angle_deg, validate_se3
from .held_relation import resolve_postlift_held_relation


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def calibrate_physical_held_relation(
    *, mode: TaskMode, shared_root: Path, candidate_key: tuple[str, str, str],
    candidate_T_key_hand: np.ndarray, samples: Sequence[Mapping],
    minimum_independent_trials: int = 5,
    max_nominal_translation_drift_m: float,
    max_nominal_rotation_drift_deg: float,
) -> dict:
    """Summarize matched physical key pose, wrist pose and finger feedback.

    A sample must come from a *different physical pickup* of this candidate.
    Its key must be visible to an independent calibrated tracker during
    commissioning; this does not require the key to remain visible at runtime.
    Source records are hashed here, but their acquisition and error bounds
    must still be independently commissioned on the robot PC.
    """
    if (not isinstance(candidate_key, tuple) or len(candidate_key) != 3 or
            any(not isinstance(part, str) or not part for part in candidate_key)):
        raise ValueError("candidate_key must be the exact v8 scene/pose/index tuple")
    if (type(minimum_independent_trials) is not int or
            minimum_independent_trials < 3):
        raise ValueError("at least three independent pickups must be required")
    if len(samples) < minimum_independent_trials:
        raise ValueError("not enough independent physical pickup samples")
    translation_scale = _positive(
        max_nominal_translation_drift_m, "nominal translation scale")
    rotation_scale = _positive(
        max_nominal_rotation_drift_deg, "nominal rotation scale")
    nominal = validate_se3(candidate_T_key_hand, name="selected v8 T_key_hand")
    seen_trials: set[str] = set()
    seen_evidence: set[Path] = set()
    relations: list[np.ndarray] = []
    admitted: list[dict] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise TypeError("physical calibration sample must be a mapping")
        trial_id = sample.get("trial_id")
        if (not isinstance(trial_id, str) or not trial_id.strip() or
                trial_id in seen_trials):
            raise ValueError("calibration requires distinct physical trial IDs")
        seen_trials.add(trial_id)
        if sample.get("source") != "physical_independent_key_and_wrist":
            raise ValueError("MuJoCo, nominal and inferred key poses are not physical calibration")
        if tuple(sample.get("candidate_key", ())) != candidate_key:
            raise ValueError("physical sample belongs to a different v8 grasp")
        evidence_path = Path(sample.get("evidence_path", "")).expanduser()
        if not evidence_path.is_absolute() or not evidence_path.is_file():
            raise ValueError("physical sample needs an existing absolute evidence file")
        evidence_path = evidence_path.resolve()
        if evidence_path in seen_evidence:
            raise ValueError("independent pickups cannot reuse one evidence file")
        seen_evidence.add(evidence_path)
        evidence_hash = sample.get("evidence_sha256")
        if evidence_hash != _file_sha256(evidence_path):
            raise ValueError("physical calibration source evidence changed")
        evidence_record = json.loads(evidence_path.read_text(encoding="utf-8"))
        bound_fields = (
            "trial_id", "source", "candidate_key", "T_robot_key_observed",
            "T_robot_hand_measured", "hand_q_measured",
            "key_translation_error_bound_m", "key_rotation_error_bound_deg",
            "wrist_translation_error_bound_m", "wrist_rotation_error_bound_deg",
        )
        if (not isinstance(evidence_record, dict) or
                any(evidence_record.get(field) != sample.get(field)
                    for field in bound_fields)):
            raise ValueError("physical calibration fields differ from hashed evidence")
        key = validate_se3(sample.get("T_robot_key_observed"),
                           name=f"{trial_id} independently observed key")
        wrist = validate_se3(sample.get("T_robot_hand_measured"),
                             name=f"{trial_id} measured wrist")
        hand_q = np.asarray(sample.get("hand_q_measured"), dtype=np.float64)
        if hand_q.shape != (6,) or not np.all(np.isfinite(hand_q)):
            raise ValueError("physical sample needs six measured Inspire joints")
        key_translation_error = _positive(
            sample.get("key_translation_error_bound_m"),
            "key translation error bound")
        key_rotation_error = _positive(
            sample.get("key_rotation_error_bound_deg"),
            "key rotation error bound")
        wrist_translation_error = _positive(
            sample.get("wrist_translation_error_bound_m"),
            "wrist translation error bound")
        wrist_rotation_error = _positive(
            sample.get("wrist_rotation_error_bound_deg"),
            "wrist rotation error bound")
        resolved = resolve_postlift_held_relation(
            mode=mode, shared_root=shared_root,
            T_robot_key_observed=key, T_robot_hand_measured=wrist,
            candidate_T_key_hand=nominal,
            max_translation_drift_m=translation_scale,
            max_rotation_drift_deg=rotation_scale)
        relations.append(resolved.T_key_hand)
        admitted.append({
            "trial_id": trial_id,
            "candidate_key": list(candidate_key),
            "evidence_path": str(evidence_path.resolve()),
            "evidence_sha256": evidence_hash,
            "symmetry_branch": resolved.symmetry_branch,
            "T_key_hand": resolved.T_key_hand.tolist(),
            "hand_q_measured": hand_q.tolist(),
            "key_translation_error_bound_m": key_translation_error,
            "key_rotation_error_bound_deg": key_rotation_error,
            "wrist_translation_error_bound_m": wrist_translation_error,
            "wrist_rotation_error_bound_deg": wrist_rotation_error,
        })

    # A measured medoid is a real sample, unlike a component-wise average of
    # SE(3) matrices. Include stated measurement error in empirical radii.
    def disagreement(i: int, j: int) -> float:
        return (np.linalg.norm(relations[i][:3, 3] - relations[j][:3, 3]) /
                translation_scale +
                pose_angle_deg(relations[i], relations[j]) / rotation_scale)

    medoid_index = min(range(len(relations)), key=lambda i: (
        max(disagreement(i, j) for j in range(len(relations))),
        sum(disagreement(i, j) for j in range(len(relations)))))
    representative = relations[medoid_index]
    observed_translation_radius = max(
        float(np.linalg.norm(relation[:3, 3] - representative[:3, 3]))
        for relation in relations)
    observed_rotation_radius = max(
        pose_angle_deg(relation, representative) for relation in relations)
    stated_translation_error = max(
        row["key_translation_error_bound_m"] +
        row["wrist_translation_error_bound_m"] for row in admitted)
    stated_rotation_error = max(
        row["key_rotation_error_bound_deg"] +
        row["wrist_rotation_error_bound_deg"] for row in admitted)
    return {
        "schema": "precision_insertion_physical_grasp_calibration_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "candidate_key": list(candidate_key),
        "candidate_T_key_hand": nominal.tolist(),
        "candidate_T_key_hand_sha256": hashlib.sha256(
            np.ascontiguousarray(nominal, dtype=np.float64).tobytes()).hexdigest(),
        "selection_scales": {
            "max_nominal_translation_drift_m": translation_scale,
            "max_nominal_rotation_drift_deg": rotation_scale,
        },
        "source": "distinct_physical_pickups_with_independent_key_and_wrist_measurement",
        "sample_count": len(admitted),
        "minimum_independent_trials": minimum_independent_trials,
        "medoid_trial_id": admitted[medoid_index]["trial_id"],
        "T_key_hand_medoid": representative.tolist(),
        "hand_q_range": {
            "minimum": np.min([row["hand_q_measured"] for row in admitted], axis=0).tolist(),
            "maximum": np.max([row["hand_q_measured"] for row in admitted], axis=0).tolist(),
        },
        "empirical_translation_radius_m": observed_translation_radius,
        "empirical_rotation_radius_deg": observed_rotation_radius,
        "largest_stated_measurement_error_m": stated_translation_error,
        "largest_stated_measurement_error_deg": stated_rotation_error,
        "descriptive_translation_envelope_m": (
            observed_translation_radius + stated_translation_error),
        "descriptive_rotation_envelope_deg": (
            observed_rotation_radius + stated_rotation_error),
        "samples": admitted,
        "not_validated": [
            "future pickup repeatability or slip after the measured lift",
            "camera/robot error-bound commissioning and evidence authenticity",
            "20 mm endpoint, uncertain held path or guarded contact",
        ],
        "scope": "commissioning_summary_only_not_online_pose_or_motion_authorization",
        "robot_ready": False,
    }


def verify_physical_held_relation(
    *, record: Mapping, mode: TaskMode, shared_root: Path,
    candidate_key: tuple[str, str, str], candidate_dir: Path,
) -> dict:
    """Rebuild a summary from immutable source files and current v8 grasp.

    This proves internal file/summary consistency, not that the source files
    are genuine physical measurements or that their stated errors are sound.
    A moved or edited source file fails closed instead of silently promoting
    stale evidence into a runtime held-key transform.
    """
    if (not isinstance(record, Mapping) or
            record.get("schema") != "precision_insertion_physical_grasp_calibration_v1" or
            record.get("candidate_key") != list(candidate_key) or
            record.get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm,
                "key_object": mode.key_object,
                "socket_object": mode.socket_object,
            }):
        raise ValueError("physical calibration targets a different grasp or mode")
    candidate_path = Path(candidate_dir).expanduser().resolve() / "wrist_se3.npy"
    nominal = validate_se3(np.load(candidate_path, allow_pickle=False),
                           name="current selected v8 T_key_hand")
    scales = record.get("selection_scales")
    samples = record.get("samples")
    if not isinstance(scales, Mapping) or not isinstance(samples, list):
        raise ValueError("physical calibration lacks reconstruction inputs")
    inputs = []
    for row in samples:
        if not isinstance(row, Mapping):
            raise ValueError("invalid physical calibration source row")
        source_path = Path(row.get("evidence_path", "")).expanduser()
        if not source_path.is_absolute() or not source_path.is_file():
            raise ValueError("physical calibration source file is missing")
        source_hash = row.get("evidence_sha256")
        if source_hash != _file_sha256(source_path):
            raise ValueError("physical calibration source evidence changed")
        source_record = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source_record, dict):
            raise ValueError("physical calibration source is not a JSON object")
        inputs.append({**source_record,
                       "evidence_path": str(source_path),
                       "evidence_sha256": source_hash})
    rebuilt = calibrate_physical_held_relation(
        mode=mode, shared_root=shared_root, candidate_key=candidate_key,
        candidate_T_key_hand=nominal, samples=inputs,
        minimum_independent_trials=record.get("minimum_independent_trials"),
        max_nominal_translation_drift_m=scales.get(
            "max_nominal_translation_drift_m"),
        max_nominal_rotation_drift_deg=scales.get(
            "max_nominal_rotation_drift_deg"))
    if json.dumps(rebuilt, sort_keys=True, allow_nan=False) != json.dumps(
            dict(record), sort_keys=True, allow_nan=False):
        raise ValueError("physical calibration summary differs from sources or v8 candidate")
    return rebuilt
