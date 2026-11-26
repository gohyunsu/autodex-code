"""Backend-independent signed-margin evaluators for constraint programs.

Physics and geometry backends are responsible for producing metric evidence
(relative transforms, pair separation, displacement and manifold distance).
This module applies the frozen task-agnostic vocabulary and sign convention.
It therefore does not pretend that an approximate local geometry library is a
physics simulator, and the same evaluator can consume MuJoCo, Isaac, CAD, or
real metrology evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from .program import validate_program


EntityPair = Tuple[str, ...]


class MissingConstraintEvidence(KeyError):
    """Raised when a backend did not provide a predicate's required evidence."""


def _entity_key(entities: Sequence[str]) -> EntityPair:
    return tuple(entities)


def _metric(mapping: Mapping[EntityPair, float], entities: Sequence[str], name: str) -> float:
    key = _entity_key(entities)
    if key not in mapping:
        raise MissingConstraintEvidence(f"missing {name} for entities {key}")
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name} for entities {key}")
    return value


def _transform(
    mapping: Mapping[EntityPair, np.ndarray],
    entities: Sequence[str],
) -> np.ndarray:
    key = _entity_key(entities)
    if key not in mapping:
        raise MissingConstraintEvidence(f"missing relative transform for entities {key}")
    transform = np.asarray(mapping[key], dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"invalid relative transform for entities {key}")
    return transform


def _rotation_angle(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.acos(cosine)


def _unit_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


@dataclass(frozen=True)
class ConstraintEvidence:
    """Metric evidence supplied by a simulator, CAD, or real-world backend."""

    relative_transforms: Mapping[EntityPair, np.ndarray] = field(default_factory=dict)
    pair_separation_m: Mapping[EntityPair, float] = field(default_factory=dict)
    entity_displacement_m: Mapping[EntityPair, float] = field(default_factory=dict)
    manifold_distance_m: Mapping[EntityPair, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MarginEvaluation:
    predicate_id: str
    predicate_type: str
    signed_margin: float
    tolerance: float
    normalized_margin: float
    satisfied: bool


def _tolerance(predicate: Mapping) -> float:
    specification = predicate.get("tolerance")
    if not specification:
        raise ValueError(
            f"predicate {predicate.get('predicate_id')} requires a metric tolerance")
    value = float(specification["value"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("constraint tolerance must be finite and non-negative")
    return value


def evaluate_predicate(predicate: Mapping, evidence: ConstraintEvidence) -> MarginEvaluation:
    """Evaluate one predicate with positive-is-satisfied sign convention."""

    predicate_type = predicate["type"]
    entities = predicate["entities"]
    tolerance = _tolerance(predicate)
    parameters = predicate.get("parameters") or {}

    if predicate_type == "pose_equality":
        actual = _transform(evidence.relative_transforms, entities)
        target = np.asarray(parameters.get("target_transform", np.eye(4)), dtype=float)
        if target.shape != (4, 4) or not np.isfinite(target).all():
            raise ValueError("target_transform must be a finite 4x4 matrix")
        residual = np.linalg.inv(target) @ actual
        translation_error = float(np.linalg.norm(residual[:3, 3]))
        orientation_mode = parameters.get("orientation_mode", "full_rotation")
        if orientation_mode == "full_rotation":
            rotation_error = _rotation_angle(residual[:3, :3])
        elif orientation_mode == "axis_alignment":
            local_axis = _unit_vector(
                parameters.get("local_axis", [0.0, 0.0, 1.0]),
                "local_axis")
            if "target_axis" in parameters:
                target_axis = _unit_vector(
                    parameters["target_axis"], "target_axis")
            else:
                # target_transform expresses the desired source pose in the
                # reference frame.  In the absence of an explicit target
                # axis, carry the declared source-local axis through that
                # desired rotation.  This keeps non-identity pose targets
                # consistent while quotienting only axial yaw.
                target_axis = target[:3, :3] @ local_axis
            actual_axis = actual[:3, :3] @ local_axis
            cosine = float(np.clip(np.dot(actual_axis, target_axis), -1.0, 1.0))
            if parameters.get("allow_axis_sign_flip", False):
                cosine = abs(cosine)
            rotation_error = math.acos(cosine)
        else:
            raise ValueError(f"unsupported orientation_mode: {orientation_mode}")
        rotation_scale = float(parameters.get("rotation_length_scale_m_per_rad", 0.05))
        if not math.isfinite(rotation_scale) or rotation_scale < 0:
            raise ValueError("rotation_length_scale_m_per_rad must be non-negative")
        metric_error = max(translation_error, rotation_scale * rotation_error)
        margin = tolerance - metric_error
    elif predicate_type in {"signed_clearance", "forbidden_contact"}:
        separation = _metric(
            evidence.pair_separation_m, entities, "pair separation")
        margin = separation - tolerance
    elif predicate_type == "required_contact":
        separation = _metric(
            evidence.pair_separation_m, entities, "pair separation")
        margin = tolerance - separation
    elif predicate_type == "disturbance_budget":
        displacement = _metric(
            evidence.entity_displacement_m, entities, "entity displacement")
        margin = tolerance - displacement
    elif predicate_type == "path_manifold":
        distance = _metric(
            evidence.manifold_distance_m, entities, "manifold distance")
        margin = tolerance - distance
    else:
        raise ValueError(f"unsupported predicate type: {predicate_type}")

    if tolerance > 0:
        normalized = margin / tolerance
    elif margin > 0:
        normalized = math.inf
    elif margin < 0:
        normalized = -math.inf
    else:
        normalized = 0.0
    return MarginEvaluation(
        predicate_id=predicate["predicate_id"],
        predicate_type=predicate_type,
        signed_margin=float(margin),
        tolerance=tolerance,
        normalized_margin=float(normalized),
        satisfied=bool(margin >= 0),
    )


def evaluate_stage(
    program: Mapping,
    stage_id: str,
    evidence: ConstraintEvidence,
) -> Dict[str, MarginEvaluation]:
    """Evaluate all predicates in one declared stage without task dispatch."""

    validate_program(program)
    for stage in program["stages"]:
        if stage["stage_id"] == stage_id:
            return {
                predicate["predicate_id"]: evaluate_predicate(predicate, evidence)
                for predicate in stage["predicates"]
            }
    raise KeyError(f"unknown stage_id: {stage_id}")
