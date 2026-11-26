"""Task-agnostic MuJoCo adapter for :mod:`precision_layer.evaluator`.

The adapter maps named MuJoCo frames and geom pairs into metric evidence.  A
task supplies only declarative bindings; no task name or task-specific control
branch appears in the extraction code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Tuple

import numpy as np

from .evaluator import ConstraintEvidence, EntityPair


@dataclass(frozen=True)
class FrameBinding:
    kind: str
    name: str

    def validate(self) -> None:
        if self.kind not in {"body", "site"}:
            raise ValueError("MuJoCo frame kind must be 'body' or 'site'")
        if not self.name:
            raise ValueError("MuJoCo frame name must be non-empty")


@dataclass(frozen=True)
class RelativeTransformBinding:
    source: FrameBinding
    target: FrameBinding


@dataclass(frozen=True)
class PairDistanceBinding:
    geom_names_a: Tuple[str, ...]
    geom_names_b: Tuple[str, ...]
    max_distance_m: float = 10.0

    def validate(self) -> None:
        if not self.geom_names_a or not self.geom_names_b:
            raise ValueError("pair distance binding requires two non-empty geom sets")
        if not math.isfinite(self.max_distance_m) or self.max_distance_m <= 0:
            raise ValueError("max_distance_m must be finite and positive")


@dataclass(frozen=True)
class DisplacementBinding:
    frame: FrameBinding
    reference_position: Tuple[float, float, float]


@dataclass(frozen=True)
class ManifoldBinding:
    frame: FrameBinding
    kind: str
    origin: Tuple[float, float, float]
    direction_or_normal: Tuple[float, float, float] | None = None

    def validate(self) -> None:
        if self.kind not in {"point", "line", "plane"}:
            raise ValueError("manifold kind must be point, line, or plane")
        if self.kind != "point" and self.direction_or_normal is None:
            raise ValueError(f"{self.kind} manifold requires direction_or_normal")


@dataclass(frozen=True)
class MuJoCoEvidenceSpec:
    relative_transforms: Mapping[EntityPair, RelativeTransformBinding] = field(
        default_factory=dict)
    pair_distances: Mapping[EntityPair, PairDistanceBinding] = field(
        default_factory=dict)
    displacements: Mapping[EntityPair, DisplacementBinding] = field(
        default_factory=dict)
    manifolds: Mapping[EntityPair, ManifoldBinding] = field(default_factory=dict)


def _frame_pose(model, data, binding: FrameBinding) -> np.ndarray:
    binding.validate()
    if binding.kind == "site":
        index = model.site(binding.name).id
        position = np.asarray(data.site_xpos[index], dtype=float)
        rotation = np.asarray(data.site_xmat[index], dtype=float).reshape(3, 3)
    else:
        index = model.body(binding.name).id
        position = np.asarray(data.xpos[index], dtype=float)
        rotation = np.asarray(data.xmat[index], dtype=float).reshape(3, 3)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def _unit(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def _manifold_distance(position: np.ndarray, binding: ManifoldBinding) -> float:
    binding.validate()
    origin = np.asarray(binding.origin, dtype=float)
    delta = position - origin
    if binding.kind == "point":
        return float(np.linalg.norm(delta))
    axis = _unit(binding.direction_or_normal, "direction_or_normal")
    if binding.kind == "line":
        return float(np.linalg.norm(delta - axis * np.dot(delta, axis)))
    return float(abs(np.dot(delta, axis)))


def extract_mujoco_evidence(model, data, spec: MuJoCoEvidenceSpec) -> ConstraintEvidence:
    """Extract metric evidence after the caller has run ``mj_forward/step``."""

    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("MuJoCo is required for this evidence adapter") from error

    relative: Dict[EntityPair, np.ndarray] = {}
    separation: Dict[EntityPair, float] = {}
    displacement: Dict[EntityPair, float] = {}
    manifold: Dict[EntityPair, float] = {}

    for entities, binding in spec.relative_transforms.items():
        source = _frame_pose(model, data, binding.source)
        target = _frame_pose(model, data, binding.target)
        relative[tuple(entities)] = np.linalg.inv(target) @ source

    for entities, binding in spec.pair_distances.items():
        binding.validate()
        distances = []
        for geom_a in binding.geom_names_a:
            geom_a_id = model.geom(geom_a).id
            for geom_b in binding.geom_names_b:
                geom_b_id = model.geom(geom_b).id
                distances.append(float(mujoco.mj_geomDistance(
                    model, data, geom_a_id, geom_b_id,
                    binding.max_distance_m, None)))
        separation[tuple(entities)] = min(distances)

    for entities, binding in spec.displacements.items():
        position = _frame_pose(model, data, binding.frame)[:3, 3]
        reference = np.asarray(binding.reference_position, dtype=float)
        if reference.shape != (3,) or not np.isfinite(reference).all():
            raise ValueError("reference_position must be a finite 3-vector")
        displacement[tuple(entities)] = float(np.linalg.norm(position - reference))

    for entities, binding in spec.manifolds.items():
        position = _frame_pose(model, data, binding.frame)[:3, 3]
        manifold[tuple(entities)] = _manifold_distance(position, binding)

    return ConstraintEvidence(
        relative_transforms=relative,
        pair_separation_m=separation,
        entity_displacement_m=displacement,
        manifold_distance_m=manifold,
    )
