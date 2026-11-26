"""Procedural MuJoCo worlds for task-agnostic precision-margin evidence.

This is an R1-alpha generator.  It converts structural benchmark cases into
primitive-geometry worlds and corrupted nominal Cartesian trajectories.  The
generator dispatches only on declared geometry/layout families and predicate
composition; it never consumes a human task name.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

from .evaluator import evaluate_predicate
from .mujoco_adapter import (
    DisplacementBinding,
    FrameBinding,
    ManifoldBinding,
    MuJoCoEvidenceSpec,
    PairDistanceBinding,
    RelativeTransformBinding,
    extract_mujoco_evidence,
)
from .program import validate_program


@dataclass(frozen=True)
class Primitive:
    kind: str
    size: Tuple[float, ...]

    def validate(self) -> None:
        expected = {
            "sphere": 1,
            "box": 3,
            "cylinder": 2,
            "capsule": 2,
            "ellipsoid": 3,
            "composite_cross": 3,
        }
        if self.kind not in expected:
            raise ValueError(f"unsupported primitive: {self.kind}")
        if len(self.size) != expected[self.kind]:
            raise ValueError(f"{self.kind} requires {expected[self.kind]} size values")
        if not all(math.isfinite(value) and value > 0 for value in self.size):
            raise ValueError("primitive sizes must be finite and positive")

    @property
    def half_extent_y(self) -> float:
        if self.kind in {"sphere", "capsule"}:
            return self.size[0]
        if self.kind == "cylinder":
            return self.size[0]
        return self.size[1]

    @property
    def half_extent_z(self) -> float:
        if self.kind == "sphere":
            return self.size[0]
        if self.kind == "capsule":
            return self.size[0] + self.size[1]
        if self.kind == "cylinder":
            return self.size[1]
        return self.size[2]


@dataclass(frozen=True)
class ProceduralScene:
    case_id: str
    partition: str
    geometry_family: str
    layout_family: str
    base_policy_class: str
    primitive: Primitive
    start_position: Tuple[float, float, float]
    goal_position: Tuple[float, float, float]
    boundary_boxes: Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...]
    movable_neighbour_position: Tuple[float, float, float] | None
    trajectory_positions: Tuple[Tuple[float, float, float], ...]
    program: Mapping


def _index(family: str) -> int:
    try:
        return int(family.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"family must end in an integer: {family}") from error


def _seed(case_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{case_id}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _primitive(partition: str, family: str, rng: np.random.Generator) -> Primitive:
    scale = float(rng.uniform(0.9, 1.1))
    index = _index(family) % 3
    if partition == "test":
        choices = (
            Primitive("capsule", (0.018 * scale, 0.025 * scale)),
            Primitive("ellipsoid", (0.025 * scale, 0.017 * scale, 0.022 * scale)),
            Primitive("composite_cross", (0.028 * scale, 0.010 * scale, 0.018 * scale)),
        )
    else:
        choices = (
            Primitive("sphere", (0.022 * scale,)),
            Primitive("box", (0.024 * scale, 0.018 * scale, 0.020 * scale)),
            Primitive("cylinder", (0.018 * scale, 0.024 * scale)),
        )
    primitive = choices[index]
    primitive.validate()
    return primitive


def _minimum_tolerance(program: Mapping) -> float:
    values = [
        float(predicate["tolerance"]["value"])
        for stage in program["stages"]
        for predicate in stage["predicates"]
        if predicate.get("tolerance") is not None
    ]
    return min(values)


def _layout(
    partition: str,
    family: str,
    primitive: Primitive,
    tolerance_m: float,
    z: float,
) -> tuple[
    Tuple[Tuple[Tuple[float, float, float], Tuple[float, float, float]], ...],
    Tuple[float, float, float] | None,
]:
    clearance = max(0.002, 2.0 * tolerance_m)
    half_gap = primitive.half_extent_y + clearance
    index = _index(family) % 2
    if partition == "test":
        if index == 0:
            # A staggered aperture is not produced by development layouts.
            boxes = (
                ((0.12, half_gap + 0.012, z), (0.035, 0.012, 0.055)),
                ((0.20, -half_gap - 0.012, z), (0.035, 0.012, 0.055)),
                ((0.28, half_gap + 0.012, z), (0.035, 0.012, 0.055)),
            )
            neighbour = (0.235, -half_gap - 0.018, z)
        else:
            # A dense alternating layout with a movable neighbour.
            boxes = (
                ((0.13, half_gap + 0.010, z), (0.025, 0.010, 0.055)),
                ((0.22, -half_gap - 0.010, z), (0.025, 0.010, 0.055)),
                ((0.31, half_gap + 0.010, z), (0.025, 0.010, 0.055)),
            )
            neighbour = (0.27, -half_gap - 0.016, z)
    elif index == 0:
        boxes = (
            ((0.18, half_gap + 0.015, z), (0.18, 0.015, 0.055)),
            ((0.18, -half_gap - 0.015, z), (0.18, 0.015, 0.055)),
        )
        neighbour = (0.24, half_gap + 0.035, z)
    else:
        boxes = (
            ((0.18, half_gap + 0.020, z), (0.040, 0.020, 0.055)),
        )
        neighbour = (0.23, -half_gap - 0.020, z)
    return boxes, neighbour


def _trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    policy_class: str,
    tolerance_m: float,
    rng: np.random.Generator,
    count: int = 41,
) -> Tuple[Tuple[float, float, float], ...]:
    positions: List[Tuple[float, float, float]] = []
    phase = float(rng.uniform(-math.pi, math.pi))
    for step in range(count):
        fraction = step / (count - 1)
        position = (1.0 - fraction) * start + fraction * goal
        if policy_class == "model_based":
            lateral = 0.75 * tolerance_m * math.sin(math.pi * fraction)
        elif policy_class == "imitation":
            lateral = tolerance_m * (
                0.8 * math.sin(2.0 * math.pi * fraction + phase)
                + 0.35 * math.sin(5.0 * math.pi * fraction))
        else:
            # Held-out policy class: piecewise local bias, not a new task.
            lateral = 1.5 * tolerance_m * (
                1.0 if 0.25 <= fraction <= 0.75 else 0.0)
        position[1] += lateral
        positions.append(tuple(float(value) for value in position))
    return tuple(positions)


def make_scene(case: Mapping, seed: int = 0) -> ProceduralScene:
    """Create a deterministic physical scene descriptor from one split case."""

    program = case["program"]
    validate_program(program)
    rng = np.random.default_rng(_seed(case["case_id"], seed))
    primitive = _primitive(case["partition"], case["geometry_family"], rng)
    tolerance = _minimum_tolerance(program)
    z = 0.12
    start = np.array((0.0, 0.0, z), dtype=float)
    goal = np.array((0.36, 0.0, z), dtype=float)
    boxes, neighbour = _layout(
        case["partition"], case["layout_family"], primitive, tolerance, z)
    trajectory = _trajectory(
        start, goal, case["base_policy_class"], tolerance, rng)
    return ProceduralScene(
        case_id=case["case_id"],
        partition=case["partition"],
        geometry_family=case["geometry_family"],
        layout_family=case["layout_family"],
        base_policy_class=case["base_policy_class"],
        primitive=primitive,
        start_position=tuple(start),
        goal_position=tuple(goal),
        boundary_boxes=boxes,
        movable_neighbour_position=neighbour,
        trajectory_positions=trajectory,
        program=program,
    )


def _values(values: Iterable[float]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def _add_primitive(body: ET.Element, primitive: Primitive) -> Tuple[str, ...]:
    common = {"density": "500", "friction": "0.8 0.02 0.001"}
    if primitive.kind != "composite_cross":
        name = "payload_geom_00"
        ET.SubElement(
            body, "geom", name=name, type=primitive.kind,
            size=_values(primitive.size), **common)
        return (name,)
    half_x, half_y, half_z = primitive.size
    ET.SubElement(
        body, "geom", name="payload_geom_00", type="box",
        size=_values((half_x, half_y, half_z)), **common)
    ET.SubElement(
        body, "geom", name="payload_geom_01", type="box",
        size=_values((half_y, half_x, half_z)), **common)
    return ("payload_geom_00", "payload_geom_01")


def build_model(scene: ProceduralScene):
    """Compile a MuJoCo model and return it with declarative evidence bindings."""

    try:
        import mujoco
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("MuJoCo is required to build procedural worlds") from error

    root = ET.Element("mujoco", model=scene.case_id)
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 0")
    world = ET.SubElement(root, "worldbody")
    payload = ET.SubElement(
        world, "body", name="payload", mocap="true",
        pos=_values(scene.start_position))
    payload_geom_names = _add_primitive(payload, scene.primitive)
    ET.SubElement(payload, "site", name="payload_frame", size="0.002")

    goal = ET.SubElement(
        world, "body", name="goal", pos=_values(scene.goal_position))
    support_half_z = 0.008
    ET.SubElement(
        goal, "geom", name="goal_contact", type="box",
        pos=_values((0.0, 0.0, -scene.primitive.half_extent_z - support_half_z)),
        size=_values((0.050, 0.050, support_half_z)),
        friction="0.8 0.02 0.001")
    ET.SubElement(goal, "site", name="goal_frame", size="0.002")

    boundary_names: List[str] = []
    for index, (position, size) in enumerate(scene.boundary_boxes):
        name = f"boundary_{index:02d}"
        ET.SubElement(
            world, "geom", name=name, type="box",
            pos=_values(position), size=_values(size),
            friction="0.8 0.02 0.001")
        boundary_names.append(name)

    neighbour_reference = None
    if scene.movable_neighbour_position is not None:
        neighbour = ET.SubElement(
            world, "body", name="movable_neighbour",
            pos=_values(scene.movable_neighbour_position))
        ET.SubElement(neighbour, "freejoint")
        ET.SubElement(
            neighbour, "geom", name="movable_neighbour_geom", type="cylinder",
            size="0.018 0.040", density="450",
            friction="0.8 0.02 0.001")
        ET.SubElement(neighbour, "site", name="movable_neighbour_frame", size="0.002")
        boundary_names.append("movable_neighbour_geom")
        neighbour_reference = scene.movable_neighbour_position

    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    predicates = {
        predicate["type"]
        for stage in scene.program["stages"]
        for predicate in stage["predicates"]
    }
    relative: Dict[tuple[str, ...], RelativeTransformBinding] = {}
    pair_distances: Dict[tuple[str, ...], PairDistanceBinding] = {}
    displacements: Dict[tuple[str, ...], DisplacementBinding] = {}
    manifolds: Dict[tuple[str, ...], ManifoldBinding] = {}
    if "pose_equality" in predicates:
        relative[("payload", "goal")] = RelativeTransformBinding(
            FrameBinding("site", "payload_frame"),
            FrameBinding("site", "goal_frame"))
    if predicates & {"signed_clearance", "forbidden_contact"}:
        pair_distances[("payload", "boundary")] = PairDistanceBinding(
            payload_geom_names, tuple(boundary_names))
    if "required_contact" in predicates:
        pair_distances[("payload", "goal")] = PairDistanceBinding(
            payload_geom_names, ("goal_contact",))
    if "disturbance_budget" in predicates:
        if neighbour_reference is None:
            raise ValueError("disturbance predicate requires a movable neighbour")
        displacements[("movable_neighbour",)] = DisplacementBinding(
            FrameBinding("site", "movable_neighbour_frame"),
            neighbour_reference)
    if "path_manifold" in predicates:
        direction = np.asarray(scene.goal_position) - np.asarray(scene.start_position)
        manifolds[("payload", "boundary")] = ManifoldBinding(
            FrameBinding("site", "payload_frame"), "line",
            scene.start_position, tuple(direction))
    specification = MuJoCoEvidenceSpec(
        relative_transforms=relative,
        pair_distances=pair_distances,
        displacements=displacements,
        manifolds=manifolds,
    )
    return model, specification


def evaluate_scene(
    scene: ProceduralScene,
    *,
    physics_steps_per_waypoint: int = 5,
) -> List[dict]:
    """Execute the corrupted mocap trajectory and return privileged labels."""

    if physics_steps_per_waypoint < 1:
        raise ValueError("physics_steps_per_waypoint must be positive")
    import mujoco

    model, specification = build_model(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mocap_id = model.body_mocapid[model.body("payload").id]
    labels: List[dict] = []
    predicate_records = [
        (stage["stage_id"], predicate)
        for stage in scene.program["stages"]
        for predicate in stage["predicates"]
    ]
    completed_stages: set[str] = set()
    satisfied_once: set[str] = set()
    for index, position in enumerate(scene.trajectory_positions):
        data.mocap_pos[mocap_id] = position
        data.mocap_quat[mocap_id] = (1.0, 0.0, 0.0, 0.0)
        for _ in range(physics_steps_per_waypoint):
            mujoco.mj_step(model, data)
        evidence = extract_mujoco_evidence(model, data, specification)
        margins = {
            predicate["predicate_id"]: evaluate_predicate(predicate, evidence)
            for _, predicate in predicate_records
        }
        # Resolve stage transitions from the declared DAG. A successor may
        # activate at the same waypoint at which its predecessor terminates;
        # no named phase or task-specific state machine is used.
        active_stages: set[str] = set()
        predicate_active = {
            predicate["predicate_id"]: False
            for _, predicate in predicate_records
        }
        changed = True
        while changed:
            changed = False
            for stage in scene.program["stages"]:
                stage_id = stage["stage_id"]
                if stage_id in completed_stages:
                    continue
                if not set(stage.get("predecessors", ())) <= completed_stages:
                    continue
                active_stages.add(stage_id)
                for predicate in stage["predicates"]:
                    predicate_id = predicate["predicate_id"]
                    activation = predicate.get("activation") or {
                        "mode": "stage_active"}
                    mode = activation.get("mode", "stage_active")
                    active = mode != "belief_event"
                    if mode == "after_predicate":
                        active = set(activation.get("depends_on", ())) <= satisfied_once
                    if active:
                        predicate_active[predicate_id] = True
                        if margins[predicate_id].satisfied and predicate_id not in satisfied_once:
                            satisfied_once.add(predicate_id)
                            changed = True
                if all(predicate_active[item] and margins[item].satisfied
                       for item in stage["terminal_success"]):
                    completed_stages.add(stage_id)
                    changed = True

        predicate_stage = {
            predicate["predicate_id"]: stage_id
            for stage_id, predicate in predicate_records
        }
        labels.append({
            "waypoint_index": index,
            "sim_time_s": float(data.time),
            "payload_position_m": list(map(float, position)),
            "active_stage_ids": sorted(active_stages),
            "completed_stage_ids": sorted(completed_stages),
            "margins": {
                predicate_id: {
                    "stage_id": predicate_stage[predicate_id],
                    "active": predicate_active[predicate_id],
                    "signed_margin_m": result.signed_margin,
                    "normalized_margin": result.normalized_margin,
                    "satisfied": result.satisfied,
                }
                for predicate_id, result in margins.items()
            },
        })
    return labels
