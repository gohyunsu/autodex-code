"""Generate task-name-free structural precision benchmark programs.

This module makes the Z1--Z4 split contract executable.  It deliberately does
not claim to generate physics-ready scenes; geometry assets and simulator
evaluators are the next R1 milestone.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .audit import audit_cases
from .program import are_programs_isomorphic, validate_program


GENERATOR_REVISION = "structural_v1"


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 0
    train_count: int = 12
    validation_count: int = 4
    test_count: int = 6

    def validate(self) -> None:
        if min(self.train_count, self.validation_count, self.test_count) < 1:
            raise ValueError("every partition must contain at least one case")


def _predicate(
    predicate_id: str,
    predicate_type: str,
    entities: Sequence[str],
    *,
    evaluator: str,
    tolerance_value: float,
    terminal_role: str = "none",
    hard: bool = True,
    activation: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "predicate_id": predicate_id,
        "type": predicate_type,
        "entities": list(entities),
        "reference_frame": entities[-1],
        "evaluator": evaluator,
        "parameters": {},
        "tolerance": {"value": tolerance_value, "unit": "m"},
        "hard": hard,
        "terminal_role": terminal_role,
    }
    if activation is not None:
        result["activation"] = dict(activation)
    return result


def _program(
    *,
    program_id: str,
    topology: str,
    tolerance_m: float,
    held_out: bool,
    created_at: str,
) -> Dict[str, Any]:
    entities: List[Dict[str, Any]] = [
        {"entity_id": "payload", "role": "manipulated", "geometry": None,
         "state_source": "robot_kinematics"},
        {"entity_id": "goal", "role": "target", "geometry": None,
         "state_source": "online_estimate"},
        {"entity_id": "boundary", "role": "environment", "geometry": None,
         "state_source": "online_estimate"},
    ]
    approach_predicates = [
        _predicate(
            "clearance", "signed_clearance", ["payload", "boundary"],
            evaluator="mesh_signed_distance", tolerance_value=tolerance_m,
            terminal_role="failure"),
        _predicate(
            "goal_alignment", "pose_equality", ["payload", "goal"],
            evaluator="relative_pose_residual", tolerance_value=tolerance_m,
            hard=False),
    ]

    if topology == "alignment_then_contact":
        stages = [
            {
                "stage_id": "free_motion",
                "predecessors": [],
                "available_action_primitives": ["cartesian_delta"],
                "predicates": approach_predicates,
                "terminal_success": ["goal_alignment"],
            },
            {
                "stage_id": "physical_interaction",
                "predecessors": ["free_motion"],
                "available_action_primitives": ["guarded_delta"],
                "predicates": [_predicate(
                    "target_contact", "required_contact", ["payload", "goal"],
                    evaluator="pair_contact_state", tolerance_value=tolerance_m,
                    terminal_role="success", activation={
                        "mode": "after_predicate",
                        "depends_on": ["goal_alignment"],
                    })],
                "terminal_success": ["target_contact"],
            },
        ]
    elif topology == "clearance_and_disturbance":
        entities.append({
            "entity_id": "movable_neighbour", "role": "environment",
            "geometry": None, "state_source": "online_estimate"})
        stages = [{
            "stage_id": "constrained_motion",
            "predecessors": [],
            "available_action_primitives": ["cartesian_delta"],
            "predicates": approach_predicates + [_predicate(
                "neighbour_motion", "disturbance_budget",
                ["movable_neighbour"], evaluator="entity_pose_displacement",
                tolerance_value=2.0 * tolerance_m,
                terminal_role="failure")],
            "terminal_success": ["goal_alignment"],
        }]
    elif topology == "manifold_then_goal":
        stages = [{
            "stage_id": "guided_motion",
            "predecessors": [],
            "available_action_primitives": ["tangent_delta"],
            "predicates": [_predicate(
                "guide_membership", "path_manifold", ["payload", "boundary"],
                evaluator="manifold_distance", tolerance_value=tolerance_m,
                terminal_role="failure"), approach_predicates[1]],
            "terminal_success": ["goal_alignment"],
        }]
    else:
        raise ValueError(f"unknown topology: {topology}")

    program = {
        "schema": "precision_constraint_program_v1",
        "program_id": program_id,
        "vocabulary_version": "v1",
        "entities": entities,
        "stages": stages,
        "generalization_contract": {
            "geometry": "held_out_family" if held_out else "seen_instance",
            "layout": "held_out_generator" if held_out else "seen_instance",
            "composition": "held_out_topology" if held_out else "seen_graph",
            "upstream_policy": (
                "held_out_policy_class" if held_out else "seen_policy"),
            "allowed_test_time_operations": [
                "state_estimation",
                "deterministic_geometry_processing",
                "deterministic_action_optimization",
            ],
            "known_geometry_at_test": True,
            "test_specific_demonstrations": 0,
            "test_specific_gradient_updates": 0,
        },
        "provenance": {
            "created_at": created_at,
            "program_revision": "1",
            "generator_revision": GENERATOR_REVISION,
            "notes": "structural split only; physics geometry not generated",
        },
    }
    validate_program(program)
    return program


def generate_cases(
    config: GeneratorConfig,
    *,
    created_at: str = "1970-01-01T00:00:00+00:00",
) -> List[Dict[str, Any]]:
    """Return a deterministic, leakage-free structural split."""

    config.validate()
    rng = random.Random(config.seed)
    cases: List[Dict[str, Any]] = []

    def add_partition(
        partition: str,
        count: int,
        topology_pool: Tuple[str, ...],
        geometry_prefix: str,
        layout_prefix: str,
        policy_classes: Tuple[str, ...],
        *,
        held_out: bool,
    ) -> None:
        for index in range(count):
            case_id = f"{partition}_{index:04d}"
            program = _program(
                program_id=f"program_{case_id}",
                topology=rng.choice(topology_pool),
                tolerance_m=rng.choice((0.0005, 0.001, 0.002, 0.005)),
                held_out=held_out,
                created_at=created_at,
            )
            cases.append({
                "case_id": case_id,
                "partition": partition,
                "geometry_family": f"{geometry_prefix}_{index % 3}",
                "layout_family": f"{layout_prefix}_{index % 2}",
                "base_policy_class": rng.choice(policy_classes),
                "program": program,
            })

    development = ("alignment_then_contact", "clearance_and_disturbance")
    add_partition(
        "train", config.train_count, development,
        "development_geometry", "development_layout",
        ("model_based", "imitation"), held_out=False)
    add_partition(
        "validation", config.validation_count, development,
        "validation_geometry", "validation_layout",
        ("model_based", "imitation"), held_out=False)
    add_partition(
        "test", config.test_count, ("manifold_then_goal",),
        "heldout_geometry", "heldout_layout",
        ("language_conditioned",), held_out=True)

    training_programs = [case["program"] for case in cases
                         if case["partition"] == "train"]
    for case in (item for item in cases if item["partition"] == "test"):
        if any(are_programs_isomorphic(case["program"], training)
               for training in training_programs):
            raise RuntimeError("generator produced a leaked test composition")
    violations = audit_cases(cases)
    if violations:
        raise RuntimeError(f"generated benchmark failed leakage audit: {violations}")
    return cases


def write_corpus(output_dir: Path, config: GeneratorConfig) -> Path:
    """Write programs and an auditable benchmark index into a new directory."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    program_dir = output_dir / "programs"
    program_dir.mkdir()
    cases = generate_cases(
        config, created_at=datetime.now(timezone.utc).isoformat())
    index_cases: List[Dict[str, Any]] = []
    for case in cases:
        program_path = program_dir / f"{case['case_id']}.json"
        program_path.write_text(
            json.dumps(case["program"], indent=2, sort_keys=True) + "\n")
        index_case = {key: value for key, value in case.items()
                      if key != "program"}
        index_case["program_path"] = str(program_path.relative_to(output_dir))
        index_cases.append(index_case)
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps({
        "schema": "precision_benchmark_index_v1",
        "cases": index_cases,
    }, indent=2, sort_keys=True) + "\n")
    return index_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--validation-count", type=int, default=4)
    parser.add_argument("--test-count", type=int, default=6)
    args = parser.parse_args(argv)
    index_path = write_corpus(args.output_dir, GeneratorConfig(
        seed=args.seed,
        train_count=args.train_count,
        validation_count=args.validation_count,
        test_count=args.test_count,
    ))
    print(index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
