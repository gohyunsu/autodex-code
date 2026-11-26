import math

import numpy as np
import pytest

from precision_layer.evaluator import (
    ConstraintEvidence,
    MissingConstraintEvidence,
    evaluate_predicate,
    evaluate_stage,
)
from precision_layer.generator import GeneratorConfig, generate_cases


def predicate(predicate_type, entities, tolerance=0.001, parameters=None):
    return {
        "predicate_id": "p",
        "type": predicate_type,
        "entities": list(entities),
        "reference_frame": entities[-1],
        "evaluator": "generic_metric_backend",
        "parameters": {} if parameters is None else parameters,
        "tolerance": {"value": tolerance, "unit": "m"},
        "hard": True,
        "terminal_role": "none",
    }


def test_clearance_contact_and_disturbance_signs():
    evidence = ConstraintEvidence(
        pair_separation_m={("body", "wall"): 0.003},
        entity_displacement_m={("neighbour",): 0.0005},
        manifold_distance_m={("body", "guide"): 0.002},
    )
    clearance = evaluate_predicate(
        predicate("signed_clearance", ("body", "wall"), 0.001), evidence)
    contact = evaluate_predicate(
        predicate("required_contact", ("body", "wall"), 0.004), evidence)
    disturbance = evaluate_predicate(
        predicate("disturbance_budget", ("neighbour",), 0.001), evidence)
    manifold = evaluate_predicate(
        predicate("path_manifold", ("body", "guide"), 0.001), evidence)

    assert clearance.signed_margin == pytest.approx(0.002)
    assert contact.signed_margin == pytest.approx(0.001)
    assert disturbance.signed_margin == pytest.approx(0.0005)
    assert manifold.signed_margin == pytest.approx(-0.001)
    assert clearance.satisfied and contact.satisfied and disturbance.satisfied
    assert not manifold.satisfied


def test_pose_equality_combines_translation_and_rotation_metric():
    transform = np.eye(4)
    transform[0, 3] = 0.002
    angle = 0.1
    transform[:3, :3] = [
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ]
    result = evaluate_predicate(predicate(
        "pose_equality", ("body", "goal"), 0.01,
        {"rotation_length_scale_m_per_rad": 0.05}), ConstraintEvidence(
            relative_transforms={("body", "goal"): transform}))
    assert result.signed_margin == pytest.approx(0.005)
    assert result.normalized_margin == pytest.approx(0.5)


def test_missing_or_nonmetric_evidence_is_not_silently_fabricated():
    with pytest.raises(MissingConstraintEvidence, match="pair separation"):
        evaluate_predicate(
            predicate("signed_clearance", ("body", "wall")),
            ConstraintEvidence())
    no_tolerance = predicate("path_manifold", ("body", "guide"))
    no_tolerance.pop("tolerance")
    with pytest.raises(ValueError, match="requires a metric tolerance"):
        evaluate_predicate(no_tolerance, ConstraintEvidence())


def test_generated_program_evaluates_without_named_task_dispatch():
    case = next(case for case in generate_cases(GeneratorConfig(
        seed=4, train_count=1, validation_count=1, test_count=1))
                if case["partition"] == "test")
    program = case["program"]
    results = evaluate_stage(program, "guided_motion", ConstraintEvidence(
        relative_transforms={("payload", "goal"): np.eye(4)},
        manifold_distance_m={("payload", "boundary"): 0.0},
    ))
    assert set(results) == {"guide_membership", "goal_alignment"}
    assert all(result.satisfied for result in results.values())
