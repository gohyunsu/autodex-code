import math

import pytest

from precision_layer.generator import GeneratorConfig, generate_cases
from precision_layer.procedural_mujoco import evaluate_scene, make_scene


def _cases():
    return generate_cases(GeneratorConfig(
        seed=4, train_count=6, validation_count=2, test_count=3))


def test_scene_generation_is_deterministic_and_holds_out_geometry_families():
    cases = _cases()
    first = make_scene(cases[0], seed=11)
    assert first == make_scene(cases[0], seed=11)

    development_shapes = {
        make_scene(case, seed=11).primitive.kind
        for case in cases
        if case["partition"] == "train"
    }
    test_shapes = {
        make_scene(case, seed=11).primitive.kind
        for case in cases
        if case["partition"] == "test"
    }
    assert development_shapes == {"sphere", "box", "cylinder"}
    assert test_shapes == {"capsule", "ellipsoid", "composite_cross"}
    assert development_shapes.isdisjoint(test_shapes)


mujoco = pytest.importorskip("mujoco")


@pytest.mark.parametrize("predicate_type", [
    "required_contact", "disturbance_budget", "path_manifold"])
def test_generated_world_returns_finite_privileged_margins(predicate_type):
    cases = _cases()
    case = next(
        item for item in cases
        if any(
            predicate["type"] == predicate_type
            for stage in item["program"]["stages"]
            for predicate in stage["predicates"]
        )
    )
    scene = make_scene(case, seed=11)
    labels = evaluate_scene(scene, physics_steps_per_waypoint=2)
    assert len(labels) == len(scene.trajectory_positions) == 41
    assert labels[-1]["payload_position_m"] == pytest.approx(scene.goal_position)
    assert all(
        math.isfinite(margin["signed_margin_m"])
        for waypoint in labels
        for margin in waypoint["margins"].values()
    )


def test_corrupted_trajectory_has_constraint_violations_not_only_successes():
    case = next(
        item for item in _cases()
        if item["partition"] == "test")
    labels = evaluate_scene(make_scene(case, seed=11), physics_steps_per_waypoint=1)
    guide_states = [
        waypoint["margins"]["guide_membership"]["satisfied"]
        for waypoint in labels
    ]
    assert any(guide_states)
    assert not all(guide_states)


def test_stage_activation_is_derived_from_program_dependencies():
    case = next(
        item for item in _cases()
        if any(
            predicate["type"] == "required_contact"
            for stage in item["program"]["stages"]
            for predicate in stage["predicates"]
        ))
    labels = evaluate_scene(make_scene(case, seed=11), physics_steps_per_waypoint=1)
    assert labels[0]["margins"]["target_contact"]["active"] is False
    assert labels[-1]["margins"]["target_contact"]["active"] is True
    assert len(labels[-1]["completed_stage_ids"]) == 2
