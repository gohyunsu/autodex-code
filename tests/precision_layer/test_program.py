import json

import pytest

from precision_layer.audit import audit_cases, main as audit_main
from precision_layer.program import (
    ProgramValidationError,
    are_programs_isomorphic,
    validate_program,
    validate_zero_shot_contract,
)


def make_program(*, held_out=True, renamed=False, contact_type="required_contact"):
    entity_ids = ("body_x", "surface_y", "goal_z") if renamed else ("payload", "boundary", "goal")
    stage_ids = ("phase_alpha", "phase_beta") if renamed else ("approach", "interaction")
    predicate_ids = ("margin_x", "goal_x", "contact_x") if renamed else ("clearance", "at_goal", "contact")
    payload, boundary, goal = entity_ids
    approach, interaction = stage_ids
    clearance, at_goal, contact = predicate_ids
    return {
        "schema": "precision_constraint_program_v1",
        "program_id": "renamed" if renamed else "base",
        "vocabulary_version": "v1",
        "entities": [
            {"entity_id": payload, "role": "manipulated", "geometry": None, "state_source": "robot_kinematics"},
            {"entity_id": boundary, "role": "environment", "geometry": None, "state_source": "fixed_calibration"},
            {"entity_id": goal, "role": "target", "geometry": None, "state_source": "fixed_calibration"},
        ],
        "stages": [
            {
                "stage_id": approach,
                "predecessors": [],
                "available_action_primitives": ["cartesian_delta"],
                "predicates": [
                    {
                        "predicate_id": clearance,
                        "type": "signed_clearance",
                        "entities": [payload, boundary],
                        "reference_frame": "world",
                        "evaluator": "mesh_signed_distance",
                        "parameters": {},
                        "hard": True,
                        "terminal_role": "failure",
                    },
                    {
                        "predicate_id": at_goal,
                        "type": "pose_equality",
                        "entities": [payload, goal],
                        "reference_frame": goal,
                        "evaluator": "relative_pose_residual",
                        "parameters": {},
                        "hard": False,
                        "terminal_role": "none",
                    },
                ],
                "terminal_success": [at_goal],
            },
            {
                "stage_id": interaction,
                "predecessors": [approach],
                "available_action_primitives": ["guarded_delta"],
                "predicates": [
                    {
                        "predicate_id": contact,
                        "type": contact_type,
                        "entities": [payload, goal],
                        "reference_frame": goal,
                        "evaluator": "pair_contact_state",
                        "parameters": {},
                        "hard": True,
                        "terminal_role": "success" if contact_type == "required_contact" else "failure",
                        "activation": {"mode": "after_predicate", "depends_on": [at_goal]},
                    }
                ],
                "terminal_success": [contact],
            },
        ],
        "generalization_contract": {
            "geometry": "held_out_family" if held_out else "seen_instance",
            "layout": "held_out_generator" if held_out else "seen_instance",
            "composition": "held_out_combination" if held_out else "seen_graph",
            "upstream_policy": "held_out_policy_class" if held_out else "seen_policy",
            "allowed_test_time_operations": ["state_estimation", "deterministic_action_optimization"],
            "known_geometry_at_test": True,
            "test_specific_demonstrations": 0,
            "test_specific_gradient_updates": 0,
        },
        "provenance": {
            "created_at": "2026-11-27T00:00:00Z",
            "program_revision": "1",
            "generator_revision": "test",
        },
    }


def test_valid_program_and_zero_shot_contract():
    program = make_program()
    validate_program(program)
    validate_zero_shot_contract(program)


def test_rejects_stage_cycle_and_unknown_entity():
    cyclic = make_program()
    cyclic["stages"][0]["predecessors"] = ["interaction"]
    with pytest.raises(ProgramValidationError, match="cycle"):
        validate_program(cyclic)

    unknown = make_program()
    unknown["stages"][0]["predicates"][0]["entities"] = ["payload", "missing"]
    with pytest.raises(ProgramValidationError, match="unknown entities"):
        validate_program(unknown)


def test_zero_shot_contract_rejects_test_learning():
    program = make_program()
    program["generalization_contract"]["test_specific_gradient_updates"] = 1
    with pytest.raises(ProgramValidationError, match="zero test gradient updates"):
        validate_zero_shot_contract(program)

    unknown_geometry = make_program()
    unknown_geometry["generalization_contract"]["known_geometry_at_test"] = False
    with pytest.raises(ProgramValidationError, match="known test geometry"):
        validate_zero_shot_contract(unknown_geometry)


def test_graph_isomorphism_ignores_ids_but_preserves_semantics():
    base = make_program()
    renamed = make_program(renamed=True)
    changed_semantics = make_program(renamed=True, contact_type="forbidden_contact")
    assert are_programs_isomorphic(base, renamed)
    assert not are_programs_isomorphic(base, changed_semantics)


def test_split_audit_detects_all_declared_leakage_axes():
    training = {
        "case_id": "train_a",
        "partition": "train",
        "geometry_family": "geometry_family_a",
        "layout_family": "layout_generator_a",
        "base_policy_class": "model_based",
        "program": make_program(held_out=False),
    }
    leaky = {
        "case_id": "test_leaky",
        "partition": "test",
        "geometry_family": "geometry_family_a",
        "layout_family": "layout_generator_a",
        "base_policy_class": "model_based",
        "program": make_program(held_out=True, renamed=True),
    }
    violations = audit_cases([training, leaky])
    assert {violation.axis for violation in violations} == {
        "Z1_geometry",
        "Z2_layout",
        "Z3_policy",
        "Z4_composition",
    }


def test_split_audit_accepts_joint_held_out_case():
    training = {
        "case_id": "train_a",
        "partition": "train",
        "geometry_family": "geometry_family_a",
        "layout_family": "layout_generator_a",
        "base_policy_class": "model_based",
        "program": make_program(held_out=False),
    }
    held_out = {
        "case_id": "test_joint_holdout",
        "partition": "test",
        "geometry_family": "geometry_family_b",
        "layout_family": "layout_generator_b",
        "base_policy_class": "vla",
        "program": make_program(held_out=True, contact_type="forbidden_contact"),
    }
    assert audit_cases([training, held_out]) == []


def test_audit_cli_resolves_programs_relative_to_index(tmp_path, capsys):
    train_path = tmp_path / "train_program.json"
    test_path = tmp_path / "test_program.json"
    train_path.write_text(json.dumps(make_program(held_out=False)))
    test_path.write_text(json.dumps(make_program(held_out=True, contact_type="forbidden_contact")))
    index = {
        "schema": "precision_benchmark_index_v1",
        "cases": [
            {
                "case_id": "train_a",
                "partition": "train",
                "geometry_family": "geometry_a",
                "layout_family": "layout_a",
                "base_policy_class": "model_based",
                "program_path": train_path.name,
            },
            {
                "case_id": "test_b",
                "partition": "test",
                "geometry_family": "geometry_b",
                "layout_family": "layout_b",
                "base_policy_class": "vla",
                "program_path": test_path.name,
            },
        ],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index))

    assert audit_main([str(index_path)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True
