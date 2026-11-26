import json

import pytest

from precision_layer.audit import audit_cases, load_index, main as audit_main
from precision_layer.generator import GeneratorConfig, generate_cases, write_corpus
from precision_layer.program import are_programs_isomorphic, validate_program


def test_generated_joint_holdout_is_valid_and_leakage_free():
    cases = generate_cases(GeneratorConfig(seed=17))
    assert {case["partition"] for case in cases} == {
        "train", "validation", "test"}
    assert audit_cases(cases) == []
    for case in cases:
        validate_program(case["program"])

    training = [case["program"] for case in cases
                if case["partition"] == "train"]
    tests = [case["program"] for case in cases
             if case["partition"] == "test"]
    assert all(not are_programs_isomorphic(test, train)
               for test in tests for train in training)


def test_generation_is_deterministic_for_fixed_metadata():
    config = GeneratorConfig(
        seed=123, train_count=3, validation_count=2, test_count=2)
    assert generate_cases(config) == generate_cases(config)


def test_written_corpus_round_trips_through_audit_cli(tmp_path, capsys):
    index_path = write_corpus(tmp_path / "corpus", GeneratorConfig(
        seed=3, train_count=4, validation_count=2, test_count=2))
    payload = json.loads(index_path.read_text())
    assert payload["schema"] == "precision_benchmark_index_v1"
    assert len(payload["cases"]) == 8
    assert audit_cases(load_index(index_path)) == []
    assert audit_main([str(index_path)]) == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True


def test_generator_refuses_nonempty_output(tmp_path):
    output = tmp_path / "corpus"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user")
    with pytest.raises(FileExistsError, match="not empty"):
        write_corpus(output, GeneratorConfig())


def test_generator_requires_every_partition():
    with pytest.raises(ValueError, match="every partition"):
        generate_cases(GeneratorConfig(test_count=0))
