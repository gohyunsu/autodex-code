"""Leakage audit for Z1--Z4 precision-layer benchmark splits."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .program import are_programs_isomorphic, validate_program, validate_zero_shot_contract


@dataclass(frozen=True)
class AuditViolation:
    axis: str
    test_case: str
    train_case: str | None
    detail: str


def audit_cases(cases: Sequence[Mapping[str, Any]]) -> List[AuditViolation]:
    """Audit declared held-out axes against authoritative case identities."""

    seen_ids = set()
    for case in cases:
        case_id = case["case_id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_ids.add(case_id)
        validate_program(case["program"])

    train = [case for case in cases if case["partition"] == "train"]
    tests = [case for case in cases if case["partition"] == "test"]
    violations: List[AuditViolation] = []

    train_geometry = {case["geometry_family"]: case["case_id"] for case in train}
    train_layout = {case["layout_family"]: case["case_id"] for case in train}
    train_policy = {case["base_policy_class"]: case["case_id"] for case in train}

    for case in tests:
        validate_zero_shot_contract(case["program"])
        contract = case["program"]["generalization_contract"]
        if contract["geometry"] == "held_out_family" and case["geometry_family"] in train_geometry:
            violations.append(AuditViolation(
                "Z1_geometry", case["case_id"], train_geometry[case["geometry_family"]],
                f"geometry family {case['geometry_family']} appears in train",
            ))
        if contract["layout"] == "held_out_generator" and case["layout_family"] in train_layout:
            violations.append(AuditViolation(
                "Z2_layout", case["case_id"], train_layout[case["layout_family"]],
                f"layout generator {case['layout_family']} appears in train",
            ))
        if contract["upstream_policy"] == "held_out_policy_class" and case["base_policy_class"] in train_policy:
            violations.append(AuditViolation(
                "Z3_policy", case["case_id"], train_policy[case["base_policy_class"]],
                f"base-policy class {case['base_policy_class']} appears in train",
            ))
        if contract["composition"] in {"held_out_combination", "held_out_topology"}:
            for train_case in train:
                if are_programs_isomorphic(case["program"], train_case["program"]):
                    violations.append(AuditViolation(
                        "Z4_composition", case["case_id"], train_case["case_id"],
                        "constraint graph is isomorphic to a training program after ID renaming",
                    ))
                    break
    return violations


def load_index(path: Path) -> List[Dict[str, Any]]:
    """Load a benchmark index and resolve program paths relative to it."""

    payload = json.loads(path.read_text())
    if payload.get("schema") != "precision_benchmark_index_v1":
        raise ValueError("unsupported benchmark index schema")
    cases: List[Dict[str, Any]] = []
    for raw in payload.get("cases", []):
        case = dict(raw)
        program_path = (path.parent / case.pop("program_path")).resolve()
        case["program"] = json.loads(program_path.read_text())
        cases.append(case)
    return cases


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    args = parser.parse_args(argv)
    violations = audit_cases(load_index(args.index))
    report = {
        "passed": not violations,
        "violation_count": len(violations),
        "violations": [violation.__dict__ for violation in violations],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
