"""Structural validation for compositional precision constraint programs.

This module deliberately knows no task names.  It validates the references and
stage graph defined by ``precision_constraint_program_v1`` and compares graph
composition while ignoring arbitrary entity, predicate, and stage IDs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Set, Tuple


class ProgramValidationError(ValueError):
    """Raised when a constraint program is structurally inconsistent."""


def _unique_index(items: Iterable[Mapping[str, Any]], key: str) -> Dict[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for item in items:
        value = item.get(key)
        if not isinstance(value, str) or not value:
            raise ProgramValidationError(f"{key} must be a non-empty string")
        if value in result:
            raise ProgramValidationError(f"duplicate {key}: {value}")
        result[value] = item
    return result


def _assert_dag(stages: Mapping[str, Mapping[str, Any]]) -> None:
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            raise ProgramValidationError(f"stage dependency cycle includes {stage_id}")
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for predecessor in stages[stage_id].get("predecessors", []):
            visit(predecessor)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in stages:
        visit(stage_id)


def validate_program(program: Mapping[str, Any]) -> None:
    """Validate graph-level invariants not expressible in JSON Schema alone."""

    if program.get("schema") != "precision_constraint_program_v1":
        raise ProgramValidationError("unsupported constraint program schema")

    entities = _unique_index(program.get("entities", []), "entity_id")
    stages = _unique_index(program.get("stages", []), "stage_id")
    if not entities:
        raise ProgramValidationError("program must contain at least one entity")
    if not stages:
        raise ProgramValidationError("program must contain at least one stage")

    for stage_id, stage in stages.items():
        unknown_predecessors = set(stage.get("predecessors", [])) - set(stages)
        if unknown_predecessors:
            raise ProgramValidationError(
                f"stage {stage_id} has unknown predecessors: {sorted(unknown_predecessors)}"
            )
        if stage_id in stage.get("predecessors", []):
            raise ProgramValidationError(f"stage {stage_id} depends on itself")

    _assert_dag(stages)

    all_predicates: Dict[str, Mapping[str, Any]] = {}
    stage_predicates: Dict[str, Set[str]] = {}
    for stage_id, stage in stages.items():
        predicates = _unique_index(stage.get("predicates", []), "predicate_id")
        duplicates = set(predicates) & set(all_predicates)
        if duplicates:
            raise ProgramValidationError(
                f"predicate IDs must be global; repeated: {sorted(duplicates)}"
            )
        all_predicates.update(predicates)
        stage_predicates[stage_id] = set(predicates)

        unknown_terminal = set(stage.get("terminal_success", [])) - set(predicates)
        if unknown_terminal:
            raise ProgramValidationError(
                f"stage {stage_id} terminal_success references predicates outside the stage: "
                f"{sorted(unknown_terminal)}"
            )

        for predicate_id, predicate in predicates.items():
            unknown_entities = set(predicate.get("entities", [])) - set(entities)
            if unknown_entities:
                raise ProgramValidationError(
                    f"predicate {predicate_id} has unknown entities: {sorted(unknown_entities)}"
                )

    for predicate_id, predicate in all_predicates.items():
        activation = predicate.get("activation") or {}
        unknown_dependencies = set(activation.get("depends_on", [])) - set(all_predicates)
        if unknown_dependencies:
            raise ProgramValidationError(
                f"predicate {predicate_id} has unknown activation dependencies: "
                f"{sorted(unknown_dependencies)}"
            )


def validate_zero_shot_contract(program: Mapping[str, Any]) -> None:
    """Enforce the primary paper's no-test-learning deployment contract."""

    validate_program(program)
    contract = program.get("generalization_contract", {})
    if contract.get("known_geometry_at_test") is not True:
        raise ProgramValidationError("primary zero-shot contract requires known test geometry")
    if contract.get("test_specific_demonstrations") != 0:
        raise ProgramValidationError("primary zero-shot contract requires zero test demonstrations")
    if contract.get("test_specific_gradient_updates") != 0:
        raise ProgramValidationError("primary zero-shot contract requires zero test gradient updates")
    operations = set(contract.get("allowed_test_time_operations", []))
    if "parameter_learning" in operations:
        raise ProgramValidationError("parameter learning is outside the primary zero-shot contract")


Node = str
NodeLabel = Tuple[Any, ...]
Edge = Tuple[Node, Node, str]


@dataclass(frozen=True)
class _ProgramGraph:
    labels: Mapping[Node, NodeLabel]
    edges: Tuple[Edge, ...]

    def edge_counter(self) -> Counter:
        return Counter(self.edges)


def _program_graph(program: Mapping[str, Any]) -> _ProgramGraph:
    validate_program(program)
    labels: Dict[Node, NodeLabel] = {}
    edges: List[Edge] = []

    for entity in program["entities"]:
        node = f"entity:{entity['entity_id']}"
        labels[node] = ("entity", entity["role"])

    for stage in program["stages"]:
        stage_node = f"stage:{stage['stage_id']}"
        labels[stage_node] = (
            "stage",
            tuple(sorted(stage.get("available_action_primitives", []))),
        )
        for predecessor in stage.get("predecessors", []):
            edges.append((f"stage:{predecessor}", stage_node, "precedes"))

        terminal = set(stage.get("terminal_success", []))
        for predicate in stage["predicates"]:
            predicate_node = f"predicate:{predicate['predicate_id']}"
            tolerance = predicate.get("tolerance")
            tolerance_unit = None if tolerance is None else tolerance.get("unit")
            labels[predicate_node] = (
                "predicate",
                predicate["type"],
                predicate["evaluator"],
                bool(predicate["hard"]),
                predicate["terminal_role"],
                tolerance_unit,
            )
            edges.append((stage_node, predicate_node, "contains"))
            if predicate["predicate_id"] in terminal:
                edges.append((stage_node, predicate_node, "stage_success"))
            for index, entity_id in enumerate(predicate["entities"]):
                edges.append((predicate_node, f"entity:{entity_id}", f"entity_slot:{index}"))
            activation = predicate.get("activation") or {}
            for dependency in activation.get("depends_on", []):
                edges.append((f"predicate:{dependency}", predicate_node, "activates"))

    return _ProgramGraph(labels=labels, edges=tuple(edges))


def _incident_signature(graph: _ProgramGraph, node: Node) -> Tuple[Any, ...]:
    incoming: List[Tuple[str, NodeLabel]] = []
    outgoing: List[Tuple[str, NodeLabel]] = []
    for source, target, edge_label in graph.edges:
        if source == node:
            outgoing.append((edge_label, graph.labels[target]))
        if target == node:
            incoming.append((edge_label, graph.labels[source]))
    return graph.labels[node], tuple(sorted(incoming)), tuple(sorted(outgoing))


def _compatible_partial_mapping(
    graph_a: _ProgramGraph,
    graph_b: _ProgramGraph,
    mapping: Mapping[Node, Node],
    new_a: Node,
    new_b: Node,
) -> bool:
    edges_a = graph_a.edge_counter()
    edges_b = graph_b.edge_counter()
    extended = dict(mapping)
    extended[new_a] = new_b
    mapped_nodes = set(extended)
    for source, target, label in graph_a.edges:
        if source in mapped_nodes and target in mapped_nodes:
            mapped_edge = (extended[source], extended[target], label)
            if edges_b[mapped_edge] < edges_a[(source, target, label)]:
                return False
    return True


def are_programs_isomorphic(program_a: Mapping[str, Any], program_b: Mapping[str, Any]) -> bool:
    """Return whether two constraint compositions differ only by arbitrary IDs.

    Geometry paths, numeric tolerances and task descriptions are intentionally
    ignored. Predicate type/evaluator, entity role, stage topology, ordered
    predicate arguments, activation and action primitives are preserved.
    """

    graph_a = _program_graph(program_a)
    graph_b = _program_graph(program_b)
    if len(graph_a.labels) != len(graph_b.labels) or len(graph_a.edges) != len(graph_b.edges):
        return False
    if Counter(graph_a.labels.values()) != Counter(graph_b.labels.values()):
        return False

    candidates: Dict[Node, List[Node]] = {}
    for node_a in graph_a.labels:
        signature_a = _incident_signature(graph_a, node_a)
        candidates[node_a] = [
            node_b
            for node_b in graph_b.labels
            if _incident_signature(graph_b, node_b) == signature_a
        ]
        if not candidates[node_a]:
            return False

    ordered = sorted(
        graph_a.labels,
        key=lambda node: (len(candidates[node]), -sum(node in edge[:2] for edge in graph_a.edges)),
    )

    def search(index: int, mapping: Dict[Node, Node], used: Set[Node]) -> bool:
        if index == len(ordered):
            remapped = Counter((mapping[s], mapping[t], label) for s, t, label in graph_a.edges)
            return remapped == graph_b.edge_counter()
        node_a = ordered[index]
        for node_b in candidates[node_a]:
            if node_b in used:
                continue
            if not _compatible_partial_mapping(graph_a, graph_b, mapping, node_a, node_b):
                continue
            mapping[node_a] = node_b
            used.add(node_b)
            if search(index + 1, mapping, used):
                return True
            used.remove(node_b)
            del mapping[node_a]
        return False

    return search(0, {}, set())
