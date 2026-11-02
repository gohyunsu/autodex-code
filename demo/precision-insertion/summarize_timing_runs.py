#!/usr/bin/env python3
"""Aggregate hierarchical latency across completed pipeline timing traces."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
for module_root in (str(REPO_ROOT), str(DEMO_DIR)):
    if module_root not in sys.path:
        sys.path.insert(0, module_root)

from precision_insertion.timing_profile import summarize_events  # noqa: E402


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _stats(values: list[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "count": len(finite),
        "minimum": min(finite) if finite else None,
        "q1": _quantile(finite, 0.25),
        "median": _quantile(finite, 0.5),
        "q3": _quantile(finite, 0.75),
        "p90": _quantile(finite, 0.9),
        "maximum": max(finite) if finite else None,
    }


def _candidates(summary: dict[str, Any]) -> list[str]:
    values = set()
    for attempt in summary.get("attempts", {}).values():
        if isinstance(attempt, dict):
            values.update(str(value) for value in
                          attempt.get("candidate_ids", []))
    return sorted(values)


def _configuration_identity(timing_dir: Path) -> tuple[str, dict[str, Any] | None]:
    manifest_path = timing_dir / "manifest.json"
    if not manifest_path.is_file():
        return "unrecorded_legacy", None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest.get("runtime_configuration")
    if not isinstance(config, dict):
        return "unrecorded_legacy", None
    digest = hashlib.sha256(json.dumps(
        config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, config


def _run_row(events: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = summarize_events(events)
    timing_dir = events.parent
    config_sha256, config = _configuration_identity(timing_dir)
    return summary, {
        "run_id": timing_dir.name.removesuffix(".timing"),
        "timing_dir": str(timing_dir),
        "unfinished_span_count": summary["unfinished_span_count"],
        "complete": summary["unfinished_span_count"] == 0,
        "candidate_ids": _candidates(summary),
        "runtime_configuration_sha256": config_sha256,
        "runtime_configuration": config,
        "attempt_ids": sorted(summary.get("attempts", {})),
        "span_count": len(summary.get("spans", [])),
        "span_envelope_s": summary.get("span_envelope_s", 0.0),
        "non_human_envelope_s": summary.get(
            "span_envelope_non_human_s", 0.0),
        "human_wait_occupied_s": summary.get("human_wait_occupied_s", 0.0),
        "vlm_inference_occupied_s": summary.get(
            "vlm_inference_occupied_s", 0.0),
    }


def _named_rows(summaries: list[dict[str, Any]], field: str,
                value_fields: tuple[str, ...]) -> dict[str, Any]:
    names = sorted({name for summary in summaries
                    for name in summary.get(field, {})})
    output = {}
    for name in names:
        records = [summary[field][name] for summary in summaries
                   if name in summary.get(field, {})]
        row = {"present_runs": len(records)}
        for value_field in value_fields:
            row[value_field] = _stats([
                record[value_field] for record in records
                if isinstance(record.get(value_field), (int, float))])
        row["total_count"] = sum(int(record.get("count", 0))
                                 for record in records)
        row["total_failures"] = sum(int(record.get("failures", 0))
                                    for record in records)
        output[name] = row
    return output


def _actions(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        per_run: dict[str, dict[str, float]] = defaultdict(
            lambda: {"elapsed_s": 0.0, "human_s": 0.0,
                     "automatic_s": 0.0, "count": 0.0})
        for action in summary.get("ordered_actions", []):
            row = per_run[str(action["name"])]
            row["elapsed_s"] += float(action["inclusive_s"])
            row["human_s"] += float(action["human_wait_s"])
            row["automatic_s"] += float(action["automatic_elapsed_s"])
            row["count"] += 1.0
        for name, row in per_run.items():
            records[name].append(row)
    return {
        name: {
            "present_runs": len(rows),
            "total_count": int(sum(row["count"] for row in rows)),
            "elapsed_s": _stats([row["elapsed_s"] for row in rows]),
            "human_s": _stats([row["human_s"] for row in rows]),
            "automatic_s": _stats([row["automatic_s"] for row in rows]),
        }
        for name, rows in sorted(records.items())
    }


def _configuration_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe whether a timing population is one reproducible condition.

    ``unrecorded_legacy`` is not a real configuration identity. In
    particular, two legacy traces do not prove that they used the same code
    or settings merely because both omitted the field.
    """
    counts = Counter(row["runtime_configuration_sha256"] for row in rows)
    unrecorded = counts.get("unrecorded_legacy", 0)
    recorded = len(rows) - unrecorded
    identities = {identity for identity in counts
                  if identity != "unrecorded_legacy"}
    if not rows:
        status = "no_runs"
    elif unrecorded == len(rows):
        status = "unknown"
    elif unrecorded:
        status = "partially_recorded"
    elif len(identities) == 1:
        status = "known_single"
    else:
        status = "known_mixed"
    return {
        "counts": dict(sorted(counts.items())),
        "recorded_run_count": recorded,
        "unrecorded_run_count": unrecorded,
        "recorded_identity_count": len(identities),
        "comparison_available": bool(rows) and unrecorded == 0,
        "mixing_status": status,
        "controlled_single_condition": status == "known_single",
    }


def summarize(timing_root: Path, *, candidate_id: str | None = None,
              runtime_configuration_sha256: str | None = None,
              include_incomplete: bool = False) -> dict[str, Any]:
    root = timing_root.expanduser().resolve()
    loaded = []
    errors = []
    for events in sorted(root.glob("*.timing/events.jsonl")):
        try:
            summary, row = _run_row(events)
        except Exception as exc:
            errors.append({"events": str(events), "error": str(exc)})
            continue
        if ((candidate_id is None or candidate_id in row["candidate_ids"]) and
                (runtime_configuration_sha256 is None or
                 row["runtime_configuration_sha256"] ==
                 runtime_configuration_sha256)):
            loaded.append((summary, row))
    selected_pairs = [(summary, row) for summary, row in loaded
                      if include_incomplete or row["complete"]]
    selected = [summary for summary, _row in selected_pairs]
    rows = [row for _, row in loaded]
    discovered_configs = _configuration_population(rows)
    statistics_configs = _configuration_population(
        [row for _summary, row in selected_pairs])
    return {
        "schema": "precision_insertion_multi_run_timing_summary_v2",
        "timing_root": str(root),
        "candidate_filter": candidate_id,
        "runtime_configuration_filter": runtime_configuration_sha256,
        "statistics_include_incomplete": include_incomplete,
        "discovered_run_count": len(rows),
        "complete_run_count": sum(row["complete"] for row in rows),
        "incomplete_run_count": sum(not row["complete"] for row in rows),
        "statistics_run_count": len(selected),
        # Keep the original flat fields for report consumers, but do not use
        # the boolean alone: all-legacy data previously appeared as "not
        # mixed" even though its identity was unknowable.
        "runtime_configuration_counts": discovered_configs["counts"],
        "mixed_runtime_configurations": (
            discovered_configs["mixing_status"] in {
                "known_mixed", "partially_recorded"}),
        "runtime_configuration_mixing_status": discovered_configs[
            "mixing_status"],
        "recorded_runtime_configuration_count": discovered_configs[
            "recorded_run_count"],
        "unrecorded_runtime_configuration_count": discovered_configs[
            "unrecorded_run_count"],
        "recorded_runtime_configuration_identity_count": discovered_configs[
            "recorded_identity_count"],
        "runtime_configuration_comparison_available": discovered_configs[
            "comparison_available"],
        "discovered_runtime_configurations": discovered_configs,
        "statistics_runtime_configurations": statistics_configs,
        "statistics_pool_is_controlled_condition": statistics_configs[
            "controlled_single_condition"],
        "load_errors": errors,
        "overall": {
            "span_envelope_s": _stats([
                summary["span_envelope_s"] for summary in selected]),
            "non_human_envelope_s": _stats([
                summary["span_envelope_non_human_s"] for summary in selected]),
            "human_wait_occupied_s": _stats([
                summary["human_wait_occupied_s"] for summary in selected]),
            "vlm_inference_occupied_s": _stats([
                summary["vlm_inference_occupied_s"] for summary in selected]),
        },
        "phases": _named_rows(
            selected, "phases", ("occupied_s", "self_s", "cumulative_s")),
        "resource_kinds": _named_rows(
            selected, "kinds", ("occupied_s", "self_s", "cumulative_s")),
        "operations": _named_rows(
            selected, "groups", ("occupied_s", "self_s", "inclusive_s")),
        "actions": _actions(selected),
        "runs": rows,
        "scope": (
            "saved_hierarchical_traces_only_default_statistics_use_"
            "complete_runs_human_wait_reported_separately"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-root", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--runtime-configuration-sha256")
    parser.add_argument("--include-incomplete", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        args.timing_root, candidate_id=args.candidate_id,
        runtime_configuration_sha256=args.runtime_configuration_sha256,
        include_incomplete=args.include_incomplete)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
