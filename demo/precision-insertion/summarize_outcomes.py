#!/usr/bin/env python3
"""Summarize the physical execution funnel and alignment branch.

The execution funnel contains only ordered physical stages.  Endpoint
observability and closed-loop convergence are reported separately because the
runtime can proceed with nominal insertion when metric visual evidence is not
available.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


FUNNEL = (
    ("physical_pickup_started", "pickup_started.json"),
    ("pickup_execution_recorded", "pickup_execution.json"),
    ("grasp_success", None),
    ("transfer_execution_recorded", "transfer_execution.json"),
    ("preinsert_reached", None),
    ("guarded_insertion_execution_recorded", "guarded_insertion_execution.json"),
    ("tip_entry_success", None),
    ("insertion_success", None),
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _latest_state(attempt: Path) -> dict[str, Any]:
    paths = sorted(attempt.glob("state_*.json"))
    return _load(paths[-1]) if paths else {}


def _label(result: dict[str, Any], state: dict[str, Any], name: str) -> bool | None:
    if name == "tip_entry_success":
        for record in (result.get("diagnostic_labels"),
                       state.get("diagnostic_labels")):
            if isinstance(record, dict) and type(record.get(name)) is bool:
                return record[name]
        return None
    labels = result.get("labels")
    if isinstance(labels, dict) and type(labels.get(name)) is bool:
        return labels[name]
    return state.get(name) if type(state.get(name)) is bool else None


def _metric_alignment(record: dict[str, Any]) -> bool:
    alignment = record.get("alignment")
    if not isinstance(alignment, dict):
        return False
    status = str(alignment.get("status", ""))
    inliers = alignment.get("inlier_cameras")
    tip = alignment.get("tip_socket_m")
    return bool(
        "metric" in status and isinstance(inliers, list) and len(inliers) >= 2
        and isinstance(tip, list) and len(tip) == 3)


def _alignment_branch(attempt: Path) -> dict[str, Any]:
    paths = sorted((attempt / "same_trial_xy" / "observations").glob("*.json"))
    observations = [_load(path) for path in paths]
    choices = [record.get("choice") if isinstance(record.get("choice"), dict)
               else {} for record in observations]
    metric = [_metric_alignment(record) for record in observations]
    # A requested shift is known to have executed only when a later observation
    # exists.  The final choice may have been planned but never commanded.
    executed_shift_indices = [
        index for index, choice in enumerate(choices[:-1])
        if choice.get("status") == "shift"]
    final_choice = choices[-1] if choices else {}
    final_alignment = (observations[-1].get("alignment", {})
                       if observations else {})
    return {
        "observation_count": len(observations),
        "first_metric_endpoint": bool(metric and metric[0]),
        "any_metric_endpoint": any(metric),
        "physical_correction_attempted": bool(executed_shift_indices),
        "physical_correction_count": len(executed_shift_indices),
        "alignment_target_reached": final_choice.get("status") == "aligned",
        "final_choice_status": final_choice.get("status"),
        "final_alignment_status": (
            final_alignment.get("status")
            if isinstance(final_alignment, dict) else None),
        "final_alignment_reason": (
            final_alignment.get("reason")
            if isinstance(final_alignment, dict) else None),
    }


def _attempt_row(attempt: Path) -> dict[str, Any]:
    result = _load(attempt / "result.json")
    state = _latest_state(attempt)
    labels = {name: _label(result, state, name) for name in (
        "grasp_success", "preinsert_reached", "tip_entry_success",
        "insertion_success")}
    evidence = {
        name: ((attempt / filename).is_file() if filename else
               labels[name] is not None)
        for name, filename in FUNNEL}
    alignment = _alignment_branch(attempt)
    return {
        "attempt_id": attempt.name,
        "candidate_id": result.get("candidate_id", state.get("candidate_id")),
        "session_calibration_sha256": state.get("session_calibration_sha256"),
        "gap_mm": state.get("gap_mm"),
        "failure_code": result.get("failure_code", state.get("failure_code")),
        "labels": labels,
        "stage_evidence": evidence,
        "alignment": alignment,
    }


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float] | None:
    if trials <= 0:
        return None
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    margin = (z / denominator * math.sqrt(
        proportion * (1.0 - proportion) / trials +
        z * z / (4.0 * trials * trials)))
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _rate(successes: int, trials: int) -> dict[str, Any]:
    return {
        "successes": successes,
        "trials": trials,
        "proportion": successes / trials if trials else None,
        "wilson_95": _wilson(successes, trials),
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def evidence(name: str) -> int:
        return sum(row["stage_evidence"][name] for row in rows)

    def positives(name: str) -> int:
        return sum(row["labels"].get(name) is True for row in rows)

    def annotated(name: str) -> int:
        return sum(type(row["labels"].get(name)) is bool for row in rows)

    pickup_started = evidence("physical_pickup_started")
    pickup_executed = evidence("pickup_execution_recorded")
    grasp = positives("grasp_success")
    transfer_executed = evidence("transfer_execution_recorded")
    preinsert = positives("preinsert_reached")
    insertion_executed = evidence("guarded_insertion_execution_recorded")
    tip_entry = positives("tip_entry_success")
    inserted = positives("insertion_success")
    branch_rows = [row for row in rows
                   if row["labels"]["preinsert_reached"] is True]
    insertion_without_metric = sum(
        row["stage_evidence"]["guarded_insertion_execution_recorded"] and
        not row["alignment"]["any_metric_endpoint"] for row in branch_rows)
    return {
        "attempt_count": len(rows),
        "execution_funnel": [
            {"stage": "physical_pickup_started", "count": pickup_started},
            {"stage": "pickup_execution_recorded", "count": pickup_executed},
            {"stage": "grasp_success", "count": grasp,
             "annotated": annotated("grasp_success")},
            {"stage": "transfer_execution_recorded", "count": transfer_executed},
            {"stage": "preinsert_reached", "count": preinsert,
             "annotated": annotated("preinsert_reached")},
            {"stage": "guarded_insertion_execution_recorded",
             "count": insertion_executed},
            {"stage": "tip_entry_success", "count": tip_entry,
             "annotated": annotated("tip_entry_success")},
            {"stage": "insertion_success", "count": inserted,
             "annotated": annotated("insertion_success")},
        ],
        "rates": {
            "end_to_end_observed_success": _rate(inserted, pickup_started),
            "conditional_grasp_all_executions": _rate(grasp, pickup_executed),
            "conditional_grasp_labeled": _rate(
                grasp, annotated("grasp_success")),
            "conditional_preinsert_after_grasp": _rate(preinsert, grasp),
            "conditional_smooth_entry_all_executions": _rate(
                tip_entry, insertion_executed),
            "conditional_smooth_entry_labeled": _rate(
                tip_entry, annotated("tip_entry_success")),
            "conditional_final_insertion_all_executions": _rate(
                inserted, insertion_executed),
            "conditional_final_insertion_labeled": _rate(
                inserted, annotated("insertion_success")),
            "tip_entry_label_coverage": _rate(
                annotated("tip_entry_success"), insertion_executed),
            "final_label_coverage": _rate(
                annotated("insertion_success"), insertion_executed),
        },
        "release_assisted_entry_count": sum(
            row["labels"]["tip_entry_success"] is False and
            row["labels"]["insertion_success"] is True for row in rows),
        "alignment_branch": {
            "preinsert_attempts": len(branch_rows),
            "with_observation": sum(
                row["alignment"]["observation_count"] > 0 for row in branch_rows),
            "first_metric_endpoint": sum(
                row["alignment"]["first_metric_endpoint"] for row in branch_rows),
            "any_metric_endpoint": sum(
                row["alignment"]["any_metric_endpoint"] for row in branch_rows),
            "physical_correction_attempted": sum(
                row["alignment"]["physical_correction_attempted"]
                for row in branch_rows),
            "alignment_target_reached": sum(
                row["alignment"]["alignment_target_reached"]
                for row in branch_rows),
            "insertion_without_any_metric_endpoint": insertion_without_metric,
            "loop_ended_before_insertion": sum(
                row["alignment"]["observation_count"] > 0 and
                not row["stage_evidence"]["guarded_insertion_execution_recorded"]
                for row in branch_rows),
            "final_choice_status_counts": dict(sorted(Counter(
                str(row["alignment"]["final_choice_status"])
                for row in branch_rows).items())),
        },
        "failure_code_counts": dict(sorted(Counter(
            str(row["failure_code"]) for row in rows
            if row["failure_code"] is not None).items())),
    }


def summarize(attempt_root: Path, *, candidate_id: str | None = None,
              session_sha256: str | None = None,
              gap_mm: float | None = None) -> dict[str, Any]:
    root = attempt_root.expanduser().resolve()
    rows = [_attempt_row(path) for path in sorted(
        path for path in root.glob("attempt_*") if path.is_dir())]
    rows = [row for row in rows
            if (candidate_id is None or row["candidate_id"] == candidate_id)
            and (session_sha256 is None or
                 row["session_calibration_sha256"] == session_sha256)
            and (gap_mm is None or row["gap_mm"] == gap_mm)]
    return {
        "schema": "precision_insertion_outcome_summary_v1",
        "attempt_root": str(root),
        "filters": {"candidate_id": candidate_id,
                    "session_calibration_sha256": session_sha256,
                    "gap_mm": gap_mm},
        **_summarize_rows(rows),
        "attempts": rows,
        "scope": (
            "saved_execution_records_and_labels_only_"
            "alignment_observability_not_a_required_execution_stage"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--session-sha256")
    parser.add_argument("--gap-mm", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        args.attempt_root, candidate_id=args.candidate_id,
        session_sha256=args.session_sha256, gap_mm=args.gap_mm)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
