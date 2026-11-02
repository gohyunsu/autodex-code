#!/usr/bin/env python3
"""Summarize saved human/VLM label agreement without rerunning a model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


ASSESSMENTS = {
    "grasp_success": "lift_assessments",
    "preinsert_reached": "preinsert_assessments",
    "insertion_success": "released_insertion_assessments",
}

CURRENT_LIFT_REPLAY_POLICY = (
    "precision_insertion_visible_grip_v5_two_view_consensus")


def _latest_report(attempt: Path, directory: str) -> Path | None:
    reports = sorted((attempt / directory).glob("*/report.json"))
    return reports[-1] if reports else None


def _prompt_digest(prompts: list[str]) -> str | None:
    unique = sorted(set(prompts))
    return (hashlib.sha256(json.dumps(
        unique, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if unique else None)


def _semantic_prompt(prompt: str) -> str:
    """Remove capture identity while preserving the model instruction."""
    normalized = re.sub(r"@\d+(?:\.\d+)?", "@<TIME>", prompt)
    return re.sub(r"\b\d{8}\b", "<CAMERA_ID>", normalized)


def _visual_policy(report: dict) -> tuple[str, str | None, str | None]:
    visual = report.get("visual")
    if not isinstance(visual, dict):
        visual = {}
    aggregation = visual.get("aggregation")
    crops = visual.get("input_crops")
    policy = report.get("visual_input_policy")
    if not isinstance(policy, str) or not policy:
        policy = visual.get("schema")
    if not isinstance(policy, str) or not policy:
        if isinstance(aggregation, dict):
            policy = aggregation.get("schema")
        if ((not isinstance(policy, str) or not policy) and
                isinstance(crops, dict)):
            policy = crops.get("schema")
    if not isinstance(policy, str) or not policy:
        policy = "unrecorded_legacy"

    prompts = []

    def collect(value):
        if isinstance(value, dict):
            prompt = value.get("prompt")
            if isinstance(prompt, str) and prompt:
                prompts.append(prompt)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(visual)
    # Image-order text and per-view prompts embed capture timestamps/camera
    # serials. Those values bind provenance but do not change the instruction.
    # Keep a full digest while grouping evaluations by a normalized semantic
    # digest that preserves the rest of every prompt.
    semantic_prompts = [_semantic_prompt(prompt) for prompt in prompts]
    return policy, _prompt_digest(semantic_prompts), _prompt_digest(prompts)


def _replay_current_lift_vote(report: dict) -> tuple[str | None, bool | None]:
    """Apply the current capture4 vote rule to already saved per-view votes.

    This does not rerun Qwen and therefore does not claim current-model
    performance.  It isolates the effect of replacing the historical v3
    one-view consensus/fingertip rescue with the current primary-crop rule:
    two YES views mean ``held``, all NO views mean ``miss``, and every mixed
    one-view result abstains.
    """
    visual = report.get("visual")
    aggregation = (visual.get("aggregation")
                   if isinstance(visual, dict) else None)
    per_view = (aggregation.get("per_view")
                if isinstance(aggregation, dict) else None)
    if not isinstance(per_view, list) or len(per_view) < 2:
        return None, None
    votes = [row.get("yes") for row in per_view
             if isinstance(row, dict) and type(row.get("yes")) is bool]
    if len(votes) != len(per_view):
        return None, None
    if sum(vote is True for vote in votes) >= 2:
        return "held", True
    if all(vote is False for vote in votes):
        return "miss", False
    return "unobservable", None


def _agreement(rows: list[dict]) -> dict:
    paired = [row for row in rows
              if type(row["manual_value"]) is bool and
              type(row["vlm_value"]) is bool]
    confusion = Counter(
        ("positive" if row["manual_value"] else "negative",
         "positive" if row["vlm_value"] else "negative")
        for row in paired)
    correct = sum(row["manual_value"] == row["vlm_value"]
                  for row in paired)
    prompt_sha256_values = sorted({str(row["prompt_sha256"])
                                   for row in rows
                                   if row.get("prompt_sha256")})
    full_prompt_sha256_values = sorted({str(row["full_prompt_sha256"])
                                        for row in rows
                                        if row.get("full_prompt_sha256")})
    return {
        "saved_assessments": len(rows),
        "manual_labels": sum(type(row["manual_value"]) is bool
                             for row in rows),
        "vlm_labels": sum(type(row["vlm_value"]) is bool
                          for row in rows),
        "paired_labels": len(paired),
        "correct": correct,
        "accuracy": (correct / len(paired) if paired else None),
        "prompt_sha256": (prompt_sha256_values[0]
                          if len(prompt_sha256_values) == 1 else None),
        "prompt_sha256_values": prompt_sha256_values,
        "full_prompt_sha256_values": full_prompt_sha256_values,
        "confusion": {
            "true_positive": confusion[("positive", "positive")],
            "true_negative": confusion[("negative", "negative")],
            "false_positive": confusion[("negative", "positive")],
            "false_negative": confusion[("positive", "negative")],
        },
    }


def summarize(attempt_root: Path) -> dict:
    """Compare labels already persisted by manual-mode live episodes."""
    root = attempt_root.expanduser().resolve()
    rows = []
    attempts = sorted(path for path in root.glob("attempt_*") if path.is_dir())
    for attempt in attempts:
        result_path = attempt / "result.json"
        result = (json.loads(result_path.read_text(encoding="utf-8"))
                  if result_path.is_file() else {})
        for stage, directory in ASSESSMENTS.items():
            report_path = _latest_report(attempt, directory)
            if report_path is None:
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            manual = report.get("manual_annotation")
            vlm = report.get("vlm_label")
            manual_value = (manual.get("value")
                            if isinstance(manual, dict) else None)
            vlm_value = vlm.get("value") if isinstance(vlm, dict) else None
            if type(manual_value) is not bool and type(vlm_value) is not bool:
                continue
            visual_policy, prompt_sha256, full_prompt_sha256 = (
                _visual_policy(report))
            replay_class, replay_value = (
                _replay_current_lift_vote(report)
                if stage == "grasp_success" else (None, None))
            rows.append({
                "attempt_id": attempt.name,
                "candidate_id": result.get("candidate_id"),
                "stage": stage,
                "manual_value": (manual_value
                                 if type(manual_value) is bool else None),
                "vlm_value": vlm_value if type(vlm_value) is bool else None,
                "vlm_class": (vlm.get("class")
                              if isinstance(vlm, dict) else None),
                "manual_note": (manual.get("note", "")
                                if isinstance(manual, dict) else ""),
                "visual_policy": visual_policy,
                "prompt_sha256": prompt_sha256,
                "full_prompt_sha256": full_prompt_sha256,
                "current_policy_replay_class": replay_class,
                "current_policy_replay_value": replay_value,
                "report": str(report_path),
            })

    stages = {}
    for stage in ASSESSMENTS:
        selected = [row for row in rows if row["stage"] == stage]
        policies = {}
        for policy in sorted({row["visual_policy"] for row in selected}):
            policy_rows = [row for row in selected
                           if row["visual_policy"] == policy]
            policies[policy] = {
                "visual_policy": policy,
                **_agreement(policy_rows),
            }
        stages[stage] = {**_agreement(selected), "policies": policies}
        if stage == "grasp_success":
            replay_rows = [{**row, "vlm_value":
                            row["current_policy_replay_value"]}
                           for row in selected
                           if row["current_policy_replay_class"] is not None]
            stages[stage]["current_policy_replay"] = {
                "policy": CURRENT_LIFT_REPLAY_POLICY,
                "source_assessments": len(replay_rows),
                "abstentions": sum(
                    row["current_policy_replay_class"] == "unobservable"
                    for row in replay_rows),
                **_agreement(replay_rows),
                "scope": (
                    "saved_per_view_vote_reaggregation_only_no_model_rerun"),
            }
    return {
        "schema": "precision_insertion_saved_label_summary_v4",
        "attempt_root": str(root),
        "attempts_scanned": len(attempts),
        "stages": stages,
        "comparisons": rows,
        "scope": (
            "saved_label_agreement_stratified_by_recorded_vlm_policy_"
            "no_model_rerun_or_physical_truth_claim"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.attempt_root)
    encoded = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
