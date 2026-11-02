#!/usr/bin/env python3
"""Audit the precision branch's modifications to shared AutoDex runtime code.

This is a development/release audit, not a live-session gate.  It compares the
checkout with the declared original-file baseline and fails only when a stock
entry point changed or a new shared-runtime edit is outside the reviewed
adapter boundary.  Precision-owned demo files are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Iterable


DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
DEFAULT_POLICY = DEMO_DIR / "config" / "source_boundary.json"


def _string_list(record: dict, key: str) -> tuple[str, ...]:
    value = record.get(key)
    if (not isinstance(value, list) or
            not all(isinstance(item, str) and item for item in value) or
            len(value) != len(set(value))):
        raise ValueError(f"{key} must be a unique list of nonempty strings")
    return tuple(value)


def load_policy(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != "precision_insertion_source_boundary_v1":
        raise ValueError("unsupported source-boundary policy schema")
    baseline = record.get("baseline_commit")
    if not isinstance(baseline, str) or len(baseline) != 40:
        raise ValueError("baseline_commit must be a full Git commit ID")
    for key in (
            "stock_entrypoints", "shared_runtime_allowlist",
            "known_inactive_branch_paths", "known_inactive_branch_prefixes",
            "shared_runtime_prefixes"):
        _string_list(record, key)
    return record


def classify_changes(paths: Iterable[str], policy: dict) -> dict:
    changed = tuple(sorted(set(paths)))
    stock = set(_string_list(policy, "stock_entrypoints"))
    allowed = set(_string_list(policy, "shared_runtime_allowlist"))
    inactive = set(_string_list(policy, "known_inactive_branch_paths"))
    inactive_prefixes = _string_list(
        policy, "known_inactive_branch_prefixes")
    runtime_prefixes = _string_list(policy, "shared_runtime_prefixes")

    stock_changes = [path for path in changed if path in stock]
    adapter_changes = [path for path in changed if path in allowed]
    inactive_changes = [
        path for path in changed
        if path in inactive or path.startswith(inactive_prefixes)]
    unexpected = [
        path for path in changed
        if path.startswith(runtime_prefixes) and
        path not in stock and path not in allowed and path not in inactive and
        not path.startswith(inactive_prefixes)]
    return {
        "changed_shared_runtime_paths": list(changed),
        "stock_entrypoint_changes": stock_changes,
        "reviewed_adapter_changes": adapter_changes,
        "known_inactive_branch_changes": inactive_changes,
        "unexpected_shared_runtime_changes": unexpected,
        "passed": not stock_changes and not unexpected,
    }


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def audit(repo: Path, policy: dict, *, baseline: str | None = None) -> dict:
    repo = repo.resolve()
    declared = baseline or policy["baseline_commit"]
    _git(repo, "cat-file", "-e", f"{declared}^{{commit}}")
    resolved_base = _git(repo, "rev-parse", declared)
    head = _git(repo, "rev-parse", "HEAD")
    changed_output = _git(
        repo, "diff", "--name-only", f"{resolved_base}..{head}", "--",
        *_string_list(policy, "shared_runtime_prefixes"))
    changed = [line for line in changed_output.splitlines() if line]
    classification = classify_changes(changed, policy)
    return {
        "schema": "precision_insertion_source_boundary_audit_v1",
        "repo_root": str(repo),
        "baseline_commit": resolved_base,
        "head_commit": head,
        "policy_schema": policy["schema"],
        **classification,
        "scope": "development_audit_not_live_execution_gate"
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--baseline", help="optional Git revision override")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit(
        args.repo, load_policy(args.policy), baseline=args.baseline)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
