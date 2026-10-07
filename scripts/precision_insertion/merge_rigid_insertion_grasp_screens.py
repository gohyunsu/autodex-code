#!/usr/bin/env python3
"""Merge complete deterministic shards from the rigid-insertion screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _candidate_key(row: dict[str, Any]) -> tuple[int, str]:
    name = str(row["candidate"])
    return (int(name), name) if name.isdigit() else (10**12, name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = [path.expanduser().resolve() for path in args.reports]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("missing report: " + ", ".join(missing))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

    invariant_fields = (
        "schema_version",
        "status",
        "scene",
        "tabletop_pose",
        "task_geometry",
        "socket_mesh",
        "contact_policy_mode",
        "sampling",
    )
    reference = reports[0]
    for report in reports[1:]:
        for field in invariant_fields:
            if report.get(field) != reference.get(field):
                parser.error(f"shard invariant mismatch: {field}")

    shards = [report.get("candidate_shard", {}) for report in reports]
    counts = {shard.get("count") for shard in shards}
    totals = {shard.get("unsharded_candidate_count") for shard in shards}
    indices = [shard.get("index") for shard in shards]
    if len(counts) != 1 or len(totals) != 1:
        parser.error("inconsistent shard count or unsharded candidate count")
    shard_count = counts.pop()
    expected_total = totals.pop()
    if shard_count != len(reports) or sorted(indices) != list(range(shard_count)):
        parser.error("reports do not contain every shard exactly once")

    rows = sorted(
        [row for report in reports for row in report["candidates"]],
        key=_candidate_key,
    )
    candidate_ids = [str(row["candidate"]) for row in rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        parser.error("duplicate candidate across shards")
    if len(rows) != expected_total:
        parser.error(
            f"merged {len(rows)} candidates, expected {expected_total}"
        )

    merged = dict(reference)
    merged["candidate_shard"] = {
        "merged": True,
        "count": shard_count,
        "indices": sorted(indices),
        "source_reports": [str(path) for path in paths],
    }
    merged["candidate_count"] = len(rows)
    merged["passed_candidates"] = [
        row["candidate"] for row in rows if row["passed"]
    ]
    merged["candidates"] = rows

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps({
        "candidate_count": merged["candidate_count"],
        "passed_count": len(merged["passed_candidates"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
