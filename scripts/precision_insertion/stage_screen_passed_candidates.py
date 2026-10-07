#!/usr/bin/env python3
"""Copy ranked screen survivors into an isolated downstream test pool."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def _quality_key(candidate: Path) -> tuple[float, float, int, str]:
    report_path = candidate / "contact_screen.json"
    if not report_path.is_file():
        return (float("inf"), float("inf"), 10**12, candidate.name)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    quality = report.get("quality", {})
    numeric_id = int(candidate.name) if candidate.name.isdigit() else 10**12
    return (
        float(quality.get("grasp_error_max", float("inf"))),
        float(quality.get("contact_distance_mean_abs_m", float("inf"))),
        numeric_id,
        candidate.name,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scene", required=True, type=Path)
    parser.add_argument("--screen-report", required=True, type=Path)
    parser.add_argument("--output-scene", required=True, type=Path)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="keep the best N by BODex quality; zero keeps every pass",
    )
    parser.add_argument(
        "--include-candidate",
        action="append",
        default=[],
        help="also include this passing candidate ID (repeatable)",
    )
    parser.add_argument(
        "--replace-backup",
        type=Path,
        help="move an existing output scene to this explicit backup first",
    )
    args = parser.parse_args()

    source = args.source_scene.expanduser().resolve()
    screen_path = args.screen_report.expanduser().resolve()
    output = args.output_scene.expanduser().resolve()
    if not source.is_dir():
        parser.error(f"source scene does not exist: {source}")
    if not screen_path.is_file():
        parser.error(f"screen report does not exist: {screen_path}")
    if args.max_candidates < 0:
        parser.error("--max-candidates cannot be negative")
    if output.exists():
        if args.replace_backup is None:
            parser.error(f"output exists; pass --replace-backup PATH: {output}")
        backup = args.replace_backup.expanduser().resolve()
        if backup.exists():
            parser.error(f"backup already exists: {backup}")
        backup.parent.mkdir(parents=True, exist_ok=True)
        output.rename(backup)

    report = json.loads(screen_path.read_text(encoding="utf-8"))
    passed_ids = {str(value) for value in report["passed_candidates"]}
    required_ids = {str(value) for value in args.include_candidate}
    invalid_required = sorted(required_ids - passed_ids)
    if invalid_required:
        parser.error(
            "explicitly included IDs did not pass the screen: "
            + ", ".join(invalid_required)
        )
    missing = sorted(
        candidate_id for candidate_id in passed_ids
        if not (source / candidate_id).is_dir()
    )
    if missing:
        parser.error("screen pass is missing from source: " + ", ".join(missing[:20]))

    ranked = sorted((source / candidate_id for candidate_id in passed_ids), key=_quality_key)
    selected = ranked[:args.max_candidates] if args.max_candidates else ranked
    selected_by_id = {candidate.name: candidate for candidate in selected}
    for candidate_id in required_ids:
        selected_by_id[candidate_id] = source / candidate_id
    selected = sorted(selected_by_id.values(), key=_quality_key)

    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(selected, start=1):
        destination = output / candidate.name
        shutil.copytree(candidate, destination)
        quality = _quality_key(candidate)
        rows.append({
            "candidate": candidate.name,
            "rank": rank,
            "grasp_error_max": quality[0],
            "contact_distance_mean_abs_m": quality[1],
            "explicitly_included": candidate.name in required_ids,
        })

    manifest = {
        "schema_version": 1,
        "status": "staged_screen_passes_not_simulated_or_physical_validated",
        "source_scene": str(source),
        "screen_report": str(screen_path),
        "output_scene": str(output),
        "screen_pass_count": len(passed_ids),
        "max_candidates": args.max_candidates,
        "selected_count": len(rows),
        "selection_order": "grasp_error_max_then_contact_distance_then_candidate_id",
        "selected": rows,
    }
    (output / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "screen_pass_count": len(passed_ids),
        "selected_count": len(rows),
        "output_scene": str(output),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
