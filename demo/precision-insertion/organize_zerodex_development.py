#!/usr/bin/env python3
"""Gather existing ZeroDex precision diagnostics under one local project path.

The original pathname becomes a symlink, so saved reports with absolute paths
remain readable. This script does not touch canonical assets or live runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import errno
import json
import os
from pathlib import Path
import re
import shutil


KEEP_RUN_NAMES = {"session_evidence", "recordings"}
KEEP_RUN_PREFIXES = ("live_", "socket_session_")
TEMP_PREFIXES = ("precision-", "precision_", "zerodex_")
KEEP_TEMP_PREFIXES = ("precision_foundpose_", "precision-insertion-managed-",
                      "precision-insertion-services-", "precision-launch-")


def _date(name: str) -> str:
    match = re.search(r"20\d{6}", name)
    return match.group(0) if match else "undated"


def organize(*, stage_runs: Path, temp_root: Path, development_root: Path) -> dict:
    development_root.mkdir(parents=True, exist_ok=True)
    moved = []
    manifest_path = development_root / "MIGRATION_MANIFEST.json"
    previous = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else None)
    existing_links = []
    for category, root in (("legacy_stage_runs", stage_runs),
                           ("legacy_tmp", temp_root)):
        for source in root.iterdir():
            if source.is_symlink() and source.resolve().is_relative_to(
                    development_root.resolve()):
                existing_links.append({"original": str(source),
                                       "canonical": str(source.resolve()),
                                       "category": category})
    sources = (
        ("legacy_stage_runs", [
            path for path in stage_runs.iterdir()
            if path.name not in KEEP_RUN_NAMES
            and not path.name.startswith(KEEP_RUN_PREFIXES)
            and not path.is_symlink()]),
        ("legacy_tmp", [
            path for path in temp_root.iterdir()
            if path.name.startswith(TEMP_PREFIXES)
            and not path.name.startswith(KEEP_TEMP_PREFIXES)
            and not path.is_symlink()]),
    )
    for category, paths in sources:
        for source in sorted(paths):
            destination = development_root / category / _date(source.name) / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            try:
                source.rename(destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.move(str(source), str(destination))
            source.symlink_to(destination, target_is_directory=destination.is_dir())
            moved.append({"original": str(source),
                          "canonical": str(destination), "category": category})
    report = {
        "schema": "precision_insertion_development_migration_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "development_root": str(development_root),
        "moved": list({entry["original"]: entry for entry in
                       (previous.get("moved", []) if previous else []) +
                       existing_links + moved}.values()),
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"development_root": str(development_root), "moved_count": len(moved),
            "by_category": {
                category: sum(item["category"] == category for item in moved)
                for category, _ in sources}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-runs", type=Path, default=(
        Path.home() / "shared_data/AutoDex/precision_insertion/zerodex_capture4/runs"))
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--development-root", type=Path, default=Path(
        os.environ.get("PRECISION_DEV_ROOT",
                       str(Path.home() / "precision_insertion_project/development/zerodex_capture4"))))
    args = parser.parse_args()
    print(json.dumps(organize(
        stage_runs=args.stage_runs, temp_root=args.temp_root,
        development_root=args.development_root), indent=2))


if __name__ == "__main__":
    main()
