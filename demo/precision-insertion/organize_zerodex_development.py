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


def organize_nas_history(*, hyunsu_root: Path, project_root: Path) -> dict:
    """Relocate precision diagnostics on the same NAS, retaining old paths."""
    source_paths = [
        hyunsu_root / "tmp/20261023_prev",
        hyunsu_root / "tmp/20261024_key_face_center_methods",
        *sorted((hyunsu_root / "tmp").glob("20261026_*")),
        *sorted(hyunsu_root.glob("precision_socket_pose_review*")),
    ]
    destination_root = project_root / "development/historical_nas"
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest_path = project_root / "development/HISTORICAL_NAS_MANIFEST.json"
    previous = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else None)
    moved = []
    existing_links = []
    for source in source_paths:
        if source.is_symlink():
            if source.resolve().is_relative_to(destination_root.resolve()):
                existing_links.append({"original": str(source),
                                       "canonical": str(source.resolve()),
                                       "old_path_is_relative_symlink": True})
            continue
        if not source.exists():
            continue
        relative = source.relative_to(hyunsu_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        source.rename(destination)
        source.symlink_to(os.path.relpath(destination, source.parent),
                          target_is_directory=destination.is_dir())
        moved.append({"original": str(source), "canonical": str(destination),
                      "old_path_is_relative_symlink": True})
    report = {
        "schema": "precision_insertion_historical_nas_migration_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "moved": list({row["original"]: row for row in
                       (previous.get("moved", []) if previous else []) +
                       existing_links + moved}.values()),
        "external_provenance_not_runtime": [
            str(hyunsu_root / "handoff"),
            str(hyunsu_root / "backup/precision-insertion_20261024"),
        ],
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "moved_count": len(moved)}


def organize_runtime_mutables(*, stage_root: Path,
                              development_root: Path) -> dict:
    """Move pre-separation run output out of the immutable runtime stage.

    Early capture4 trials wrote ``runs/`` and inline ``*_request_*`` frames
    beside launch assets.  New launches use the project ``runs/`` tree.  This
    migration keeps every old absolute path readable through a relative
    symlink while placing the actual bytes below ``development/``.
    """
    stage_root = stage_root.expanduser().resolve()
    development_root = development_root.expanduser().resolve()
    destination_root = development_root / "legacy_runtime_mutables"
    destination_root.mkdir(parents=True, exist_ok=True)
    candidates = [stage_root / "runs", *sorted(stage_root.glob("*_request_*"))]
    moved, existing = [], []
    for source in candidates:
        if source.is_symlink():
            target = source.resolve()
            if target.is_relative_to(destination_root):
                existing.append({"original": str(source),
                                 "canonical": str(target)})
            continue
        if not source.exists():
            continue
        category = ("stage_runs" if source.name == "runs" else
                    "raw_captures")
        destination = destination_root / category / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(destination)
        source.rename(destination)
        source.symlink_to(os.path.relpath(destination, source.parent),
                          target_is_directory=destination.is_dir())
        moved.append({"original": str(source),
                      "canonical": str(destination)})
    manifest_path = development_root / "RUNTIME_MUTABLE_MIGRATION.json"
    previous = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {"moved": []})
    records = {row["original"]: row for row in
               [*previous.get("moved", []), *existing, *moved]}
    report = {
        "schema": "precision_insertion_runtime_mutable_migration_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "stage_root": str(stage_root),
        "development_root": str(development_root),
        "moved": list(records.values()),
    }
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"manifest": str(manifest_path), "moved_count": len(moved),
            "existing_count": len(existing)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-runs", type=Path, default=(
        Path.home() / "shared_data/AutoDex/precision_insertion/zerodex_capture4/runs"))
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--development-root", type=Path, default=Path(
        os.environ.get("PRECISION_DEV_ROOT",
                       str(Path.home() / "precision_insertion_project/development/zerodex_capture4"))))
    parser.add_argument("--organize-nas-history", action="store_true")
    parser.add_argument("--organize-runtime-mutables", action="store_true")
    parser.add_argument("--stage-root", type=Path, default=Path(
        os.environ.get("PRECISION_STAGE_ROOT",
                       str(Path.home() / "shared_data/AutoDex/precision_insertion/zerodex_capture4"))))
    parser.add_argument("--nas-project-root", type=Path, default=Path(
        os.environ.get("PRECISION_NAS_PROJECT_ROOT",
                       "/mnt/paradex2/hyunsu/precision_insertion")))
    args = parser.parse_args()
    if args.organize_runtime_mutables:
        print(json.dumps(organize_runtime_mutables(
            stage_root=args.stage_root,
            development_root=args.development_root), indent=2))
        return
    if args.organize_nas_history:
        print(json.dumps(organize_nas_history(
            hyunsu_root=args.nas_project_root.parent,
            project_root=args.nas_project_root), indent=2))
        return
    print(json.dumps(organize(
        stage_runs=args.stage_runs, temp_root=args.temp_root,
        development_root=args.development_root), indent=2))


if __name__ == "__main__":
    main()
