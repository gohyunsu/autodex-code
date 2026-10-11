"""Non-destructively assemble verified v8 reset seeds into one staging root.

Different offline BODex/MuJoCo runs place directed cells in separate handoff
trees. The existing direct-v8 loader accepts one parent root per repose
preflight, so this module copies explicitly selected *verified* cells into a
new root. It never installs into AutoDex's canonical NAS candidate tree and
never changes source scenes, meshes, evidence, or candidate IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

from autodex.utils.path import RESET_RELEASE_HEIGHTS_CM

from .config import TaskMode
from .reorient_assets import audit_v8_reorient_assets
from .reset_candidates import verify_v8_reset_seed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble_v8_reorient_handoff(
    *, shared_root: Path, mode: TaskMode,
    sources_by_cell: Mapping[tuple[int, int, int], Path],
    output_root: Path,
) -> dict:
    """Copy stable, source-hash-valid cells without pretending they are plans.

    Keys are ``(release_height_cm, from_v8_pose, to_v8_pose)`` and values are
    parents of ``reset_<height>``. All source seeds are verified against the
    current v8 key mesh, scenes, MuJoCo pass and squeeze/gravity trajectory
    *before* a new output directory is created. A failed source aborts the
    whole assembly; partial output is left for review if copying later fails.
    """
    root = Path(shared_root).expanduser().resolve()
    output_input = Path(output_root).expanduser()
    if (not root.is_dir() or not output_input.is_absolute() or
            not isinstance(sources_by_cell, Mapping) or
            not sources_by_cell):
        raise ValueError("handoff needs a shared root, explicit cells and output")
    output = output_input.resolve()
    if output.exists():
        raise FileExistsError(f"reorient handoff already exists: {output}")
    canonical = root / "AutoDex/candidates"
    if (output_input.absolute().is_relative_to(canonical) or
            output.is_relative_to(canonical.resolve()) or
            any(output.is_relative_to(
                (canonical / "inspire" / f"reset_{height}").resolve())
                for height in RESET_RELEASE_HEIGHTS_CM)):
        raise ValueError("handoff cannot write into canonical AutoDex candidates")
    rows, planned = [], []
    for cell, source_input in sorted(sources_by_cell.items()):
        if (not isinstance(cell, tuple) or len(cell) != 3 or
                any(type(value) is not int for value in cell)):
            raise ValueError("reorient source key must be (height, from, to)")
        height, pose_i, pose_j = cell
        if pose_i == pose_j:
            raise ValueError("reorient source needs two different tabletop poses")
        source_path = Path(source_input).expanduser()
        if not source_path.is_absolute():
            raise ValueError("reorient source roots must be absolute")
        source = source_path.resolve()
        if output.is_relative_to(source):
            raise ValueError("handoff output cannot be inside a source tree")
        cell_dir = (source / f"reset_{height}" / mode.key_object /
                    f"reorient_{height}" / f"{pose_i}_{pose_j}")
        if not cell_dir.is_dir():
            raise FileNotFoundError(f"directed reset cell is missing: {cell_dir}")
        stable_ids = []
        for seed in sorted(cell_dir.iterdir(), key=lambda p: p.name):
            if not seed.is_dir() or not seed.name.isdigit():
                continue
            result_path = seed / "sim_eval.json"
            if not result_path.is_file():
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("success") is not True:
                continue
            verified = verify_v8_reset_seed(
                shared_root=root, mode=mode, height_cm=height,
                from_pose_stem=pose_i, to_pose_stem=pose_j,
                seed_id=seed.name,
                candidate_root=source / f"reset_{height}")
            if seed.is_symlink() or any(path.is_symlink()
                                        for path in seed.rglob("*")):
                raise ValueError(f"reset seed contains a symlink: {seed}")
            stable_ids.append(seed.name)
            target = (output / f"reset_{height}" / mode.key_object /
                      f"reorient_{height}" / f"{pose_i}_{pose_j}" /
                      seed.name)
            planned.append((seed, target, verified["source_evidence_sha256"]))
        if not stable_ids:
            raise ValueError(f"directed reset cell has no verified pass: {cell}")
        rows.append({
            "height_cm": height,
            "from_v8_pose": pose_i, "to_v8_pose": pose_j,
            "source_root": str(source),
            "verified_seed_ids": sorted(stable_ids, key=int),
        })
    output.mkdir(parents=True, exist_ok=False)
    copied = []
    for source, target, evidence_hash in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, symlinks=False)
        if _sha(target / "source_evidence.json") != evidence_hash:
            raise ValueError(f"copied reset evidence changed: {target}")
        copied.append({"candidate_dir": str(target),
                       "source_evidence_sha256": evidence_hash})
    # The exact existing loader re-verifies every copied seed against the
    # destination tree; a file copy alone is not evidence of eligibility.
    for row in rows:
        for seed_id in row["verified_seed_ids"]:
            verify_v8_reset_seed(
                shared_root=root, mode=mode, height_cm=row["height_cm"],
                from_pose_stem=row["from_v8_pose"],
                to_pose_stem=row["to_v8_pose"], seed_id=seed_id,
                candidate_root=(output / f"reset_{row['height_cm']}"))
    audit = audit_v8_reorient_assets(
        shared_root=root, mode=mode, candidate_root=output)
    for row in rows:
        match = next(item for item in audit["directed_pairs"] if
                     (item["from_v8_pose"], item["to_v8_pose"]) ==
                     (row["from_v8_pose"], row["to_v8_pose"]))
        if (match["stable_reset_seed_counts_by_height_cm"].get(
                str(row["height_cm"])) != len(row["verified_seed_ids"])):
            raise ValueError("assembled reset audit differs from copied seeds")
    report = {
        "schema": "precision_insertion_v8_reorient_handoff_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "shared_root": str(root), "output_root": str(output),
        "cells": rows, "candidate_count": len(copied),
        "candidates": copied,
        "limitations": [
            "only listed directed cells have verified reset seeds",
            "MuJoCo gravity stability is not an accepted pose-fidelity limit",
            "no socket-aware Franka repose path or physical landing validated",
            "absolute v8 scene paths may differ on another host",
        ],
        "scope": "verified_staged_reset_seeds_not_robot_motion_permission",
        "robot_ready": False,
    }
    manifest = output / "handoff_manifest.json"
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report
