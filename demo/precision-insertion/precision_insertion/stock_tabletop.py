"""Verify one complete original AutoDex v8 tabletop simulation-filter cell.

This module only audits BODex/MuJoCo output. It does not alter the stock
filter or treat its grasp-stability result as insertion success.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


STOCK_FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy",
               "bodex_info.npy")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_tabletop_cell(
    *, raw_root: Path, pass_root: Path, key_object: str,
    scene_id: str, expected_seed_count: int,
) -> tuple[dict, list[tuple[str, Path]]]:
    """Return counts and exact stock-passing seed directories for one cell."""
    if type(expected_seed_count) is not int or expected_seed_count < 1:
        raise ValueError("expected_seed_count must be positive")
    if not scene_id.isdigit() or str(int(scene_id)) != scene_id:
        raise ValueError("scene_id must be a canonical numeric v8 scene")
    raw_scene = Path(raw_root) / key_object / "table" / scene_id
    pass_scene = Path(pass_root) / key_object / "table" / scene_id
    observed_raw = {p.name: p for p in raw_scene.iterdir() if p.is_dir()}
    expected = {str(i) for i in range(expected_seed_count)}
    if set(observed_raw) != expected:
        raise ValueError(f"incomplete BODex cell {key_object}/table/{scene_id}: "
                         f"{len(observed_raw)}/{expected_seed_count}")
    counts = {"generated": expected_seed_count, "scene_clear": 0,
              "squeeze_contact": 0, "mujoco_stable": 0,
              "failed_scene_collision": 0, "failed_no_object_contact": 0,
              "failed_mujoco_or_error": 0}
    passed = []
    for seed_id in range(expected_seed_count):
        raw = observed_raw[str(seed_id)]
        collision = np.load(raw / "coll_valid.npy", allow_pickle=False)
        if collision.shape != () or collision.dtype != np.dtype(bool):
            raise ValueError(f"invalid collision flag: {raw}")
        result = json.loads((raw / "sim_eval.json").read_text(encoding="utf-8"))
        if result.get("hand") != "inspire" or result.get("version") != "v8":
            raise ValueError(f"not an Inspire v8 stock result: {raw}")
        if not bool(collision):
            if result.get("reason") != "scene_collision" or result.get("success") is not False:
                raise ValueError(f"inconsistent scene-collision result: {raw}")
            counts["failed_scene_collision"] += 1
            continue
        counts["scene_clear"] += 1
        if result.get("reason") == "no_object_contact":
            if result.get("success") is not False:
                raise ValueError(f"inconsistent no-contact result: {raw}")
            counts["failed_no_object_contact"] += 1
            continue
        if result.get("reason") is not None or type(result.get("success")) is not bool:
            raise ValueError(f"unexpected stock filter result: {raw}")
        counts["squeeze_contact"] += 1
        if not result["success"]:
            counts["failed_mujoco_or_error"] += 1
            continue
        candidate = pass_scene / str(seed_id)
        if not all((candidate / name).is_file() for name in STOCK_FILES):
            raise FileNotFoundError(f"stable result not copied by stock filter: {candidate}")
        if any(_sha(raw / name) != _sha(candidate / name)
               for name in STOCK_FILES):
            raise ValueError(f"stock pass arrays differ from raw proposal: {candidate}")
        counts["mujoco_stable"] += 1
        passed.append((str(seed_id), candidate))
    observed_pass = ({p.name for p in pass_scene.iterdir() if p.is_dir()}
                     if pass_scene.is_dir() else set())
    if observed_pass != {seed_id for seed_id, _ in passed}:
        raise ValueError(f"stock pass pool disagrees with MuJoCo results: {pass_scene}")
    return counts, passed
