"""Read-only v8 repose target/seed assessment after insertion-grasp exhaustion.

An insertable target tabletop pose and a stable reset grasp are independent
facts. Neither is a Franka reset plan, physical release, or permission to
move. The caller must still preflight pickup, lift, transport, placement,
release and retreat in the session-frozen socket collision world.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from autodex.utils.path import RESET_RELEASE_HEIGHTS_CM

from .candidates import select_pose_candidates
from .config import TaskMode
from .geometry import validate_se3
from .reset_candidates import load_v8_reset_seeds


def assess_repose_options(
    *, shared_root: Path, mode: TaskMode, catalog: dict,
    current_pose_stem: str, target_stems: Iterable[str],
    T_robot_key: np.ndarray, max_center_in_hand_drift_m: float,
    max_symmetry_axis_tilt_deg: float,
    attempted_insertion: Iterable[tuple[str, str, str]] = (),
    covered_scenes: Iterable[int] = (),
    attempted_reset: Iterable[tuple[int, str, str]] = (),
    candidate_root: Path | None = None,
) -> dict:
    """Cross-check insertable target poses with direct-v8 reset seed cells.

    ``candidate_root`` is the parent of ``reset_<h>`` directories, usually
    the canonical AutoDex Inspire candidate tree. An explicit override is
    useful to audit a local handoff before its NAS installation. A reported
    seed is merely input to a *future* socket-aware full-chain preflight.
    """
    root = Path(shared_root).expanduser().resolve()
    if Path(catalog.get("shared_root", "")).expanduser().resolve() != root:
        raise ValueError("repose catalog and reset assets use different shared roots")
    pose = validate_se3(T_robot_key, name="fresh reset T_robot_key")
    drift = float(max_center_in_hand_drift_m)
    tilt = float(max_symmetry_axis_tilt_deg)
    if (not math.isfinite(drift) or drift <= 0 or
            not math.isfinite(tilt) or tilt <= 0):
        raise ValueError("commissioned reset drift and tilt limits are required")
    current_raw = str(current_pose_stem)
    if not current_raw.isdigit():
        raise ValueError("current tabletop pose must be a numeric v8 stem")
    current = f"{int(current_raw):03d}"
    current_file = (root / "object_processing" / mode.key_object /
                    "processed_data/info/tabletop" / f"{current}.npy")
    if not current_file.is_file():
        raise FileNotFoundError(f"current v8 tabletop pose missing: {current_file}")
    raw_targets = tuple(str(stem) for stem in target_stems)
    if any(not stem.isdigit() for stem in raw_targets):
        raise ValueError("repose targets must be numeric v8 pose stems")
    targets = tuple(f"{int(stem):03d}" for stem in raw_targets)
    if len(targets) != len(set(targets)) or current in targets:
        raise ValueError("repose targets must be unique other v8 pose stems")
    attempted = tuple((int(h), str(stem), str(seed))
                      for h, stem, seed in attempted_reset)
    if any(h not in RESET_RELEASE_HEIGHTS_CM or not stem.isdigit() or
           not seed.isdigit() for h, stem, seed in attempted):
        raise ValueError("invalid attempted v8 reset seed key")
    base = (None if candidate_root is None else
            Path(candidate_root).expanduser().resolve())
    available_status = (
        "reset_seed_available_requires_full_chain_preflight" if base is None
        else "staged_reset_seed_requires_install_and_full_chain_preflight")

    rows = []
    for target in targets:
        insertion = select_pose_candidates(
            catalog, expected_mode=mode, tabletop_pose_stem=target,
            attempted=attempted_insertion, covered_scenes=covered_scenes)
        if insertion["status"] in {"catalog_incomplete", "catalog_stale"}:
            return {
                "schema": "precision_insertion_repose_assessment_v1",
                "status": "catalog_unavailable", "reason": insertion["reason"],
                "current_pose_stem": current, "targets": [],
                "robot_ready": False,
            }
        if insertion["status"] != "candidates_available":
            # A previously suggested target can become exhausted or stale.
            rows.append({
                "target_pose_stem": target,
                "status": "no_insertable_grasp_for_target",
                "insertion_candidate_count": 0,
                "reset_seed_count_by_height_cm": {},
                "reset_seed_refs": [],
            })
            continue
        seed_refs = []
        counts = {}
        for height in RESET_RELEASE_HEIGHTS_CM:
            excluded = tuple(seed for h, stem, seed in attempted
                             if h == height and int(stem) == int(target))
            seeds = load_v8_reset_seeds(
                shared_root=root, mode=mode, height_cm=height,
                from_pose_stem=current, to_pose_stem=target,
                T_robot_key=pose,
                max_center_in_hand_drift_m=drift,
                max_symmetry_axis_tilt_deg=tilt,
                attempted_ids=excluded,
                candidate_root=(None if base is None else
                                base / f"reset_{height}"))
            infos = [] if seeds is None else seeds["scene_info"]
            counts[str(height)] = len(infos)
            seed_refs.extend({
                "height_cm": height, "seed_id": info["grasp_idx"],
                "source": info["source"],
            } for info in infos)
        rows.append({
            "target_pose_stem": target,
            "status": (available_status if seed_refs else
                       "no_reset_seed_for_insertable_pose"),
            "insertion_candidate_count": len(insertion["candidates"]),
            "reset_seed_count_by_height_cm": counts,
            "reset_seed_refs": seed_refs,
        })
    rows.sort(key=lambda row: (
        not bool(row["reset_seed_refs"]),
        -row["insertion_candidate_count"], row["target_pose_stem"]))
    return {
        "schema": "precision_insertion_repose_assessment_v1",
        "status": (available_status
                   if any(row["reset_seed_refs"] for row in rows) else
                   "no_executable_repose_path"),
        "current_pose_stem": current, "targets": rows,
        "candidate_root": (str(base) if base is not None else
                           str(root / "AutoDex/candidates/inspire")),
        "candidate_root_is_canonical": base is None,
        "fidelity_limits": {
            "max_center_in_hand_drift_m": drift,
            "max_symmetry_axis_tilt_deg": tilt,
        },
        "not_validated": [
            "socket-aware Franka pickup/lift/reorient/place/release/retreat",
            "physical grasp, reset, or insertion success",
        ],
        "robot_ready": False,
    }
