"""Commission an insertion grasp's physical held relation from v8 samples.

The offline endpoint catalog identifies the exact, eligible grasp.  This
module creates no samples and cannot promote a MuJoCo pose to a measurement.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np

from .assets import AssetPaths
from .candidates import select_pose_candidates
from .config import TaskMode
from .geometry import validate_se3
from .physical_grasp_calibration import (
    calibrate_physical_held_relation, load_physical_sample_paths,
    verify_physical_held_relation,
)


_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def calibration_path_for_candidate(
    output_root: Path, mode: TaskMode, candidate_id: str,
) -> Path:
    """Resolve one mode/grasp-specific file for the live physical route."""
    parts = candidate_id.split("/") if isinstance(candidate_id, str) else []
    if len(parts) != 3 or any(_PART.fullmatch(part) is None for part in parts):
        raise ValueError("candidate ID must be scene_type/scene_id/grasp_id")
    return (Path(output_root).expanduser().resolve() / mode.family /
            mode.key_object / mode.socket_object / Path(*parts) /
            "physical_grasp_calibration.json")


def build_insertion_grasp_calibration(
    *, shared_root: Path, mode: TaskMode, catalog_path: Path,
    candidate_id: str, sample_paths: Sequence[Path], output_root: Path,
    minimum_independent_trials: int = 5,
    max_nominal_translation_drift_m: float,
    max_nominal_rotation_drift_deg: float,
) -> Path:
    """Bind physical samples to an eligible full-key v8 endpoint grasp.

    The output is a calibration summary, not permission to transfer or insert.
    A distinct commissioned future-trial bound is still required at runtime.
    """
    root = Path(shared_root).expanduser().resolve()
    catalog_file = Path(catalog_path).expanduser().resolve()
    catalog_bytes = catalog_file.read_bytes()
    catalog = json.loads(catalog_bytes)
    if not isinstance(catalog, dict):
        raise ValueError("endpoint catalog must be a JSON object")
    target_file = calibration_path_for_candidate(output_root, mode, candidate_id)
    key = tuple(candidate_id.split("/"))
    rows = [row for row in catalog.get("candidates", [])
            if isinstance(row, dict) and row.get("key") == list(key)]
    if len(rows) != 1:
        raise ValueError("candidate is absent or duplicated in endpoint catalog")
    selected = select_pose_candidates(
        catalog, expected_mode=mode,
        tabletop_pose_stem=rows[0]["tabletop_pose_stem"])
    matches = [row for row in selected["candidates"]
               if row["key"] == list(key)]
    if selected["status"] != "candidates_available" or len(matches) != 1:
        raise ValueError("selected grasp is not currently endpoint-eligible")
    candidate_dir = Path(matches[0]["candidate_dir"]).expanduser().resolve()
    expected_dir = AssetPaths(root, mode).candidate_dir / Path(*key)
    if candidate_dir != expected_dir.resolve():
        raise ValueError("selected grasp differs from the canonical v8 pool")
    nominal = validate_se3(np.load(candidate_dir / "wrist_se3.npy",
                                   allow_pickle=False), name="selected v8 T_key_hand")
    samples = load_physical_sample_paths(sample_paths)
    summary = calibrate_physical_held_relation(
        mode=mode, shared_root=root, candidate_key=key,
        candidate_T_key_hand=nominal, samples=samples,
        minimum_independent_trials=minimum_independent_trials,
        max_nominal_translation_drift_m=max_nominal_translation_drift_m,
        max_nominal_rotation_drift_deg=max_nominal_rotation_drift_deg)
    verify_physical_held_relation(
        record=summary, mode=mode, shared_root=root,
        candidate_key=key, candidate_dir=candidate_dir)
    if target_file.parent.exists():
        raise FileExistsError(f"grasp calibration already exists: {target_file.parent}")
    target_file.parent.mkdir(parents=True, exist_ok=False)
    target_file.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n", encoding="utf-8")
    binding = {
        "schema": "precision_insertion_grasp_calibration_binding_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object, "socket_object": mode.socket_object},
        "candidate_key": list(key),
        "candidate_dir": str(candidate_dir),
        "catalog_path": str(catalog_file),
        "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "physical_grasp_calibration": target_file.name,
        "physical_grasp_calibration_sha256": hashlib.sha256(
            target_file.read_bytes()).hexdigest(),
        "sample_count": summary["sample_count"],
        "scope": "physical_sample_summary_not_future_trial_bound_or_motion_permit",
        "robot_ready": False,
    }
    (target_file.parent / "binding.json").write_text(
        json.dumps(binding, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return target_file
