#!/usr/bin/env python3
"""Install only verified, fidelity-eligible cylinder reset seeds into v8.

This is an asset installation, not a Franka reset-path or physical-success
claim. Existing canonical candidates are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
for path in (str(DEMO_DIR), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.reset_candidates import (  # noqa: E402
    fidelity_within_limits, verify_v8_reset_seed,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(
    *, shared_root: Path, source_reset_root: Path, gap_mm: float,
    from_pose_stem: int, to_pose_stem: int,
    max_center_drift_m: float, max_axis_tilt_deg: float,
    output_manifest: Path,
) -> dict:
    """Prevalidate a directed cell, then copy eligible seeds without overwrite."""
    root = Path(shared_root).expanduser().resolve()
    source = Path(source_reset_root).expanduser().resolve()
    manifest = Path(output_manifest).expanduser().resolve()
    mode = select_mode("cylinder", gap_mm)
    if (source.name != "reset_12" or not source.is_dir() or
            not root.is_dir() or manifest.exists() or
            from_pose_stem == to_pose_stem):
        raise ValueError("reset installation needs a new manifest and exact reset_12 source")
    canonical = root / "AutoDex/candidates/inspire/reset_12"
    if source == canonical.resolve() or manifest.is_relative_to(canonical):
        raise ValueError("source and manifest must be separate from canonical candidates")
    cell = f"{from_pose_stem}_{to_pose_stem}"
    source_cell = source / mode.key_object / "reorient_12" / cell
    if not source_cell.is_dir() or source_cell.is_symlink():
        raise FileNotFoundError(f"staged cylinder reset cell is missing: {source_cell}")
    verified_rows = []
    rejected = []
    for seed in sorted(source_cell.iterdir(), key=lambda path: path.name):
        if not seed.is_dir() or not seed.name.isdigit():
            continue
        if seed.is_symlink() or any(path.is_symlink() for path in seed.rglob("*")):
            raise ValueError(f"reset seed has a symlink: {seed}")
        evidence = verify_v8_reset_seed(
            shared_root=root, mode=mode, height_cm=12,
            from_pose_stem=from_pose_stem, to_pose_stem=to_pose_stem,
            seed_id=seed.name, candidate_root=source)
        if fidelity_within_limits(
                evidence["fidelity"],
                max_center_in_hand_drift_m=max_center_drift_m,
                max_symmetry_axis_tilt_deg=max_axis_tilt_deg,
                family="cylinder"):
            target = canonical / mode.key_object / "reorient_12" / cell / seed.name
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"canonical reset seed already exists: {target}")
            source_hashes = {
                str(path.relative_to(seed)): _sha(path)
                for path in sorted(seed.rglob("*")) if path.is_file()}
            verified_rows.append((seed, target, source_hashes))
        else:
            rejected.append(seed.name)
    if not verified_rows:
        raise ValueError(f"no fidelity-eligible reset seed for {cell}")
    for seed, target, source_hashes in verified_rows:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(seed, target)
        if {str(path.relative_to(target)): _sha(path)
                for path in sorted(target.rglob("*")) if path.is_file()} != source_hashes:
            raise ValueError(f"copied reset seed differs from source: {target}")
        verify_v8_reset_seed(
            shared_root=root, mode=mode, height_cm=12,
            from_pose_stem=from_pose_stem, to_pose_stem=to_pose_stem,
            seed_id=seed.name)
    result = {
        "schema": "precision_insertion_cylinder_reset_install_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object},
        "cell": cell, "source_reset_root": str(source),
        "canonical_reset_root": str(canonical),
        "max_center_drift_m": max_center_drift_m,
        "max_axis_tilt_deg": max_axis_tilt_deg,
        "installed": [
            {"seed_id": seed.name, "canonical_path": str(target),
             "source_sha256": source_hashes}
            for seed, target, source_hashes in verified_rows],
        "fidelity_rejected_seed_ids": rejected,
        "scope": "v8_full_key_MuJoCo_and_fidelity_seed_not_Franka_path",
        "robot_ready": False,
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--source-reset-root", type=Path, required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--from-pose-stem", type=int, required=True)
    parser.add_argument("--to-pose-stem", type=int, required=True)
    parser.add_argument("--max-center-drift-mm", type=float, required=True)
    parser.add_argument("--max-axis-tilt-deg", type=float, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    result = install(
        shared_root=args.shared_root,
        source_reset_root=args.source_reset_root,
        gap_mm=args.gap_mm, from_pose_stem=args.from_pose_stem,
        to_pose_stem=args.to_pose_stem,
        max_center_drift_m=args.max_center_drift_mm / 1000.0,
        max_axis_tilt_deg=args.max_axis_tilt_deg,
        output_manifest=args.output_manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
