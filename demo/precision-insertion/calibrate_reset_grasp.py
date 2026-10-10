#!/usr/bin/env python3
"""Build a reset-seed relation summary from independent physical pickups.

Read-only with respect to AutoDex assets and robot state. The output is a
commissioning *description*, never a robot-ready reset candidate or a
future-trial worst-case error bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.physical_grasp_calibration import (  # noqa: E402
    calibrate_physical_held_relation, verify_physical_held_relation,
)
from precision_insertion.reset_candidates import (  # noqa: E402
    verify_v8_reset_seed,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_reset_grasp_calibration(
    *, shared_root: Path, family: str, gap_mm: float,
    candidate_root: Path, height_cm: int, from_pose_stem: str,
    to_pose_stem: str, seed_id: str, sample_paths: list[Path],
    minimum_independent_trials: int,
    max_nominal_translation_drift_m: float,
    max_nominal_rotation_drift_deg: float, output_dir: Path,
) -> Path:
    """Verify the full-key seed and source records before exclusive output."""
    mode = select_mode(family, gap_mm)
    root = Path(shared_root).expanduser().resolve()
    selected = verify_v8_reset_seed(
        shared_root=root, mode=mode, height_cm=height_cm,
        from_pose_stem=from_pose_stem, to_pose_stem=to_pose_stem,
        seed_id=seed_id, candidate_root=candidate_root)
    sources = []
    seen: set[Path] = set()
    for raw in sample_paths:
        if not Path(raw).expanduser().is_absolute():
            raise ValueError("physical sample paths must be absolute")
        path = Path(raw).expanduser().resolve()
        if path in seen:
            raise ValueError("one physical pickup sample cannot be reused")
        seen.add(path)
        sample = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(sample, dict):
            raise ValueError("physical pickup sample must be a JSON object")
        sources.append({**sample, "evidence_path": str(path),
                        "evidence_sha256": _sha(path)})
    summary = calibrate_physical_held_relation(
        mode=mode, shared_root=root,
        candidate_key=selected["candidate_key"],
        candidate_T_key_hand=selected["T_key_hand"], samples=sources,
        minimum_independent_trials=minimum_independent_trials,
        max_nominal_translation_drift_m=(
            max_nominal_translation_drift_m),
        max_nominal_rotation_drift_deg=max_nominal_rotation_drift_deg)
    verify_physical_held_relation(
        record=summary, mode=mode, shared_root=root,
        candidate_key=selected["candidate_key"],
        candidate_dir=selected["candidate_dir"])
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"calibration output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    summary_path = target / "physical_grasp_calibration.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    verify_physical_held_relation(
        record=json.loads(summary_path.read_text(encoding="utf-8")),
        mode=mode, shared_root=root,
        candidate_key=selected["candidate_key"],
        candidate_dir=selected["candidate_dir"])
    binding = {
        "schema": "precision_insertion_reset_grasp_calibration_binding_v1",
        "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                 "key_object": mode.key_object,
                 "socket_object": mode.socket_object},
        "candidate_key": list(selected["candidate_key"]),
        "candidate_dir": str(selected["candidate_dir"]),
        "source_evidence_sha256": selected["source_evidence_sha256"],
        "physical_grasp_calibration": summary_path.name,
        "physical_grasp_calibration_sha256": _sha(summary_path),
        "sample_count": summary["sample_count"],
        "scope": "physical_sample_summary_not_future_trial_bound_or_reset_plan",
        "robot_ready": False,
    }
    with (target / "binding.json").open("x", encoding="utf-8") as stream:
        json.dump(binding, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--family", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True,
                        help="exact reset_<height> directory; staged is allowed")
    parser.add_argument("--height-cm", type=int, choices=(4, 8, 12), required=True)
    parser.add_argument("--from-pose", required=True)
    parser.add_argument("--to-pose", required=True)
    parser.add_argument("--seed-id", required=True)
    parser.add_argument("--sample", type=Path, action="append", required=True,
                        help="repeat for independent physical pickup JSON files")
    parser.add_argument("--minimum-independent-trials", type=int, default=5)
    parser.add_argument("--max-nominal-translation-drift-mm", type=float,
                        required=True, help="SE(3) medoid selection scale")
    parser.add_argument("--max-nominal-rotation-drift-deg", type=float,
                        required=True, help="SE(3) medoid selection scale")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new directory; existing output is never overwritten")
    args = parser.parse_args(argv)
    result = build_reset_grasp_calibration(
        shared_root=args.shared_root, family=args.family, gap_mm=args.gap_mm,
        candidate_root=args.candidate_root, height_cm=args.height_cm,
        from_pose_stem=args.from_pose, to_pose_stem=args.to_pose,
        seed_id=args.seed_id, sample_paths=args.sample,
        minimum_independent_trials=args.minimum_independent_trials,
        max_nominal_translation_drift_m=(
            args.max_nominal_translation_drift_mm / 1000.),
        max_nominal_rotation_drift_deg=(
            args.max_nominal_rotation_drift_deg),
        output_dir=args.output_dir)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
