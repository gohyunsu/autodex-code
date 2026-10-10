#!/usr/bin/env python3
"""Stage prior cylinder grip-proxy reset seeds for full-key AutoDex filtering.

The proxy and full key share an object frame but have different shape-derived
virtual reset pillars. This only copies raw proposal transforms into a new
isolated full-key tree. Stock cuRobo scene collision, squeeze contact, MuJoCo
gravity stability, fixture-aware Franka planning and physical validation are
all separate subsequent gates. Never import this stage into runtime reset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from precision_insertion.geometry import validate_se3


PROXY = "precision_key_cylinder_r15_h80_grip_proxy"
KEY = "precision_key_cylinder_r15_h80"
FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy", "bodex_info.npy")
PAIRS = ("0_1", "1_0")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_scene_pair(shared_root: Path, cell: str) -> dict:
    root = shared_root / "AutoDex" / "scene" / "inspire"
    proxy_path = root / PROXY / "reorient_12" / f"{cell}.json"
    key_path = root / KEY / "reorient_12" / f"{cell}.json"
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    pi, pj = cell.split("_")
    for scene in (proxy, key):
        meta = scene["meta"]
        if (meta.get("pose_i") != f"{int(pi):03d}" or
                meta.get("pose_j") != f"{int(pj):03d}" or
                meta.get("scene_type") != "reorient_12" or
                meta.get("h") != 0.12 or meta.get("version") != "v8"):
            raise ValueError(f"reorient scene {cell} does not match v8 cell")
    p_target = proxy["scene"]["mesh"]["target"]
    k_target = key["scene"]["mesh"]["target"]
    proxy_mesh = (shared_root / "object_processing" / PROXY /
                  "processed_data/mesh/simplified.obj").resolve()
    key_mesh = (shared_root / "object_processing" / KEY /
                "processed_data/mesh/simplified.obj").resolve()
    if (not np.allclose(p_target["pose"], k_target["pose"], atol=1e-9) or
            p_target["scale"] != k_target["scale"] or
            Path(p_target["file_path"]).resolve() != proxy_mesh or
            Path(k_target["file_path"]).resolve() != key_mesh or
            not proxy_mesh.is_file() or not key_mesh.is_file()):
        raise ValueError(f"proxy/full-key reset object frames differ: {cell}")
    return {
        "cell": cell,
        "proxy_scene": str(proxy_path), "proxy_sha256": _sha256(proxy_path),
        "full_key_scene": str(key_path), "full_key_sha256": _sha256(key_path),
        "obstacle_geometries_identical": (
            proxy["scene"]["cuboid"] == key["scene"]["cuboid"]),
    }


def stage(
    *, raw_root: Path, stage_root: Path, shared_root: Path,
    expected_per_cell: int,
) -> dict:
    """Create a raw full-key candidate *input*, not filtered candidates."""
    raw = Path(raw_root).expanduser().resolve()
    target = Path(stage_root).expanduser().resolve()
    shared = Path(shared_root).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite staged reset run: {target}")
    if expected_per_cell < 1:
        raise ValueError("expected seeds per cell must be positive")
    scenes = [_check_scene_pair(shared, cell) for cell in PAIRS]
    sources = []
    for cell in PAIRS:
        source = raw / PROXY / "reorient_12" / cell
        if not source.is_dir():
            raise FileNotFoundError(f"raw proxy reset cell missing: {source}")
        seeds = sorted((item for item in source.iterdir() if item.is_dir()),
                       key=lambda item: int(item.name) if item.name.isdigit() else -1)
        if {item.name for item in seeds} != {
                str(i) for i in range(expected_per_cell)}:
            raise ValueError(f"reset cell {cell} is not exactly {expected_per_cell} seeds")
        for seed in seeds:
            missing = [name for name in FILES if not (seed / name).is_file()]
            if missing:
                raise FileNotFoundError(f"incomplete reset proposal {seed}: {missing}")
            validate_se3(np.load(seed / "wrist_se3.npy", allow_pickle=False),
                         name=f"reset proposal {cell}/{seed.name} T_key_hand")
            for name in ("pregrasp_pose.npy", "grasp_pose.npy"):
                hand_q = np.load(seed / name, allow_pickle=False)
                if hand_q.shape != (6,) or not np.all(np.isfinite(hand_q)):
                    raise ValueError(f"invalid Inspire joints in {seed / name}")
            sources.append((cell, seed))
    for cell, source in sources:
        destination = target / KEY / "reorient_12" / cell / source.name
        destination.mkdir(parents=True)
        for name in FILES:
            shutil.copy2(source / name, destination / name)
    report = {
        "schema": "precision_insertion_cylinder_reset_proxy_stage_v1",
        "status": "raw_full_key_filter_input_only",
        "proxy_object": PROXY, "full_key_object": KEY,
        "raw_root": str(raw), "stage_root": str(target),
        "expected_per_cell": expected_per_cell,
        "total_raw_proposals": len(sources), "scenes": scenes,
        "warning": (
            "Proxy pillars may differ from full-key pillars. No seed is usable "
            "until the original full-key scene/contact/MuJoCo filter and "
            "socket-aware reset path preflight pass."),
        "robot_ready": False,
    }
    with (target / "stage_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--expected-per-cell", type=int, required=True)
    args = parser.parse_args()
    try:
        report = stage(
            raw_root=args.raw_root, stage_root=args.stage_root,
            shared_root=args.shared_root,
            expected_per_cell=args.expected_per_cell)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError) as exc:
        parser.exit(2, f"reset staging rejected: {exc}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
