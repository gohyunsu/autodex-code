#!/usr/bin/env python3
"""Copy grip-proxy BODex proposals into an isolated full-key sim-filter run.

The proxy supplies only object-frame grasp transforms. The original AutoDex
sim_filter subsequently reads the *full physical key* scene and URDF. This
script does not select, validate, or promote any grasp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


PROXY = "precision_key_cylinder_r15_h80_grip_proxy"
KEY = "precision_key_cylinder_r15_h80"
FILES = ("wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy", "bodex_info.npy")
SCENES = ("0", "1")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_scene_pair(shared_root: Path, scene_id: str) -> dict:
    root = shared_root / "AutoDex" / "scene" / "inspire"
    proxy_path = root / PROXY / "table" / f"{scene_id}.json"
    key_path = root / KEY / "table" / f"{scene_id}.json"
    proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))
    p_scene, k_scene = proxy["scene"], key["scene"]
    p_target, k_target = p_scene["mesh"]["target"], k_scene["mesh"]["target"]
    if not np.allclose(p_target["pose"], k_target["pose"], atol=1e-9):
        raise ValueError(f"proxy/full-key tabletop transforms differ: {scene_id}")
    if p_target["scale"] != k_target["scale"] or p_scene["cuboid"] != k_scene["cuboid"]:
        raise ValueError(f"proxy/full-key table scene differs: {scene_id}")
    if proxy["meta"]["pose_idx"] != key["meta"]["pose_idx"]:
        raise ValueError(f"proxy/full-key tabletop pose IDs differ: {scene_id}")
    if proxy["meta"].get("runtime_object") != KEY:
        raise ValueError(f"proxy does not declare runtime object {KEY}: {scene_id}")
    return {
        "scene_id": scene_id,
        "tabletop_pose_stem": key["meta"]["pose_idx"],
        "T_world_key_cart": k_target["pose"],
        "proxy_scene": str(proxy_path),
        "proxy_scene_sha256": sha256(proxy_path),
        "full_key_scene": str(key_path),
        "full_key_scene_sha256": sha256(key_path),
    }


def stage(raw_root: Path, stage_root: Path, shared_root: Path, expected: int) -> dict:
    if stage_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing run: {stage_root}")
    if expected < 1:
        raise ValueError("expected seed count must be positive")
    scenes = [check_scene_pair(shared_root, scene_id) for scene_id in SCENES]
    sources = []
    for scene_id in SCENES:
        source = raw_root / PROXY / "table" / scene_id
        actual = sorted(path.name for path in source.iterdir() if path.is_dir())
        wanted = sorted(str(i) for i in range(expected))
        if actual != wanted:
            raise ValueError(f"scene {scene_id}: expected exactly {expected} seed IDs, got {len(actual)}")
        for name in actual:
            seed = source / name
            missing = [file for file in FILES if not (seed / file).is_file()]
            if missing:
                raise FileNotFoundError(f"incomplete proposal {seed}: {missing}")
            sources.append((scene_id, name, seed))
    for scene_id, name, source in sources:
        dest = stage_root / KEY / "table" / scene_id / name
        dest.mkdir(parents=True)
        for file in FILES:
            shutil.copy2(source / file, dest / file)
    report = {
        "schema": "precision_insertion_proxy_to_full_key_stage_v1",
        "status": "raw_proposals_only_not_filtered",
        "proxy_object": PROXY,
        "full_key_object": KEY,
        "raw_root": str(raw_root),
        "stage_root": str(stage_root),
        "seed_count_per_scene": expected,
        "seed_count_total": len(sources),
        "scenes": scenes,
        "warning": "No grasp is eligible until original AutoDex collision/contact/MuJoCo filters and exact 20 mm socket screen pass.",
    }
    (stage_root / "stage_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--expected-per-scene", type=int, default=1000)
    args = parser.parse_args()
    report = stage(args.raw_root.resolve(), args.stage_root.resolve(),
                   args.shared_root.resolve(), args.expected_per_scene)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
