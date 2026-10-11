#!/usr/bin/env python3
"""Install a verified full-length cylinder candidate addendum, append-only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from export_full_length_cylinder_handoff import (
    KEY, _scene_semantics, verify,
)
from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rehydrate(*, bundle_root: Path, target_shared_root: Path,
              install: bool = False) -> dict:
    bundle = Path(bundle_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    manifest = verify(bundle)
    if not target.is_dir():
        raise FileNotFoundError(target)
    paths = AssetPaths(target, select_mode("cylinder", 20))
    for path, expected in (
            (paths.raw_mesh(KEY), manifest["physical_key_raw_mesh_sha256"]),
            (paths.key_planning_mesh,
             manifest["physical_key_planning_mesh_sha256"])):
        if not path.is_file() or _sha(path) != expected:
            raise ValueError(f"recipient physical key mesh differs: {path}")
    for scene_id, semantics in manifest["scene_semantics"].items():
        scene = target / "AutoDex/scene/inspire" / KEY / "table" / f"{scene_id}.json"
        if (not scene.is_file() or _scene_semantics(scene) != semantics or
                Path(json.loads(scene.read_text(encoding="utf-8"))["scene"]
                     ["mesh"]["target"]["file_path"]).resolve() !=
                paths.key_planning_mesh.resolve()):
            raise ValueError(f"recipient key tabletop scene differs: {scene}")
    for stem, expected in manifest["tabletop_pose_sha256"].items():
        pose = paths.key_tabletop_dir / f"{stem}.npy"
        if not pose.is_file() or _sha(pose) != expected:
            raise ValueError(f"recipient tabletop pose differs: {pose}")
    payload = (bundle / "payload/shared_data/AutoDex/candidates/inspire/v8" /
               KEY)
    declared = set(manifest["candidate_ids"])
    observed = {str(path.relative_to(payload)) for path in payload.glob(
        "table/*/*") if path.is_dir()}
    if (not declared or declared != observed or
            any(Path(identifier).parts[:2] not in
                (("table", "0"), ("table", "1")) or
                len(Path(identifier).parts) != 3
                for identifier in declared)):
        raise ValueError("candidate payload differs from manifest")
    planned = []
    identical = 0
    for identifier in sorted(declared):
        source = payload / identifier
        destination = paths.candidate_dir / identifier
        source_files = {p.name: _sha(p) for p in source.iterdir() if p.is_file()}
        if not source_files or any(p.is_dir() or p.is_symlink()
                                   for p in source.iterdir()):
            raise ValueError(f"unexpected candidate payload structure: {source}")
        if destination.exists() or destination.is_symlink():
            if (not destination.is_dir() or destination.is_symlink() or
                    {p.name: _sha(p) for p in destination.iterdir() if p.is_file()}
                    != source_files or
                    any(p.is_dir() or p.is_symlink()
                        for p in destination.iterdir())):
                raise ValueError(f"recipient candidate conflicts: {destination}")
            identical += 1
        else:
            if destination.with_name(destination.name + ".incomplete").exists():
                raise FileExistsError(f"incomplete install exists: {destination}")
            planned.append((source, destination))
    result = {
        "schema": "precision_insertion_full_length_cylinder_rehydration_v1",
        "handoff_manifest_sha256": _sha(bundle / "MANIFEST.json"),
        "recipient_shared_root": str(target),
        "candidate_count": len(declared),
        "identical_existing_candidates": identical,
        "candidates_to_install": len(planned),
        "installed": bool(install),
        "next_step": "rebuild all six complete endpoint catalogs on recipient",
        "robot_ready": False,
    }
    if install:
        for source, destination in planned:
            working = destination.with_name(destination.name + ".incomplete")
            working.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, working, symlinks=False)
            source_files = {p.name: _sha(p) for p in source.iterdir()}
            copied_files = {p.name: _sha(p) for p in working.iterdir()}
            if copied_files != source_files:
                raise ValueError(f"candidate changed during copy: {source}")
            working.rename(destination)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rehydrate(
        bundle_root=args.bundle_root,
        target_shared_root=args.target_shared_root,
        install=args.install), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
