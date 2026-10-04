#!/usr/bin/env python3
"""Create a writable ``~/shared_data`` overlay over the read-only ParaDex2 NAS.

Existing NAS objects remain visible through per-entry symlinks.  New precision
insertion assets and experiment outputs stay local and writable.  The script is
idempotent and never replaces an existing non-symlink path.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _ensure_dir(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"expected a writable directory, found symlink: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _link(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"conflicting symlink: {destination} -> {os.readlink(destination)}")
        return
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source, target_is_directory=source.is_dir())
    except FileExistsError:
        # A previous invocation may still be completing a slow NFS directory
        # walk. Re-check instead of treating the harmless race as corruption.
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return
        raise


def _link_children(source: Path, destination: Path, *, exclude: set[str] | None = None) -> None:
    _ensure_dir(destination)
    excluded = exclude or set()
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        if child.name not in excluded:
            _link(child, destination / child.name)


def setup(shared_root: Path, nas_root: Path, paradex_repo: Path) -> None:
    shared_root = shared_root.expanduser().resolve()
    nas_root = nas_root.expanduser().resolve()
    paradex_repo = paradex_repo.expanduser().resolve()
    if not nas_root.is_dir():
        raise FileNotFoundError(f"ParaDex2 NAS root not found: {nas_root}")
    if not (paradex_repo / "paradex").is_dir():
        raise FileNotFoundError(f"ParaDex checkout not found: {paradex_repo}")

    _ensure_dir(shared_root)
    _link_children(
        nas_root,
        shared_root,
        exclude={"AutoDex", "object_processing", "shared_data", "paradex"},
    )

    # Object data: existing NAS objects are linked one-by-one so new objects
    # can coexist in the local directory.
    _link_children(nas_root / "object_processing", shared_root / "object_processing")

    nas_project = nas_root / "AutoDex"
    local_project = shared_root / "AutoDex"
    _link_children(
        nas_project,
        local_project,
        exclude={"foundpose_assets", "scene", "candidates", "experiment", "precision_insertion"},
    )

    _link_children(nas_project / "foundpose_assets", local_project / "foundpose_assets")

    # Scene overlay: hand/object are the useful merge boundaries.
    _ensure_dir(local_project / "scene")
    for hand_dir in sorted((nas_project / "scene").iterdir(), key=lambda path: path.name):
        if hand_dir.is_dir():
            _link_children(hand_dir, local_project / "scene" / hand_dir.name)
        else:
            _link(hand_dir, local_project / "scene" / hand_dir.name)

    # Candidate overlay: keep all other pools linked wholesale, but expose
    # inspire/v8 per object so precision-key candidates can be added locally.
    local_candidates = local_project / "candidates"
    _ensure_dir(local_candidates)
    for hand_dir in sorted((nas_project / "candidates").iterdir(), key=lambda path: path.name):
        if hand_dir.name != "inspire" or not hand_dir.is_dir():
            _link(hand_dir, local_candidates / hand_dir.name)
            continue
        local_inspire = local_candidates / "inspire"
        _ensure_dir(local_inspire)
        for pool in sorted(hand_dir.iterdir(), key=lambda path: path.name):
            if pool.name != "v8" or not pool.is_dir():
                _link(pool, local_inspire / pool.name)
            else:
                _link_children(pool, local_inspire / "v8")

    _ensure_dir(local_project / "experiment")
    _ensure_dir(local_project / "precision_insertion")

    # Official ParaDex checkout plus the lab's deployed machine profile.  The
    # profile is configuration data, not Python source.
    current = paradex_repo / "system" / "current"
    deployed = nas_root / "paradex" / "system" / "current"
    if not deployed.is_dir():
        raise FileNotFoundError(f"deployed ParaDex config not found: {deployed}")
    _link(deployed, current)

    print(f"shared overlay: {shared_root}")
    print(f"ParaDex code:    {paradex_repo}")
    print(f"ParaDex config:  {current} -> {deployed}")
    print("Camera contract: use the deployed AutoDex profile and verify it with "
          "scripts/precision_insertion/verify_autodex_camera_profile.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument("--nas-root", type=Path, default=Path("/mnt/paradex2"))
    parser.add_argument("--paradex-repo", type=Path, default=Path.home() / "paradex")
    args = parser.parse_args()
    setup(args.shared_root, args.nas_root, args.paradex_repo)


if __name__ == "__main__":
    main()
