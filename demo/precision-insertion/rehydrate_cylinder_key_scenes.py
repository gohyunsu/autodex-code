#!/usr/bin/env python3
"""Rebind the two source-bound cylinder key tabletop scenes in a staging root.

The default is a read-only preflight. Installation accepts only an absent
recipient scene or an exact copy of the verified source scene, archives the
original bytes, and never changes a different recipient scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile


KEY = "precision_key_cylinder_r15_h80"
SCENE_REL = Path("AutoDex/scene/inspire") / KEY / "table"
KEY_REL = Path("object_processing") / KEY / "processed_data"
REPORT_REL = Path("AutoDex/precision_insertion/cylindrical/key_scene_relocation")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)
            + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rehydrate(*, source_shared_root: Path, source_origin_shared_root: Path,
              target_shared_root: Path, install: bool = False) -> dict:
    source = Path(source_shared_root).expanduser().resolve()
    origin = Path(source_origin_shared_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    if source == target or origin == target or not source.is_dir() or not target.is_dir():
        raise ValueError("source, historical origin and staging target must differ")
    report_dir = target / REPORT_REL
    if report_dir.exists() or report_dir.is_symlink():
        raise FileExistsError(report_dir)
    planned = []
    records = []
    for stem in (0, 1):
        relative = SCENE_REL / f"{stem}.json"
        source_file, destination = source / relative, target / relative
        if (source_file.is_symlink() or destination.is_symlink() or
                not source_file.is_file()):
            raise ValueError(f"unsafe or absent cylinder key scene: {relative}")
        original = source_file.read_bytes()
        scene = json.loads(original)
        if (not isinstance(scene, dict) or
                set(scene.get("scene", {}).get("mesh", {})) != {"target"} or
                not isinstance(scene.get("meta"), dict) or
                scene["meta"].get("pose_idx") != f"{stem:03d}"):
            raise ValueError(f"unexpected cylinder key scene structure: {relative}")
        mesh = scene["scene"]["mesh"]["target"]
        replacements = {
            "file_path": KEY_REL / "mesh/simplified.obj",
            "urdf_path": KEY_REL / "urdf/coacd.urdf",
        }
        for field, subpath in replacements.items():
            if mesh.get(field) != str(origin / subpath):
                raise ValueError(f"unexpected historical {field}: {relative}")
            if not (target / subpath).is_file():
                raise FileNotFoundError(target / subpath)
            mesh[field] = str(target / subpath)
        rebound = _json_bytes(scene)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() not in {
                    original, rebound}:
                raise ValueError(f"recipient scene differs from verified source: {destination}")
            needs_write = destination.read_bytes() == original
        else:
            needs_write = True
        if needs_write:
            planned.append((destination, original, rebound))
        records.append({"scene": str(relative), "source_sha256": _sha(original),
                        "recipient_sha256": _sha(rebound),
                        "needs_write": needs_write})
    report = {
        "schema": "precision_insertion_cylinder_key_scene_relocation_v1",
        "source_shared_root": str(source),
        "source_origin_shared_root": str(origin),
        "target_shared_root": str(target),
        "scenes": records, "files_to_install": len(planned),
        "installed": install, "robot_ready": False,
    }
    if not install:
        return report
    report_dir.mkdir(parents=True, exist_ok=False)
    for destination, original, rebound in planned:
        archived = report_dir / "original_scenes" / destination.name
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(original)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
            temp.write(rebound)
            temp_path = Path(temp.name)
        os.replace(temp_path, destination)
    for row in records:
        if _sha((target / row["scene"]).read_bytes()) != row["recipient_sha256"]:
            raise ValueError(f"recipient scene changed during installation: {row['scene']}")
    (report_dir / "RELOCATION.json").write_bytes(_json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shared-root", type=Path, required=True)
    parser.add_argument("--source-origin-shared-root", type=Path, required=True)
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rehydrate(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
