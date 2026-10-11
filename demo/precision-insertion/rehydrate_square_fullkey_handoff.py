#!/usr/bin/env python3
"""Relocate a verified four-gap square v8 handoff without overwriting files.

Only v8 scene and fixture absolute paths, plus their candidate scene hashes,
are rebound. Original BODex/MuJoCo evidence remains unchanged. The recipient
must rebuild current-pool endpoint catalogs against its own path and code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from export_square_fullkey_handoff import GAPS, SOCKET, verify
from export_square_tabletop_handoff import _sha
from precision_insertion.config import select_mode
from precision_insertion.endpoint import validate_task_geometry
from rehydrate_square_tabletop_handoff import (
    FIXTURE_PARENT, ROBOT_URDF, _check_target_path,
    _create_complete_file, _has_unbound_origin, _json_bytes,
    _read_object, _rebind, _relocate_json,
)


def _sha_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _scene_relative(key: str, index: int) -> Path:
    return Path("AutoDex/scene/inspire") / key / "table" / f"{index}.json"


def _relocate_scene(relative: Path, content: bytes, *, key: str,
                    origin: Path, target: Path) -> bytes:
    if origin == target:
        return content
    value = _read_object(content, str(relative))
    key_root = Path("object_processing") / key
    for field, subpath in (
        ("file_path", key_root / "processed_data/mesh/simplified.obj"),
        ("urdf_path", key_root / "processed_data/urdf/coacd.urdf"),
    ):
        _rebind(value, ("scene", "mesh", "target", field), subpath,
                origin=origin, target=target)
    if _has_unbound_origin(value, origin=origin, target=target):
        raise ValueError(f"unbound source path in scene {relative}")
    return _json_bytes(value)


def rehydrate(*, bundle_root: Path, target_shared_root: Path,
              install: bool = False) -> dict:
    bundle = Path(bundle_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    manifest = verify(bundle)
    origin_name = manifest.get("origin_shared_root")
    if not isinstance(origin_name, str) or not Path(origin_name).is_absolute():
        raise ValueError("handoff needs an absolute source shared-data root")
    origin = Path(origin_name)
    if not target.is_dir():
        raise FileNotFoundError(f"recipient shared-data root is missing: {target}")
    urdf = target / ROBOT_URDF
    if (not urdf.is_file() or
            _sha(urdf) != manifest.get("source_robot_urdf_sha256")):
        raise ValueError("recipient Franka/Inspire URDF differs from source")
    entries = manifest.get("modes")
    if not isinstance(entries, list) or len(entries) != len(GAPS):
        raise ValueError("handoff needs exactly four square modes")
    keys = {}
    candidate_ids = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("gap_mm") not in GAPS:
            raise ValueError("invalid square mode entry")
        gap = entry["gap_mm"]
        key = select_mode("square", gap).key_object
        ids = entry.get("candidate_ids")
        if (gap in keys or entry.get("key_object") != key or
                not isinstance(ids, list) or not ids or
                len(ids) != len(set(ids))):
            raise ValueError("duplicate or incomplete square mode entry")
        for identifier in ids:
            if (not isinstance(identifier, str) or
                    len(identifier.split("/")) != 3 or
                    identifier.split("/")[0] != "table" or
                    identifier.split("/")[1] not in {str(i) for i in range(5)} or
                    not identifier.split("/")[2].isdigit()):
                raise ValueError(f"invalid v8 grasp ID: {identifier}")
        keys[gap] = key
        candidate_ids[key] = set(ids)
    payload = bundle / "payload/shared_data"
    if not payload.is_dir():
        raise FileNotFoundError(payload)
    files = sorted(path for path in payload.rglob("*") if path.is_file())
    relatives = {path.relative_to(payload) for path in files}
    scene_relatives = {_scene_relative(key, index): (key, index)
                       for key in keys.values() for index in range(5)}
    if not set(scene_relatives) <= relatives:
        raise ValueError("handoff lacks one or more v8 square tabletop scenes")
    for key, ids in candidate_ids.items():
        candidate_parent = Path("AutoDex/candidates/inspire/v8") / key
        found = {"/".join(path.parts[5:8]) for path in relatives
                 if path.parts[:5] == candidate_parent.parts and
                 len(path.parts) >= 9 and path.parts[5] == "table" and
                 path.parts[6] in {str(i) for i in range(5)} and
                 path.parts[7].isdigit()}
        if found != ids:
            raise ValueError(f"candidate tree differs from manifest: {key}")
    allowed_objects = set(keys.values()) | {SOCKET}
    for relative in relatives:
        parts = relative.parts
        if parts[:1] == ("object_processing",) and len(parts) >= 3:
            allowed = parts[1] in allowed_objects and "scene" not in parts[2:]
        elif parts[:3] == ("AutoDex", "scene", "inspire"):
            allowed = relative in scene_relatives
        elif parts[:4] == ("AutoDex", "candidates", "inspire", "v8") and len(parts) >= 6:
            allowed = parts[4] in candidate_ids
        else:
            allowed = (parts[:len(FIXTURE_PARENT.parts)] ==
                       FIXTURE_PARENT.parts)
        if not allowed:
            raise ValueError(f"unexpected handoff payload path: {relative}")
    source_geometry = _read_object(
        (payload / FIXTURE_PARENT / "task_geometry.json").read_bytes(),
        "source task geometry")
    validate_task_geometry(source_geometry, select_mode("square", 1.5))
    source_scene = {relative: (payload / relative).read_bytes()
                    for relative in scene_relatives}
    relocated_scene = {
        relative: _relocate_scene(
            relative, source_scene[relative], key=key,
            origin=origin, target=target)
        for relative, (key, _) in scene_relatives.items()}
    manifest_hash = _sha(bundle / "MANIFEST.json")
    planned = []
    same = 0
    for source in files:
        relative = source.relative_to(payload)
        destination = _check_target_path(target, relative)
        content = source.read_bytes()
        if relative in relocated_scene:
            content = relocated_scene[relative]
        elif relative in {
                FIXTURE_PARENT / "task_geometry.json",
                FIXTURE_PARENT / "fixture_pose.template.json",
                FIXTURE_PARENT / "pose_measurement_asset.json"}:
            content = _relocate_json(relative, content,
                                     origin=origin, target=target)
        elif (origin != target and
              relative.name == "simulation_validation.json" and
              relative.parts[:4] == ("AutoDex", "candidates", "inspire", "v8")):
            key = relative.parts[4]
            scene_index = int(relative.parts[6])
            scene_path = _scene_relative(key, scene_index)
            record = _read_object(content, str(relative))
            old_hash = record.get("source_scene_sha256")
            if old_hash is not None:
                if old_hash != _sha_bytes(source_scene[scene_path]) or "relocation" in record:
                    raise ValueError(f"candidate scene evidence differs: {relative}")
                record["source_scene_sha256"] = _sha_bytes(
                    relocated_scene[scene_path])
                record["relocation"] = {
                    "schema": "precision_insertion_square_scene_relocation_v1",
                    "source_scene_sha256": old_hash,
                    "recipient_scene_sha256": record["source_scene_sha256"],
                    "handoff_manifest_sha256": manifest_hash,
                    "recipient_shared_root": str(target),
                    "scope": "absolute_scene_paths_only_not_new_simulation",
                }
                content = _json_bytes(record)
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != content:
                raise ValueError(f"recipient file conflicts with handoff: {destination}")
            same += 1
        else:
            planned.append((destination, content))
    result = {
        "schema": "precision_insertion_square_fullkey1000_relocation_v1",
        "origin_shared_root": str(origin),
        "recipient_shared_root": str(target),
        "handoff_manifest_sha256": manifest_hash,
        "payload_files": len(files), "identical_existing_files": same,
        "files_to_install": len(planned), "installed": bool(install),
        "robot_ready": False,
        "next_step": "rebuild complete endpoint catalog for each square gap",
    }
    if install:
        for path, content in planned:
            _create_complete_file(path, content)
        validate_task_geometry(
            _read_object((target / FIXTURE_PARENT / "task_geometry.json").read_bytes(),
                         "recipient task geometry"), select_mode("square", 1.5))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rehydrate(bundle_root=args.bundle_root,
                               target_shared_root=args.target_shared_root,
                               install=args.install), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
