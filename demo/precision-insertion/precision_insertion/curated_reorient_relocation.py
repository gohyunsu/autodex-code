"""Relocate a curated v8 reset handoff without laundering source evidence.

The source key and scene files must remain available (for example in an
extracted NAS handoff). This is a host-path rebind, not reset planning or a
robot-motion permission. Existing target scenes are never silently replaced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .config import TaskMode
from .reorient_assets import _scene_path, _sim_filter_scene_path, _validate_scene
from .reset_candidates import (
    _candidate_arrays, _same_numeric_record, _valid_scene_pair,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_scene(scene: dict, *, object_dir: Path, key_object: str) -> dict:
    """Compare collision geometry while normalizing only two host paths."""
    normalized = json.loads(json.dumps(scene, allow_nan=False))
    target = normalized["scene"]["mesh"]["target"]
    paths = {
        "file_path": object_dir / "processed_data/mesh/simplified.obj",
        "urdf_path": object_dir / "processed_data/urdf/coacd.urdf",
    }
    for field, expected in paths.items():
        if Path(target[field]) != expected:
            raise ValueError(f"source reset scene has unexpected {field}")
        target[field] = f"<KEY_{field.upper()}>"
    for field in ("geometry_object", "grasp_target_object"):
        if normalized["meta"].pop(field, key_object) != key_object:
            raise ValueError(f"source reset scene has wrong {field}")
    return normalized


def _matching_key_files(source: Path, target: Path) -> dict[str, str]:
    """Require recipient v8 key bytes, excluding host-bound scene JSONs."""
    hashes = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"source key contains symlink: {path}")
        if not path.is_file() or "scene" in path.relative_to(source).parts:
            continue
        relative = path.relative_to(source)
        received = target / relative
        if not received.is_file() or _sha(received) != _sha(path):
            raise ValueError(f"recipient key asset is missing or changed: {relative}")
        hashes[str(relative)] = _sha(path)
    if not hashes:
        raise ValueError("no source key files were checked")
    return hashes


def _verify_source_snapshot(source: Path, *, manifest_sha: str,
                            origin: Path, mode: TaskMode) -> None:
    """When a portable source snapshot is supplied, verify its file ledger."""
    path = source / "SNAPSHOT.json"
    if not path.exists():
        return
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if (snapshot.get("schema") !=
            "precision_insertion_curated_reorient_source_snapshot_v1" or
            snapshot.get("source_handoff_manifest_sha256") != manifest_sha or
            snapshot.get("source_origin_shared_root") != str(origin) or
            snapshot.get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm,
                "key_object": mode.key_object} or
            not isinstance(snapshot.get("files"), list) or
            not snapshot["files"]):
        raise ValueError("portable source snapshot identity differs")
    seen = set()
    for row in snapshot["files"]:
        relative = Path(row["relative_path"])
        if (relative.is_absolute() or ".." in relative.parts or
                relative in seen):
            raise ValueError("portable source snapshot file list is invalid")
        seen.add(relative)
        file = source / relative
        if file.is_symlink() or not file.is_file() or _sha(file) != row["sha256"]:
            raise ValueError(f"portable source snapshot file changed: {relative}")


def rehydrate_curated_reorient(
    *, source_shared_root: Path, source_handoff_root: Path,
    target_shared_root: Path, target_handoff_root: Path,
    mode: TaskMode, archive_source_bound_scenes: bool = False,
) -> dict:
    """Rebind selected cells to target paths, preserving all original bytes.

    The handoff manifest names the historical source root. Source files may
    have been extracted elsewhere, but scene hashes, geometry, raw MuJoCo
    trajectory, key mesh and every seed must verify before any target write.
    Output is an explicit noncanonical staging root containing ``reset_12``.
    """
    source = Path(source_shared_root).expanduser().resolve()
    incoming = Path(source_handoff_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    output_input = Path(target_handoff_root).expanduser()
    if not output_input.is_absolute():
        raise ValueError("target handoff root must be absolute")
    output = output_input.resolve()
    canonical = target / "AutoDex/candidates"
    if (not source.is_dir() or not incoming.is_dir() or not target.is_dir() or
            output.exists() or not output.is_relative_to(target) or
            output.is_relative_to(incoming) or
            output.is_relative_to(canonical.resolve()) or
            output_input.absolute().is_relative_to(canonical)):
        raise ValueError("invalid non-overwriting staged reorient relocation")
    manifest_path = incoming / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = manifest.get("mode")
    if (manifest.get("schema") != "precision_insertion_v8_reorient_handoff_v1" or
            not isinstance(identity, dict) or
            identity != {"family": mode.family, "gap_mm": mode.gap_mm,
                         "key_object": mode.key_object} or
            not isinstance(manifest.get("cells"), list) or
            not isinstance(manifest.get("candidates"), list)):
        raise ValueError("curated handoff identity or manifest is invalid")
    origin = Path(manifest["shared_root"])
    if not origin.is_absolute() or source == target:
        raise ValueError("source origin or recipient is invalid")
    _verify_source_snapshot(
        source, manifest_sha=_sha(manifest_path), origin=origin, mode=mode)
    source_obj = source / "object_processing" / mode.key_object
    target_obj = target / "object_processing" / mode.key_object
    origin_obj = origin / "object_processing" / mode.key_object
    key_hashes = _matching_key_files(source_obj, target_obj)
    from src.grasp_generation.reorient.gen_scene import gen_reorient_scene

    cells = []
    seeds = []
    scene_writes = []
    scene_archives = []
    listed = {}
    for candidate in manifest["candidates"]:
        raw = Path(candidate["candidate_dir"])
        if not raw.is_relative_to(Path(manifest["output_root"])):
            raise ValueError("handoff candidate escapes recorded source root")
        relative = raw.relative_to(Path(manifest["output_root"]))
        if relative in listed:
            raise ValueError("duplicate handoff candidate")
        listed[relative] = candidate["source_evidence_sha256"]
    for row in manifest["cells"]:
        h, i, j = (row["height_cm"], row["from_v8_pose"],
                   row["to_v8_pose"])
        if (type(h) is not int or h != 12 or
                any(type(v) is not int or v < 0 for v in (i, j)) or i == j or
                not isinstance(row.get("verified_seed_ids"), list)):
            raise ValueError("invalid curated directed reset cell")
        cell = f"{i}_{j}"
        source_scenes = (
            _scene_path(source_obj, h, i, j),
            _sim_filter_scene_path(source, mode, h, i, j),
        )
        source_json = [json.loads(path.read_text(encoding="utf-8"))
                       for path in source_scenes]
        normalized = [
            _normalized_scene(scene, object_dir=origin_obj,
                              key_object=mode.key_object)
            for scene in source_json
        ]
        if not _same_numeric_record(normalized[0], normalized[1]):
            raise ValueError(f"source reset scenes disagree: {cell}")
        generated = gen_reorient_scene(
            mode.key_object, i, j, h / 100,
            obj_root=str(target / "object_processing"))
        generated["meta"]["scene_type"] = f"reorient_{h}"
        _validate_scene(generated, mode=mode, object_dir=target_obj,
                        h_cm=h, i=i, j=j)
        if not _same_numeric_record(normalized[0], _normalized_scene(
                generated, object_dir=target_obj,
                key_object=mode.key_object)):
            raise ValueError(f"recipient reset geometry differs: {cell}")
        destinations = (
            _scene_path(target_obj, h, i, j),
            _sim_filter_scene_path(target, mode, h, i, j),
        )
        for source_path, destination in zip(source_scenes, destinations):
            if destination.is_symlink():
                raise ValueError(f"recipient reset scene is a symlink: {destination}")
            if not destination.exists():
                scene_writes.append((destination, generated))
                continue
            existing = json.loads(destination.read_text(encoding="utf-8"))
            try:
                _validate_scene(existing, mode=mode, object_dir=target_obj,
                                h_cm=h, i=i, j=j)
                same_target_geometry = _same_numeric_record(_normalized_scene(
                    existing, object_dir=target_obj,
                    key_object=mode.key_object), normalized[0])
            except (ValueError, KeyError, TypeError):
                same_target_geometry = False
            if same_target_geometry:
                continue
            if archive_source_bound_scenes and _sha(destination) == _sha(source_path):
                archive = (output / "source_bound_scenes_archive" /
                           destination.relative_to(target))
                scene_archives.append((destination, archive))
                scene_writes.append((destination, generated))
                continue
            raise ValueError(f"recipient reset scene differs: {destination}")
        folder = (incoming / f"reset_{h}" / mode.key_object /
                  f"reorient_{h}" / cell)
        expected_ids = {str(seed_id) for seed_id in row["verified_seed_ids"]}
        actual_ids = {path.name for path in folder.iterdir()
                      if path.is_dir() and path.name.isdigit()}
        if not expected_ids or expected_ids != actual_ids:
            raise ValueError(f"curated reset seed set differs: {cell}")
        for seed_id in sorted(expected_ids, key=int):
            seed = folder / seed_id
            if seed.is_symlink() or any(p.is_symlink() for p in seed.rglob("*")):
                raise ValueError(f"curated reset seed contains symlink: {seed}")
            relative = seed.relative_to(incoming)
            if listed.get(relative) != _sha(seed / "source_evidence.json"):
                raise ValueError(f"curated source evidence differs: {seed}")
            _candidate_arrays(seed, mode=mode, cell=cell, h_cm=h,
                              scenes=source_scenes)
            seeds.append((seed, relative, source_scenes, destinations))
        cells.append({"height_cm": h, "from_v8_pose": i,
                      "to_v8_pose": j, "seed_ids": sorted(expected_ids, key=int)})
    if (len(seeds) != manifest.get("candidate_count") or
            len(seeds) != len(listed) or len(cells) != len({
                (r["height_cm"], r["from_v8_pose"], r["to_v8_pose"])
                for r in cells})):
        raise ValueError("curated handoff has duplicate or unlisted seeds")
    for original, archive in scene_archives:
        if archive.exists():
            raise FileExistsError(f"reset scene archive exists: {archive}")

    # Every source check precedes writes. A later filesystem error leaves
    # explicit staged evidence for review; it never edits canonical seeds.
    output.mkdir(parents=True, exist_ok=False)
    for original, archive in scene_archives:
        archive.parent.mkdir(parents=True, exist_ok=True)
        original.rename(archive)
    for destination, scene in scene_writes:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(scene, stream, indent=2, allow_nan=False)
            stream.write("\n")
    rows = []
    for seed, relative, source_scenes, destinations in seeds:
        dst = output / relative
        shutil.copytree(seed, dst, symlinks=False)
        original = dst / "source_evidence.json"
        original.rename(dst / "source_evidence_original.json")
        evidence = json.loads((dst / "source_evidence_original.json").read_text(
            encoding="utf-8"))
        evidence["scene_sha256"] = {
            "bodex": _sha(destinations[0]),
            "sim_filter": _sha(destinations[1]),
        }
        evidence["relocation"] = {
            "schema": "precision_insertion_curated_reset_rebind_v1",
            "source_evidence_sha256": _sha(dst / "source_evidence_original.json"),
            "source_scene_sha256": [_sha(path) for path in source_scenes],
            "source_shared_root": str(source),
            "source_origin_shared_root": str(origin),
            "target_shared_root": str(target),
            "geometry_equivalent_after_path_normalization": True,
            "robot_ready": False,
        }
        with original.open("x", encoding="utf-8") as stream:
            json.dump(evidence, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write("\n")
        h = int(relative.parts[0].split("_")[1])
        cell = relative.parts[3]
        _candidate_arrays(dst, mode=mode, cell=cell, h_cm=h,
                          scenes=destinations)
        rows.append({"candidate": str(relative),
                     "source_evidence_sha256": evidence["relocation"][
                         "source_evidence_sha256"],
                     "target_evidence_sha256": _sha(original)})
    for row in cells:
        _valid_scene_pair(target, target_obj, mode, row["height_cm"],
                          row["from_v8_pose"], row["to_v8_pose"])
    report = {
        "schema": "precision_insertion_curated_reorient_relocation_v1",
        "mode": manifest["mode"], "source_handoff_manifest_sha256": _sha(manifest_path),
        "source_shared_root": str(source), "target_shared_root": str(target),
        "target_handoff_root": str(output), "key_asset_sha256": key_hashes,
        "cells": cells, "seeds": rows,
        "archived_source_bound_scenes": [
            {"source": str(path), "archive": str(archive),
             "sha256": _sha(archive)} for path, archive in scene_archives],
        "scope": "staged_v8_reset_seeds_not_franka_paths_or_motion_permission",
        "robot_ready": False,
    }
    with (output / "RELOCATION.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report
