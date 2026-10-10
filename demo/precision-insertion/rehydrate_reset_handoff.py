#!/usr/bin/env python3
"""Rebind staged cylinder v8 reset seeds to another shared-data root.

This never installs into the canonical AutoDex reset tree or authorizes robot
motion. The recipient key assets must already be byte-identical. Source and
regenerated target scenes must have the same numeric collision geometry;
only their absolute mesh/URDF paths may differ. Existing files are not
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.reorient_assets import (  # noqa: E402
    _scene_path, _sim_filter_scene_path, _validate_scene,
)
from precision_insertion.reset_candidates import (  # noqa: E402
    EVIDENCE_SCHEMA, GRASP_FILES, _candidate_arrays, _valid_scene_pair,
)


KEY = "precision_key_cylinder_r15_h80"
HEIGHT_CM = 12


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_scene(scene: dict, object_dir: Path) -> dict:
    """Normalize *only* the two expected host paths, never pose geometry."""
    result = json.loads(json.dumps(scene, allow_nan=False))
    target = result["scene"]["mesh"]["target"]
    paths = {
        "file_path": object_dir / "processed_data/mesh/simplified.obj",
        "urdf_path": object_dir / "processed_data/urdf/coacd.urdf",
    }
    for field, expected in paths.items():
        if Path(target[field]).resolve() != expected.resolve():
            raise ValueError(f"reorient scene has unexpected {field}")
        target[field] = f"<KEY_{field.upper()}>"
    # The stock simulation mirror may add descriptive object IDs; the v8
    # scene validator permits them only when they name this same full key.
    for field in ("geometry_object", "grasp_target_object"):
        value = result["meta"].pop(field, KEY)
        if value != KEY:
            raise ValueError(f"reorient scene has a different {field}")
    return result


def _matching_key_assets(
    source: Path, target: Path, *, install_missing: bool,
) -> dict[str, str]:
    src = source / "object_processing" / KEY
    dst = target / "object_processing" / KEY
    if not src.is_dir() or (not install_missing and not dst.is_dir()):
        raise FileNotFoundError("install the byte-identical key object assets first")
    hashes = {}
    missing = []
    for path in sorted(src.rglob("*")):
        if not path.is_file() or "scene" in path.relative_to(src).parts:
            continue
        relative = path.relative_to(src)
        received = dst / relative
        digest = _sha(path)
        if received.exists() and (not received.is_file() or
                                  _sha(received) != digest):
            raise ValueError(f"recipient key asset is missing or changed: {relative}")
        if not received.exists():
            missing.append((path, received))
        hashes[str(relative)] = digest
    if not hashes:
        raise ValueError("no key object assets were checked")
    if missing and not install_missing:
        raise FileNotFoundError(
            f"recipient key asset is missing: {missing[0][1]}")
    for path, received in missing:
        received.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as source_stream, received.open("xb") as target_stream:
            shutil.copyfileobj(source_stream, target_stream)
    return hashes


def rehydrate(
    *, source_shared_root: Path, target_shared_root: Path,
    source_candidate_root: Path, target_candidate_root: Path,
    source_audit_path: Path,
    source_origin_shared_root: Path | None = None,
    archive_source_bound_scenes: bool = False,
    install_missing_key_assets: bool = False,
) -> dict:
    """Verify first; stage relocated evidence and install without overwrite."""
    source = Path(source_shared_root).expanduser().resolve()
    target = Path(target_shared_root).expanduser().resolve()
    origin = (source if source_origin_shared_root is None else
              Path(source_origin_shared_root).expanduser().resolve())
    incoming = Path(source_candidate_root).expanduser().resolve()
    output = Path(target_candidate_root).expanduser().resolve()
    audit_file = Path(source_audit_path).expanduser().resolve()
    expected_parent = target / "AutoDex/precision_insertion/cylindrical/reorient_handoff"
    if (source == target or output != expected_parent / "reset_12" or
            output.exists() or not incoming.is_dir() or
            not output.is_relative_to(target)):
        raise ValueError("source/recipient or non-overwriting staged reset_12 path is invalid")
    mode = select_mode("cylinder", 20)
    asset_hashes = _matching_key_assets(
        source, target, install_missing=install_missing_key_assets)
    audit_sha = _sha(audit_file)
    source_obj = source / "object_processing" / KEY
    origin_obj = origin / "object_processing" / KEY
    target_obj = target / "object_processing" / KEY

    # Compare every directed scene after replacing only the absolute paths.
    # A target scene with different table/pillar geometry is never repaired.
    from src.grasp_generation.reorient.gen_scene import gen_reorient_scene

    scene_writes: list[tuple[Path, dict]] = []
    scene_archives: list[tuple[Path, Path]] = []
    scene_pairs = {}
    for cell in ("0_1", "1_0"):
        i, j = map(int, cell.split("_"))
        source_scenes = (
            _scene_path(source_obj, HEIGHT_CM, i, j),
            _sim_filter_scene_path(source, mode, HEIGHT_CM, i, j),
        )
        source_scene = json.loads(source_scenes[0].read_text(encoding="utf-8"))
        source_sim_scene = json.loads(source_scenes[1].read_text(
            encoding="utf-8"))
        if (_canonical_scene(source_scene, origin_obj) !=
                _canonical_scene(source_sim_scene, origin_obj)):
            raise ValueError(f"source BODex/simulation scenes differ: {cell}")
        generated = gen_reorient_scene(
            KEY, i, j, HEIGHT_CM / 100.0,
            obj_root=str(target / "object_processing"))
        generated["meta"]["scene_type"] = f"reorient_{HEIGHT_CM}"
        _validate_scene(generated, mode=mode, object_dir=target_obj,
                        h_cm=HEIGHT_CM, i=i, j=j)
        if (_canonical_scene(source_scene, origin_obj) !=
                _canonical_scene(generated, target_obj)):
            raise ValueError(f"regenerated reset scene changed geometry: {cell}")
        destinations = (
            _scene_path(target_obj, HEIGHT_CM, i, j),
            _sim_filter_scene_path(target, mode, HEIGHT_CM, i, j),
        )
        for path, source_path in zip(destinations, source_scenes):
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8"))
                try:
                    _validate_scene(existing, mode=mode, object_dir=target_obj,
                                    h_cm=HEIGHT_CM, i=i, j=j)
                    already_valid = (
                        _canonical_scene(existing, target_obj) ==
                        _canonical_scene(generated, target_obj))
                except ValueError:
                    already_valid = False
                if already_valid:
                    continue
                if (archive_source_bound_scenes and
                        _sha(path) == _sha(source_path)):
                    archive = (output.parent / "source_bound_scenes_archive" /
                               path.relative_to(target))
                    if archive.exists():
                        raise FileExistsError(
                            f"source-bound scene archive already exists: {archive}")
                    scene_archives.append((path, archive))
                    scene_writes.append((path, generated))
                    continue
                raise ValueError(f"recipient reset scene changed geometry: {path}")
            else:
                scene_writes.append((path, generated))
        scene_pairs[cell] = (source_scenes, destinations)

    selected = []
    for cell, (source_scenes, _) in scene_pairs.items():
        folder = incoming / KEY / "reorient_12" / cell
        for seed in sorted(folder.glob("*")):
            if not seed.is_dir() or not seed.name.isdigit():
                continue
            _candidate_arrays(seed, mode=mode, cell=cell,
                              h_cm=HEIGHT_CM, scenes=source_scenes)
            original = seed / "source_evidence.json"
            evidence = json.loads(original.read_text(encoding="utf-8"))
            if (evidence.get("schema") != EVIDENCE_SCHEMA or
                    evidence.get("source_audit_sha256") != audit_sha):
                raise ValueError(f"reset seed and source audit differ: {seed}")
            selected.append((cell, seed, evidence))
    if not selected:
        raise ValueError("handoff has no provenance-bound reset seed")

    # All comparisons above precede writes. Existing scene files are verified,
    # not overwritten; new scenes are needed by the target-side seed loader.
    for path, archive in scene_archives:
        archive.parent.mkdir(parents=True, exist_ok=True)
        path.rename(archive)
    for path, scene in scene_writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(scene, stream, indent=2, allow_nan=False)
            stream.write("\n")
    target_scenes = {
        cell: _valid_scene_pair(target, target_obj, mode,
                                HEIGHT_CM, *map(int, cell.split("_")))
        for cell in scene_pairs
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="reset_12.rehydrating.",
                                 dir=output.parent))
    shutil.copyfile(audit_file, work / "SOURCE_AUDIT.json")
    source_scene_hashes = {}
    for cell, (source_scenes, _) in scene_pairs.items():
        source_scene_hashes[cell] = {}
        for label, scene_path in zip(("bodex", "sim_filter"), source_scenes):
            retained = work / "original_scenes" / label / f"{cell}.json"
            retained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(scene_path, retained)
            source_scene_hashes[cell][label] = _sha(retained)
    rows = []
    for cell, seed, evidence in selected:
        destination = work / KEY / "reorient_12" / cell / seed.name
        destination.mkdir(parents=True)
        for name in GRASP_FILES:
            shutil.copyfile(seed / name, destination / name)
        original_path = seed / "source_evidence.json"
        shutil.copyfile(original_path, destination / "source_evidence_original.json")
        scenes = target_scenes[cell]
        rebound = dict(evidence)
        rebound["scene_sha256"] = {
            "bodex": _sha(scenes[0]), "sim_filter": _sha(scenes[1])}
        rebound["relocation"] = {
            "schema": "precision_insertion_reset_scene_rebind_v1",
            "source_evidence_sha256": _sha(original_path),
            "source_shared_root": str(source),
            "source_origin_shared_root": str(origin),
            "target_shared_root": str(target),
            "scene_geometry_equivalent_after_path_normalization": True,
            "robot_ready": False,
        }
        with (destination / "source_evidence.json").open("x", encoding="utf-8") as stream:
            json.dump(rebound, stream, indent=2, allow_nan=False)
            stream.write("\n")
        _candidate_arrays(destination, mode=mode, cell=cell,
                          h_cm=HEIGHT_CM, scenes=scenes)
        rows.append({"cell": cell, "seed_id": seed.name,
                     "source_evidence_sha256": _sha(original_path),
                     "target_evidence_sha256": _sha(destination / "source_evidence.json")})
    report = {
        "schema": "precision_insertion_reset_handoff_rehydration_v1",
        "source_shared_root": str(source), "target_shared_root": str(target),
        "source_origin_shared_root": str(origin),
        "source_candidate_root": str(incoming),
        "target_candidate_root": str(output),
        "source_audit_sha256": audit_sha,
        "retained_source_scene_sha256_by_cell": source_scene_hashes,
        "matching_key_asset_sha256": asset_hashes,
        "regenerated_scene_count": len(scene_writes),
        "archived_source_bound_scenes": [
            {"original": str(path), "archive": str(archive),
             "sha256": _sha(archive)} for path, archive in scene_archives],
        "seeds": rows,
        "scope": "staged_v8_reset_grasps_not_franka_paths_or_physical_reset",
        "robot_ready": False,
    }
    with (work / "RELOCATION.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    work.rename(output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shared-root", type=Path, required=True)
    parser.add_argument("--source-origin-shared-root", type=Path,
                        help="historical absolute root embedded in source scene JSON; "
                             "use when the handoff was extracted elsewhere")
    parser.add_argument("--target-shared-root", type=Path, required=True)
    parser.add_argument("--source-candidate-root", type=Path, required=True,
                        help="exact staged reset_12 directory")
    parser.add_argument("--target-candidate-root", type=Path, required=True,
                        help="new target shared_data/AutoDex/precision_insertion/"
                             "cylindrical/reorient_handoff/reset_12")
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--archive-source-bound-scenes", action="store_true",
                        help="move only byte-identical historical scene JSONs "
                             "to a non-overwriting archive before regeneration")
    parser.add_argument("--install-missing-key-assets", action="store_true",
                        help="copy absent byte-identical key files from the "
                             "handoff, never overwrite a changed target file")
    args = parser.parse_args()
    result = rehydrate(
        source_shared_root=args.source_shared_root,
        target_shared_root=args.target_shared_root,
        source_candidate_root=args.source_candidate_root,
        target_candidate_root=args.target_candidate_root,
        source_audit_path=args.source_audit,
        source_origin_shared_root=args.source_origin_shared_root,
        archive_source_bound_scenes=args.archive_source_bound_scenes,
        install_missing_key_assets=args.install_missing_key_assets)
    print(json.dumps({"target_candidate_root": result["target_candidate_root"],
                      "seed_count": len(result["seeds"]),
                      "robot_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
