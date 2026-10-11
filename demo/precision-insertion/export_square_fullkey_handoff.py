#!/usr/bin/env python3
"""Package the four current square full-key v8 pools without replacing assets.

The bundle is source-root-bound. Its catalogues are audit-only; relocation to
another shared-data root must rebind scene paths and candidate scene hashes,
then rebuild all endpoint catalogues on that recipient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from export_square_tabletop_handoff import _copy_tree, _json, _sha
from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.live_session_runner import preflight_endpoint_catalog_file


GAPS = (0.3, 0.5, 1.0, 1.5)
SOCKET = "precision_socket_unified"
SCHEMA = "precision_insertion_square_fullkey1000_handoff_v1"


def verify(bundle_root: Path) -> dict:
    bundle = Path(bundle_root).expanduser().resolve()
    manifest = _json(bundle / "MANIFEST.json")
    hashes = manifest.get("file_sha256")
    if (manifest.get("schema") != SCHEMA or
            manifest.get("robot_ready") is not False or
            not isinstance(hashes, dict)):
        raise ValueError("not a square full-key handoff")
    actual = {}
    for file in sorted(bundle.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"handoff symlink is forbidden: {file}")
        if file.is_file() and file != bundle / "MANIFEST.json":
            actual[str(file.relative_to(bundle))] = _sha(file)
    if actual != hashes:
        raise ValueError("handoff file set or bytes differ from manifest")
    return manifest


def export(*, shared_root: Path, catalog_paths: dict[float, Path],
           output_root: Path, code_commit: str) -> dict:
    root = Path(shared_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    work = target.with_name(target.name + ".incomplete")
    if target.exists() or work.exists():
        raise FileExistsError(f"handoff or incomplete directory exists: {target}")
    if (set(catalog_paths) != set(GAPS) or len(code_commit) < 7 or
            any(character not in "0123456789abcdef" for character in code_commit)):
        raise ValueError("four square catalogs and a lowercase Git hash are required")
    modes = {}
    for gap in GAPS:
        mode = select_mode("square", gap)
        catalog_file = Path(catalog_paths[gap]).expanduser().resolve()
        catalog = preflight_endpoint_catalog_file(
            mode=mode, shared_root=root, catalog_path=catalog_file)
        paths = AssetPaths(root, mode)
        candidate_dirs = sorted(
            path for path in paths.candidate_dir.glob("table/*/*")
            if path.is_dir())
        if (catalog.get("total_candidate_directories") != len(candidate_dirs) or
                len(catalog["candidates"]) != len(candidate_dirs) or
                catalog.get("robot_ready") is not False):
            raise ValueError(f"candidate pool changed for square gap {gap}")
        modes[gap] = (mode, paths, catalog_file, catalog, candidate_dirs)
    robot_urdf = modes[1.5][1].robot_urdf
    if not robot_urdf.is_file():
        raise FileNotFoundError(robot_urdf)

    work.mkdir(parents=True)
    payload = work / "payload/shared_data"
    objects = payload / "object_processing"
    for object_name in (SOCKET, *(modes[gap][0].key_object for gap in GAPS)):
        _copy_tree(root / "object_processing" / object_name,
                   objects / object_name, omit_scene=True)
    fixture = root / "AutoDex/precision_insertion/fixtures/unified_socket"
    _copy_tree(fixture,
               payload / "AutoDex/precision_insertion/fixtures/unified_socket")
    audits = work / "audit_only"
    audits.mkdir()
    entries = []
    for gap in GAPS:
        mode, paths, catalog_file, catalog, candidate_dirs = modes[gap]
        scenes = sorted(paths.scene_dir.glob("*.json"))
        if [p.stem for p in scenes] != [str(i) for i in range(5)]:
            raise ValueError(f"square v8 tabletop scenes incomplete: {mode.key_object}")
        for scene in scenes:
            destination = (payload / "AutoDex/scene/inspire" /
                           mode.key_object / "table" / scene.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(scene, destination)
        _copy_tree(paths.candidate_dir,
                   payload / "AutoDex/candidates/inspire/v8" / mode.key_object)
        shutil.copy2(catalog_file, audits / catalog_file.name)
        entries.append({
            "gap_mm": gap, "key_object": mode.key_object,
            "candidate_ids": [str(path.relative_to(paths.candidate_dir))
                              for path in candidate_dirs],
            "eligible_count": catalog["eligible_count"],
            "source_catalog": str(catalog_file),
            "source_catalog_sha256": _sha(catalog_file),
        })
    (work / "README.md").write_text(
        "# Square full-key v8 tabletop handoff\n\n"
        f"Code: `gohyunsu/autodex-code` `feat/precision-insertion` `{code_commit}`.\n\n"
        "Contains all current v8 tabletop candidate directories for the four "
        "square keys, their metric object-processing assets, the common socket "
        "and fixture, and source v8 scenes. All four source-verified catalogs "
        "are under `audit_only/`, not live input on another host. The source "
        "scene JSONs and grasp validation hashes bind absolute source paths. "
        "A recipient must verify this manifest, non-destructively rebind "
        "those files, then rebuild its complete endpoint catalogs. FoundPose "
        "PTHs are in a separate handoff. These are offline simulated grasp "
        "candidates, not robot motion or physical insertion results.\n",
        encoding="utf-8")
    hashes = {str(path.relative_to(work)): _sha(path)
              for path in sorted(work.rglob("*")) if path.is_file()}
    manifest = {
        "schema": SCHEMA, "code_commit": code_commit,
        "origin_shared_root": str(root),
        "source_robot_urdf_sha256": _sha(robot_urdf),
        "modes": entries, "file_sha256": hashes,
        "catalogs_audit_only": True, "robot_ready": False,
    }
    with (work / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    work.rename(target)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-root", type=Path)
    group.add_argument("--output-root", type=Path)
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--catalog", nargs=2, action="append",
                        metavar=("GAP_MM", "CATALOG_JSON"))
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.verify_root is not None:
        result = verify(args.verify_root)
    else:
        if (args.shared_root is None or args.code_commit is None or
                not args.catalog):
            parser.error("export needs shared root, four catalogs and code commit")
        catalogs = {float(gap): Path(path) for gap, path in args.catalog}
        if len(catalogs) != len(args.catalog):
            parser.error("catalog gaps must be unique")
        result = export(shared_root=args.shared_root, catalog_paths=catalogs,
                        output_root=args.output_root, code_commit=args.code_commit)
    print(json.dumps({"modes": len(result["modes"]),
                      "file_count": len(result["file_sha256"]),
                      "robot_ready": False}, indent=2))


if __name__ == "__main__":
    main()
