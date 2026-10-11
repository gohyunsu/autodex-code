#!/usr/bin/env python3
"""Publish the current cylinder 001→000/1687 reset seed as a standalone addendum.

This source run has a different audit from the older NAS 001→000/631 bundle.
Its exact source scenes, key assets and audit are retained so the recipient
can rebind them with ``rehydrate_reset_handoff.py`` before v8 installation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


KEY = "precision_key_cylinder_r15_h80"
SEED = "1687"
SCHEMA = "precision_insertion_cylinder_reset_1687_addendum_v1"
AUDIT_REL = ("AutoDex/precision_insertion/cylindrical/"
             "reorient_filter_audit_3000_seed11012_20261011.json")
INSTALL_REL = ("AutoDex/precision_insertion/cylindrical/"
               "cylinder_reset_install_1_0_seed1687_stock_v8_20261011.json")
CANDIDATE_REL = ("AutoDex/precision_insertion/cylindrical/"
                 "reorient_handoff_seed11012/reset_12/" + KEY +
                 "/reorient_12/1_0/" + SEED)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify(root: Path) -> dict:
    bundle = Path(root).expanduser().resolve()
    manifest = _json(bundle / "MANIFEST.json")
    hashes = manifest.get("file_sha256")
    if (manifest.get("schema") != SCHEMA or
            manifest.get("robot_ready") is not False or
            not isinstance(hashes, dict)):
        raise ValueError("not a current cylinder reset handoff")
    actual = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"handoff symlink forbidden: {path}")
        if path.is_file() and path != bundle / "MANIFEST.json":
            actual[str(path.relative_to(bundle))] = _sha(path)
    if actual != hashes:
        raise ValueError("reset handoff file set or bytes differ from manifest")
    return manifest


def export(shared_root: Path, output_root: Path) -> dict:
    shared = Path(shared_root).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    work = target.with_name(target.name + ".incomplete")
    if target.exists() or work.exists():
        raise FileExistsError(f"handoff already exists: {target}")
    candidate = shared / "AutoDex/candidates/inspire/reset_12" / KEY / (
        "reorient_12/1_0/" + SEED)
    install = _json(shared / INSTALL_REL)
    audit = shared / AUDIT_REL
    evidence = _json(candidate / "source_evidence.json")
    if (install.get("schema") != "precision_insertion_cylinder_reset_install_v1" or
            install.get("selected_seed_ids") != [SEED] or
            install.get("cell") != "1_0" or
            install.get("robot_ready") is not False or
            evidence.get("schema") != "precision_insertion_v8_reset_candidate_evidence_v1" or
            evidence.get("seed_id") != SEED or
            evidence.get("source_audit_sha256") != _sha(audit)):
        raise ValueError("current canonical seed and source audit disagree")
    expected = install["installed"][0]["source_sha256"]
    actual = {str(path.relative_to(candidate)): _sha(path)
              for path in sorted(candidate.rglob("*")) if path.is_file()}
    if (len(install["installed"]) != 1 or actual != expected or
            any(path.is_symlink() for path in candidate.rglob("*"))):
        raise ValueError("current canonical reset seed differs from install record")
    key_object = shared / "object_processing" / KEY
    mesh = key_object / "processed_data/mesh/simplified.obj"
    info = key_object / "processed_data/info/simplified.json"
    if (evidence.get("key_mesh_sha256") != _sha(mesh) or
            evidence.get("key_info_sha256") != _sha(info) or
            any(path.is_symlink() for path in key_object.rglob("*"))):
        raise ValueError("source key object differs from reset provenance")
    scenes = []
    for cell in ("0_1", "1_0"):
        for relative in (
            f"object_processing/{KEY}/scene/reorient_12/{cell}.json",
            f"AutoDex/scene/inspire/{KEY}/reorient_12/{cell}.json",
        ):
            path = shared / relative
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError(path)
            scenes.append(relative)
    if evidence["scene_sha256"] != {
            "bodex": _sha(shared / scenes[2]),
            "sim_filter": _sha(shared / scenes[3])}:
        raise ValueError("source scene differs from reset provenance")

    payload = work / "payload/shared_data"
    payload.mkdir(parents=True)
    shutil.copytree(key_object, payload / "object_processing" / KEY,
                    symlinks=False)
    for relative in scenes[1::2]:
        destination = payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shared / relative, destination)
    destination = payload / CANDIDATE_REL
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(candidate, destination, symlinks=False)
    audit_only = work / "audit_only"
    audit_only.mkdir()
    shutil.copy2(audit, audit_only / "SOURCE_AUDIT.json")
    shutil.copy2(shared / INSTALL_REL, audit_only / "INSTALL_RECORD.json")
    (work / "README.md").write_text(
        "# Current cylinder 001→000/1687 reset seed addendum\n\n"
        "This is the selected full-key v8/MuJoCo-stable seed, not a complete "
        "Franka repose path or physical result. It supersedes the *choice* "
        "of the older staged 001→000/631 seed; do not overwrite that bundle. "
        "Verify MANIFEST.json first. The payload has source-root-bound "
        "scenes and evidence. Rehydrate it into a distinct recipient "
        "`cylindrical/reorient_handoff_seed11012/reset_12` staging root "
        "using `rehydrate_reset_handoff.py`, with "
        f"`--source-origin-shared-root {shared}`, before running "
        "`install_cylinder_reset_seeds.py --from-pose-stem 1 "
        "--to-pose-stem 0 --seed-id 1687`. Keep the older 000→001/191 "
        "seed in its own staging root. Neither seed certifies a live reset.\n",
        encoding="utf-8")
    hashes = {str(path.relative_to(work)): _sha(path)
              for path in sorted(work.rglob("*")) if path.is_file()}
    manifest = {
        "schema": SCHEMA, "origin_shared_root": str(shared),
        "seed_id": SEED, "cell": "1_0", "file_sha256": hashes,
        "source_audit_sha256": _sha(audit),
        "scope": "source_bound_full_key_v8_seed_not_robot_motion",
        "robot_ready": False,
    }
    with (work / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    work.rename(target)
    verify(target)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-root", type=Path)
    group.add_argument("--shared-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.verify_root:
        manifest = verify(args.verify_root)
    else:
        if args.output_root is None:
            parser.error("export requires --output-root")
        manifest = export(args.shared_root, args.output_root)
    print(json.dumps({"file_count": len(manifest["file_sha256"]),
                      "seed_id": manifest["seed_id"],
                      "robot_ready": False}, indent=2))


if __name__ == "__main__":
    main()
