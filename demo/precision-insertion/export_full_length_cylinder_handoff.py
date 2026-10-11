#!/usr/bin/env python3
"""Export the append-only full-length cylinder v8 grasp addendum.

Only new candidate directories are transferable. Catalogues and source
promotion evidence are audit-only because they contain producer paths. The
receiving AutoDex PC rebuilds catalogues against its own fixture assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from precision_insertion.assets import AssetPaths
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode
from precision_insertion.live_session_runner import preflight_endpoint_catalog_file


SCHEMA = "precision_insertion_cylinder_full_length_addendum_v1"
KEY = "precision_key_cylinder_r15_h80"
CATALOG = "endpoint_catalog_full_length_collision_only_gap_{gap}mm_20261011.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _scene_semantics(path: Path) -> dict:
    record = _json(path)
    scene = record["scene"]
    if set(scene["mesh"]) != {"target"}:
        raise ValueError(f"unexpected physical key scene mesh: {path}")
    target = scene["mesh"]["target"]
    return {"meta": record["meta"], "cuboid": scene["cuboid"],
            "target_pose": target["pose"], "target_scale": target["scale"]}


def verify(bundle_root: Path) -> dict:
    bundle = Path(bundle_root).expanduser().resolve()
    manifest = _json(bundle / "MANIFEST.json")
    declared = manifest.get("file_sha256")
    if (manifest.get("schema") != SCHEMA or
            manifest.get("robot_ready") is not False or
            not isinstance(declared, dict)):
        raise ValueError("not a full-length cylinder v8 addendum")
    actual = {}
    for path in sorted(bundle.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in handoff: {path}")
        if path.is_file() and path != bundle / "MANIFEST.json":
            actual[str(path.relative_to(bundle))] = _sha(path)
    if actual != declared:
        raise ValueError("handoff file set or SHA256 differs from manifest")
    return manifest


def export(*, shared_root: Path, promotion_path: Path,
           output_root: Path, code_commit: str) -> dict:
    root = Path(shared_root).expanduser().resolve()
    promotion_file = Path(promotion_path).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    working = target.with_name(target.name + ".incomplete")
    if target.exists() or working.exists():
        raise FileExistsError(f"handoff destination already exists: {target}")
    if (len(code_commit) < 7 or
            any(char not in "0123456789abcdef" for char in code_commit)):
        raise ValueError("code_commit must be a lowercase Git SHA")
    promotion = _json(promotion_file)
    selected = promotion.get("selected")
    if (promotion.get("schema") !=
            "precision_insertion_cylinder_full_length_promotion_v1" or
            promotion.get("robot_ready") is not False or
            promotion.get("selected_count") != len(selected or []) or
            not selected):
        raise ValueError("missing complete full-length promotion evidence")
    mode = select_mode("cylinder", 20)
    assets = AssetPaths(root, mode)
    if promotion.get("full_key_planning_mesh_sha256") != _sha(
            assets.key_planning_mesh):
        raise ValueError("physical key mesh changed since promotion")
    identifiers = [row["installed_id"] for row in selected]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate new candidate IDs")
    for identifier in identifiers:
        parts = Path(identifier).parts
        if (len(parts) != 3 or parts[:2] not in
                (("table", "0"), ("table", "1")) or
                not parts[2].isdigit() or int(parts[2]) < 20000):
            raise ValueError(f"unexpected appended v8 ID: {identifier}")
        source = assets.candidate_dir / identifier
        if not source.is_dir() or source.is_symlink():
            raise FileNotFoundError(source)
        if any(path.is_symlink() or not path.is_file()
               for path in source.iterdir()):
            raise ValueError(f"candidate has a non-file or symlink: {source}")
    audit_root = root / "AutoDex/precision_insertion/cylindrical"
    catalogs = {}
    for gap in CYLINDER_RADIAL_GAPS_MM:
        path = audit_root / CATALOG.format(gap=int(gap))
        catalog = preflight_endpoint_catalog_file(
            mode=select_mode("cylinder", gap), shared_root=root,
            catalog_path=path)
        if catalog.get("robot_ready") is not False:
            raise ValueError(f"catalogue claims robot readiness: {path}")
        by_id = {"/".join(row["key"]): row for row in catalog["candidates"]}
        if not set(identifiers) <= set(by_id):
            raise ValueError(f"new candidate missing from catalogue: {path}")
        label = f"{int(gap):02d}mm"
        if any(by_id[row["installed_id"]]["eligible"] !=
               (label in row["eligible_socket_gaps"]) for row in selected):
            raise ValueError(f"catalogue disagrees with 20 mm promotion: {path}")
        catalogs[label] = (path, catalog["eligible_count"])
    scenes = {}
    tabletop = {}
    for scene_id, stem in (("0", "000"), ("1", "001")):
        scene = root / "AutoDex/scene/inspire" / KEY / "table" / f"{scene_id}.json"
        if Path(_json(scene)["scene"]["mesh"]["target"]["file_path"]).resolve() != (
                assets.key_planning_mesh.resolve()):
            raise ValueError(f"key scene targets a different mesh: {scene}")
        scenes[scene_id] = _scene_semantics(scene)
        tabletop[stem] = _sha(assets.key_tabletop_dir / f"{stem}.npy")

    working.mkdir(parents=True)
    payload = working / "payload/shared_data/AutoDex/candidates/inspire/v8" / KEY
    for identifier in identifiers:
        shutil.copytree(assets.candidate_dir / identifier,
                        payload / identifier, symlinks=False)
    audits = working / "audit_only"
    audits.mkdir()
    shutil.copy2(promotion_file, audits / "promotion_manifest.json")
    for path, _count in catalogs.values():
        shutil.copy2(path, audits / path.name)
    (working / "README.md").write_text(
        "# Full-length cylinder v8 grasp addendum\n\n"
        f"Code: `gohyunsu/autodex-code` `feat/precision-insertion` `{code_commit}`.\n\n"
        "Install the 112 new candidate IDs with the demo-local "
        "`rehydrate_full_length_cylinder_handoff.py` command. It verifies "
        "the existing key mesh/tabletop scene and never overwrites a "
        "different candidate. The six JSON catalogues in `audit_only/` "
        "contain source-machine paths: rebuild all six complete catalogues "
        "on the receiving PC after installation, using "
        "`run_pipeline.py screen-catalog --min-hand-clearance-mm 0`. "
        "The physical key, socket fixture and FoundPose assets are supplied "
        "by earlier handoffs and must already be installed. To reproduce "
        "BODex rather than merely run the filtered candidates, regenerate "
        "the 80 mm proxy with `build_full_length_cylinder_proposal.py` "
        "on the recipient root. No Franka path, insertion dynamics or "
        "physical success is claimed by this addendum.\n",
        encoding="utf-8")
    files = {str(path.relative_to(working)): _sha(path)
             for path in sorted(working.rglob("*")) if path.is_file()}
    manifest = {
        "schema": SCHEMA, "code_commit": code_commit,
        "origin_shared_root": str(root),
        "physical_key_object": KEY,
        "physical_key_raw_mesh_sha256": _sha(assets.raw_mesh(KEY)),
        "physical_key_planning_mesh_sha256": _sha(assets.key_planning_mesh),
        "scene_semantics": scenes, "tabletop_pose_sha256": tabletop,
        "candidate_ids": sorted(identifiers),
        "catalog_eligible_counts": {label: count for label, (_path, count)
                                    in catalogs.items()},
        "source_promotion_sha256": _sha(promotion_file),
        "file_sha256": files, "catalogs_audit_only": True,
        "robot_ready": False,
    }
    with (working / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    working.rename(target)
    verify(target)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-root", type=Path)
    group.add_argument("--output-root", type=Path)
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--promotion-manifest", type=Path)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    if args.verify_root is not None:
        report = verify(args.verify_root)
    else:
        if (args.shared_root is None or args.promotion_manifest is None or
                args.code_commit is None):
            parser.error("export needs shared root, promotion and code commit")
        report = export(
            shared_root=args.shared_root,
            promotion_path=args.promotion_manifest,
            output_root=args.output_root, code_commit=args.code_commit)
    print(json.dumps({"candidate_count": len(report["candidate_ids"]),
                      "file_count": len(report["file_sha256"]),
                      "robot_ready": False}, indent=2))


if __name__ == "__main__":
    main()
