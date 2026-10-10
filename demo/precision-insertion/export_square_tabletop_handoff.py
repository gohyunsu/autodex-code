#!/usr/bin/env python3
"""Export a source-verified square v8 asset bundle without touching NAS.

All catalogues, candidate provenance, scenes and fixture JSONs in the bundle
retain source-host absolute paths. They are *not* live-ready on another host;
the recipient must rebind the known paths, preserve source evidence and
rebuild its endpoint catalogue before using any candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.candidates import select_pose_candidates  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402


KEY = "precision_key_1p5mm"
SOCKET = "precision_socket_unified"
MODE = select_mode("square", 1.5)
SCHEMA = "precision_insertion_square_tabletop_handoff_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _files(root: Path, *, omit_scene: bool = False) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"missing handoff input: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"handoff cannot contain a symlink: {path}")
        if path.is_file() and not (omit_scene and
                                   "scene" in path.relative_to(root).parts):
            files.append(path)
    if not files:
        raise ValueError(f"no files in handoff input: {root}")
    return files


def _copy_tree(source: Path, target: Path, *, omit_scene: bool = False) -> int:
    files = _files(source, omit_scene=omit_scene)
    for path in files:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    return len(files)


def export(
    *, shared_root: Path, catalog_path: Path, fidelity_report_path: Path,
    output_root: Path, code_commit: str,
) -> dict:
    root = Path(shared_root).expanduser().resolve()
    catalog_file = Path(catalog_path).expanduser().resolve()
    fidelity_file = Path(fidelity_report_path).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    work = target.with_name(target.name + ".incomplete")
    if target.exists() or work.exists():
        raise FileExistsError(f"handoff or incomplete directory exists: {target}")
    if (len(code_commit) < 7 or
            any(character not in "0123456789abcdef" for character in code_commit)):
        raise ValueError("code_commit must be a lowercase Git hash")
    paths = AssetPaths(root, MODE)
    catalog = _json(catalog_file)
    if (catalog.get("shared_root") != str(root) or
            catalog.get("mode", {}).get("key_object") != KEY or
            catalog.get("mode", {}).get("socket_object") != SOCKET or
            catalog.get("complete_scan") is not True or
            catalog.get("total_candidate_directories") != 8 or
            catalog.get("eligible_count") != 7 or
            catalog.get("robot_ready") is not False):
        raise ValueError("expected complete source-root square 1.5 mm v8 catalog")
    selected = select_pose_candidates(
        catalog, expected_mode=MODE, tabletop_pose_stem="004")
    selected_ids = {"/".join(row["key"]) for row in selected["candidates"]}
    if (selected["status"] != "candidates_available" or
            len(selected_ids) != 7 or
            {identifier.split("/")[2] for identifier in selected_ids} !=
            {"35", "70", "79", "95", "99", "104", "5102"}):
        raise ValueError("square promotion candidate list changed")
    for stem in ("000", "001", "002", "003"):
        if select_pose_candidates(
                catalog, expected_mode=MODE,
                tabletop_pose_stem=stem)["candidates"]:
            raise ValueError("source catalog has an unexpected square pose grasp")
    promotion_path = paths.candidate_dir / "table/4/promotion_manifest.json"
    promotion = _json(promotion_path)
    if (promotion.get("schema") != "precision_insertion_square_scene_promotion_v1"
            or promotion.get("candidate_ids") !=
            ["35", "70", "79", "95", "99", "104", "5102"] or
            promotion.get("robot_ready") is not False):
        raise ValueError("square pilot promotion manifest changed")
    fidelity = _json(fidelity_file)
    if (fidelity.get("schema") !=
            "precision_insertion_square_grasp_fidelity_audit_v1" or
            fidelity.get("catalog_sha256") != _sha(catalog_file) or
            fidelity.get("candidate_count") != 7 or
            fidelity.get("robot_ready") is not False):
        raise ValueError("square MuJoCo fidelity report is stale")
    if not paths.robot_urdf.is_file():
        raise FileNotFoundError(f"AutoDex robot URDF missing: {paths.robot_urdf}")

    work.mkdir(parents=True)
    payload = work / "payload/shared_data"
    objects = payload / "object_processing"
    for name in (KEY, SOCKET):
        _copy_tree(root / "object_processing" / name,
                   objects / name, omit_scene=True)
    scene_root = paths.scene_dir
    source_scenes = sorted(scene_root.glob("*.json"))
    if [p.stem for p in source_scenes] != ["0", "1", "2", "3", "4"]:
        raise ValueError("square v8 source tabletop scenes are incomplete")
    for scene in source_scenes:
        destination = (payload / "AutoDex/scene/inspire" / KEY /
                       "table" / scene.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scene, destination)
    _copy_tree(paths.candidate_dir,
               payload / "AutoDex/candidates/inspire/v8" / KEY)
    fixture = root / "AutoDex/precision_insertion/fixtures/unified_socket"
    _copy_tree(fixture,
               payload / "AutoDex/precision_insertion/fixtures/unified_socket")
    audits = work / "audit_only"
    audits.mkdir()
    shutil.copy2(catalog_file, audits / "source_endpoint_catalog.json")
    shutil.copy2(fidelity_file, audits / "source_square_fidelity.json")
    (work / "README.md").write_text(
        "# Square 1.5 mm tabletop candidate handoff\n\n"
        f"Code: `gohyunsu/autodex-code` `feat/precision-insertion` `{code_commit}`.\n\n"
        "This package carries the full 1.5 mm key/unified socket CAD, "
        "source-host v8 tabletop scenes, fixture templates, and eight v8 "
        "grasp candidates. Seven pose-004 grasps pass the *nominal* 20 mm "
        "key/whole-Inspire/socket endpoint; the old pose-000 grasp fails. "
        "All seven are from a contact-policy-report-only 10,000-proposal "
        "pilot and show MuJoCo key-in-hand drift. None is physically or "
        "robot ready.\n\n"
        "MANIFEST.json binds every packaged file by SHA-256. The JSON scene "
        "and fixture paths, candidate source-scene hashes and audit-only "
        "catalog are bound to the source shared-data root. Do not directly "
        "use those JSON files at another root. Verify the manifest, rebind "
        "only expected absolute path fields, preserve the original evidence, "
        "and rebuild the endpoint catalog on the recipient PC. The package "
        "does not include QA-approved FoundPose PTHs, a measured socket pose, "
        "a guarded controller or a physical success claim.\n",
        encoding="utf-8")
    hashes = {str(file.relative_to(work)): _sha(file)
              for file in sorted(work.rglob("*")) if file.is_file()}
    manifest = {
        "schema": SCHEMA, "code_commit": code_commit,
        "origin_shared_root": str(root),
        "source_catalog_sha256": _sha(catalog_file),
        "source_fidelity_sha256": _sha(fidelity_file),
        "source_robot_urdf_sha256": _sha(paths.robot_urdf),
        "candidate_ids": sorted(selected_ids),
        "file_sha256": hashes,
        "catalogs_audit_only": True, "robot_ready": False,
    }
    with (work / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    work.rename(target)
    return manifest


def verify(root: Path) -> dict:
    bundle = Path(root).expanduser().resolve()
    manifest = _json(bundle / "MANIFEST.json")
    hashes = manifest.get("file_sha256")
    if (manifest.get("schema") != SCHEMA or
            manifest.get("robot_ready") is not False or
            not isinstance(hashes, dict)):
        raise ValueError("not a square tabletop handoff manifest")
    actual = {}
    for file in sorted(bundle.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"handoff symlink is forbidden: {file}")
        if file.is_file() and file != bundle / "MANIFEST.json":
            actual[str(file.relative_to(bundle))] = _sha(file)
    if actual != hashes:
        raise ValueError("handoff file set or bytes differ from manifest")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify-root", type=Path)
    group.add_argument("--output-root", type=Path)
    parser.add_argument("--shared-root", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--fidelity-report", type=Path)
    parser.add_argument("--code-commit")
    args = parser.parse_args()
    try:
        if args.verify_root is not None:
            report = verify(args.verify_root)
        else:
            if any(value is None for value in (
                    args.shared_root, args.catalog, args.fidelity_report,
                    args.code_commit)):
                parser.error("export needs shared root, catalog, fidelity and commit")
            report = export(
                shared_root=args.shared_root, catalog_path=args.catalog,
                fidelity_report_path=args.fidelity_report,
                output_root=args.output_root, code_commit=args.code_commit)
    except (FileExistsError, FileNotFoundError, KeyError, TypeError,
            ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"candidate_count": len(report["candidate_ids"]),
                      "file_count": len(report["file_sha256"]),
                      "robot_ready": False}, indent=2))


if __name__ == "__main__":
    main()
