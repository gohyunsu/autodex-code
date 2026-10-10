#!/usr/bin/env python3
"""Export verified offline v8 tabletop grasps as a non-overwriting NAS addendum.

Catalogues contain this host's absolute paths and are audit-only in the
bundle. Rebuild them on the recipient PC after installing the candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from precision_insertion.candidates import select_pose_candidates
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM, select_mode


KEY = "precision_key_cylinder_r15_h80"
CATALOG_PATTERN = "endpoint_catalog_staged_verified_gap_{gap}mm_20261011.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def export(*, shared_root: Path, promotion_path: Path,
           output_root: Path, code_commit: str) -> dict:
    shared = Path(shared_root).expanduser().resolve()
    promotion_file = Path(promotion_path).expanduser().resolve()
    target = Path(output_root).expanduser().resolve()
    work = target.with_name(target.name + ".incomplete")
    if target.exists() or work.exists():
        raise FileExistsError(f"handoff or incomplete directory exists: {target}")
    if (len(code_commit) < 7 or
            any(character not in "0123456789abcdef" for character in code_commit)):
        raise ValueError("code_commit must be a lowercase Git hash")
    promotion = _json(promotion_file)
    source = shared / "AutoDex/candidates/inspire/v8" / KEY
    if (promotion.get("schema") !=
            "precision_insertion_cylinder_tabletop_v8_promotion_v1" or
            promotion.get("robot_ready") is not False or
            promotion.get("status") != "offline_simulated_candidates_not_physical" or
            promotion.get("candidate_root") != str(source) or
            promotion.get("count") != 12 or
            len(promotion.get("selected_ids", [])) != 12):
        raise ValueError("expected complete, offline-only promoted v8 source")
    selected = set(promotion["selected_ids"])
    found = {str(path.relative_to(source)) for path in source.glob("table/*/*")
             if path.is_dir()}
    if found != selected:
        raise ValueError("v8 candidate tree differs from promotion manifest")
    audit_dir = shared / "AutoDex/precision_insertion/cylindrical"
    catalog_files = []
    for gap in CYLINDER_RADIAL_GAPS_MM:
        label = int(gap)
        path = audit_dir / CATALOG_PATTERN.format(gap=label)
        catalog = _json(path)
        mode = select_mode("cylinder", gap)
        if (catalog.get("complete_scan") is not True or
                catalog.get("eligible_count") != 12 or
                catalog.get("robot_ready") is not False):
            raise ValueError(f"incomplete catalog for radial gap {gap}")
        visible = set()
        for stem in ("000", "001"):
            result = select_pose_candidates(
                catalog, expected_mode=mode, tabletop_pose_stem=stem)
            if result["status"] != "candidates_available":
                raise ValueError(f"stale catalog for radial gap {gap}/{stem}")
            visible.update("/".join(row["key"]) for row in result["candidates"])
        if visible != selected:
            raise ValueError(f"catalog does not cover selected v8 pool: {gap}")
        catalog_files.append(path)

    work.mkdir(parents=True)
    payload = work / "payload/shared_data/AutoDex/candidates/inspire/v8" / KEY
    for identifier in sorted(selected):
        shutil.copytree(source / identifier, payload / identifier)
    audits = work / "audit_only"
    audits.mkdir()
    shutil.copy2(promotion_file, audits / "promotion_manifest.json")
    for path in catalog_files:
        shutil.copy2(path, audits / path.name)
    readme = (
        "# Cylinder tabletop v8 candidate addendum\n\n"
        f"Code: `gohyunsu/autodex-code` `feat/precision-insertion` `{code_commit}`.\n\n"
        "This package contains 12 **offline simulated** full-key Inspire v8 "
        "grasps for the 80 mm cylindrical key. They passed the stock AutoDex "
        "full-key squeeze/gravity filter and nominal plus both achieved "
        "MuJoCo 20 mm endpoint screens for all six radial socket gaps. "
        "They are **not physical or robot-ready grasps**. Squeeze-induced "
        "key-in-hand drift was not accepted against any commissioned limit.\n\n"
        "Install only the `payload/shared_data` candidate directories into "
        "the recipient shared-data tree after checking that no target IDs "
        "exist. Do not overwrite existing candidates. The JSON files in "
        "`audit_only/` contain `/home/hyunsu/shared_data` absolute paths "
        "and must **not** be used as live catalogues on `/home/robot`. "
        "Rebuild there with the demo `run_pipeline.py screen-catalog` for "
        "each actual socket gap, using that PC's `--shared-root`, a positive "
        "commissioned clearance, and a new output filename. The 1 µm "
        "offline numerical tolerance used here is not a safety margin.\n\n"
        "FoundPose key/socket PTHs remain in the earlier NAS handoff's "
        "`pending_foundpose/` and require real AutoDex-image QA before "
        "canonical installation. A session must measure ChArUco and the "
        "socket, then live planning, guarded contact and physical grasp "
        "verification remain required. This bundle never commands a robot.\n"
    )
    (work / "README.md").write_text(readme, encoding="utf-8")
    hashes = {str(path.relative_to(work)): _hash(path)
              for path in sorted(work.rglob("*")) if path.is_file()}
    manifest = {
        "schema": "precision_insertion_cylinder_tabletop_v8_handoff_v1",
        "code_commit": code_commit,
        "source_promotion_sha256": _hash(promotion_file),
        "candidate_ids": sorted(selected), "file_sha256": hashes,
        "catalogs_audit_only": True, "robot_ready": False,
    }
    with (work / "MANIFEST.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    work.rename(target)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--promotion-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    result = export(shared_root=args.shared_root,
                    promotion_path=args.promotion_manifest,
                    output_root=args.output_root, code_commit=args.code_commit)
    print(json.dumps({"output_root": str(args.output_root),
                      "candidate_count": len(result["candidate_ids"]),
                      "file_count": len(result["file_sha256"]),
                      "robot_ready": False}, indent=2))


if __name__ == "__main__":
    main()
