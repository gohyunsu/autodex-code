#!/usr/bin/env python3
"""Rebind installed v8 reset evidence after a shared-data path relocation.

The candidate arrays, MuJoCo trajectory, key mesh and numeric scene geometry
must still match their source evidence.  Only the absolute mesh/URDF paths in
the reset scenes may differ.  Original evidence is retained next to every
installed seed before the current-scene hashes are installed.
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
for path in (str(DEMO_DIR), str(REPO_ROOT), str(REPO_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.curated_reorient_relocation import (  # noqa: E402
    _normalized_scene,
)
from precision_insertion.reorient_assets import (  # noqa: E402
    _same_scene_geometry, _scene_path, _sim_filter_scene_path,
    _validate_scene,
)
from precision_insertion.reset_candidates import (  # noqa: E402
    GRASP_FILES, _candidate_arrays, _same_numeric_record, _valid_scene_pair,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebind_installed(
    *, source_shared_root: Path, source_origin_shared_root: Path,
    target_shared_root: Path, cells: tuple[str, ...] = ("0_1", "1_0"),
) -> dict:
    source = Path(source_shared_root).expanduser().resolve()
    origin = Path(source_origin_shared_root).expanduser()
    target = Path(target_shared_root).expanduser().resolve()
    if (source == target or not source.is_dir() or not target.is_dir() or
            not origin.is_absolute() or not cells or
            any(len(cell.split("_")) != 2 or
                not all(part.isdigit() for part in cell.split("_"))
                for cell in cells)):
        raise ValueError("reset evidence rebind needs distinct valid shared roots")
    mode = select_mode("cylinder", 20)
    source_obj = source / "object_processing" / mode.key_object
    origin_obj = origin / "object_processing" / mode.key_object
    target_obj = target / "object_processing" / mode.key_object
    candidate_root = (target / "AutoDex/candidates/inspire/reset_12" /
                      mode.key_object / "reorient_12")

    pending = []
    rows = []
    temporary_root = Path(tempfile.mkdtemp(prefix="reset_evidence_rebind."))
    try:
        for cell in cells:
            i, j = (int(part) for part in cell.split("_"))
            source_scenes = (
                _scene_path(source_obj, 12, i, j),
                _sim_filter_scene_path(source, mode, 12, i, j),
            )
            source_json = [json.loads(path.read_text(encoding="utf-8"))
                           for path in source_scenes]
            for scene in source_json:
                # The portable archive stores bytes elsewhere while retaining
                # the exact historical host paths recorded by the evidence.
                _validate_scene(
                    scene, mode=mode, object_dir=origin_obj,
                    h_cm=12, i=i, j=j)
            if not _same_scene_geometry(*source_json):
                raise ValueError(f"source reset scene pair changed: {cell}")
            target_scenes = _valid_scene_pair(
                target, target_obj, mode, 12, i, j)
            source_scene = source_json[0]
            target_scene = json.loads(target_scenes[0].read_text(
                encoding="utf-8"))
            if not _same_numeric_record(
                    _normalized_scene(
                        source_scene, object_dir=origin_obj,
                        key_object=mode.key_object),
                    _normalized_scene(
                        target_scene, object_dir=target_obj,
                        key_object=mode.key_object)):
                raise ValueError(f"reset scene geometry changed: {cell}")
            folder = candidate_root / cell
            seeds = sorted(
                (path for path in folder.iterdir()
                 if path.is_dir() and path.name.isdigit()),
                key=lambda path: int(path.name))
            if not seeds:
                raise FileNotFoundError(f"installed reset cell is empty: {folder}")
            for seed in seeds:
                current_path = seed / "source_evidence.json"
                current = json.loads(current_path.read_text(encoding="utf-8"))
                target_hashes = {
                    "bodex": _sha(target_scenes[0]),
                    "sim_filter": _sha(target_scenes[1]),
                }
                if (current.get("scene_sha256") == target_hashes and
                        current.get("relocation", {}).get("schema") ==
                        "precision_insertion_installed_reset_rebind_v1"):
                    _candidate_arrays(
                        seed, mode=mode, cell=cell, h_cm=12,
                        scenes=target_scenes)
                    rows.append({
                        "cell": cell, "seed_id": seed.name,
                        "status": "verified_rebound",
                        "source_evidence_sha256": current["relocation"][
                            "source_evidence_sha256"],
                        "target_evidence_sha256": _sha(current_path),
                    })
                    continue
                _candidate_arrays(
                    seed, mode=mode, cell=cell, h_cm=12,
                    scenes=source_scenes)
                source_evidence_sha = _sha(current_path)
                rebound = dict(current)
                rebound["scene_sha256"] = target_hashes
                rebound["relocation"] = {
                    "schema": "precision_insertion_installed_reset_rebind_v1",
                    "source_evidence_sha256": source_evidence_sha,
                    "source_shared_root": str(source),
                    "source_origin_shared_root": str(origin),
                    "target_shared_root": str(target),
                    "source_scene_sha256": {
                        "bodex": _sha(source_scenes[0]),
                        "sim_filter": _sha(source_scenes[1]),
                    },
                    "scene_geometry_equivalent_after_path_normalization": True,
                    "robot_ready": False,
                }
                staged = temporary_root / cell / seed.name
                staged.mkdir(parents=True)
                for name in GRASP_FILES:
                    shutil.copyfile(seed / name, staged / name)
                (staged / "source_evidence.json").write_text(
                    json.dumps(rebound, indent=2, sort_keys=True,
                               allow_nan=False) + "\n",
                    encoding="utf-8")
                _candidate_arrays(
                    staged, mode=mode, cell=cell, h_cm=12,
                    scenes=target_scenes)
                pending.append((seed, current_path, rebound,
                                source_evidence_sha))

        # All source/target semantic and candidate checks finish before writes.
        for seed, current_path, rebound, source_evidence_sha in pending:
            original = seed / "source_evidence_original.json"
            if original.exists():
                if _sha(original) != source_evidence_sha:
                    raise ValueError(
                        f"retained original reset evidence differs: {original}")
            else:
                shutil.copyfile(current_path, original)
            temporary = seed / "source_evidence.json.rebinding"
            temporary.write_text(
                json.dumps(rebound, indent=2, sort_keys=True,
                           allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(current_path)
            rows.append({
                "cell": rebound["cell"], "seed_id": rebound["seed_id"],
                "status": "verified_rebound",
                "source_evidence_sha256": source_evidence_sha,
                "target_evidence_sha256": _sha(current_path),
            })

        # Reverify the canonical installed form after all atomic replacements.
        for row in rows:
            cell = row["cell"]
            i, j = (int(part) for part in cell.split("_"))
            scenes = _valid_scene_pair(target, target_obj, mode, 12, i, j)
            _candidate_arrays(
                candidate_root / cell / row["seed_id"], mode=mode,
                cell=cell, h_cm=12, scenes=scenes)
        report = {
            "schema": "precision_insertion_installed_reset_relocation_v1",
            "source_shared_root": str(source),
            "source_origin_shared_root": str(origin),
            "target_shared_root": str(target),
            "candidate_root": str(candidate_root),
            "seeds": sorted(rows, key=lambda row: (row["cell"],
                                                     int(row["seed_id"]))),
            "scope": "host_path_rebind_not_new_grasp_or_robot_motion_permission",
            "robot_ready": False,
        }
        manifest = candidate_root.parents[1] / "RESET_SCENE_RELOCATION.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved != report:
                raise ValueError(f"installed reset relocation manifest differs: {manifest}")
        else:
            manifest.write_text(
                json.dumps(report, indent=2, sort_keys=True,
                           allow_nan=False) + "\n", encoding="utf-8")
        return report
    finally:
        shutil.rmtree(temporary_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-shared-root", type=Path, required=True)
    parser.add_argument("--source-origin-shared-root", type=Path, required=True)
    parser.add_argument("--target-shared-root", type=Path, required=True)
    args = parser.parse_args()
    report = rebind_installed(
        source_shared_root=args.source_shared_root,
        source_origin_shared_root=args.source_origin_shared_root,
        target_shared_root=args.target_shared_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
