"""Add immutable source key/scenes beside a curated NAS reset handoff."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .config import TaskMode
from .reorient_assets import _scene_path, _sim_filter_scene_path
from .reset_candidates import _candidate_arrays, verify_v8_reset_seed


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_curated_reorient_source(
    *, shared_root: Path, handoff_root: Path, snapshot_root: Path,
    mode: TaskMode,
) -> dict:
    """Copy exact source assets to a new root; never rewrite scene paths.

    The output is a *source* tree for host relocation, not the active target
    object-processing directory. Every listed stable seed verifies against
    the original v8 loader before and after the snapshot copy.
    """
    source = Path(shared_root).expanduser().resolve()
    handoff = Path(handoff_root).expanduser().resolve()
    output_input = Path(snapshot_root).expanduser()
    if not output_input.is_absolute():
        raise ValueError("source snapshot output must be absolute")
    output = output_input.resolve()
    canonical = source / "AutoDex/candidates"
    if (not source.is_dir() or not handoff.is_dir() or output.exists() or
            output.is_relative_to(canonical.resolve()) or
            output_input.absolute().is_relative_to(canonical) or
            output.is_relative_to(handoff / "reset_12") or
            output.is_relative_to(source / "object_processing") or
            output.is_relative_to(source / "AutoDex/scene")):
        raise ValueError("source snapshot must be a new noncanonical directory")
    manifest_path = handoff / "handoff_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "precision_insertion_v8_reorient_handoff_v1" or
            manifest.get("shared_root") != str(source) or
            manifest.get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm,
                "key_object": mode.key_object}):
        raise ValueError("curated source snapshot identity differs")
    object_dir = source / "object_processing" / mode.key_object
    scene_dir = source / "AutoDex/scene/inspire" / mode.key_object
    if not object_dir.is_dir() or not scene_dir.is_dir():
        raise FileNotFoundError("source key object or reset scene mirror is missing")
    for tree in (object_dir, scene_dir):
        if tree.is_symlink() or any(path.is_symlink() for path in tree.rglob("*")):
            raise ValueError("source key or reset scene tree contains a symlink")
    cells = []
    count = 0
    for row in manifest["cells"]:
        h, i, j = (row["height_cm"], row["from_v8_pose"],
                   row["to_v8_pose"])
        if type(h) is not int or h != 12 or i == j:
            raise ValueError("invalid curated source reset cell")
        ids = row["verified_seed_ids"]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("curated source reset cell has no unique seeds")
        for seed_id in ids:
            verify_v8_reset_seed(
                shared_root=source, mode=mode, height_cm=h,
                from_pose_stem=i, to_pose_stem=j, seed_id=seed_id,
                candidate_root=handoff / f"reset_{h}")
            count += 1
        cells.append((h, i, j, list(ids)))
    if count != manifest.get("candidate_count"):
        raise ValueError("curated source seed count differs")
    output.mkdir(parents=True, exist_ok=False)
    copied_obj = output / "object_processing" / mode.key_object
    copied_scene = output / "AutoDex/scene/inspire" / mode.key_object
    shutil.copytree(object_dir, copied_obj)
    shutil.copytree(scene_dir, copied_scene)
    paths = []
    for tree in (object_dir, scene_dir):
        for original in sorted(tree.rglob("*")):
            if not original.is_file():
                continue
            copied = output / original.relative_to(source)
            if _sha(original) != _sha(copied):
                raise ValueError(f"source snapshot copy changed: {copied}")
            paths.append({"relative_path": str(original.relative_to(source)),
                          "sha256": _sha(copied)})
    for h, i, j, ids in cells:
        scenes = (_scene_path(copied_obj, h, i, j),
                  _sim_filter_scene_path(output, mode, h, i, j))
        for seed_id in ids:
            seed = (handoff / f"reset_{h}" / mode.key_object /
                    f"reorient_{h}" / f"{i}_{j}" / seed_id)
            _candidate_arrays(seed, mode=mode, cell=f"{i}_{j}",
                              h_cm=h, scenes=scenes)
    report = {
        "schema": "precision_insertion_curated_reorient_source_snapshot_v1",
        "mode": manifest["mode"], "source_origin_shared_root": str(source),
        "source_handoff_manifest_sha256": _sha(manifest_path),
        "snapshot_root": str(output), "verified_seed_count": count,
        "files": paths,
        "scope": "source_key_and_scenes_for_host_relocation_not_active_assets",
        "robot_ready": False,
    }
    with (output / "SNAPSHOT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report
