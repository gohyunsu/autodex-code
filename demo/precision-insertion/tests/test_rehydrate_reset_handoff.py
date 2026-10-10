"""Relocating reset seeds must preserve geometry and provenance, not paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.grasp_fidelity import trajectory_closure_audit  # noqa: E402
from precision_insertion.reorient_assets import prepare_v8_reorient_scenes  # noqa: E402
from precision_insertion.reset_candidates import (  # noqa: E402
    EVIDENCE_SCHEMA, GRASP_FILES, _candidate_arrays, _valid_scene_pair,
)
from rehydrate_reset_handoff import rehydrate  # noqa: E402


MODE = select_mode("cylinder", 20)
KEY = MODE.key_object


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    object_dir = source / "object_processing" / KEY
    tabletop = object_dir / "processed_data/info/tabletop"
    tabletop.mkdir(parents=True)
    for index, angle in ((0, 0.0), (1, 90.0)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "y", angle, degrees=True).as_matrix()
        np.save(tabletop / f"{index:03d}.npy", pose)
    mesh = trimesh.creation.cylinder(radius=.015, height=.08)
    simplified = object_dir / "processed_data/mesh/simplified.obj"
    simplified.parent.mkdir(parents=True)
    mesh.export(simplified)
    raw = object_dir / "raw_mesh" / f"{KEY}.obj"
    raw.parent.mkdir()
    mesh.export(raw)
    info = object_dir / "processed_data/info/simplified.json"
    info.write_text(json.dumps({"obb": [.03, .03, .08]}))
    urdf = object_dir / "processed_data/urdf/coacd.urdf"
    urdf.parent.mkdir()
    urdf.write_text("<robot name='test'/>")
    prepare_v8_reorient_scenes(
        shared_root=source, mode=MODE,
        manifest_path=tmp_path / "source_scene_manifest.json",
        heights_cm=(12,))
    for cell in ("0_1", "1_0"):
        mirror = source / "AutoDex/scene/inspire" / KEY / (
            f"reorient_12/{cell}.json")
        scene = json.loads(mirror.read_text())
        scene["meta"].update({"geometry_object": KEY,
                              "grasp_target_object": KEY})
        mirror.write_text(json.dumps(scene))
    shutil.copytree(
        object_dir, target / "object_processing" / KEY,
        ignore=shutil.ignore_patterns("scene"))
    candidate_root = source / "staged/reset_12"
    seed = candidate_root / KEY / "reorient_12/0_1/7"
    seed.mkdir(parents=True)
    np.save(seed / "wrist_se3.npy", np.eye(4))
    np.save(seed / "pregrasp_pose.npy", np.zeros(6))
    np.save(seed / "grasp_pose.npy", np.ones(6) * .1)
    np.save(seed / "bodex_info.npy", np.zeros(2))
    (seed / "sim_eval.json").write_text(json.dumps({
        "success": True, "hand": "inspire", "version": "v8"}))
    pose7 = [0., 0., 0., 1., 0., 0., 0.]
    trajectory = {
        "phase": ["pregrasp", "squeeze", "force_gravity"],
        "object_pose": [pose7] * 3,
        "robot_qpos": [pose7 + [0.] * 6] * 3,
    }
    (seed / "sim_traj.json").write_text(json.dumps(trajectory))
    audit = tmp_path / "source_audit.json"
    audit.write_text('{"scope":"test_source_audit"}')
    scenes = _valid_scene_pair(source, object_dir, MODE, 12, 0, 1)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "full_key_object": KEY, "height_cm": 12,
        "cell": "0_1", "seed_id": "7",
        "candidate_sha256": {name: _sha(seed / name) for name in GRASP_FILES},
        "scene_sha256": {"bodex": _sha(scenes[0]),
                         "sim_filter": _sha(scenes[1])},
        "key_mesh_sha256": _sha(simplified),
        "key_info_sha256": _sha(info),
        "key_height_m": .08,
        "post_squeeze_fidelity": trajectory_closure_audit(
            trajectory, key_height_m=.08),
        "raw_proposal": "source/test/7", "stock_pass": "source/pass/7",
        "source_audit_sha256": _sha(audit),
        "robot_ready": False,
    }
    (seed / "source_evidence.json").write_text(json.dumps(evidence))
    output = (target / "AutoDex/precision_insertion/cylindrical/"
              "reorient_handoff/reset_12")
    return source, target, candidate_root, output, audit, seed


def test_rehydrates_exact_scenes_and_keeps_original_evidence(tmp_path):
    source, target, incoming, output, audit, original = _case(tmp_path)
    result = rehydrate(
        source_shared_root=source, target_shared_root=target,
        source_candidate_root=incoming, target_candidate_root=output,
        source_audit_path=audit)
    assert result["robot_ready"] is False
    assert result["regenerated_scene_count"] == 4
    assert [(row["cell"], row["seed_id"]) for row in result["seeds"]] == [
        ("0_1", "7")]
    relocated = output / KEY / "reorient_12/0_1/7"
    assert (relocated / "source_evidence_original.json").read_bytes() == (
        original / "source_evidence.json").read_bytes()
    assert (output / "SOURCE_AUDIT.json").read_bytes() == audit.read_bytes()
    assert (output / "original_scenes/bodex/0_1.json").read_bytes() == (
        source / "object_processing" / KEY /
        "scene/reorient_12/0_1.json").read_bytes()
    assert json.loads((output / "RELOCATION.json").read_text()) == result
    scenes = _valid_scene_pair(
        target, target / "object_processing" / KEY, MODE, 12, 0, 1)
    _candidate_arrays(relocated, mode=MODE, cell="0_1",
                      h_cm=12, scenes=scenes)
    with pytest.raises(ValueError, match="non-overwriting"):
        rehydrate(
            source_shared_root=source, target_shared_root=target,
            source_candidate_root=incoming, target_candidate_root=output,
            source_audit_path=audit)


def test_rejects_modified_recipient_key_before_writes(tmp_path):
    source, target, incoming, output, audit, _ = _case(tmp_path)
    raw = target / "object_processing" / KEY / "raw_mesh" / f"{KEY}.obj"
    raw.write_bytes(raw.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="recipient key asset"):
        rehydrate(
            source_shared_root=source, target_shared_root=target,
            source_candidate_root=incoming, target_candidate_root=output,
            source_audit_path=audit)
    assert not output.exists()
    assert not (target / "AutoDex/scene/inspire" / KEY).exists()


def test_extracted_handoff_works_without_historical_source_path(tmp_path):
    source, target, _incoming, output, audit, _ = _case(tmp_path)
    extracted = tmp_path / "archive_extracted"
    shutil.copytree(source, extracted)
    shutil.rmtree(source)  # only this test's synthetic former source root
    result = rehydrate(
        source_shared_root=extracted,
        source_origin_shared_root=source,
        target_shared_root=target,
        source_candidate_root=extracted / "staged/reset_12",
        target_candidate_root=output,
        source_audit_path=audit)
    assert result["source_origin_shared_root"] == str(source)
    assert not source.exists()
    assert len(result["seeds"]) == 1


def test_installs_only_missing_key_files_without_overwriting(tmp_path):
    source, target, incoming, output, audit, _ = _case(tmp_path)
    copied_key = target / "object_processing" / KEY
    shutil.rmtree(copied_key)  # only this test's synthetic recipient asset
    result = rehydrate(
        source_shared_root=source, target_shared_root=target,
        source_candidate_root=incoming, target_candidate_root=output,
        source_audit_path=audit, install_missing_key_assets=True)
    assert result["robot_ready"] is False
    assert (copied_key / "processed_data/mesh/simplified.obj").is_file()
    assert not (copied_key / "scene/reorient_0").exists()


def test_rejects_source_bound_scene_already_installed_on_target(tmp_path):
    source, target, incoming, output, audit, _ = _case(tmp_path)
    stale = (target / "object_processing" / KEY / "scene/reorient_12/0_1.json")
    stale.parent.mkdir(parents=True)
    shutil.copyfile(source / "object_processing" / KEY /
                    "scene/reorient_12/0_1.json", stale)
    with pytest.raises(ValueError, match="recipient reset scene changed"):
        rehydrate(
            source_shared_root=source, target_shared_root=target,
            source_candidate_root=incoming, target_candidate_root=output,
            source_audit_path=audit)
    assert not output.exists()


def test_explicitly_archives_only_identical_source_bound_scenes(tmp_path):
    source, target, incoming, output, audit, _ = _case(tmp_path)
    source_scenes = [
        source / "object_processing" / KEY / "scene/reorient_12/0_1.json",
        source / "AutoDex/scene/inspire" / KEY / "reorient_12/0_1.json",
    ]
    for src in source_scenes:
        dest = target / src.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    result = rehydrate(
        source_shared_root=source, target_shared_root=target,
        source_candidate_root=incoming, target_candidate_root=output,
        source_audit_path=audit, archive_source_bound_scenes=True)
    archived = result["archived_source_bound_scenes"]
    assert len(archived) == 2
    for row in archived:
        assert Path(row["archive"]).read_bytes() == (
            source / Path(row["original"]).relative_to(target)).read_bytes()
    assert len(result["seeds"]) == 1


def test_changed_stale_scene_cannot_be_archived(tmp_path):
    source, target, incoming, output, audit, _ = _case(tmp_path)
    stale = target / "object_processing" / KEY / "scene/reorient_12/0_1.json"
    stale.parent.mkdir(parents=True)
    changed = json.loads((source / "object_processing" / KEY /
                          "scene/reorient_12/0_1.json").read_text())
    changed["scene"]["cuboid"]["table_i"]["dims"][0] = 1.0
    stale.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="recipient reset scene changed"):
        rehydrate(
            source_shared_root=source, target_shared_root=target,
            source_candidate_root=incoming, target_candidate_root=output,
            source_audit_path=audit, archive_source_bound_scenes=True)
    assert not output.exists()
