"""Curated reset relocation binds the same seeds to target-host v8 scenes."""

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
from precision_insertion.curated_reorient_relocation import (  # noqa: E402
    rehydrate_curated_reorient,
)
from precision_insertion.grasp_fidelity import (  # noqa: E402
    trajectory_closure_audit, trajectory_rigid_closure_audit,
)
from precision_insertion.reorient_assets import (  # noqa: E402
    _scene_path, _sim_filter_scene_path, prepare_v8_reorient_scenes,
)
from precision_insertion.reset_candidates import (  # noqa: E402
    EVIDENCE_SCHEMA, GRASP_FILES, verify_v8_reset_seed,
)
from precision_insertion.reorient_source_snapshot import (  # noqa: E402
    snapshot_curated_reorient_source,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(tmp_path: Path, family: str):
    gap = 1.5 if family == "square" else 20
    mode = select_mode(family, gap)
    i, j = (0, 4) if family == "square" else (0, 1)
    source = tmp_path / "source"
    target = tmp_path / "target"
    target.mkdir()
    obj = source / "object_processing" / mode.key_object
    tabletop = obj / "processed_data/info/tabletop"
    tabletop.mkdir(parents=True)
    for index, angle in ((i, 0.), (j, 90.)):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_euler(
            "y", angle, degrees=True).as_matrix()
        np.save(tabletop / f"{index:03d}.npy", pose)
    mesh = (trimesh.creation.box(extents=(.03, .03, .08))
            if family == "square" else
            trimesh.creation.cylinder(radius=.015, height=.08))
    simplified = obj / "processed_data/mesh/simplified.obj"
    simplified.parent.mkdir(parents=True)
    mesh.export(simplified)
    raw = obj / "raw_mesh" / f"{mode.key_object}.obj"
    raw.parent.mkdir()
    mesh.export(raw)
    info = obj / "processed_data/info/simplified.json"
    info.write_text(json.dumps({
        "obb": [.03, .03, .08], "gravity_center": [0., 0., 0.]}))
    urdf = obj / "processed_data/urdf/coacd.urdf"
    urdf.parent.mkdir()
    urdf.write_text("<robot name='test'/>")
    prepare_v8_reorient_scenes(
        shared_root=source, mode=mode,
        manifest_path=tmp_path / "scene_manifest.json", heights_cm=(12,))
    shutil.copytree(obj, target / "object_processing" / mode.key_object,
                    ignore=shutil.ignore_patterns("scene"))
    handoff = tmp_path / "curated"
    seed = (handoff / "reset_12" / mode.key_object /
            "reorient_12" / f"{i}_{j}" / "7")
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
    scenes = (_scene_path(obj, 12, i, j),
              _sim_filter_scene_path(source, mode, 12, i, j))
    fidelity = (trajectory_rigid_closure_audit(
        trajectory, key_center_local_m=np.zeros(3)) if family == "square"
        else trajectory_closure_audit(trajectory, key_height_m=.08))
    evidence = {
        "schema": EVIDENCE_SCHEMA, "full_key_object": mode.key_object,
        "height_cm": 12, "cell": f"{i}_{j}", "seed_id": "7",
        "candidate_sha256": {name: _sha(seed / name) for name in GRASP_FILES},
        "scene_sha256": {"bodex": _sha(scenes[0]),
                         "sim_filter": _sha(scenes[1])},
        "key_mesh_sha256": _sha(simplified),
        "key_info_sha256": _sha(info),
        "post_squeeze_fidelity": fidelity,
        "robot_ready": False,
    }
    if family == "square":
        evidence["key_center_local_m"] = [0., 0., 0.]
    else:
        evidence["key_height_m"] = .08
    (seed / "source_evidence.json").write_text(json.dumps(evidence))
    manifest = {
        "schema": "precision_insertion_v8_reorient_handoff_v1",
        "mode": {"family": family, "gap_mm": gap,
                 "key_object": mode.key_object},
        "shared_root": str(source), "output_root": str(handoff),
        "candidate_count": 1,
        "cells": [{"height_cm": 12, "from_v8_pose": i,
                   "to_v8_pose": j, "verified_seed_ids": ["7"]}],
        "candidates": [{"candidate_dir": str(seed),
                        "source_evidence_sha256": _sha(
                            seed / "source_evidence.json")}],
    }
    (handoff / "handoff_manifest.json").write_text(json.dumps(manifest))
    output = target / "AutoDex/precision_insertion/staged_reorient"
    return source, handoff, target, output, mode, seed, (i, j)


@pytest.mark.parametrize("family", ["square", "cylinder"])
def test_rebinds_curated_seed_and_keeps_original_evidence(tmp_path, family):
    source, handoff, target, output, mode, seed, (i, j) = _case(
        tmp_path, family)
    report = rehydrate_curated_reorient(
        source_shared_root=source, source_handoff_root=handoff,
        target_shared_root=target, target_handoff_root=output, mode=mode)
    assert report["robot_ready"] is False
    assert len(report["seeds"]) == 1
    relocated = output / "reset_12" / mode.key_object / (
        f"reorient_12/{i}_{j}/7")
    assert (relocated / "source_evidence_original.json").read_bytes() == (
        seed / "source_evidence.json").read_bytes()
    verified = verify_v8_reset_seed(
        shared_root=target, mode=mode, height_cm=12,
        from_pose_stem=i, to_pose_stem=j, seed_id="7",
        candidate_root=output / "reset_12")
    assert verified["robot_ready"] is False
    with pytest.raises(ValueError, match="non-overwriting"):
        rehydrate_curated_reorient(
            source_shared_root=source, source_handoff_root=handoff,
            target_shared_root=target, target_handoff_root=output, mode=mode)


def test_changed_key_or_candidate_fails_before_writes(tmp_path):
    source, handoff, target, output, mode, seed, _ = _case(
        tmp_path, "square")
    received = (target / "object_processing" / mode.key_object /
                "processed_data/mesh/simplified.obj")
    received.write_bytes(received.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="recipient key asset"):
        rehydrate_curated_reorient(
            source_shared_root=source, source_handoff_root=handoff,
            target_shared_root=target, target_handoff_root=output, mode=mode)
    assert not output.exists()
    shutil.copyfile(source / "object_processing" / mode.key_object /
                    "processed_data/mesh/simplified.obj", received)
    (seed / "grasp_pose.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed reset candidate"):
        rehydrate_curated_reorient(
            source_shared_root=source, source_handoff_root=handoff,
            target_shared_root=target, target_handoff_root=output, mode=mode)
    assert not output.exists()


def test_source_bound_target_scene_requires_explicit_archive(tmp_path):
    source, handoff, target, output, mode, _, (i, j) = _case(
        tmp_path, "cylinder")
    src_scene = _scene_path(source / "object_processing" / mode.key_object,
                            12, i, j)
    dst_scene = _scene_path(target / "object_processing" / mode.key_object,
                            12, i, j)
    dst_scene.parent.mkdir(parents=True)
    shutil.copyfile(src_scene, dst_scene)
    with pytest.raises(ValueError, match="recipient reset scene differs"):
        rehydrate_curated_reorient(
            source_shared_root=source, source_handoff_root=handoff,
            target_shared_root=target, target_handoff_root=output, mode=mode)
    assert not output.exists()
    report = rehydrate_curated_reorient(
        source_shared_root=source, source_handoff_root=handoff,
        target_shared_root=target, target_handoff_root=output, mode=mode,
        archive_source_bound_scenes=True)
    assert len(report["archived_source_bound_scenes"]) == 1
    assert _sha(Path(report["archived_source_bound_scenes"][0]["archive"])) == (
        _sha(src_scene))


@pytest.mark.parametrize("family", ["square", "cylinder"])
def test_snapshot_survives_source_tree_relocation(tmp_path, family):
    source, handoff, target, output, mode, _, _ = _case(tmp_path, family)
    snapshot = tmp_path / "portable_source_shared_data"
    bundled = snapshot_curated_reorient_source(
        shared_root=source, handoff_root=handoff,
        snapshot_root=snapshot, mode=mode)
    assert bundled["verified_seed_count"] == 1
    assert bundled["robot_ready"] is False
    report = rehydrate_curated_reorient(
        source_shared_root=snapshot, source_handoff_root=handoff,
        target_shared_root=target, target_handoff_root=output, mode=mode)
    assert len(report["seeds"]) == 1
    assert report["robot_ready"] is False


def test_changed_portable_snapshot_fails_before_target_writes(tmp_path):
    source, handoff, target, output, mode, _, _ = _case(tmp_path, "square")
    snapshot = tmp_path / "portable_source_shared_data"
    snapshot_curated_reorient_source(
        shared_root=source, handoff_root=handoff,
        snapshot_root=snapshot, mode=mode)
    scene = _scene_path(snapshot / "object_processing" / mode.key_object,
                        12, 0, 4)
    scene.write_bytes(scene.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="portable source snapshot file changed"):
        rehydrate_curated_reorient(
            source_shared_root=snapshot, source_handoff_root=handoff,
            target_shared_root=target, target_handoff_root=output, mode=mode)
    assert not output.exists()
