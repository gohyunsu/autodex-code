"""Proxy staging and full-key reset-filter evidence stay separate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_cylinder_reorient_pilot import audit  # noqa: E402
from promote_v8_reset_candidates import promote  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.reorient_assets import audit_v8_reorient_assets  # noqa: E402
from precision_insertion.reset_candidates import load_v8_reset_seeds  # noqa: E402
from stage_cylinder_reorient_proposals import KEY, PAIRS, PROXY, stage  # noqa: E402


def _stage_fixture(root: Path, *, expected: int = 2) -> tuple[Path, Path]:
    raw = root / "raw"
    scene_root = root / "AutoDex/scene/inspire"
    for obj in (PROXY, KEY):
        mesh = root / "object_processing" / obj / "processed_data/mesh/simplified.obj"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("o test\n")
        urdf = root / "object_processing" / obj / "processed_data/urdf/coacd.urdf"
        urdf.parent.mkdir(parents=True)
        urdf.write_text("<robot name='test'/>\n")
        for cell in PAIRS:
            i, j = map(int, cell.split("_"))
            scene = {
                "meta": {"scene_type": "reorient_12", "pose_i": f"{i:03d}",
                         "pose_j": f"{j:03d}", "h": 0.12,
                         "thickness": 0.01, "version": "v8"},
                "scene": {
                    "mesh": {"target": {"file_path": str(mesh),
                                        "urdf_path": str(urdf),
                                        "pose": [0, 0, 0, 1, 0, 0, 0],
                                        "scale": [1, 1, 1]}},
                    "cuboid": {"table_i": {}, "table_j": {}},
                },
            }
            path = scene_root / obj / "reorient_12" / f"{cell}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(scene))
            if obj == KEY:
                bodex = (root / "object_processing" / KEY / "scene" /
                         "reorient_12" / f"{cell}.json")
                bodex.parent.mkdir(parents=True, exist_ok=True)
                bodex.write_text(json.dumps(scene))
    tabletop = (root / "object_processing" / KEY /
                "processed_data/info/tabletop")
    tabletop.mkdir(parents=True)
    for pose in range(2):
        np.save(tabletop / f"{pose:03d}.npy", np.eye(4))
    raw_mesh = root / "object_processing" / KEY / "raw_mesh" / f"{KEY}.obj"
    raw_mesh.parent.mkdir()
    raw_mesh.write_text("o test\n")
    key_info = (root / "object_processing" / KEY /
                "processed_data/info/simplified.json")
    key_info.write_text(json.dumps({"obb": [0.03, 0.03, 0.08]}))
    for cell in PAIRS:
        for seed in range(expected):
            seed_dir = raw / PROXY / "reorient_12" / cell / str(seed)
            seed_dir.mkdir(parents=True)
            np.save(seed_dir / "wrist_se3.npy", np.eye(4))
            np.save(seed_dir / "pregrasp_pose.npy", np.zeros(6))
            np.save(seed_dir / "grasp_pose.npy", np.ones(6))
            np.save(seed_dir / "bodex_info.npy", np.zeros(1))
    return raw, root / "staged"


def test_stage_and_audit_stock_filter_results(tmp_path):
    raw, staged = _stage_fixture(tmp_path)
    manifest = stage(raw_root=raw, stage_root=staged,
                     shared_root=tmp_path, expected_per_cell=2)
    assert manifest["total_raw_proposals"] == 4
    assert manifest["robot_ready"] is False
    with pytest.raises(FileExistsError):
        stage(raw_root=raw, stage_root=staged,
              shared_root=tmp_path, expected_per_cell=2)

    outcomes = {
        ("0_1", 0): (True, None, True),
        ("0_1", 1): (False, "scene_collision", False),
        ("1_0", 0): (True, "no_object_contact", False),
        ("1_0", 1): (True, None, False),
    }
    for (cell, seed), (clear, reason, success) in outcomes.items():
        seed_dir = staged / KEY / "reorient_12" / cell / str(seed)
        np.save(seed_dir / "coll_valid.npy", clear)
        result = {"hand": "inspire", "version": "v8", "success": success}
        if reason is not None:
            result["reason"] = reason
        (seed_dir / "sim_eval.json").write_text(json.dumps(result))
    output = tmp_path / "audit.json"
    report = audit(stage_root=staged, output=output)
    assert report["totals"] == {
        "raw": 4, "scene_clear": 3, "squeeze_contact": 2,
        "mujoco_stable": 1, "scene_collision": 1,
        "no_object_contact": 1, "mujoco_unstable": 1,
    }
    assert report["cells"][0]["mujoco_stable_seed_ids"] == [0]
    assert report["robot_ready"] is False
    with pytest.raises(FileExistsError):
        audit(stage_root=staged, output=output)


def test_audit_rejects_incomplete_or_inconsistent_filter(tmp_path):
    raw, staged = _stage_fixture(tmp_path, expected=1)
    stage(raw_root=raw, stage_root=staged,
          shared_root=tmp_path, expected_per_cell=1)
    with pytest.raises(FileNotFoundError):
        audit(stage_root=staged, output=tmp_path / "audit.json")
    for cell in PAIRS:
        seed_dir = staged / KEY / "reorient_12" / cell / "0"
        np.save(seed_dir / "coll_valid.npy", False)
        (seed_dir / "sim_eval.json").write_text(json.dumps({
            "hand": "inspire", "version": "v8", "success": True,
            "reason": "scene_collision"}))
    with pytest.raises(ValueError, match="inconsistent collision"):
        audit(stage_root=staged, output=tmp_path / "audit.json")
    assert not (tmp_path / "audit.json").exists()


def test_promotes_only_stock_mujoco_pass_and_loads_direct_v8_cell(tmp_path):
    raw, staged = _stage_fixture(tmp_path, expected=1)
    raw_wrist = np.eye(4)
    raw_wrist[0, 3] = 0.02
    np.save(raw / PROXY / "reorient_12/0_1/0/wrist_se3.npy", raw_wrist)
    stage(raw_root=raw, stage_root=staged,
          shared_root=tmp_path, expected_per_cell=1)
    stock = tmp_path / "stock_candidates"
    for cell in PAIRS:
        seed = staged / KEY / "reorient_12" / cell / "0"
        success = cell == "0_1"
        np.save(seed / "coll_valid.npy", success)
        result = {"hand": "inspire", "version": "v8", "success": success}
        if not success:
            result["reason"] = "scene_collision"
        (seed / "sim_eval.json").write_text(json.dumps(result))
        if success:
            qpose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            moved = [0.001, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            (seed / "sim_traj.json").write_text(json.dumps({
                "phase": ["pregrasp", "squeeze", "force_gravity"],
                "object_pose": [qpose, moved, moved],
                "robot_qpos": [qpose + [0.0] * 6] * 3,
            }))
            stock_seed = stock / KEY / "reorient_12" / cell / "0"
            stock_seed.mkdir(parents=True)
            for name in ("wrist_se3.npy", "pregrasp_pose.npy",
                         "grasp_pose.npy", "bodex_info.npy"):
                (stock_seed / name).write_bytes((seed / name).read_bytes())
    report_path = tmp_path / "audit.json"
    audit(stage_root=staged, output=report_path)
    promotion_path = tmp_path / "promotion.json"
    stock_wrist = stock / KEY / "reorient_12/0_1/0/wrist_se3.npy"
    np.save(stock_wrist, np.eye(4))
    with pytest.raises(ValueError, match="stock/raw reset proposal mismatch"):
        promote(shared_root=tmp_path, stage_root=staged,
                stock_candidate_root=stock, audit_path=report_path,
                output_manifest=promotion_path)
    assert not promotion_path.exists()
    assert not (tmp_path / "AutoDex/candidates/inspire/reset_12" /
                KEY / "reorient_12/0_1/0").exists()
    stock_wrist.write_bytes((staged / KEY /
                             "reorient_12/0_1/0/wrist_se3.npy").read_bytes())
    promoted = promote(
        shared_root=tmp_path, stage_root=staged,
        stock_candidate_root=stock, audit_path=report_path,
        output_manifest=promotion_path)
    assert promoted["promoted_count"] == 1
    assert promoted["robot_ready"] is False
    handoff_root = tmp_path / "handoff/reset_12"
    handoff = promote(
        shared_root=tmp_path, stage_root=staged,
        stock_candidate_root=stock, audit_path=report_path,
        output_manifest=tmp_path / "handoff_promotion.json",
        output_candidate_root=handoff_root)
    assert handoff["installed_in_canonical_reset_tree"] is False
    mode = select_mode("cylinder", 20)
    asset_audit = audit_v8_reorient_assets(shared_root=tmp_path, mode=mode)
    assert asset_audit["schema"].endswith("v4")
    assert asset_audit["directed_pairs"][0][
        "stable_reset_seed_counts_by_height_cm"]["12"] == 1
    handoff_audit = audit_v8_reorient_assets(
        shared_root=tmp_path, mode=mode,
        candidate_root=handoff_root.parent)
    assert handoff_audit["candidate_root_is_canonical"] is False
    assert handoff_audit["directed_pairs"][0][
        "stable_reset_seed_counts_by_height_cm"]["12"] == 1
    T_robot_key = np.eye(4)
    T_robot_key[:3, 3] = [0.5, 0.1, 0.2]
    seeds = load_v8_reset_seeds(
        shared_root=tmp_path, mode=mode, height_cm=12,
        from_pose_stem="000", to_pose_stem="001", T_robot_key=T_robot_key,
        max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=8.0)
    assert seeds["n_total"] == 1
    assert np.allclose(seeds["wrist_se3"][0], T_robot_key @ raw_wrist)
    assert seeds["scene_info"][0]["v8_cell"] == "0_1"
    assert seeds["robot_ready"] is False
    assert load_v8_reset_seeds(
        shared_root=tmp_path, mode=mode, height_cm=12,
        from_pose_stem=0, to_pose_stem=1, T_robot_key=T_robot_key,
        max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=8.0,
        candidate_root=handoff_root)["n_total"] == 1
    assert load_v8_reset_seeds(
        shared_root=tmp_path, mode=mode, height_cm=12,
        from_pose_stem=1, to_pose_stem=0, T_robot_key=T_robot_key,
        max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=8.0) is None
    assert load_v8_reset_seeds(
        shared_root=tmp_path, mode=mode, height_cm=12,
        from_pose_stem=0, to_pose_stem=1, T_robot_key=T_robot_key,
        max_center_in_hand_drift_m=0.003,
        max_symmetry_axis_tilt_deg=8.0,
        attempted_ids=("000",)) is None
    assert load_v8_reset_seeds(
        shared_root=tmp_path, mode=mode, height_cm=12,
        from_pose_stem=0, to_pose_stem=1, T_robot_key=T_robot_key,
        max_center_in_hand_drift_m=0.0005,
        max_symmetry_axis_tilt_deg=8.0) is None
    candidate = (tmp_path / "AutoDex/candidates/inspire/reset_12" / KEY /
                 "reorient_12/0_1/0/grasp_pose.npy")
    np.save(candidate, np.zeros(6))
    with pytest.raises(ValueError, match="changed reset candidate file"):
        load_v8_reset_seeds(
            shared_root=tmp_path, mode=mode, height_cm=12,
            from_pose_stem=0, to_pose_stem=1, T_robot_key=T_robot_key,
            max_center_in_hand_drift_m=0.003,
            max_symmetry_axis_tilt_deg=8.0)
