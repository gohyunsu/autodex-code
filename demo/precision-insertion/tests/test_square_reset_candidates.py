"""Square reset seeds retain full yaw error and stock-filter provenance."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.reorient_assets import audit_v8_reorient_assets  # noqa: E402
from precision_insertion.reset_candidates import (  # noqa: E402
    load_v8_reset_seeds, verify_v8_reset_seed,
)
from calibrate_reset_grasp import build_reset_grasp_calibration  # noqa: E402
from stage_square_reorient_passes import stage_passes  # noqa: E402
from test_physical_grasp_calibration import _sample  # noqa: E402


MODE = select_mode("square", 1.5)
CELL = "0_1"


def _fixture(root: Path) -> tuple[Path, Path]:
    object_dir = root / "object_processing" / MODE.key_object
    tabletop = object_dir / "processed_data/info/tabletop"
    tabletop.mkdir(parents=True)
    for index in (0, 1):
        np.save(tabletop / f"{index:03d}.npy", np.eye(4))
    mesh = object_dir / "processed_data/mesh/simplified.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("o key\n", encoding="utf-8")
    raw_mesh = object_dir / "raw_mesh" / f"{MODE.key_object}.obj"
    raw_mesh.parent.mkdir()
    raw_mesh.write_text("o key\n", encoding="utf-8")
    info = object_dir / "processed_data/info/simplified.json"
    info.write_text(json.dumps({"gravity_center": [0., 0., 0.],
                                "obb": [.02, .02, .06]}), encoding="utf-8")
    urdf = object_dir / "processed_data/urdf/coacd.urdf"
    urdf.parent.mkdir()
    urdf.write_text("<robot name='key'/>\n", encoding="utf-8")
    scene = {
        "meta": {"scene_type": "reorient_12", "pose_i": "000",
                 "pose_j": "001", "h": .12, "thickness": .01,
                 "version": "v8"},
        "scene": {"mesh": {"target": {
            "file_path": str(mesh), "urdf_path": str(urdf),
            "pose": [0, 0, 0, 1, 0, 0, 0], "scale": [1, 1, 1],
        }}, "cuboid": {"table_i": {}, "table_j": {}}},
    }
    bodex_scene = object_dir / "scene/reorient_12" / f"{CELL}.json"
    bodex_scene.parent.mkdir(parents=True)
    bodex_scene.write_text(json.dumps(scene), encoding="utf-8")
    sim_scene = (root / "AutoDex/scene/inspire" / MODE.key_object /
                 "reorient_12" / f"{CELL}.json")
    sim_scene.parent.mkdir(parents=True)
    sim_scene.write_text(json.dumps(scene), encoding="utf-8")

    raw = root / "raw"
    stock = root / "stock"
    initial = [0., 0., 0., 1., 0., 0., 0.]
    yaw10 = [0., 0., 0., np.cos(np.deg2rad(5.)), 0., 0.,
             np.sin(np.deg2rad(5.))]
    trajectory = {
        "phase": ["pregrasp", "squeeze", "force_gravity"],
        "object_pose": [initial, yaw10, yaw10],
        "robot_qpos": [initial + [0.] * 6] * 3,
    }
    for seed_id in (0, 1):
        seed = raw / MODE.key_object / "reorient_12" / CELL / str(seed_id)
        seed.mkdir(parents=True)
        np.save(seed / "wrist_se3.npy", np.eye(4))
        np.save(seed / "pregrasp_pose.npy", np.zeros(6))
        np.save(seed / "grasp_pose.npy", np.ones(6) * .1)
        np.save(seed / "bodex_info.npy", np.zeros(2))
        np.save(seed / "coll_valid.npy", seed_id == 0)
        (seed / "sim_eval.json").write_text(json.dumps({
            "success": seed_id == 0, "hand": "inspire", "version": "v8",
            **({} if seed_id == 0 else {"reason": "scene_collision"}),
        }), encoding="utf-8")
        if seed_id == 0:
            (seed / "sim_traj.json").write_text(json.dumps(trajectory),
                                                encoding="utf-8")
            target = stock / MODE.key_object / "reorient_12" / CELL / "0"
            target.mkdir(parents=True)
            for name in ("wrist_se3.npy", "pregrasp_pose.npy",
                         "grasp_pose.npy", "bodex_info.npy"):
                shutil.copy2(seed / name, target / name)
    return raw, stock


def test_square_reset_stage_and_full_yaw_gate(tmp_path):
    raw, stock = _fixture(tmp_path)
    output = tmp_path / "handoff"
    result = stage_passes(
        shared_root=tmp_path, gap_mm=1.5, raw_root=raw,
        stock_candidate_root=stock, cell=CELL, expected_seed_count=2,
        output_root=output)
    assert result["counts"] == {
        "raw": 2, "scene_clear": 1, "squeeze_contact": 1,
        "mujoco_stable": 1, "scene_collision": 1,
        "no_object_contact": 0, "mujoco_unstable": 0,
    }
    fidelity = result["stable_seeds"][0]["post_squeeze_fidelity"]
    assert fidelity["end_squeeze"]["full_relative_rotation_deg"] == pytest.approx(10.)
    assert fidelity["end_squeeze"]["center_in_hand_displacement_m"] == 0.
    audit = audit_v8_reorient_assets(
        shared_root=tmp_path, mode=MODE, candidate_root=output,
        max_center_in_hand_drift_m=.003,
        max_symmetry_axis_tilt_deg=5.)
    row = next(row for row in audit["directed_pairs"] if
               (row["from_v8_pose"], row["to_v8_pose"]) == (0, 1))
    assert row["stable_reset_seed_counts_by_height_cm"]["12"] == 1
    assert row["fidelity_eligible_seed_counts_by_height_cm"]["12"] == 0
    assert audit["reset_fidelity_rotation_measure"] == "full_relative_rotation_deg"
    assert load_v8_reset_seeds(
        shared_root=tmp_path, mode=MODE, height_cm=12,
        from_pose_stem=0, to_pose_stem=1, T_robot_key=np.eye(4),
        max_center_in_hand_drift_m=.003,
        max_symmetry_axis_tilt_deg=5.,
        candidate_root=output / "reset_12") is None
    admitted = load_v8_reset_seeds(
        shared_root=tmp_path, mode=MODE, height_cm=12,
        from_pose_stem=0, to_pose_stem=1, T_robot_key=np.eye(4),
        max_center_in_hand_drift_m=.003,
        max_symmetry_axis_tilt_deg=15.,
        candidate_root=output / "reset_12")
    assert admitted["n_total"] == 1
    assert admitted["fidelity_limits"]["rotation_measure"] == (
        "full_relative_rotation_deg")
    with pytest.raises(FileExistsError):
        stage_passes(
            shared_root=tmp_path, gap_mm=1.5, raw_root=raw,
            stock_candidate_root=stock, cell=CELL, expected_seed_count=2,
            output_root=output)


def test_square_reset_stage_rejects_changed_stock_copy_before_writes(tmp_path):
    raw, stock = _fixture(tmp_path)
    np.save(stock / MODE.key_object / "reorient_12" / CELL / "0" /
            "grasp_pose.npy", np.zeros(6))
    output = tmp_path / "handoff"
    with pytest.raises(ValueError, match="stock candidate differs"):
        stage_passes(
            shared_root=tmp_path, gap_mm=1.5, raw_root=raw,
            stock_candidate_root=stock, cell=CELL, expected_seed_count=2,
            output_root=output)
    assert not output.exists()


def test_reset_physical_calibration_reuses_v8_seed_and_real_sample_contract(
        tmp_path):
    raw, stock = _fixture(tmp_path)
    handoff = tmp_path / "handoff"
    stage_passes(
        shared_root=tmp_path, gap_mm=1.5, raw_root=raw,
        stock_candidate_root=stock, cell=CELL, expected_seed_count=2,
        output_root=handoff)
    candidate_root = handoff / "reset_12"
    selected = verify_v8_reset_seed(
        shared_root=tmp_path, mode=MODE, height_cm=12,
        from_pose_stem="000", to_pose_stem="001", seed_id="0",
        candidate_root=candidate_root)
    assert selected["candidate_key"] == ("reset", "0_1", "0")
    assert selected["robot_ready"] is False
    assert selected["fidelity"]["end_squeeze"][
        "full_relative_rotation_deg"] == pytest.approx(10.)
    with pytest.raises(ValueError, match="exact reset height"):
        verify_v8_reset_seed(
            shared_root=tmp_path, mode=MODE, height_cm=12,
            from_pose_stem=0, to_pose_stem=1, seed_id="0",
            candidate_root=handoff)
    samples = [_sample(tmp_path, index, selected["candidate_key"])
               for index in range(5)]
    paths = [Path(sample["evidence_path"]) for sample in samples]
    output = tmp_path / "physical_reset_calibration"
    built = build_reset_grasp_calibration(
        shared_root=tmp_path, family="square", gap_mm=1.5,
        candidate_root=candidate_root, height_cm=12,
        from_pose_stem="000", to_pose_stem="001", seed_id="0",
        sample_paths=paths, minimum_independent_trials=5,
        max_nominal_translation_drift_m=.003,
        max_nominal_rotation_drift_deg=15., output_dir=output)
    assert built == output
    summary = json.loads((built / "physical_grasp_calibration.json")
                         .read_text(encoding="utf-8"))
    binding = json.loads((built / "binding.json").read_text(
        encoding="utf-8"))
    assert summary["candidate_key"] == ["reset", "0_1", "0"]
    assert summary["sample_count"] == 5
    assert binding["robot_ready"] is False
    with pytest.raises(FileExistsError):
        build_reset_grasp_calibration(
            shared_root=tmp_path, family="square", gap_mm=1.5,
            candidate_root=candidate_root, height_cm=12,
            from_pose_stem="000", to_pose_stem="001", seed_id="0",
            sample_paths=paths, minimum_independent_trials=5,
            max_nominal_translation_drift_m=.003,
            max_nominal_rotation_drift_deg=15., output_dir=output)
    with pytest.raises(ValueError, match="cannot be reused"):
        build_reset_grasp_calibration(
            shared_root=tmp_path, family="square", gap_mm=1.5,
            candidate_root=candidate_root, height_cm=12,
            from_pose_stem="000", to_pose_stem="001", seed_id="0",
            sample_paths=[paths[0]] * 5, minimum_independent_trials=5,
            max_nominal_translation_drift_m=.003,
            max_nominal_rotation_drift_deg=15.,
            output_dir=tmp_path / "must_not_exist")
    assert not (tmp_path / "must_not_exist").exists()
