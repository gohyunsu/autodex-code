#!/usr/bin/env python3
"""Create symmetry-reduced cylinder/gripper seed assets.

These records are not sampled BODex grasps.  They are deterministic parallel-
jaw geometry for the two cylindrical tabletop classes.  At runtime the same
construction is repeated from the measured 6D key pose; the files provide a
stable catalog/scene identifier and the canonical 20 mm endpoint screen.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from precision_insertion.analytic_gripper import (
    SIDE_DOWN_WRIST_VARIANTS, UPRIGHT_APPROACH_VARIANTS,
    analytic_franka_gripper_grasp_for_candidate,
)
from precision_insertion.config import select_mode
from precision_insertion.geometry import validate_se3
from autodex.utils.conversion import se32cart


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) +
            "\n").encode("utf-8")


def _write_exact(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"different gripper asset already exists: {path}")
        return
    path.write_bytes(payload)


def _save_npy_exact(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(value)
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        if saved.dtype != array.dtype or not np.array_equal(saved, array):
            raise FileExistsError(f"different gripper asset already exists: {path}")
        return
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)


def build(shared_root: Path) -> dict:
    root = Path(shared_root).expanduser().resolve()
    mode = select_mode("cylinder", 5)
    object_dir = root / "object_processing" / mode.key_object / "processed_data"
    tabletop_dir = object_dir / "info" / "tabletop"
    planning_mesh = object_dir / "mesh" / "simplified.obj"
    object_urdf = object_dir / "urdf" / "coacd.urdf"
    for path in (planning_mesh, object_urdf, tabletop_dir / "000.npy",
                 tabletop_dir / "001.npy"):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_root = (root / "AutoDex" / "candidates" / "franka_gripper" /
                      "v8" / mode.key_object / "table")
    scene_root = (root / "AutoDex" / "scene" / "franka_gripper" /
                  mode.key_object / "table")
    rows = []
    for scene_id, stem in enumerate(("000", "001")):
        key_pose = validate_se3(
            np.load(tabletop_dir / f"{stem}.npy", allow_pickle=False),
            name=f"cylinder tabletop {stem}")
        variant_count = (UPRIGHT_APPROACH_VARIANTS if scene_id == 0 else
                         SIDE_DOWN_WRIST_VARIANTS)
        scene = {
            "scene": {
                "mesh": {"target": {
                    "scale": [1.0, 1.0, 1.0],
                    "pose": se32cart(key_pose).tolist(),
                    "file_path": str(planning_mesh),
                    "urdf_path": str(object_urdf),
                }},
                "cuboid": {"table": {
                    "dims": [2.0, 2.0, 0.2],
                    "pose": [0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],
                }},
            },
            "meta": {
                "pose_idx": stem,
                "param": {"placement": (
                    "end_down" if scene_id == 0 else "side_down")},
                "symmetry_reduced": True,
                "end_effector": "franka_gripper",
            },
        }
        scene_path = scene_root / f"{scene_id}.json"
        _write_exact(scene_path, _json_bytes(scene))
        for variant in range(variant_count):
            identifier = ("table", str(scene_id), f"{variant:03d}")
            grasp = analytic_franka_gripper_grasp_for_candidate(
                key_pose, table_normal_robot=np.asarray([0.0, 0.0, 1.0]),
                identifier=identifier)
            candidate = candidate_root / str(scene_id) / f"{variant:03d}"
            _save_npy_exact(candidate / "wrist_se3.npy",
                            grasp.T_key_hand.astype(np.float64))
            _save_npy_exact(candidate / "pregrasp_pose.npy",
                            grasp.pregrasp_q.astype(np.float64))
            _save_npy_exact(candidate / "grasp_pose.npy",
                            grasp.grasp_q.astype(np.float64))
            contract = {
                "schema": "precision_insertion_franka_gripper_analytic_grasp_v1",
                "key_object": mode.key_object,
                "gripper": "franka_stock_parallel_gripper",
                "tabletop_pose_stem": stem,
                "tabletop_class": grasp.tabletop_class,
                "variant_index": variant,
                "variant_count": variant_count,
                "construction": "measured_6d_pose_deterministic_parallel_jaw",
                "runtime_contract": (
                    "candidate ID selects one symmetry-equivalent grasp; runtime "
                    "reconstructs it from the fresh measured T_robot_key"),
                "symmetry": {
                    "axial_roll": "continuous_equivalence",
                    "end_exchange": "equivalent",
                    "tabletop_classes": ["end_down", "side_down"],
                },
                "physical_validation": False,
                "robot_ready": False,
            }
            # Candidate 000 predates the explicit variant metadata. Keep its
            # byte-identical contract so asset generation remains additive.
            if variant == 0:
                contract.pop("variant_index")
                contract.pop("variant_count")
                contract["runtime_contract"] = (
                    "canonical files are catalog identifiers; runtime reconstructs "
                    "the grasp from the fresh measured T_robot_key")
            _write_exact(candidate / "analytic_grasp.json", _json_bytes(contract))
            rows.append({"scene_id": scene_id, "tabletop_pose_stem": stem,
                         "tabletop_class": grasp.tabletop_class,
                         "variant_index": variant,
                         "candidate_dir": str(candidate),
                         "scene": str(scene_path)})
    return {
        "schema": "precision_insertion_franka_gripper_seed_assets_v1",
        "shared_root": str(root), "end_effector": "franka_gripper",
        "key_object": mode.key_object, "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path(os.environ.get(
        "PRECISION_SHARED_ROOT",
        "/mnt/paradex2/hyunsu/precision_insertion/runtime/active_shared_data")))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build(args.shared_root)
    payload = _json_bytes(result)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        _write_exact(output, payload)
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
