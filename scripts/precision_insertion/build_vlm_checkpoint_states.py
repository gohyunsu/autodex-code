#!/usr/bin/env python3
"""Build actual-asset states for the precision-insertion VLM storyboards.

The source trajectory supplies the real FR3/Inspire joint configuration,
full-resolution key/socket mesh paths, and the rigid key-to-hand transform.
Recorded success states are copied exactly.  Counterfactual states are clearly
marked as illustrations in the adjacent manifest:

* ``lift_miss`` and ``lift_slip`` change only the observed key state;
* ``preinsert_misaligned`` and ``insertion_rim_jam`` preserve the recorded
  ``T_key_hand`` and solve a new seven-joint FR3 endpoint IK;
* finish states show a retracted, open real Inspire hand and an independently
  observed key state after the press attempt.

These frames are visual explanations for a read-only VLM checkpoint.  They are
not cuRobo plans, collision proofs, dynamics results, or physical evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from yourdfpy import URDF


SHARED = Path.home() / "shared_data"
DEFAULT_SOURCE = (
    SHARED / "AutoDex/precision_insertion/visualizations/verification_20mm/"
    "pose_004/candidate_40/tabletop_004_to_insertion_preview.npz"
)
DEFAULT_OUTPUT = (
    SHARED / "AutoDex/precision_insertion/presentation_assets/08_vlm_checkpoints/"
    "actual_asset_states.npz"
)


@dataclass
class FrankaHandKinematics:
    """Seven-joint FK/IK ending at the mounted Inspire ``base_link``."""

    urdf: URDF

    def __post_init__(self) -> None:
        joints = {joint.name: joint for joint in self.urdf.robot.joints}
        self._joints = joints
        self._chain = [
            ("fr3_base_joint", None),
            *((f"fr3_joint{index}", index - 1) for index in range(1, 8)),
            ("fr3_joint8", None),
            ("flange_to_hand", None),
        ]
        self.lower = np.asarray([
            joints[f"fr3_joint{index}"].limit.lower for index in range(1, 8)
        ], dtype=np.float64)
        self.upper = np.asarray([
            joints[f"fr3_joint{index}"].limit.upper for index in range(1, 8)
        ], dtype=np.float64)

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint_name, q_index in self._chain:
            joint = self._joints[joint_name]
            transform = transform @ np.asarray(joint.origin, dtype=np.float64)
            if q_index is not None:
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = Rotation.from_rotvec(
                    np.asarray(joint.axis, dtype=np.float64) * q_arm[q_index]
                ).as_matrix()
                transform = transform @ motion
        return transform

    def solve(self, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, dict]:
        def residual(q_arm: np.ndarray) -> np.ndarray:
            actual = self.fk(q_arm)
            translation = (actual[:3, 3] - target[:3, 3]) * 30.0
            rotation = Rotation.from_matrix(
                target[:3, :3].T @ actual[:3, :3]
            ).as_rotvec() * 3.0
            return np.concatenate([translation, rotation])

        result = least_squares(
            residual,
            np.clip(np.asarray(seed, dtype=np.float64), self.lower, self.upper),
            bounds=(self.lower, self.upper),
            max_nfev=1500,
            ftol=1.0e-12,
            xtol=1.0e-12,
            gtol=1.0e-12,
        )
        actual = self.fk(result.x)
        translation_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        rotation_error = np.linalg.norm(Rotation.from_matrix(
            target[:3, :3].T @ actual[:3, :3]
        ).as_rotvec())
        diagnostic = {
            "translation_error_mm": float(translation_error * 1000.0),
            "rotation_error_deg": float(np.degrees(rotation_error)),
            "function_evaluations": int(result.nfev),
            "active_joint_limits": [int(value) for value in result.active_mask],
        }
        if translation_error > 5.0e-4 or rotation_error > np.radians(0.5):
            raise RuntimeError(f"checkpoint IK did not converge: {diagnostic}")
        return np.asarray(result.x, dtype=np.float64), diagnostic


def _tilted_pose(
    source: np.ndarray,
    *,
    translation_xyz_m: tuple[float, float, float],
    world_tilt_xyz_deg: tuple[float, float, float],
) -> np.ndarray:
    result = np.asarray(source, dtype=np.float64).copy()
    delta = Rotation.from_euler("xyz", world_tilt_xyz_deg, degrees=True).as_matrix()
    result[:3, :3] = delta @ result[:3, :3]
    result[:3, 3] += np.asarray(translation_xyz_m, dtype=np.float64)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source_path.is_file():
        parser.error(f"missing source trajectory: {source_path}")

    with np.load(source_path, allow_pickle=False) as source:
        qpos = np.asarray(source["qpos"], dtype=np.float64)
        key_poses = np.asarray(source["object_pose"], dtype=np.float64)
        socket_pose = np.asarray(source["socket_pose"], dtype=np.float64)
        seated_key_pose = np.asarray(
            source["desired_seated_key_pose"], dtype=np.float64
        )
        joint_names = np.asarray(source["joint_names"])
        object_mesh_path = str(source["object_mesh_path"].item())
        socket_mesh_path = str(source["socket_mesh_path"].item())
        robot_urdf_path = str(source["robot_urdf_path"].item())

    urdf_path = Path(robot_urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        parser.error(f"missing combined FR3/Inspire URDF: {urdf_path}")
    urdf = URDF.load(str(urdf_path), build_scene_graph=False, load_meshes=False)
    kinematics = FrankaHandKinematics(urdf)

    # Exact source frames (zero-based): lift end, reorient end, preinsert,
    # partial insertion, and the 20 mm verification endpoint.
    lift_i, retracted_i, preinsert_i, partial_i, terminal_i = 41, 77, 93, 100, 107
    grasp_hand = qpos[terminal_i, 7:].copy()
    open_hand = qpos[0, 7:].copy()

    world_hand_preinsert = kinematics.fk(qpos[preinsert_i, :7])
    T_key_hand = np.linalg.inv(key_poses[preinsert_i]) @ world_hand_preinsert

    preinsert_misaligned = _tilted_pose(
        key_poses[preinsert_i],
        translation_xyz_m=(0.018, 0.006, 0.0),
        world_tilt_xyz_deg=(0.0, 12.0, 8.0),
    )
    q_misaligned_arm, misaligned_ik = kinematics.solve(
        preinsert_misaligned @ T_key_hand, qpos[preinsert_i, :7]
    )
    q_misaligned = np.concatenate([q_misaligned_arm, grasp_hand])

    rim_jam = _tilted_pose(
        key_poses[partial_i],
        translation_xyz_m=(0.009, 0.0, 0.003),
        world_tilt_xyz_deg=(0.0, 8.0, 0.0),
    )
    q_rim_arm, rim_ik = kinematics.solve(
        rim_jam @ T_key_hand, qpos[partial_i, :7]
    )
    q_rim = np.concatenate([q_rim_arm, grasp_hand])

    lift_slip = _tilted_pose(
        key_poses[lift_i],
        translation_xyz_m=(0.008, 0.0, -0.032),
        world_tilt_xyz_deg=(0.0, 18.0, 0.0),
    )
    finish_observer_q = np.concatenate([qpos[retracted_i, :7], open_hand])

    states = [
        ("lift_retained", qpos[lift_i], key_poses[lift_i], "recorded"),
        ("lift_miss", qpos[lift_i], key_poses[0], "counterfactual_observation"),
        ("lift_slip", qpos[lift_i], lift_slip, "counterfactual_observation"),
        ("preinsert_aligned", qpos[preinsert_i], key_poses[preinsert_i], "recorded"),
        ("preinsert_misaligned", q_misaligned, preinsert_misaligned, "ik_illustration"),
        ("preinsert_occluded", qpos[preinsert_i], key_poses[preinsert_i], "recorded_different_view"),
        ("insertion_20mm", qpos[terminal_i], key_poses[terminal_i], "recorded"),
        ("insertion_partial", qpos[partial_i], key_poses[partial_i], "recorded"),
        ("insertion_rim_jam", q_rim, rim_jam, "ik_illustration"),
        ("finish_seated", finish_observer_q, seated_key_pose, "kinematic_illustration"),
        ("finish_press_jam", finish_observer_q, rim_jam, "kinematic_illustration"),
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        qpos=np.asarray([state[1] for state in states], dtype=np.float32),
        object_pose=np.asarray([state[2] for state in states], dtype=np.float64),
        socket_pose=socket_pose,
        joint_names=joint_names,
        object_mesh_path=np.asarray(object_mesh_path),
        socket_mesh_path=np.asarray(socket_mesh_path),
        robot_urdf_path=np.asarray(robot_urdf_path),
        state_name=np.asarray([state[0] for state in states]),
    )

    manifest = {
        "schema_version": 1,
        "scope": "actual_asset_vlm_checkpoint_illustrations",
        "source_trajectory": str(source_path),
        "output_trajectory": str(output),
        "robot_urdf": robot_urdf_path,
        "key_mesh": object_mesh_path,
        "socket_mesh": socket_mesh_path,
        "physical_validation": False,
        "planning_validation": False,
        "warning": (
            "Recorded states preserve source kinematics; counterfactual and IK "
            "states are explanatory renders only."
        ),
        "rigid_T_key_hand": T_key_hand.tolist(),
        "ik_diagnostics": {
            "preinsert_misaligned": misaligned_ik,
            "insertion_rim_jam": rim_ik,
        },
        "states": [
            {
                "frame_1based": index + 1,
                "name": name,
                "provenance": provenance,
            }
            for index, (name, _q, _pose, provenance) in enumerate(states)
        ],
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)
    print(manifest_path)
    print(json.dumps(manifest["ik_diagnostics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
