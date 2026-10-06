#!/usr/bin/env python3
"""Export an actual Inspire hand and BODex contacts in the key frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh

from validate_whole_hand_contact_policy import _hand_link_meshes


SHARED = Path.home() / "shared_data"
DEFAULT_CANDIDATE = (
    SHARED / "AutoDex/contact_screen_staging/inspire/"
    "precision_insertion_tabletop_v1/precision_key_1p5mm/table/4/290"
)
DEFAULT_ROBOT = (
    SHARED / "AutoDex/content/assets/robot/fr3_inspire_description/"
    "fr3_inspire.urdf"
)
DEFAULT_OUTPUT = (
    SHARED / "AutoDex/precision_insertion/presentation_assets/"
    "03_contact_policy/source"
)


def _export_hand(candidate: Path, robot: Path, q_file: str, output: Path) -> None:
    q = np.load(candidate / q_file).reshape(-1)
    key_to_hand = np.load(candidate / "wrist_se3.npy")
    meshes = []
    for mesh in _hand_link_meshes(robot, q).values():
        moved = mesh.copy()
        moved.apply_transform(key_to_hand)
        meshes.append(moved)
    combined = trimesh.util.concatenate(meshes)
    combined.export(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_ROBOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    candidate = args.candidate_dir.expanduser().resolve()
    robot = args.robot_urdf.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (candidate / "wrist_se3.npy", candidate / "bodex_info.npy", robot):
        if not path.is_file():
            parser.error(f"missing input: {path}")
    output.mkdir(parents=True, exist_ok=True)
    _export_hand(candidate, robot, "pregrasp_pose.npy", output / "hand_pregrasp_key_frame.ply")
    _export_hand(candidate, robot, "grasp_pose.npy", output / "hand_grasp_key_frame.ply")
    data = np.load(candidate / "bodex_info.npy", allow_pickle=True).item()
    raw = np.asarray(data["contact_point"], dtype=np.float64)
    contacts = raw.reshape(-1, raw.shape[-1])[:, :3]
    np.save(output / "declared_contacts_key_frame.npy", contacts)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
