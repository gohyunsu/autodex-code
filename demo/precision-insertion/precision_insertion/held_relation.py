"""Resolve an observed post-lift key/wrist relation modulo key symmetry.

The round D-infinity key's axial yaw and identical-end exchange are not
observable in the images. Re-expressing the same physical cylinder with a
different CAD frame must not look like grasp slip or change its center.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .config import TaskMode
from .geometry import pose_angle_deg, validate_se3
from .symmetry import load_axial_symmetry


@dataclass(frozen=True)
class HeldRelation:
    T_robot_key_canonical: np.ndarray
    T_key_hand: np.ndarray
    center_in_robot_m: np.ndarray
    translation_drift_m: float
    rotation_drift_deg: float
    symmetry_branch: str

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_observed_held_relation_v1",
            "T_robot_key_canonical": self.T_robot_key_canonical.tolist(),
            "T_key_hand": self.T_key_hand.tolist(),
            "center_in_robot_m": self.center_in_robot_m.tolist(),
            "translation_drift_m": self.translation_drift_m,
            "rotation_drift_deg": self.rotation_drift_deg,
            "symmetry_branch": self.symmetry_branch,
            "scope": "geometry_from_observed_key_and_live_wrist_not_grasp_success",
        }


def _nearest_axial_rotation(matrix: np.ndarray) -> np.ndarray:
    """Closest local-Z rotation in Frobenius norm to a 3x3 matrix."""
    theta = math.atan2(matrix[1, 0] - matrix[0, 1],
                       matrix[0, 0] + matrix[1, 1])
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def resolve_postlift_held_relation(
    *, mode: TaskMode, shared_root: Path,
    T_robot_key_observed: np.ndarray, T_robot_hand_measured: np.ndarray,
    candidate_T_key_hand: np.ndarray,
    max_translation_drift_m: float, max_rotation_drift_deg: float,
) -> HeldRelation:
    """Choose the equivalent CAD pose nearest the grasp's planned relation.

    The limits define only the branch-selection scale; the caller must still
    veto a relation outside either commissioned limit. This routine never
    interprets a center shift as a symmetry operation.
    """
    limits = (float(max_translation_drift_m), float(max_rotation_drift_deg))
    if not all(math.isfinite(value) and value > 0 for value in limits):
        raise ValueError("grasp drift limits must be finite and positive")
    observed = validate_se3(T_robot_key_observed, name="observed T_robot_key")
    hand = validate_se3(T_robot_hand_measured, name="measured T_robot_hand")
    nominal = validate_se3(candidate_T_key_hand, name="candidate T_key_hand")
    if mode.family == "square":
        center = observed[:3, 3].copy()
        alternatives = [("identity", observed)]
    elif mode.family == "cylinder":
        info = (Path(shared_root).expanduser().resolve() /
                "object_processing" / mode.key_object / "processed_data" /
                "info")
        symmetry = load_axial_symmetry(info.parent.parent.parent, mode.key_object)
        if (not symmetry.end_exchange or
                not np.allclose(np.abs(symmetry.axis_local), [0, 0, 1], atol=1e-8)):
            raise ValueError("cylindrical key requires local-Z Dinf symmetry")
        data = json.loads((info / "symmetry.json").read_text(encoding="utf-8"))
        local_center = np.asarray(data.get("center"), dtype=np.float64)
        if local_center.shape != (3,) or not np.all(np.isfinite(local_center)):
            raise ValueError("cylinder symmetry center is missing or invalid")
        center = observed[:3, :3] @ local_center + observed[:3, 3]
        desired = hand[:3, :3] @ nominal[:3, :3].T
        relative = observed[:3, :3].T @ desired
        flip_axis = symmetry.end_exchange_axis_local
        flip = 2.0 * np.outer(flip_axis, flip_axis) - np.eye(3)
        alternatives = []
        for branch, F in (("axial_yaw", np.eye(3)),
                          ("end_exchange_and_axial_yaw", flip)):
            local_rotation = _nearest_axial_rotation(relative @ F.T) @ F
            equivalent = np.eye(4)
            equivalent[:3, :3] = observed[:3, :3] @ local_rotation
            equivalent[:3, 3] = center - equivalent[:3, :3] @ local_center
            alternatives.append((branch, validate_se3(
                equivalent, name="symmetry-equivalent observed key pose")))
    else:
        raise ValueError(f"unsupported key family: {mode.family}")

    scored: list[tuple[float, float, float, str, np.ndarray, np.ndarray]] = []
    for branch, equivalent in alternatives:
        relation = validate_se3(
            np.linalg.inv(equivalent) @ hand,
            name="observed T_key_hand")
        translation = float(np.linalg.norm(
            relation[:3, 3] - nominal[:3, 3]))
        rotation = pose_angle_deg(relation, nominal)
        score = max(translation / limits[0], rotation / limits[1])
        scored.append((score, translation, rotation, branch, equivalent, relation))
    _, translation, rotation, branch, equivalent, relation = min(
        scored, key=lambda row: (row[0], row[1] / limits[0] + row[2] / limits[1]))
    return HeldRelation(
        equivalent, relation, center, translation, rotation, branch)
