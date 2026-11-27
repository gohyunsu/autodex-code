"""Deterministic Franka-gripper grasps for the cylindrical precision key.

The cylindrical key has only two tabletop pose classes: its axis is either
approximately parallel to the support plane (``side_down``) or approximately
normal to it (``end_down``).  A fresh measured key pose and the measured table
normal are sufficient to construct the gripper pose; BODex sampling is not
needed for this end effector.

The returned transform is the mounted Franka hand-base pose used by cuRobo.
It is *not* a robot command.  The normal planner still checks IK, table/world
collision, approach and lift before execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import validate_se3


KEY_AXIS_LOCAL = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
KEY_CENTER_LOCAL_M = np.asarray([0.0, 0.0, 0.040], dtype=np.float64)
KEY_DIAMETER_M = 0.030
ROBOTIQ_PALM_TO_KEY_CENTER_M = 0.125
ROBOTIQ_KEY_GRASP_Q_RAD = 0.53
UPRIGHT_GRASP_ABOVE_CENTER_M = 0.015
UPRIGHT_APPROACH_VARIANTS = 8
SIDE_DOWN_WRIST_VARIANTS = 2


def _unit(value, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-10:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _canonical_sign(axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Choose one representative of the cylinder's end-exchange symmetry."""
    # Prefer the table +X direction, then +Y.  The fallback is deterministic
    # and changes no physical grasp because the two ends are symmetry-related.
    for reference in (np.asarray([1.0, 0.0, 0.0]),
                      np.asarray([0.0, 1.0, 0.0])):
        tangent = reference - normal * float(np.dot(reference, normal))
        if np.linalg.norm(tangent) > 1e-8:
            tangent = tangent / np.linalg.norm(tangent)
            return axis if float(np.dot(axis, tangent)) >= 0.0 else -axis
    return axis


@dataclass(frozen=True)
class AnalyticGripperGrasp:
    tabletop_class: str
    T_robot_hand: np.ndarray
    T_robot_pregrasp: np.ndarray
    T_key_hand: np.ndarray
    key_axis_robot: np.ndarray
    table_normal_robot: np.ndarray
    pregrasp_q: np.ndarray
    grasp_q: np.ndarray

    def candidate_override(self, *, identifier: tuple[str, str, str]) -> tuple:
        """Return the explicit-candidate tuple accepted by ``GraspPlanner``."""
        return (
            self.T_robot_hand[None, ...].copy(),
            self.pregrasp_q[None, ...].astype(np.float32),
            self.grasp_q[None, ...].astype(np.float32),
            [tuple(identifier)],
            [self.pregrasp_q.copy()],
        )


def classify_cylinder_tabletop(
    T_robot_key: np.ndarray,
    *,
    table_normal_robot: np.ndarray,
) -> str:
    """Choose the nearer of the only two D-infinity tabletop classes.

    This intentionally has no confidence/failure threshold: every measured
    axis is assigned to the nearer class, as required by the runtime policy.
    """
    key = validate_se3(T_robot_key, name="measured T_robot_key")
    normal = _unit(table_normal_robot, name="table normal")
    axis = _unit(key[:3, :3] @ KEY_AXIS_LOCAL, name="measured key axis")
    alignment = abs(float(np.dot(axis, normal)))
    return "end_down" if alignment >= np.sqrt(0.5) else "side_down"


def match_cylinder_tabletop_stem(
    T_robot_key: np.ndarray,
    *,
    tabletop_dir: Path,
    table_normal_robot: np.ndarray,
) -> dict:
    """Map the analytic lying/standing class to an existing AutoDex stem.

    The class decision deliberately has no rejection threshold.  Existing
    tabletop NPY files are used only as stable scene/catalog identifiers; the
    live measured pose, not the template transform, drives the grasp.
    """
    pose = validate_se3(T_robot_key, name="measured T_robot_key")
    normal = _unit(table_normal_robot, name="table normal")
    observed_axis = _unit(pose[:3, :3] @ KEY_AXIS_LOCAL,
                          name="measured key axis")
    tabletop_class = classify_cylinder_tabletop(
        pose, table_normal_robot=normal)
    rows = []
    for path in sorted(Path(tabletop_dir).glob("*.npy")):
        template = validate_se3(np.load(path, allow_pickle=False),
                                name=f"tabletop template {path.stem}")
        if classify_cylinder_tabletop(
                template, table_normal_robot=normal) != tabletop_class:
            continue
        template_axis = _unit(template[:3, :3] @ KEY_AXIS_LOCAL,
                              name=f"tabletop axis {path.stem}")
        error = float(np.degrees(np.arccos(np.clip(
            abs(float(np.dot(observed_axis, template_axis))), -1.0, 1.0))))
        rows.append((error, path.stem))
    if not rows:
        raise FileNotFoundError(
            f"no tabletop asset represents analytic class {tabletop_class}")
    error, stem = min(rows)
    return {
        "schema": "precision_insertion_tabletop_class_v1",
        "stem": stem,
        "tabletop_class": tabletop_class,
        "rotation_error_deg": error,
        "max_rotation_error_deg": None,
        "method": "franka_gripper_cylinder_nearest_axis_class",
        "source": "fresh_key_pose_used_directly_template_is_identifier_only",
    }


def analytic_franka_gripper_grasp(
    T_robot_key: np.ndarray,
    *,
    table_normal_robot: np.ndarray,
    robot_base_position_robot: np.ndarray = np.zeros(3),
    palm_to_key_center_m: float | None = None,
    pregrasp_clearance_m: float = 0.080,
    approach_azimuth_rad: float = 0.0,
    axis_sign: int = 1,
    end_effector: str = "franka_gripper",
) -> AnalyticGripperGrasp:
    """Construct one grasp for a lying or standing cylindrical key.

    ``hand z`` is the approach direction from the palm toward the key and
    ``hand y`` is the Franka jaw-closing direction.  For a lying key the
    approach is top-down. For a standing key it is horizontal; the zero-
    azimuth side is away from the robot base and an optional azimuth selects
    another symmetry-equivalent radial approach.
    """
    key = validate_se3(T_robot_key, name="measured T_robot_key")
    normal = _unit(table_normal_robot, name="table normal")
    base = np.asarray(robot_base_position_robot, dtype=np.float64).reshape(-1)
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("robot base position must be a finite three-vector")
    if end_effector not in {"franka_gripper", "robotiq_2f85"}:
        raise ValueError("analytic parallel grasp needs a supported gripper")
    if palm_to_key_center_m is None:
        palm_to_key_center_m = (
            ROBOTIQ_PALM_TO_KEY_CENTER_M
            if end_effector == "robotiq_2f85" else 0.1034)
    for value, name in ((palm_to_key_center_m, "palm-to-key-center"),
                        (pregrasp_clearance_m, "pregrasp clearance")):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    azimuth = float(approach_azimuth_rad)
    if not np.isfinite(azimuth):
        raise ValueError("approach azimuth must be finite")
    if type(axis_sign) is not int or axis_sign not in (-1, 1):
        raise ValueError("axis sign must be +1 or -1")

    observed_axis = _unit(key[:3, :3] @ KEY_AXIS_LOCAL,
                          name="measured key axis")
    key_center = key[:3, :3] @ KEY_CENTER_LOCAL_M + key[:3, 3]
    tabletop_class = classify_cylinder_tabletop(
        key, table_normal_robot=normal)

    if tabletop_class == "side_down":
        # Drop the small measured tilt from the grasp gauge: the cylinder is
        # supported by the table and its D-infinity roll is unobservable.
        axis = observed_axis - normal * float(np.dot(observed_axis, normal))
        axis = _canonical_sign(_unit(axis, name="table-parallel key axis"),
                               normal)
        hand_x = float(axis_sign) * axis
        hand_z = -normal
        grasp_center = key_center
    else:
        hand_x = (observed_axis if float(np.dot(observed_axis, normal)) >= 0.0
                  else -observed_axis)
        hand_x = _unit(hand_x, name="upright key axis")
        radial = key_center - base
        radial = radial - normal * float(np.dot(radial, normal))
        if np.linalg.norm(radial) <= 1e-8:
            radial = np.asarray([1.0, 0.0, 0.0])
            radial = radial - normal * float(np.dot(radial, normal))
            if np.linalg.norm(radial) <= 1e-8:
                radial = np.asarray([0.0, 1.0, 0.0])
                radial = radial - normal * float(np.dot(radial, normal))
        outward = _unit(radial, name="horizontal gripper side")
        # The upright cylinder is invariant to axial yaw.  Sample an explicit
        # orbit around its measured support normal so cuRobo can choose an IK-
        # reachable side without changing the physical grasp or key pose.
        outward = _unit(
            np.cos(azimuth) * outward +
            np.sin(azimuth) * np.cross(normal, outward),
            name="yaw-shifted horizontal gripper side")
        hand_z = -outward
        # Grasp the upper half of the standing peg. Express this from the
        # D-infinity object center and measured support normal, not from one
        # arbitrarily labelled cylinder end, so end exchange changes nothing.
        grasp_center = key_center + normal * UPRIGHT_GRASP_ABOVE_CENTER_M

    hand_y = _unit(np.cross(hand_z, hand_x), name="jaw-closing axis")
    # Re-orthogonalize after using the measured upright axis.
    hand_x = _unit(np.cross(hand_y, hand_z), name="gripper longitudinal axis")
    key_axis = hand_x.copy()
    # Stock Franka Hand closes along local y. Robotiq 2F-85 closes along
    # local x, with its straight tool extension along local z. Reorder the
    # exact same physical key/jaw/approach frame instead of reusing the wrong
    # stock-hand convention.
    rotation = (np.column_stack((hand_y, -hand_x, hand_z))
                if end_effector == "robotiq_2f85" else
                np.column_stack((hand_x, hand_y, hand_z)))
    if np.linalg.det(rotation) < 0.0:
        raise ValueError("analytic gripper frame is left-handed")

    hand = np.eye(4, dtype=np.float64)
    hand[:3, :3] = rotation
    hand[:3, 3] = grasp_center - hand_z * float(palm_to_key_center_m)
    pregrasp = hand.copy()
    pregrasp[:3, 3] -= hand_z * float(pregrasp_clearance_m)
    key_hand = np.linalg.inv(key) @ hand
    validate_se3(key_hand, name="analytic T_key_hand")
    return AnalyticGripperGrasp(
        tabletop_class=tabletop_class,
        T_robot_hand=hand,
        T_robot_pregrasp=pregrasp,
        T_key_hand=key_hand,
        key_axis_robot=key_axis,
        table_normal_robot=normal,
        pregrasp_q=(np.asarray([0.0], dtype=np.float64)
                    if end_effector == "robotiq_2f85" else
                    np.asarray([0.04, 0.04], dtype=np.float64)),
        grasp_q=(np.asarray([ROBOTIQ_KEY_GRASP_Q_RAD], dtype=np.float64)
                 if end_effector == "robotiq_2f85" else
                 np.asarray([KEY_DIAMETER_M / 2.0] * 2, dtype=np.float64)),
    )


def analytic_franka_gripper_grasp_for_candidate(
    T_robot_key: np.ndarray,
    *,
    table_normal_robot: np.ndarray,
    identifier: tuple[str, str, str],
    robot_base_position_robot: np.ndarray = np.zeros(3),
    end_effector: str = "franka_gripper",
) -> AnalyticGripperGrasp:
    """Reconstruct one catalogued symmetry-equivalent analytic grasp.

    Scene ``0`` is the upright/end-down tabletop class and uses eight radial
    approaches. Scene ``1`` is the side-down class and uses the two wrist
    orientations related by a 180-degree flip about the top-down approach.
    The mapping is deterministic and shared by asset generation, live
    planning and failed-attempt restoration.
    """
    if (not isinstance(identifier, tuple) or len(identifier) != 3 or
            identifier[0] != "table" or
            not identifier[1].isdigit() or not identifier[2].isdigit()):
        raise ValueError("analytic gripper candidate needs table/scene/grasp ID")
    scene_id = int(identifier[1])
    variant = int(identifier[2])
    grasp_class = classify_cylinder_tabletop(
        T_robot_key, table_normal_robot=table_normal_robot)
    expected_scene = {"end_down": 0, "side_down": 1}[grasp_class]
    if scene_id != expected_scene:
        raise ValueError("analytic gripper candidate differs from tabletop class")
    if grasp_class == "end_down":
        if not 0 <= variant < UPRIGHT_APPROACH_VARIANTS:
            raise ValueError("upright gripper variant is outside the catalog")
        return analytic_franka_gripper_grasp(
            T_robot_key, table_normal_robot=table_normal_robot,
            robot_base_position_robot=robot_base_position_robot,
            approach_azimuth_rad=(2.0 * np.pi * variant /
                                  UPRIGHT_APPROACH_VARIANTS),
            end_effector=end_effector)
    if not 0 <= variant < SIDE_DOWN_WRIST_VARIANTS:
        raise ValueError("side-down gripper variant is outside the catalog")
    return analytic_franka_gripper_grasp(
        T_robot_key, table_normal_robot=table_normal_robot,
        robot_base_position_robot=robot_base_position_robot,
        axis_sign=(1 if variant == 0 else -1),
        end_effector=end_effector)
