"""Project *predicted* Franka/Inspire, held key and fixed socket together.

This adapts AutoDex's existing depth-tested RobotOverlayRenderer; it does not
copy a rasterizer or infer the actual key pose from its own overlay. The robot
uses measured 13-DOF joints. The key pose comes from an explicitly identified
key/hand hypothesis, and must not be called an observation. Camera images and
calibration must already have passed the session's provenance checks.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np
from PIL import Image

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import _load_mesh
from .frame_provenance import image_sha256
from .geometry import validate_se3
from .observer import HeldSceneView


_RELATION_SOURCES = frozenset({
    "observed_multiview_key_plus_wrist",
    "verified_physical_grasp_calibration",
    "mujoco_achieved_diagnostic",
    "v8_nominal_diagnostic",
})
_KEY_COLOR_BGR = (66, 128, 239)
_SOCKET_COLOR_BGR = (224, 101, 93)


@dataclass(frozen=True)
class HeldScenePrediction:
    mode: TaskMode
    names: tuple[str, ...]
    meshes: tuple[object, ...]
    poses_robot: tuple[np.ndarray, ...]
    finger_labels: Mapping[str, str | None]
    T_robot_hand: np.ndarray
    T_robot_key_predicted: np.ndarray
    T_robot_socket: np.ndarray
    relation_source: str

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_held_scene_prediction_v1",
            "mode": {"family": self.mode.family, "gap_mm": self.mode.gap_mm},
            "mesh_names": list(self.names),
            "T_robot_hand_measured_fk": self.T_robot_hand.tolist(),
            "T_robot_key_predicted": self.T_robot_key_predicted.tolist(),
            "T_robot_socket_frozen": self.T_robot_socket.tolist(),
            "relation_source": self.relation_source,
            "scope": "rendered_hypothesis_not_observed_key_pose_or_motion_permission",
            "robot_ready": False,
        }


@dataclass(frozen=True)
class HeldSceneComparison:
    """Paired VLM images and pixel digests; not camera producer provenance."""

    views: tuple[HeldSceneView, ...]
    pixel_digests: Mapping[str, Mapping[str, str]]
    prediction: HeldScenePrediction

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_held_scene_comparison_v1",
            "prediction": self.prediction.to_record(),
            "views": {
                view.camera_id: {
                    "timestamp_s": view.timestamp_s,
                    **self.pixel_digests[view.camera_id],
                } for view in self.views
            },
            "scope": "paired_pixels_not_verified_capture_or_observed_key_pose",
            "robot_ready": False,
        }


def _finger_label(name: str) -> str | None:
    for prefix, label in (
        ("right_thumb_", "thumb"), ("right_index_", "index"),
        ("right_middle_", "middle"), ("right_ring_", "ring"),
        ("right_little_", "pinky"),
    ):
        if name.startswith(prefix):
            return label
    return None


def _load_robot(path: Path):
    from yourdfpy import URDF

    return URDF.load(str(path), build_scene_graph=True, load_meshes=True)


def _visual_affine(value, name: str) -> np.ndarray:
    """URDF visual mesh graph may include unit/asset scale, unlike a pose."""
    matrix = np.asarray(value, dtype=np.float64)
    if (matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)) or
            not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9) or
            np.linalg.det(matrix[:3, :3]) <= 0):
        raise ValueError(f"{name} must be a finite nonsingular visual affine")
    return matrix.copy()


def build_held_scene_prediction(
    *, shared_root: Path, mode: TaskMode, full_q_measured: np.ndarray,
    T_robot_socket_frozen: np.ndarray, T_key_hand_hypothesis: np.ndarray,
    relation_source: str, robot_loader: Callable = _load_robot,
) -> HeldScenePrediction:
    """Assemble exact available visual meshes in the FR3 robot-base frame.

    The caller must verify that the 13 joints are real feedback and that the
    key/hand hypothesis belongs to this grasp. A squeeze command alone is
    deliberately distinguished from an observed or physically calibrated
    relation by ``relation_source``. This function does no robot/camera I/O.
    """
    if relation_source not in _RELATION_SOURCES:
        raise ValueError("held-key relation source must be explicit")
    q = np.asarray(full_q_measured, dtype=np.float64)
    if q.shape != (13,) or not np.all(np.isfinite(q)):
        raise ValueError("overlay needs 13 finite measured FR3/Inspire joints")
    T_socket = validate_se3(T_robot_socket_frozen, name="frozen T_robot_socket")
    T_key_hand = validate_se3(T_key_hand_hypothesis,
                              name="hypothesized T_key_hand")
    paths = AssetPaths(Path(shared_root).expanduser().resolve(), mode)
    robot = robot_loader(paths.robot_urdf)
    joints = [joint.name for joint in robot.actuated_joints]
    if (len(joints) != 13 or
            not all(name.startswith("fr3_") for name in joints[:7])):
        raise ValueError("overlay URDF is not the FR3/Inspire 13-joint model")
    robot.update_cfg(dict(zip(joints, q)))
    hand = validate_se3(robot.get_transform("base_link", robot.base_link),
                        name="measured-joint URDF hand FK")
    key = validate_se3(hand @ np.linalg.inv(T_key_hand),
                       name="predicted held key pose")

    scene = robot.scene
    names: list[str] = []
    meshes: list[object] = []
    poses: list[np.ndarray] = []
    labels: dict[str, str | None] = {}
    for name in sorted(scene.geometry):
        if name in {"predicted_key", "frozen_socket"}:
            raise ValueError("URDF geometry name conflicts with task object")
        transform, _ = scene.graph.get(name)
        names.append(name)
        meshes.append(scene.geometry[name])
        poses.append(_visual_affine(transform, name=f"URDF visual {name}"))
        labels[name] = _finger_label(name)
    if (not names or not any(name.startswith("fr3_") for name in names) or
            not any(name.startswith("right_") for name in names)):
        raise ValueError("full Franka and Inspire visual meshes are required")
    names.extend(("predicted_key", "frozen_socket"))
    meshes.extend((_load_mesh(paths.raw_mesh(mode.key_object)),
                   _load_mesh(paths.raw_mesh(mode.socket_object))))
    poses.extend((key, T_socket))
    labels.update({"predicted_key": None, "frozen_socket": None})
    return HeldScenePrediction(
        mode, tuple(names), tuple(meshes), tuple(poses), labels,
        hand, key, T_socket, relation_source)


def render_held_scene_overlays(
    *, prediction: HeldScenePrediction,
    frames_bgr: Mapping[str, np.ndarray],
    intrinsics_undistorted: Mapping[str, np.ndarray],
    T_camera_robot: Mapping[str, np.ndarray],
    renderer_factory: Callable | None = None,
) -> dict[str, np.ndarray]:
    """Return translucent CAD overlays while preserving the raw frame inputs.

    ``T_camera_robot`` and K must refer to the same undistorted image pixels.
    The image on the left of a VLM comparison should remain the untouched raw
    frame. Synthetic meshes depth-test against each other, not against the
    real scene: occlusion by an unmodelled table/object is *not* resolved.
    Never feed this returned overlay alone as a measured key pose.
    On the AutoDex PC, this reuses the installed GPU RobotOverlayRenderer;
    no fallback wireframe or untested pixel interpolation is substituted.
    """
    if not isinstance(prediction, HeldScenePrediction):
        raise TypeError("overlay needs an assembled held-scene prediction")
    serials = sorted(frames_bgr)
    if (len(serials) < 1 or set(serials) != set(intrinsics_undistorted) or
            set(serials) != set(T_camera_robot)):
        raise ValueError("raw frames and calibrated cameras must have identical IDs")
    first = np.asarray(frames_bgr[serials[0]])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("overlay needs unchanged uint8 BGR frames")
    height, width = first.shape[:2]
    intrinsics: dict[str, dict] = {}
    extrinsics: dict[str, np.ndarray] = {}
    for serial in serials:
        frame = np.asarray(frames_bgr[serial])
        if frame.shape != first.shape or frame.dtype != np.uint8:
            raise ValueError("all overlay frames must have the same image geometry")
        K = np.asarray(intrinsics_undistorted[serial], dtype=np.float64)
        if (K.shape != (3, 3) or not np.all(np.isfinite(K)) or
                K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose(K[2], [0.0, 0.0, 1.0], atol=1e-9)):
            raise ValueError("overlay camera intrinsics are invalid")
        intrinsics[serial] = {"intrinsics_undistort": K}
        extrinsics[serial] = validate_se3(
            T_camera_robot[serial], name=f"{serial} T_camera_robot")[:3, :]
    if renderer_factory is None:
        from src.visualization.overlay_robot_video import RobotOverlayRenderer

        renderer_factory = RobotOverlayRenderer
    renderer = renderer_factory(
        prediction.meshes, prediction.names, prediction.finger_labels,
        intrinsics, extrinsics, height, width)
    if tuple(renderer.serials) != tuple(serials):
        raise ValueError("renderer camera order differs from source frames")
    # Reuse the existing depth-buffered renderer while giving the two task
    # objects distinct translucent colors. Its LUT is deliberately local to
    # this instance; no stock renderer module or global palette is changed.
    if getattr(renderer, "n_links", None) != len(prediction.names):
        raise ValueError("renderer dropped a task or robot mesh")
    import torch

    for name, bgr, alpha in (
        ("predicted_key", _KEY_COLOR_BGR, 0.45),
        ("frozen_socket", _SOCKET_COLOR_BGR, 0.35),
    ):
        index = prediction.names.index(name) + 1
        renderer.color_lut[index] = torch.as_tensor(
            bgr, dtype=renderer.color_lut.dtype,
            device=renderer.color_lut.device)
        renderer.alpha_lut[index] = float(alpha)
    rendered = renderer.render(
        prediction.poses_robot,
        [np.asarray(frames_bgr[serial]).copy() for serial in serials])
    if (len(rendered) != len(serials) or
            any(np.asarray(image).shape != first.shape or
                np.asarray(image).dtype != np.uint8 for image in rendered)):
        raise ValueError("renderer returned invalid camera overlays")
    return dict(zip(serials, rendered))


def build_held_scene_comparison(
    *, prediction: HeldScenePrediction,
    frames_bgr: Mapping[str, np.ndarray],
    frame_timestamps_s: Mapping[str, float],
    intrinsics_undistorted: Mapping[str, np.ndarray],
    T_camera_robot: Mapping[str, np.ndarray],
    renderer_factory: Callable | None = None,
) -> HeldSceneComparison:
    """Prepare same-pixel raw/CAD pairs for the read-only preinsert observer.

    These timestamps and camera matrices are caller-supplied; this adapter
    does not verify exposure/frame IDs or session calibration. A live caller
    must first admit the camera bundle through the demo's provenance checks.
    """
    if set(frame_timestamps_s) != set(frames_bgr):
        raise ValueError("held-scene timestamps must match all camera frames")
    timestamps = {serial: float(stamp)
                  for serial, stamp in frame_timestamps_s.items()}
    if not all(math.isfinite(stamp) and stamp > 0
               for stamp in timestamps.values()):
        raise ValueError("held-scene exposure times must be finite and positive")
    overlays = render_held_scene_overlays(
        prediction=prediction, frames_bgr=frames_bgr,
        intrinsics_undistorted=intrinsics_undistorted,
        T_camera_robot=T_camera_robot, renderer_factory=renderer_factory)
    views = []
    digests = {}
    for serial in sorted(frames_bgr):
        raw = np.asarray(frames_bgr[serial])
        overlay = np.asarray(overlays[serial])
        digests[serial] = {
            "raw_image_sha256": image_sha256(raw),
            "overlay_image_sha256": image_sha256(overlay),
        }
        views.append(HeldSceneView(
            serial, timestamps[serial],
            Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)),
            Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))))
    return HeldSceneComparison(tuple(views), digests, prediction)
