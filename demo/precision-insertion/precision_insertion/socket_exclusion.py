"""Project the session-frozen socket CAD into key-perception camera frames.

The projected convex hull is a conservative image exclusion region, *not* a
pixel-accurate silhouette or evidence that the fixture has not moved. It is
used only to veto a SAM key mask that actually covers the known fixed socket.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3
from .world import validated_frozen_socket_pose


@dataclass(frozen=True)
class SocketImageExclusion:
    masks: dict[str, np.ndarray]
    per_camera: dict[str, dict]
    socket_object: str
    socket_mesh_sha256: str
    socket_pose_robot: np.ndarray
    dilation_px: int

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_socket_image_exclusion_v1",
            "socket_object": self.socket_object,
            "socket_mesh_sha256": self.socket_mesh_sha256,
            "socket_pose_robot": self.socket_pose_robot.tolist(),
            "dilation_px": self.dilation_px,
            "per_camera": self.per_camera,
            "method": "convex_hull_of_projected_frozen_collision_mesh_vertices",
            "scope": "conservative_key_mask_veto_not_socket_remeasurement",
        }


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) != 4:
                    raise ValueError("socket OBJ has a malformed vertex")
                vertices.append([float(value) for value in parts[1:]])
    points = np.asarray(vertices, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or
            not np.all(np.isfinite(points))):
        raise ValueError("socket collision OBJ needs finite 3D vertices")
    return points


def project_frozen_socket_exclusion(
    *, mode: TaskMode, shared_root: Path, calibration,
    images_bgr: Mapping[str, np.ndarray],
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_full: Mapping[str, np.ndarray],
    dilation_px: int,
) -> SocketImageExclusion:
    """Return a per-camera socket hull using the *frozen* session world.

    All vertices must lie in front of each camera. If this cannot be shown,
    the image-exclusion evidence is unavailable and key admission fails.
    """
    if type(dilation_px) is not int or dilation_px < 0:
        raise ValueError("socket exclusion dilation must be nonnegative pixels")
    if (not images_bgr or set(images_bgr) - set(intrinsics_undist) or
            set(images_bgr) - set(extrinsics_full)):
        raise ValueError("socket projection needs calibrated key cameras")
    root = Path(shared_root).expanduser().resolve()
    socket_pose = validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    mesh_path = AssetPaths(root, mode).socket_collision_mesh.resolve()
    points = _obj_vertices(mesh_path)
    mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    world_robot = validate_se3(calibration.record.get("c2r"),
                               name="session C2R")
    masks = {}
    diagnostics = {}
    for serial, image in images_bgr.items():
        if (not isinstance(image, np.ndarray) or image.ndim != 3 or
                image.shape[2] != 3 or image.dtype != np.uint8 or
                not image.size):
            raise ValueError(f"invalid key camera image: {serial}")
        height, width = image.shape[:2]
        if dilation_px >= min(height, width):
            raise ValueError("socket exclusion dilation exceeds image size")
        K = np.asarray(intrinsics_undist[serial], dtype=float)
        if (K.shape != (3, 3) or not np.all(np.isfinite(K)) or
                K[0, 0] <= 0 or K[1, 1] <= 0 or
                not np.allclose(K[2], [0, 0, 1], atol=1e-9)):
            raise ValueError(f"invalid undistorted intrinsics: {serial}")
        camera_world = validate_se3(extrinsics_full[serial],
                                    name=f"{serial} T_camera_world")
        camera_socket = validate_se3(
            camera_world @ world_robot @ socket_pose,
            name=f"{serial} T_camera_socket")
        camera_points = (points @ camera_socket[:3, :3].T +
                         camera_socket[:3, 3])
        if np.any(camera_points[:, 2] <= 1e-6):
            raise ValueError(f"socket mesh is behind or crosses camera: {serial}")
        pixels_h = camera_points @ K.T
        pixels = pixels_h[:, :2] / pixels_h[:, 2, None]
        if not np.all(np.isfinite(pixels)) or np.max(np.abs(pixels)) > 1e6:
            raise ValueError(f"invalid projected socket pixels: {serial}")
        hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
        if len(hull) < 3:
            raise ValueError(f"socket projection has no area: {serial}")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        if dilation_px:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * dilation_px + 1, 2 * dilation_px + 1))
            mask = cv2.dilate(mask, kernel)
        masks[serial] = mask.astype(bool)
        diagnostics[serial] = {
            "projected_hull_bbox_px": [
                int(np.floor(pixels[:, 0].min())),
                int(np.floor(pixels[:, 1].min())),
                int(np.ceil(pixels[:, 0].max())),
                int(np.ceil(pixels[:, 1].max())),
            ],
            "excluded_pixels": int(np.count_nonzero(mask)),
            "image_size_wh": [width, height],
        }
    return SocketImageExclusion(
        masks, diagnostics, mode.socket_object, mesh_hash,
        socket_pose, dilation_px)


def key_mask_socket_overlap_fraction(
    key_mask: np.ndarray, exclusion_mask: np.ndarray,
) -> float:
    """Fraction of a proposed key mask that covers the fixed socket hull."""
    key = np.asarray(key_mask, dtype=bool)
    socket = np.asarray(exclusion_mask, dtype=bool)
    if key.ndim != 2 or key.shape != socket.shape or not np.any(key):
        raise ValueError("key mask and socket exclusion need matching nonempty images")
    return float(np.count_nonzero(key & socket) / np.count_nonzero(key))
