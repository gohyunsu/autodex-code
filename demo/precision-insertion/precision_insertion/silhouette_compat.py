"""Demo-local silhouette compatibility for solid-colour precision OBJ/MTL.

The unchanged AutoDex renderer hands a trimesh ``TextureVisuals`` with no
texture image to FoundationPose, which calls ``material.image.convert`` and
crashes. This adapter changes only the visual representation for that exact
case. The same raw OBJ, vertices, faces, units and frame are retained; it does
not alter FoundPose's representation or relax any perception-quality gate.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import trimesh

from autodex.perception.silhouette import SilhouetteOptimizer


class SolidMtlSilhouetteOptimizer(SilhouetteOptimizer):
    """Use vertex colour only when an OBJ's MTL has no texture image."""

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        mesh = super()._load_mesh(mesh_path)
        if (isinstance(mesh.visual, trimesh.visual.TextureVisuals) and
                getattr(mesh.visual.material, "image", None) is None):
            color = np.asarray(
                getattr(mesh.visual.material, "main_color", None),
                dtype=np.uint8).reshape(-1)
            if color.size == 3:
                color = np.append(color, np.uint8(255))
            if color.size != 4:
                raise ValueError("solid MTL has no usable RGBA colour")
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh, vertex_colors=np.tile(color, (len(mesh.vertices), 1)))
            if (not np.array_equal(mesh.vertices, vertices) or
                    not np.array_equal(mesh.faces, faces)):
                raise ValueError("silhouette compatibility changed CAD geometry")
        return mesh


def prepare_key_silhouette(
    *, init_orchestrator, object_name: str, raw_mesh: Path,
) -> str:
    """Prepare the robot-side renderer before AutoDex key init/capture.

    This does not patch a global class or modify stock source. ``init_object``
    must subsequently be called with ``load_silhouette=False`` so its daemon
    initialization runs unchanged while this compatible local renderer stays
    installed on the same stock orchestrator instance.
    """
    stock = getattr(init_orchestrator, "stock", init_orchestrator)
    if (not isinstance(object_name, str) or not object_name or
            not isinstance(raw_mesh, Path)):
        raise ValueError("key silhouette needs an object name and raw mesh path")
    mesh = raw_mesh.expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(mesh)
    device = getattr(stock, "device", None)
    if not isinstance(device, str) or not device:
        raise ValueError("stock init orchestrator has no renderer device")
    digest = hashlib.sha256(mesh.read_bytes()).hexdigest()
    current = getattr(stock, "_sil", None)
    if (isinstance(current, SolidMtlSilhouetteOptimizer) and
            getattr(current, "_obj_name", None) == object_name and
            getattr(current, "_precision_mesh_sha256", None) == digest and
            current.device == device):
        return digest
    renderer = SolidMtlSilhouetteOptimizer(str(mesh), device=device)
    renderer._obj_name = object_name
    renderer._precision_mesh_sha256 = digest
    stock._sil = renderer
    return digest
