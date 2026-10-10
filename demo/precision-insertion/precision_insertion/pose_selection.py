"""Match one fresh perceived key pose to the selected v8 tabletop stem."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .assets import AssetPaths
from .config import TaskMode
from .geometry import validate_se3
from .symmetry import match_axisymmetric_tabletop_pose


def classify_key_tabletop_pose(
    *, mode: TaskMode, shared_root: Path, pose_robot_key: np.ndarray,
    max_rotation_error_deg: float,
) -> dict:
    """Use AutoDex square classes or the demo's local-z D∞ cylinder class.

    This checks rotational class consistency, not FoundPose confidence or
    whether the key is really supported by the measured table. It never
    silently picks the nearest class beyond the explicit error bound.
    """
    threshold = float(max_rotation_error_deg)
    if not np.isfinite(threshold) or not 0 < threshold < 90:
        raise ValueError("tabletop rotation error bound must be in (0, 90) deg")
    pose = validate_se3(pose_robot_key, name="fresh T_robot_key")
    root = Path(shared_root).expanduser().resolve()
    paths = AssetPaths(root, mode)
    if mode.family == "cylinder":
        _, stem, error = match_axisymmetric_tabletop_pose(
            pose, object_root=paths.object_root,
            object_name=mode.key_object,
            max_axis_error_deg=threshold)
        method = "v8_local_z_Dinf_axis"
    elif mode.family == "square":
        from src.experiment.reset.tabletop_pose import classify_tabletop_pose

        result = classify_tabletop_pose(
            pose, mode.key_object, str(paths.object_root))
        if result is None:
            raise FileNotFoundError("square key v8 tabletop poses are missing")
        stem = Path(result["filename"]).stem
        error = float(result["rot_err_deg"])
        if not np.isfinite(error) or error > threshold:
            raise ValueError(
                f"square key does not match a v8 tabletop pose "
                f"(error {error:.2f} deg)")
        method = "AutoDex_v8_tabletop_rotation"
    else:
        raise ValueError(f"unsupported key family: {mode.family}")
    if not (paths.key_tabletop_dir / f"{stem}.npy").is_file():
        raise FileNotFoundError(f"selected v8 tabletop pose missing: {stem}")
    return {
        "schema": "precision_insertion_tabletop_class_v1",
        "stem": stem,
        "rotation_error_deg": error,
        "max_rotation_error_deg": threshold,
        "method": method,
        "source": "fresh_key_pose_not_physical_pose_certification",
    }
