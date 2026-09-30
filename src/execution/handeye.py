"""Arm-aware selection of camera-world to robot-base calibration.

Paradex's legacy ``load_current_C2R()`` selects the newest hand-eye session
globally.  That is unsafe on a rig shared by xArm and Franka: whichever arm was
calibrated last silently supplies the transform for both robots.  The helpers
here select the newest *matching-arm* session and persist its provenance beside
every copied ``C2R.npy``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from paradex.calibration.utils import handeye_calib_path


ARM_DOF = {"xarm": 6, "franka": 7}
ARM_EEF_LINK = {"xarm": "link6", "franka": "fr3_link8"}


def _index_dirs(session: Path) -> list[Path]:
    def key(path: Path):
        return (0, int(path.name)) if path.name.isdigit() else (1, path.name)

    return sorted((path for path in session.iterdir() if path.is_dir()), key=key)


def _session_arm(session: Path) -> tuple[Optional[str], str]:
    """Return ``(arm, evidence)`` without guessing an ambiguous session."""
    index_dirs = _index_dirs(session)
    meta_candidates = [session / "meta.json"]
    meta_candidates.extend(path / "meta.json" for path in index_dirs)
    for meta_path in meta_candidates:
        if not meta_path.is_file():
            continue
        try:
            with meta_path.open() as handle:
                meta = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
        arm = meta.get("arm")
        if arm in ARM_DOF:
            return str(arm), f"metadata:{meta_path.relative_to(session)}"
        eef_link = meta.get("eef_link")
        for name, expected_link in ARM_EEF_LINK.items():
            if eef_link == expected_link:
                return name, f"metadata_eef:{meta_path.relative_to(session)}"

    # Older captures predate meta.json. Infer only from an unambiguous qpos
    # dimension; six and seven are the deployed xArm and Franka contracts.
    observed_dofs = set()
    for index_dir in index_dirs:
        qpos_path = index_dir / "qpos.npy"
        if not qpos_path.is_file():
            continue
        try:
            qpos = np.load(qpos_path, mmap_mode="r")
        except (OSError, ValueError):
            continue
        if qpos.ndim == 1:
            observed_dofs.add(int(qpos.shape[0]))
    if len(observed_dofs) == 1:
        dof = observed_dofs.pop()
        for name, expected_dof in ARM_DOF.items():
            if dof == expected_dof:
                return name, f"qpos_dof:{dof}"
    return None, "ambiguous"


def _c2r_path(session: Path) -> Optional[Path]:
    for index_dir in _index_dirs(session):
        path = index_dir / "C2R.npy"
        if path.is_file():
            return path
    direct = session / "C2R.npy"
    return direct if direct.is_file() else None


def _validated_c2r(path: Path) -> np.ndarray:
    matrix = np.asarray(np.load(path), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid C2R matrix in {path}: shape={matrix.shape}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"invalid homogeneous C2R matrix in {path}")
    return matrix


def load_arm_C2R(
    arm: str,
    *,
    calibration_root: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Load the newest valid hand-eye calibration belonging to ``arm``.

    Sessions are ordered by their timestamp-style directory names, matching the
    existing Paradex convention. A newer calibration for the other robot is
    skipped instead of silently reused.
    """
    if arm not in ARM_DOF:
        raise ValueError(f"unsupported arm {arm!r}; expected one of {tuple(ARM_DOF)}")
    root = Path(calibration_root or handeye_calib_path).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"hand-eye calibration root not found: {root}")

    rejected = []
    sessions = sorted(
        (path for path in root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for session in sessions:
        session_arm, evidence = _session_arm(session)
        if session_arm != arm:
            continue
        path = _c2r_path(session)
        if path is None:
            rejected.append(f"{session.name}: missing C2R.npy")
            continue
        try:
            matrix = _validated_c2r(path)
        except (OSError, ValueError) as exc:
            rejected.append(f"{session.name}: {exc}")
            continue
        info = {
            "arm": arm,
            "eef_link": ARM_EEF_LINK[arm],
            "session": session.name,
            "source": str(path),
            "arm_evidence": evidence,
        }
        return matrix, info

    detail = f" Rejected matching sessions: {'; '.join(rejected)}" if rejected else ""
    raise FileNotFoundError(
        f"no valid {arm} hand-eye calibration under {root}.{detail}")


def save_arm_C2R(
    save_path: str | Path,
    arm: str,
    *,
    calibration_root: str | Path | None = None,
) -> tuple[np.ndarray, dict]:
    """Persist arm-matched ``C2R.npy`` and ``C2R_meta.json`` in ``save_path``."""
    matrix, info = load_arm_C2R(arm, calibration_root=calibration_root)
    destination = Path(save_path)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "C2R.npy", matrix)
    with (destination / "C2R_meta.json").open("w") as handle:
        json.dump(info, handle, indent=2)
    print(f"[handeye] {arm} -> {info['session']}")
    return matrix, info
