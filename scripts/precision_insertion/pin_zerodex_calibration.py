#!/usr/bin/env python3
"""Freeze one complete ZeroDex camera + Franka hand-eye calibration snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


DEFAULT_CAM_SESSION = "20261002_141639"
DEFAULT_HANDEYE_SESSION = "20261002_145508"
DEFAULT_SERIALS = ("25305462", "25322639", "25322642", "26053248")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pin(
    shared_root: Path,
    cam_session: str,
    handeye_session: str,
    serials: tuple[str, ...],
) -> Path:
    shared_root = shared_root.expanduser().resolve()
    cam_source = shared_root / "cam_param_jisoo" / cam_session
    handeye_source = shared_root / "handeye_calibration_jisoo" / handeye_session / "0"
    intrinsics_path = cam_source / "intrinsics.json"
    extrinsics_path = cam_source / "extrinsics.json"
    c2r_path = handeye_source / "C2R.npy"
    qpos_path = handeye_source / "qpos.npy"
    for path in (intrinsics_path, extrinsics_path, c2r_path, qpos_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    intrinsics = json.loads(intrinsics_path.read_text())
    extrinsics = json.loads(extrinsics_path.read_text())
    requested = set(serials)
    intr_keys, extr_keys = set(intrinsics), set(extrinsics)
    if requested != intr_keys or requested != extr_keys:
        raise ValueError(
            "pinned serial set must exactly match both calibration files; "
            f"requested={sorted(requested)}, intrinsics={sorted(intr_keys)}, "
            f"extrinsics={sorted(extr_keys)}"
        )

    c2r = np.asarray(np.load(c2r_path), dtype=np.float64)
    qpos = np.asarray(np.load(qpos_path), dtype=np.float64)
    if c2r.shape != (4, 4) or not np.allclose(c2r[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"invalid C2R: {c2r_path}")
    if qpos.shape != (7,):
        raise ValueError(f"hand-eye session is not identifiable as Franka 7-DoF: {qpos_path}")

    snapshot_name = f"zerodex_4cam_{cam_session}_franka_{handeye_session}"
    destination = shared_root / "AutoDex" / "precision_insertion" / "calibration" / snapshot_name
    cam_destination = destination / "cam_param"
    cam_destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(intrinsics_path, cam_destination / "intrinsics.json")
    shutil.copy2(extrinsics_path, cam_destination / "extrinsics.json")
    shutil.copy2(c2r_path, destination / "C2R.npy")
    shutil.copy2(qpos_path, destination / "handeye_qpos.npy")

    provenance = {
        "schema_version": 1,
        "status": "pinned_pending_physical_rig_confirmation",
        "camera_source": str(cam_source),
        "handeye_source": str(handeye_source),
        "camera_serials": list(serials),
        "camera_count": len(serials),
        "arm": "franka",
        "arm_evidence": "handeye qpos has 7 joints",
        "files": {
            "cam_param/intrinsics.json": _sha256(cam_destination / "intrinsics.json"),
            "cam_param/extrinsics.json": _sha256(cam_destination / "extrinsics.json"),
            "C2R.npy": _sha256(destination / "C2R.npy"),
            "handeye_qpos.npy": _sha256(destination / "handeye_qpos.npy"),
        },
        "warning": (
            "ZeroDex config/cameras.py still lists six cameras. Confirm that the physical rig "
            "currently uses exactly these four serials before changing status to verified."
        ),
    }
    (destination / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument("--cam-session", default=DEFAULT_CAM_SESSION)
    parser.add_argument("--handeye-session", default=DEFAULT_HANDEYE_SESSION)
    parser.add_argument("--serials", nargs="+", default=list(DEFAULT_SERIALS))
    args = parser.parse_args()
    pin(args.shared_root, args.cam_session, args.handeye_session, tuple(args.serials))


if __name__ == "__main__":
    main()
