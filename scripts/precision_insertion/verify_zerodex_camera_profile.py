#!/usr/bin/env python3
"""Audit the pinned ZeroDex cameras against calibration and active ParaDex.

The default audit is read-only and reports blockers. ``--require-runtime``
returns non-zero unless every camera has the expected owner in the active
ParaDex profile and the frozen calibration has been physically confirmed.
It never edits ``system/current`` because guessing a capture-PC IP can direct
camera ownership commands at the wrong machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def audit(profile_path: Path, shared_root: Path, paradex_root: Path) -> dict:
    profile_path = profile_path.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    paradex_root = paradex_root.expanduser().resolve()
    profile = _json(profile_path)
    serials = [str(value) for value in profile["camera_serials"]]
    expected = {
        str(pc): [str(value) for value in values]
        for pc, values in profile["expected_pc_serials"].items()
    }
    blockers: list[str] = []
    failures: list[str] = []

    if len(serials) != len(set(serials)):
        failures.append("camera_serials contains duplicates")
    expected_flat = [serial for values in expected.values() for serial in values]
    if sorted(expected_flat) != sorted(serials):
        failures.append("expected_pc_serials does not partition camera_serials")
    if list(expected) != list(profile["pc_list"]):
        failures.append("expected_pc_serials key order differs from pc_list")

    snapshot = shared_root / profile["calibration_snapshot_relative"]
    calib_dir = shared_root / profile["calib_dir_relative"]
    required = [
        calib_dir / "intrinsics.json",
        calib_dir / "extrinsics.json",
        snapshot / "C2R.npy",
        snapshot / "provenance.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    failures.extend(f"missing calibration asset: {path}" for path in missing)
    provenance = {}
    if not missing:
        intrinsics = _json(calib_dir / "intrinsics.json")
        extrinsics = _json(calib_dir / "extrinsics.json")
        provenance = _json(snapshot / "provenance.json")
        for label, keys in (
            ("intrinsics", intrinsics),
            ("extrinsics", extrinsics),
            ("provenance", provenance.get("camera_serials", [])),
        ):
            if sorted(str(value) for value in keys) != sorted(serials):
                failures.append(f"{label} serial set differs from camera profile")
        if provenance.get("status") != "verified_on_physical_rig":
            blockers.append("frozen camera/hand-eye calibration is not physically confirmed")

    pc_path = paradex_root / "system" / "current" / "pc.json"
    network_path = paradex_root / "system" / "current" / "network.json"
    active_pc_info = {}
    if not pc_path.is_file():
        failures.append(f"active ParaDex pc.json missing: {pc_path}")
    else:
        active_pc_info = _json(pc_path)
        owners: dict[str, list[str]] = {
            serial: [
                pc for pc, info in active_pc_info.items()
                if serial in [str(value) for value in info.get("cam_list", [])]
            ]
            for serial in serials
        }
        for expected_pc, expected_serials in expected.items():
            if expected_pc not in active_pc_info:
                blockers.append(f"active ParaDex profile has no PC named {expected_pc}")
            for serial in expected_serials:
                if owners[serial] != [expected_pc]:
                    blockers.append(
                        f"camera {serial}: expected owner {expected_pc}, active owners={owners[serial]}"
                    )

    if not network_path.is_file():
        failures.append(f"active ParaDex network.json missing: {network_path}")
    else:
        network = _json(network_path)
        if "franka" not in network:
            blockers.append("active ParaDex network profile has no Franka endpoint")
        if "inspire" not in network:
            blockers.append("active ParaDex network profile has no Inspire endpoint")

    status = "PASS" if not failures and not blockers else (
        "FAIL" if failures else "BLOCKED"
    )
    return {
        "status": status,
        "profile": str(profile_path),
        "camera_serials": serials,
        "calibration_snapshot": str(snapshot),
        "active_paradex_pc_json": str(pc_path),
        "failures": failures,
        "blockers": blockers,
        "run_pipeline_arguments_when_ready": [
            "--pc_list", *profile["pc_list"],
            "--calib_dir", str(calib_dir),
            "--camera-sync", profile["capture_sync"],
        ],
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=repo_root / "assets/precision_insertion/zerodex_camera_profile.json",
    )
    parser.add_argument("--shared-root", type=Path, default=Path.home() / "shared_data")
    parser.add_argument("--paradex-root", type=Path, default=Path.home() / "paradex")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    result = audit(args.profile, args.shared_root, args.paradex_root)
    print(json.dumps(result, indent=2))
    if result["failures"] or (args.require_runtime and result["blockers"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
