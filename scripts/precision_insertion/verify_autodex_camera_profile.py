#!/usr/bin/env python3
"""Audit precision insertion against the active AutoDex camera contract.

This command is read-only.  It resolves the deployed camera serials from
ParaDex ``system/current/pc.json``, verifies that one AutoDex calibration
covers all of them, and checks that the established hardware-trigger and
timestamp-camera configuration is present.  It never substitutes a ZeroDex
camera profile.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _latest_complete_calibration(root: Path) -> Path | None:
    sessions = sorted(
        (
            path for path in root.iterdir()
            if path.is_dir()
            and (path / "intrinsics.json").is_file()
            and (path / "extrinsics.json").is_file()
        ),
        key=lambda path: path.name,
    ) if root.is_dir() else []
    return sessions[-1] if sessions else None


def audit(
    profile_path: Path,
    paradex_root: Path,
    calibration_root: Path,
    calibration_dir: Path | None,
    *,
    check_handeye: bool = True,
) -> dict:
    profile = _json(profile_path.expanduser().resolve())
    paradex_root = paradex_root.expanduser().resolve()
    current = paradex_root / "system" / "current"
    pc_path = current / "pc.json"
    network_path = current / "network.json"
    failures: list[str] = []
    blockers: list[str] = []
    notes: list[str] = []

    pc_info = _json(pc_path) if pc_path.is_file() else {}
    if not pc_info:
        failures.append(f"active ParaDex PC profile is missing: {pc_path}")
    pc_list = [str(value) for value in profile["pc_list"]]
    serials: list[str] = []
    for pc in pc_list:
        entry = pc_info.get(pc)
        if not isinstance(entry, dict):
            failures.append(f"active ParaDex profile has no PC named {pc}")
            continue
        if not entry.get("ip"):
            failures.append(f"{pc} has no IP address")
        cameras = [str(value) for value in entry.get("cam_list", [])]
        if not cameras:
            failures.append(f"{pc} has no camera serials")
        serials.extend(cameras)
    duplicates = sorted({value for value in serials if serials.count(value) > 1})
    if duplicates:
        failures.append(f"camera serials have multiple owners: {duplicates}")

    network = _json(network_path) if network_path.is_file() else {}
    signal = network.get("signal_generator")
    timestamp = network.get("timestamp")
    if not (isinstance(signal, dict) and isinstance(signal.get("param"), dict)):
        blockers.append(
            "network.json lacks AutoDex signal_generator.param for UTGE900"
        )
    if not (isinstance(timestamp, dict)
            and isinstance(timestamp.get("param"), dict)):
        blockers.append(
            "network.json lacks AutoDex timestamp.param for TimestampMonitor"
        )
    if "franka" not in network:
        blockers.append("network.json has no Franka endpoint")
    if "inspire" not in network:
        blockers.append("network.json has no Inspire endpoint")

    calibration_root = calibration_root.expanduser().resolve()
    selected = (calibration_dir.expanduser().resolve()
                if calibration_dir is not None
                else _latest_complete_calibration(calibration_root))
    intrinsics: dict = {}
    extrinsics: dict = {}
    if selected is None:
        failures.append(f"no complete camera calibration under {calibration_root}")
    else:
        intrinsics_path = selected / "intrinsics.json"
        extrinsics_path = selected / "extrinsics.json"
        if not intrinsics_path.is_file() or not extrinsics_path.is_file():
            failures.append(f"incomplete camera calibration: {selected}")
        else:
            intrinsics = _json(intrinsics_path)
            extrinsics = _json(extrinsics_path)
            active = set(serials)
            missing_intrinsics = sorted(active - set(intrinsics))
            missing_extrinsics = sorted(active - set(extrinsics))
            if missing_intrinsics:
                blockers.append(
                    f"calibration misses active intrinsics: {missing_intrinsics}"
                )
            if missing_extrinsics:
                blockers.append(
                    f"calibration misses active extrinsics: {missing_extrinsics}"
                )
            extras = sorted((set(intrinsics) | set(extrinsics)) - active)
            if extras:
                notes.append(
                    f"calibration has {len(extras)} inactive serial(s), ignored: {extras}"
                )

    handeye = None
    if check_handeye:
        try:
            from src.execution.handeye import load_arm_C2R

            _matrix, handeye = load_arm_C2R("franka")
        except Exception as exc:
            blockers.append(
                f"no valid arm-matched Franka hand-eye calibration: {exc}")
    else:
        blockers.append("Franka hand-eye audit explicitly skipped")

    status = "PASS" if not failures and not blockers else (
        "FAIL" if failures else "BLOCKED"
    )
    return {
        "schema_version": 1,
        "audited_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": status,
        "profile": str(profile_path.expanduser().resolve()),
        "active_pc_json": str(pc_path),
        "active_network_json": str(network_path),
        "pc_list": pc_list,
        "active_camera_serials": serials,
        "active_camera_count": len(serials),
        "calibration_dir": str(selected) if selected is not None else None,
        "calibrated_camera_count": len(intrinsics),
        "franka_handeye": handeye,
        "capture_sync": "hardware",
        "run_pipeline_arguments_when_ready": (
            ["--pc_list", *pc_list, "--calib_dir", str(selected)]
            if selected is not None else ["--pc_list", *pc_list]
        ),
        "failures": failures,
        "blockers": blockers,
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", type=Path,
        default=REPO_ROOT / "assets/precision_insertion/autodex_camera_profile.json",
    )
    parser.add_argument("--paradex-root", type=Path, default=Path.home() / "paradex")
    parser.add_argument(
        "--calibration-root", type=Path,
        default=Path.home() / "shared_data" / "cam_param",
    )
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-handeye", action="store_true",
        help="Development-only: skip slow/NFS hand-eye discovery and leave a blocker.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Optional JSON audit output, normally under "
             "~/shared_data/AutoDex/precision_insertion.",
    )
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.profile, args.paradex_root, args.calibration_root, args.calib_dir,
        check_handeye=not args.skip_handeye,
    )
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.partial")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    if result["failures"] or (args.require_runtime and result["blockers"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
