#!/usr/bin/env python3
"""Build a local four-camera profile from commissioned camera assets.

This only copies/rebinds camera and runtime files. Grasp endpoint catalogs
must be screened separately with the ZeroDex straight-mount URDF; the old
AutoDex right-angle catalogs are deliberately not copied into this profile.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


CAMERAS = ("26053248", "25322639", "25305462", "25322642")
def _write_json(path: Path, record) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def build(*, source_stage: Path, shared_root: Path, calibration_dir: Path,
          c2r_npy: Path, output_dir: Path, qwen_model_dir: Path,
          capture_exchange_dir: Path) -> dict:
    source_stage = source_stage.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_exchange_dir = capture_exchange_dir.expanduser().resolve()
    capture_exchange_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration").mkdir(exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    for name in ("intrinsics.json", "extrinsics.json"):
        shutil.copy2(calibration_dir / name, output_dir / "calibration" / name)
    board_info = Path(__file__).resolve().parent / "config" / "zerodex_board9.json"
    shutil.copy2(board_info, output_dir / "charuco_board9.json")
    shutil.copy2(board_info, capture_exchange_dir / "charuco_board9.json")
    shutil.copy2(c2r_npy, output_dir / "calibration" / "C2R.npy")
    shutil.copy2(source_stage / "static_preflight_board9_gap20_20261013" /
                 "base_scene.json", output_dir / "base_scene.json")

    launch = json.loads((source_stage / "live_board9_gap20_20261013" /
                         "launch.json").read_text(encoding="utf-8"))
    launch.update(
        shared_root=str(shared_root), capture_root=str(output_dir),
        evidence_dir=str(output_dir / "runs" / "session_evidence"),
        base_scene_json=str(output_dir / "base_scene.json"),
        calib_dir=str(output_dir / "calibration"),
        c2r_npy=str(output_dir / "calibration" / "C2R.npy"),
        pc_list=["capture4"], capture_ips=["147.46.219.136"],
        camera_ids=list(CAMERAS),
        camera_acquisition_mode="stream",
        init_frame_transport="inline_png",
        minimum_socket_refined_iou=0.0,
        socket_prompt="large hollow white cylindrical socket with a wide flat circular base",
    )
    _write_json(output_dir / "launch.json", launch)

    runtime = json.loads((source_stage /
                          "franka_runtime_board9_gap20_distributed.json"
                          ).read_text(encoding="utf-8"))
    runtime["vlm"] = {
        key: value for key, value in runtime["vlm"].items()
        if key not in {"workers", "worker_root"}
    }
    runtime["vlm"].update(mode="local", model_id=str(qwen_model_dir))
    runtime["held_motion_monitor"]["daemon_binary_path"] = (
        str(shared_root.parent / "paradex" / "cpp" / "franka_daemon" /
            "build" / "franka_daemon"))
    _write_json(output_dir / "franka_runtime.json", runtime)

    catalogs = []
    missing_catalogs = []
    for gap in (1, 3, 5, 10, 15, 20):
        target = output_dir / f"recipient_gap{gap}_endpoint_catalog_straight.json"
        if target.is_file():
            catalogs.append(str(target))
        else:
            missing_catalogs.append(str(target))
    return {"launch": str(output_dir / "launch.json"),
            "robot_config": str(output_dir / "franka_runtime.json"),
            "catalogs": catalogs, "missing_straight_mount_catalogs": missing_catalogs,
            "camera_ids": list(CAMERAS)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-stage", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path,
                        default=Path.home() / "shared_data")
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--c2r-npy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "shared_data" / "AutoDex" /
                                "precision_insertion" / "zerodex_capture4")
    parser.add_argument("--qwen-model-dir", type=Path,
                        default=Path.home() / "shared_data" / "AutoDex" /
                                "models" / "Qwen3-VL-2B-Instruct")
    parser.add_argument("--capture-exchange-dir", type=Path,
                        default=Path("/mnt/paradex2/hyunsu/zerodex_capture4_runtime"))
    args = parser.parse_args(argv)
    print(json.dumps(build(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
