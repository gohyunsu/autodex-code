#!/usr/bin/env python3
"""Build a local four-camera profile from commissioned camera assets.

This only copies/rebinds camera and runtime files. Grasp endpoint catalogs
must be screened separately with the ZeroDex straight-mount URDF; the old
AutoDex right-angle catalogs are deliberately not copied into this profile.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


# On the current board, 25305462 contributes a repeatable ~4 px corner-8
# mismatch. Use the other three only for board triangulation; retain all four
# for socket/key perception and three selected views for VLM.
CAMERAS = ("26053248", "25322639", "25305462", "25322642")
BOARD_FIT_CAMERAS = ("26053248", "25322639", "25322642")


def _serials(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _write_json(path: Path, record) -> None:
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def build(*, source_stage: Path, shared_root: Path, calibration_dir: Path,
          c2r_npy: Path, output_dir: Path, qwen_model_dir: Path,
          capture_ip: str, camera_ids: tuple[str, ...],
          board_fit_camera_ids: tuple[str, ...],
          default_gap_mm: int = 5) -> dict:
    source_stage = source_stage.expanduser().resolve()
    shared_root = shared_root.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calibration").mkdir(exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)
    for name in ("intrinsics.json", "extrinsics.json"):
        shutil.copy2(calibration_dir / name, output_dir / "calibration" / name)
    board_info = Path(__file__).resolve().parent / "config" / "zerodex_board9.json"
    shutil.copy2(board_info, output_dir / "charuco_board9.json")
    shutil.copy2(c2r_npy, output_dir / "calibration" / "C2R.npy")
    shutil.copy2(source_stage / "static_preflight_board9_gap20_20261013" /
                 "base_scene.json", output_dir / "base_scene.json")

    launch = json.loads((source_stage / "live_board9_gap20_20261013" /
                         "launch.json").read_text(encoding="utf-8"))
    launch.update(
        mode={"family": "cylinder", "gap_mm": default_gap_mm},
        shared_root=str(shared_root), capture_root=str(output_dir),
        evidence_dir=str(output_dir / "runs" / "session_evidence"),
        base_scene_json=str(output_dir / "base_scene.json"),
        calib_dir=str(output_dir / "calibration"),
        c2r_npy=str(output_dir / "calibration" / "C2R.npy"),
        pc_list=["capture4"], capture_ips=[capture_ip],
        camera_ids=list(camera_ids),
        board_fit_camera_ids=list(board_fit_camera_ids),
        camera_acquisition_mode="stream",
        init_frame_transport="inline_png",
        upright_cylinder_only=False,
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
            "camera_ids": list(camera_ids),
            "board_fit_camera_ids": list(board_fit_camera_ids)}


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
    parser.add_argument("--capture-ip", default=os.environ.get(
        "PRECISION_CAPTURE4_IP", "147.46.219.136"))
    parser.add_argument("--camera-ids", type=_serials, default=_serials(
        os.environ.get("PRECISION_CAPTURE4_CAMERA_IDS", ",".join(CAMERAS))))
    parser.add_argument("--board-fit-camera-ids", type=_serials,
                        default=_serials(os.environ.get(
                            "PRECISION_BOARD_FIT_CAMERA_IDS",
                            ",".join(BOARD_FIT_CAMERAS))))
    parser.add_argument("--default-gap-mm", type=int, default=5,
                        help="gap in the generated launch template; the runner may override it")
    args = parser.parse_args(argv)
    print(json.dumps(build(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
