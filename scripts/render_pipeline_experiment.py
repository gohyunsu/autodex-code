#!/usr/bin/env python3
"""One-command renderer for an AutoDex experiment recording.

Example:
    python scripts/render_pipeline_experiment.py \
      --exp-name v8_video_13 --obj apple --video 1000085551.mp4

The command resolves the latest matching pipeline run, regenerates every grasp
from calibrated camera 25322649, composites the external video, then verifies
the camera/calibration inputs, white PNG backgrounds, sync residual, and full
MP4 decode.  Verification is written beside the output video.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

# The wrapper should also work when invoked by the system Python. Rendering
# and pixel-level verification require the AutoDex environment, so re-exec
# there before importing its binary packages.
try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    autodex_python = Path("/home/robot/anaconda3/envs/autodex/bin/python")
    if autodex_python.is_file() and Path(sys.executable).resolve() != autodex_python:
        os.execv(str(autodex_python),
                 [str(autodex_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.render_pipeline_video import (  # noqa: E402
    GRASP_CAMERA_SERIAL,
    resolve_run_dir,
)


def resolve_video_path(video: str | Path, downloads_dir: str | Path) -> Path:
    candidate = Path(video).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    elif candidate.is_file():
        resolved = candidate.resolve()
    else:
        resolved = (Path(downloads_dir).expanduser() / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"external video not found: {resolved}")
    return resolved


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _variant_name(no_timeline: bool, outcome_style: str) -> str:
    name = "no_timeline" if no_timeline else "timeline"
    if outcome_style == "emoji":
        name += "_emoji"
    return name


def _variant_suffix(no_timeline: bool, outcome_style: str) -> str:
    name = _variant_name(no_timeline, outcome_style)
    return "" if name == "timeline" else f"_{name}"


def _to_4x4(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.size == 12:
        result = np.eye(4, dtype=np.float64)
        result[:3, :] = matrix.reshape(3, 4)
        return result
    if matrix.size == 16:
        return matrix.reshape(4, 4)
    raise ValueError("camera extrinsic must contain 12 or 16 values")


def verify_output(
    run_dir: Path, *, camera_serial: str, no_timeline: bool = False,
    outcome_style: str = "bar",
) -> dict[str, Any]:
    suffix = _variant_suffix(no_timeline, outcome_style)
    expected_mode = _variant_name(no_timeline, outcome_style)
    metadata_path = run_dir / "edit" / f"pipeline_video{suffix}.json"
    metadata = _read_json(metadata_path)
    output = Path(metadata["output_video"])
    if not output.is_file():
        raise FileNotFoundError(f"rendered video not found: {output}")
    visualization = metadata.get("grasp_visualization") or {}
    actual_outcome_style = visualization.get("outcome_style", "bar")
    if actual_outcome_style != outcome_style:
        raise RuntimeError(
            f"metadata outcome style is {actual_outcome_style}, "
            f"expected {outcome_style}")
    if str(visualization.get("camera_serial")) != camera_serial:
        raise RuntimeError(
            f"metadata camera is {visualization.get('camera_serial')}, "
            f"expected {camera_serial}")

    attempts = metadata.get("attempts") or []
    if not attempts:
        raise RuntimeError("render metadata contains no grasp attempts")
    grasp_checks = []
    for attempt in attempts:
        episode_dir = Path(attempt["episode_dir"])
        if not episode_dir.is_absolute():
            episode_dir = (run_dir / episode_dir).resolve()
        render_path = Path(attempt["render_path"])
        if not render_path.is_absolute():
            render_path = run_dir / render_path
        if f"_cam{camera_serial}.png" not in render_path.name:
            raise RuntimeError(f"camera-specific cache name missing: {render_path}")
        image = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot read grasp render: {render_path}")
        corners = (image[0, 0], image[0, -1], image[-1, 0], image[-1, -1])
        corners_white = all(bool(np.all(pixel == 255)) for pixel in corners)
        if not corners_white:
            raise RuntimeError(f"grasp render background is not RGB white: {render_path}")

        cam_param = episode_dir / "cam_param"
        intrinsics = _read_json(cam_param / "intrinsics.json")
        extrinsics = _read_json(cam_param / "extrinsics.json")
        if camera_serial not in intrinsics or camera_serial not in extrinsics:
            raise RuntimeError(
                f"camera {camera_serial} calibration missing in {episode_dir}")
        c2r = np.asarray(np.load(episode_dir / "C2R.npy"), dtype=np.float64).reshape(4, 4)
        camera_to_robot = np.linalg.inv(c2r) @ np.linalg.inv(
            _to_4x4(extrinsics[camera_serial]))
        rotation = camera_to_robot[:3, :3]
        orthogonality_error = float(
            np.max(np.abs(rotation.T @ rotation - np.eye(3))))
        determinant = float(np.linalg.det(rotation))
        if orthogonality_error > 1e-5 or abs(determinant - 1.0) > 1e-5:
            raise RuntimeError(
                f"invalid camera rotation for {attempt['episode_id']}: "
                f"orthogonality={orthogonality_error}, det={determinant}")
        grasp_checks.append({
            "episode_id": attempt["episode_id"],
            "render_path": str(render_path),
            "camera_serial": camera_serial,
            "background_corners_rgb255": True,
            "rotation_orthogonality_error": orthogonality_error,
            "rotation_determinant": determinant,
        })

    sync_audit = metadata.get("sync_audit") or {}
    residual = float(sync_audit.get("anchor_residual_s", float("inf")))
    if abs(residual) > 1e-6:
        raise RuntimeError(f"non-zero Enter-anchor sync residual: {residual}")

    layout = metadata.get("layout") or {}
    if no_timeline:
        width, height = layout.get("canvas", [0, 0])
        rail_x, rail_y, rail_width, rail_height = layout.get(
            "history_column", [0, 0, 0, 0])
        if metadata.get("render_mode") != expected_mode:
            raise RuntimeError(
                f"metadata mode is {metadata.get('render_mode')}, "
                f"expected {expected_mode}")
        if layout.get("pipeline_diagram_enabled") is not False:
            raise RuntimeError("pipeline diagram is enabled in no-timeline layout")
        if (rail_y != 0 or rail_height != height or rail_width * 4 != height
                or rail_x + rail_width != width
                or int(layout.get("stack_rows", 0)) != 4):
            raise RuntimeError(f"invalid no-timeline grasp stack layout: {layout}")

    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(output),
         "-f", "null", "-"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"full video decode failed:\n{decode.stderr}")

    report = {
        "schema_version": 1,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": metadata.get("run_id"),
        "output_video": str(output),
        "video_bytes": output.stat().st_size,
        "camera_serial": camera_serial,
        "grasp_count": len(grasp_checks),
        "all_grasp_calibrations_valid": True,
        "all_grasp_background_corners_rgb255": True,
        "sync_anchor": sync_audit.get("anchor_event"),
        "sync_anchor_residual_s": residual,
        "full_video_decode": "passed",
        "render_mode": metadata.get("render_mode", "timeline"),
        "outcome_style": actual_outcome_style,
        "layout": layout,
        "grasps": grasp_checks,
    }
    report_path = run_dir / "edit" / f"render_verification{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve, render, and verify one AutoDex experiment video")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--obj", required=True)
    parser.add_argument("--video", required=True,
                        help="Absolute path or filename under --downloads-dir")
    parser.add_argument("--hand", default="inspire")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--experiment-root",
        default=str(Path.home() / "shared_data" / "AutoDex" / "experiment"))
    parser.add_argument("--downloads-dir", default=str(Path.home() / "Downloads"))
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--video-offset-s", type=float, default=0.0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--render-python", default=None)
    parser.add_argument("--reuse-grasp-renders", action="store_true",
                        help="Reuse camera-specific PNGs instead of forcing regeneration")
    parser.add_argument("--keep-trailing-interruptions", action="store_true")
    parser.add_argument("--no-timeline", action="store_true",
                        help="Render and verify the timeline-free layout variant")
    parser.add_argument(
        "--outcome-style", choices=("bar", "emoji"), default="bar",
        help="Executed-grasp outcome marker (default: bar)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Verify an existing output without rendering")
    args = parser.parse_args()

    object_dir = (Path(args.experiment_root).expanduser() / args.exp_name
                  / args.hand / args.obj)
    run_dir = resolve_run_dir(object_dir, args.run_id)
    video_path = resolve_video_path(args.video, args.downloads_dir)
    renderer = REPO_ROOT / "scripts" / "render_pipeline_video.py"
    known_python = Path("/home/robot/anaconda3/envs/autodex/bin/python")
    pipeline_python = str(known_python if known_python.is_file() else Path(sys.executable))

    if not args.verify_only:
        subprocess.run(
            [pipeline_python, str(REPO_ROOT / "scripts" /
                                  "build_pipeline_edit_package.py"),
             "--run", str(run_dir)],
            cwd=REPO_ROOT, check=True,
        )
        command = [
            pipeline_python, str(renderer),
            "--experiment", str(run_dir),
            "--video", str(video_path),
            "--speed", str(args.speed),
            "--video-offset-s", str(args.video_offset_s),
            "--crf", str(args.crf), "--preset", args.preset,
        ]
        if not args.reuse_grasp_renders:
            command.append("--force-grasp-renders")
        if args.keep_trailing_interruptions:
            command.append("--keep-trailing-interruptions")
        if args.no_timeline:
            command.append("--no-timeline")
        command += ["--outcome-style", args.outcome_style]
        if args.output:
            command += ["--output", str(Path(args.output).expanduser().resolve())]
        if args.render_python:
            command += ["--render-python", args.render_python]
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    report = verify_output(
        run_dir, camera_serial=GRASP_CAMERA_SERIAL,
        no_timeline=args.no_timeline, outcome_style=args.outcome_style)
    verification_suffix = _variant_suffix(
        args.no_timeline, args.outcome_style)
    verification_name = f"render_verification{verification_suffix}.json"
    print(json.dumps({
        "run": str(run_dir),
        "video": report["output_video"],
        "grasp_count": report["grasp_count"],
        "camera_serial": report["camera_serial"],
        "sync_anchor_residual_s": report["sync_anchor_residual_s"],
        "full_video_decode": report["full_video_decode"],
        "render_mode": report["render_mode"],
        "outcome_style": report["outcome_style"],
        "verification": str(run_dir / "edit" / verification_name),
    }, indent=2))


if __name__ == "__main__":
    main()
