#!/usr/bin/env python3
"""Read-only FoundPose replay of hash-bound, saved AutoDex camera views.

This deliberately reuses stock ``FoundPoseInit`` and the demo's canonical
asset paths. A saved SAM mask is an input, not a verified object identity.
No live camera, ChArUco session, robot, or VLM is contacted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.geometry import pose_angle_deg  # noqa: E402
from precision_insertion.symmetry import load_axial_symmetry  # noqa: E402


def _hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_view(shot_dir: Path, calib_dir: Path, camera_id: str,
               mask_path: Path) -> dict:
    shot_raw = (shot_dir / "shot.json").read_bytes()
    shot = json.loads(shot_raw)
    if shot.get("status") != "COMPLETE" or camera_id not in shot.get("images", {}):
        raise ValueError("view is absent from a COMPLETE shot")
    info = shot["images"][camera_id]
    image_name = info.get("file")
    if not isinstance(image_name, str) or Path(image_name).name != image_name:
        raise ValueError("shot image path must be a filename")
    image_path = shot_dir / image_name
    image_raw = image_path.read_bytes()
    if (_hash(image_raw) != info.get("sha256") or
            len(image_raw) != info.get("bytes")):
        raise ValueError("shot image differs from recorded bytes/hash")
    intr_raw = (calib_dir / "intrinsics.json").read_bytes()
    extr_raw = (calib_dir / "extrinsics.json").read_bytes()
    if (_hash(intr_raw) != shot.get("intrinsics_sha256") or
            _hash(extr_raw) != shot.get("extrinsics_sha256")):
        raise ValueError("camera calibration differs from shot provenance")
    intr = json.loads(intr_raw)[camera_id]
    extr = np.asarray(json.loads(extr_raw)[camera_id], dtype=np.float64)
    if extr.shape == (3, 4):
        extr = np.vstack((extr, [0., 0., 0., 1.]))
    if (extr.shape != (4, 4) or not np.isfinite(extr).all() or
            not np.allclose(extr[3], [0., 0., 0., 1.])):
        raise ValueError("camera extrinsics must be a finite homogeneous pose")
    raw_bgr = cv2.imdecode(np.frombuffer(image_raw, np.uint8),
                           cv2.IMREAD_COLOR)
    if (raw_bgr is None or raw_bgr.shape[:2] !=
            (info.get("height"), info.get("width"))):
        raise ValueError("shot image dimensions differ from provenance")
    k_original = np.asarray(intr["original_intrinsics"], dtype=np.float64)
    k_undistorted = np.asarray(intr["intrinsics_undistort"], dtype=np.float64)
    distortion = np.asarray(intr["dist_params"], dtype=np.float64)
    if (k_original.shape != (3, 3) or k_undistorted.shape != (3, 3) or
            not np.isfinite(k_original).all() or
            not np.isfinite(k_undistorted).all()):
        raise ValueError("camera intrinsics are invalid")
    undistorted = cv2.undistort(raw_bgr, k_original, distortion,
                                None, k_undistorted)
    mask_raw = mask_path.read_bytes()
    mask = cv2.imdecode(np.frombuffer(mask_raw, np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    if (mask is None or mask.shape != undistorted.shape[:2] or
            not np.isin(mask, [0, 255]).all()):
        raise ValueError("saved mask must be a binary 0/255 undistorted frame")
    return {
        "camera_id": camera_id,
        "rgb": cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB),
        "mask": mask > 0,
        "K": k_undistorted,
        "extrinsics": extr,
        "image_path": str(image_path),
        "image_sha256": _hash(image_raw),
        "mask_path": str(mask_path),
        "mask_sha256": _hash(mask_raw),
        "shot_json_sha256": _hash(shot_raw),
        "intrinsics_sha256": _hash(intr_raw),
        "extrinsics_sha256": _hash(extr_raw),
        "condition_id": shot.get("condition_id"),
    }


def _render_mask(mesh_path: Path, pose_world: np.ndarray,
                 extrinsics: np.ndarray, K: np.ndarray,
                 image_shape: tuple[int, int]) -> np.ndarray:
    """Independent EGL silhouette diagnostic; not stock refinement."""
    import pyrender
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("runtime mesh is not a single Trimesh")
    # AutoDex solid-MTL OBJ has no texture image; keep geometry and use an
    # opaque vertex color for the depth-only render.
    mesh.visual = trimesh.visual.ColorVisuals(
        mesh=mesh, vertex_colors=np.tile(
            np.array([180, 180, 180, 255], dtype=np.uint8),
            (len(mesh.vertices), 1)))
    cv_to_gl = np.diag([1., -1., -1., 1.])
    mesh_pose_gl = cv_to_gl @ extrinsics @ pose_world
    h, w = image_shape
    scene = pyrender.Scene()
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False),
              pose=mesh_pose_gl)
    scene.add(pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]), fy=float(K[1, 1]),
        cx=float(K[0, 2]), cy=float(K[1, 2]),
        znear=0.01, zfar=5.0), pose=np.eye(4))
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    try:
        depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    finally:
        renderer.delete()
    return depth > 0


def _validate_condition(condition: object, family: str,
                        object_name: str, part: str,
                        socket_name: str | None = None) -> None:
    if not isinstance(condition, str) or not condition.startswith(family + "/"):
        raise ValueError("shot condition does not match the requested family")
    parts = condition.split("/")
    if part == "key":
        if family == "cylinder":
            # Cylinder handoff conditions are indexed by socket gap, not by
            # the one shared cylindrical key ID.
            if (socket_name is None or len(parts) != 4 or
                    parts[1] != socket_name or parts[2] != "with_key" or
                    not parts[3].startswith("pose_")):
                raise ValueError("shot condition does not identify the requested key scene")
        elif object_name not in parts:
            raise ValueError("shot condition does not identify the requested key")
    if part == "socket" and not (
            family == "square" and condition == "square/socket_only" or
            family == "cylinder" and len(parts) == 3 and
            parts[1:] == [object_name, "socket_only"]):
        raise ValueError("shot condition does not identify the requested socket")


def _pose_spread(rows: list[dict], *, center_local: np.ndarray,
                 axis_local: np.ndarray, end_exchange: bool) -> dict:
    valid = [row for row in rows if row["status"] ==
             "pose_estimated_not_validated"]
    origin_mm, center_mm, axis_deg, rotation_deg = [], [], [], []
    for i, left in enumerate(valid):
        a = np.asarray(left["pose_world_m"], dtype=np.float64)
        center_a = a[:3, :3] @ center_local + a[:3, 3]
        for right in valid[i + 1:]:
            b = np.asarray(right["pose_world_m"], dtype=np.float64)
            center_b = b[:3, :3] @ center_local + b[:3, 3]
            origin_mm.append(float(1000 * np.linalg.norm(
                a[:3, 3] - b[:3, 3])))
            center_mm.append(float(1000 * np.linalg.norm(center_a - center_b)))
            dot = float(np.clip((a[:3, :3] @ axis_local) @
                                (b[:3, :3] @ axis_local), -1., 1.))
            if end_exchange:
                dot = abs(dot)
            axis_deg.append(float(math.degrees(math.acos(dot))))
            rotation_deg.append(pose_angle_deg(a, b))
    return {
        "max_pairwise_frame_origin_mm": max(origin_mm, default=None),
        "max_pairwise_center_mm": max(center_mm, default=None),
        "max_pairwise_axis_deg": max(axis_deg, default=None),
        "max_pairwise_rotation_deg": max(rotation_deg, default=None),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--family", choices=("square", "cylinder"), required=True)
    parser.add_argument("--gap-mm", type=float, required=True)
    parser.add_argument("--part", choices=("key", "socket"), required=True)
    parser.add_argument("--shot-dir", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--gotrack-root", type=Path, required=True)
    parser.add_argument("--view", nargs=2, action="append", required=True,
                        metavar=("CAMERA_ID", "SAVED_MASK_PNG"))
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON report; refuses overwrite")
    args = parser.parse_args(argv)
    mode = select_mode(args.family, args.gap_mm)
    name = mode.key_object if args.part == "key" else mode.socket_object
    paths = AssetPaths(args.shared_root.expanduser().resolve(), mode)
    mesh_path = paths.raw_mesh(name)
    repre_path = paths.foundpose_repre(name)
    if not mesh_path.is_file() or not repre_path.is_file():
        raise FileNotFoundError("exact runtime mesh and representation are required")
    gotrack = args.gotrack_root.expanduser().resolve()
    for source in (gotrack, gotrack / "external/bop_toolkit",
                   gotrack / "external/dinov2"):
        if not source.is_dir():
            raise FileNotFoundError(f"MV-GoTrack source is incomplete: {source}")
        sys.path.insert(0, str(source))
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("EGL_PLATFORM", "surfaceless")
    shot_dir = args.shot_dir.expanduser().resolve()
    calib_dir = args.calib_dir.expanduser().resolve()
    cameras = [item[0] for item in args.view]
    if len(cameras) != len(set(cameras)):
        raise ValueError("saved-view replay needs unique camera IDs")
    views = [_load_view(shot_dir, calib_dir, camera,
                        Path(mask).expanduser().resolve())
             for camera, mask in args.view]
    if len({view["shot_json_sha256"] for view in views}) != 1:
        raise ValueError("all views must be from the same shot")
    _validate_condition(views[0]["condition_id"], mode.family, name, args.part,
                        mode.socket_object)

    # Fail early on output collision before loading a 700+ MB representation.
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"report already exists: {output}")
    from autodex.perception.foundpose_init import FoundPoseInit

    started = time.perf_counter()
    model = FoundPoseInit(str(mesh_path), str(paths.foundpose_assets_root(name)),
                          name)
    model_load_s = time.perf_counter() - started
    import trimesh
    cad_mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if not isinstance(cad_mesh, trimesh.Trimesh):
        raise ValueError("runtime mesh is not a single Trimesh")
    # The frame origin can move to the opposite end for a D∞ key. Its CAD
    # bounds centre is a more meaningful, end-exchange-invariant diagnostic.
    center_local = np.asarray(cad_mesh.bounds.mean(axis=0), dtype=np.float64)
    symmetry = (load_axial_symmetry(paths.object_root, name)
                if mode.family == "cylinder" else None)
    axis_local = (symmetry.axis_local if symmetry is not None else
                  np.array([0., 0., 1.]))
    rows = []
    for view in views:
        started = time.perf_counter()
        result = model.estimate_one_view(view["rgb"], view["mask"],
                                         view["K"], view["extrinsics"])
        row = {key: value for key, value in view.items()
               if key not in {"rgb", "mask", "K", "extrinsics"}}
        row["mask_pixels"] = int(view["mask"].sum())
        row["elapsed_s"] = time.perf_counter() - started
        if result is None:
            row["status"] = "no_foundpose_pose"
        else:
            pose = np.asarray(result["pose_world"], dtype=np.float64)
            if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
                    not np.allclose(pose[3], [0., 0., 0., 1.])):
                raise ValueError("FoundPose returned an invalid world pose")
            rendered = _render_mask(mesh_path, pose, view["extrinsics"],
                                    view["K"], view["mask"].shape)
            intersection = int(np.logical_and(rendered, view["mask"]).sum())
            union = int(np.logical_or(rendered, view["mask"]).sum())
            row.update({
                "status": "pose_estimated_not_validated",
                "pose_world_m": pose.tolist(),
                "cad_bounds_center_world_m": (
                    pose[:3, :3] @ center_local + pose[:3, 3]).tolist(),
                "quality": float(result["quality"]),
                "inliers": int(result["inliers"]),
                "template_id": int(result["template_id"]),
                "predicted_pixels": int(rendered.sum()),
                "mask_iou": intersection / union if union else 0.0,
            })
        rows.append(row)
    spread = _pose_spread(
        rows, center_local=center_local, axis_local=axis_local,
        end_exchange=bool(symmetry is not None and symmetry.end_exchange))
    report = {
        "schema": "precision_insertion_saved_foundpose_probe_v2",
        "object": name, "family": mode.family, "gap_mm": mode.gap_mm,
        "part": args.part, "shot_dir": str(shot_dir),
        "condition_id": views[0]["condition_id"],
        "mesh_path": str(mesh_path), "mesh_sha256": _hash_file(mesh_path),
        "repre_path": str(repre_path),
        "repre_sha256": _hash_file(repre_path),
        "model_load_s": model_load_s, "views": rows,
        "cad_bounds_center_local_m": center_local.tolist(),
        "axis_end_exchange_equivalent": bool(
            symmetry is not None and symmetry.end_exchange),
        **spread,
        "mask_provenance": "user_supplied_saved_mask_hash_recorded_not_authenticated_to_shot",
        "scope": "saved_mask_pose_diagnostic_not_independent_ground_truth",
        "camera_acquisition_times_verified": False,
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
