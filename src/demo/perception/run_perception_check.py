#!/usr/bin/env python3
"""Interactively verify AutoDex's live FoundPose perception pipeline.

This entry point deliberately does *not* plan or move the robot.  It uses the
same live initialisation path as ``src/demo/inference/run_demo.py``:

    frame -> undistort -> SAM3 mask -> per-view FoundPose
          -> cross-view IoU selection -> silhouette refinement

The capture-PC ``init_daemon`` instances must already be running::

    bash scripts/init_daemons.sh start
    python src/demo/perception/run_perception_check.py

At the prompt, enter a v8 object name.  Once an object has run, a blank line
repeats it; ``list`` prints the supported v8 objects and ``q`` exits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
DEFAULT_ASSETS_ROOT = Path.home() / "shared_data/AutoDex/foundpose_assets"
DEFAULT_OUTPUT_ROOT = Path.home() / "shared_data/AutoDex/experiment/perception_check"
V8_OBJECT_LIST = REPO_ROOT / "src/grasp_generation/obj_list_v8.txt"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-list", type=Path, default=V8_OBJECT_LIST,
                        help="v8 object allow-list (one name per line)")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT,
                        help="FoundPose asset root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="directory for per-run artifacts")
    parser.add_argument("--calib-dir", type=Path, default=None,
                        help="camera calibration directory; default is the latest cam_param")
    parser.add_argument("--pc-list", nargs="+", default=DEFAULT_PC_LIST,
                        help="capture PCs running init_daemon")
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    parser.add_argument("--stream-fps", type=int, default=10)
    parser.add_argument("--stream-warmup-s", type=float, default=2.0)
    parser.add_argument("--no-auto-start-stream", dest="auto_start_stream",
                        action="store_false",
                        help="assume an appropriate camera stream is already running")
    parser.set_defaults(auto_start_stream=True)
    parser.add_argument("--prompt", default="object on the checkerboard",
                        help="SAM3 text prompt")
    parser.add_argument("--timeout-s", type=float, default=60.0,
                        help="maximum wait for capture-PC mask/pose payloads")
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-lr", type=float, default=0.002)
    parser.add_argument("--sil-loss-max", type=float, default=0.003,
                        help="reject a pose above this silhouette loss")
    parser.add_argument("--sil-debug", action="store_true",
                        help="save silhouette optimiser debug images")
    parser.add_argument("--web", action="store_true",
                        help="run the same perception checker through a local web UI")
    parser.add_argument("--web-host", default="127.0.0.1",
                        help="web UI bind host (default: loopback only)")
    parser.add_argument("--web-port", type=int, default=8091,
                        help="web UI port when --web is set (default: 8091)")
    return parser.parse_args(argv)


def _read_object_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"v8 object list not found: {path}")
    names = [
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"v8 object list is empty: {path}")
    return sorted(set(names))


def _print_object_list(names: Iterable[str], columns: int = 4) -> None:
    names = list(names)
    width = max(len(name) for name in names) + 2
    print(f"[v8 objects] {len(names)} supported")
    for start in range(0, len(names), columns):
        print("  " + "".join(f"{name:<{width}}" for name in names[start:start + columns]).rstrip())


def _object_paths(name: str, object_root: Path,
                  assets_root: Path) -> tuple[Path, Path, Path]:
    mesh = object_root / name / "raw_mesh" / f"{name}.obj"
    # FoundPoseInit treats its ``assets_root`` argument as one object's
    # directory.  This is the same contract used by run_auto/run_demo:
    # ``ASSETS_BASE / object_name``, not the shared parent directory.
    object_assets_root = assets_root / name
    representation = (
        object_assets_root / "object_repre" / "v1" / name / "1" / "repre.pth"
    )
    return mesh, object_assets_root, representation


def _validate_object(name: str, supported: set[str], object_root: Path,
                     assets_root: Path) -> tuple[Optional[dict[str, Path]], list[str]]:
    if name not in supported:
        suggestions = difflib.get_close_matches(name, sorted(supported), n=3, cutoff=0.55)
        message = f"{name!r} is not in the v8 object list"
        if suggestions:
            message += "; did you mean " + ", ".join(repr(item) for item in suggestions) + "?"
        return None, [message]

    mesh, object_assets_root, representation = _object_paths(
        name, object_root, assets_root)
    problems = []
    if not mesh.is_file():
        problems.append(f"v8 mesh missing: {mesh}")
    if not representation.is_file():
        problems.append(f"FoundPose representation missing: {representation}")
    if problems:
        return None, problems
    return {
        "mesh": mesh,
        "assets_root": object_assets_root,
        "representation": representation,
    }, []


def _new_run_dir(output_root: Path, object_name: str) -> Path:
    parent = output_root / object_name
    parent.mkdir(parents=True, exist_ok=True)
    base = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / base
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _save_json(path: Path, value: Any) -> None:
    with path.open("w") as file:
        json.dump(_jsonable(value), file, indent=2, sort_keys=True)
        file.write("\n")


def _save_masks(masks: dict[str, dict[str, Any]], output_dir: Path) -> None:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    for serial, payload in masks.items():
        mask = payload.get("mask")
        if mask is None:
            continue
        cv2.imwrite(str(output_dir / f"{serial}.png"),
                    (np.asarray(mask, dtype=bool).astype(np.uint8) * 255))


def _make_overlay_grid(masks: dict[str, dict[str, Any]], poses: dict[str, dict[str, Any]],
                       capture_dir: Path, output_path: Path) -> Optional[Path]:
    """Save a compact per-camera mask/FoundPose diagnostic grid."""
    import cv2

    serials = sorted(set(masks) | set(poses))
    if not serials:
        return None
    tiles: list[np.ndarray] = []
    for serial in serials:
        mask = (masks.get(serial) or {}).get("mask")
        image_path = capture_dir / "images" / f"{serial}.png"
        image = cv2.imread(str(image_path)) if image_path.is_file() else None
        if image is None and mask is None:
            image = np.zeros((240, 320, 3), dtype=np.uint8)
        elif image is None:
            mask_arr = np.asarray(mask, dtype=bool)
            image = np.zeros((*mask_arr.shape, 3), dtype=np.uint8)
        else:
            image = image.copy()

        if mask is not None:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape != image.shape[:2]:
                mask_arr = cv2.resize(mask_arr.astype(np.uint8),
                                      (image.shape[1], image.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            green = np.zeros_like(image)
            green[:, :, 1] = 255
            image[mask_arr] = cv2.addWeighted(image, 0.45, green, 0.55, 0)[mask_arr]

        pose = poses.get(serial) or {}
        label = f"{serial} | {'FP OK' if pose.get('ok') else 'FP FAIL'}"
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(image, (320, 240)))

    ncols = min(5, len(tiles))
    nrows = math.ceil(len(tiles) / ncols)
    grid = np.full((nrows * 240, ncols * 320, 3), 32, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, ncols)
        grid[row * 240:(row + 1) * 240, col * 320:(col + 1) * 320] = tile
    cv2.imwrite(str(output_path), grid)
    return output_path


def _path_resource(key: str, label: str, path: Path, *, required: bool = True) -> dict[str, Any]:
    """Describe one local input/output used by the perception check."""
    try:
        exists = path.exists()
        is_dir = path.is_dir() if exists else False
        size = path.stat().st_size if exists and path.is_file() else None
    except OSError:
        exists = False
        is_dir = False
        size = None
    return {
        "key": key,
        "label": label,
        "path": str(path),
        "required": required,
        "exists": exists,
        "kind": "directory" if is_dir else "file",
        "size_bytes": size,
    }


def _object_resources(object_name: str, object_list: Path, object_root: Path,
                      assets_root: Path) -> list[dict[str, Any]]:
    """Return the concrete local resources required to initialise one object."""
    mesh, object_assets_root, representation = _object_paths(
        object_name, object_root, assets_root)
    return [
        _path_resource("v8_list", "v8 object allow-list", object_list),
        _path_resource("object_root", "v8 object root", object_root),
        _path_resource("mesh", "perception mesh", mesh),
        _path_resource("foundpose_assets", "FoundPose object assets", object_assets_root),
        _path_resource("representation", "FoundPose representation", representation),
        _path_resource("foundpose_summary", "FoundPose onboarding summary",
                       object_assets_root / "summary.json", required=False),
        _path_resource("representation_config", "FoundPose representation config",
                       representation.parent / "config.json", required=False),
        _path_resource("foundpose_model", "FoundPose model mesh",
                       object_assets_root / object_name / "models" / "obj_000001.ply",
                       required=False),
    ]


def _load_json_mapping(path: Path) -> dict[str, Any]:
    """Read a small metadata JSON file without making it a hard requirement."""
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _object_asset_profile(object_name: str, paths: dict[str, Path]) -> dict[str, Any]:
    """Small, safe-to-read metadata describing an opaque FoundPose archive."""
    assets_root = paths["assets_root"]
    summary = _load_json_mapping(assets_root / "summary.json")
    config = _load_json_mapping(paths["representation"].parent / "config.json")
    models_info = summary.get("models_info_entry")
    if not isinstance(models_info, dict):
        models_info = {}
    return {
        "object": object_name,
        "foundpose_summary_available": bool(summary),
        "representation_config_available": bool(config),
        "mesh_scale": summary.get("mesh_scale"),
        "geometry_mm": {
            key: models_info.get(key)
            for key in ("diameter", "size_x", "size_y", "size_z")
            if models_info.get(key) is not None
        },
        "template_count": summary.get("num_templates"),
        "feature_vector_count": summary.get("num_feature_vectors"),
        "extractor": summary.get("extractor_name", config.get("extractor_name")),
        "viewspheres": summary.get("num_viewspheres"),
        "min_viewpoints": summary.get("min_num_viewpoints"),
        "inplane_rotations": summary.get("num_inplane_rotations"),
        "pca_components": config.get("pca_components"),
        "cluster_count": config.get("cluster_num"),
        "depth_range_mm": summary.get("depth_range_mm"),
    }


def _mesh_preview_paths(preview_root: Path, object_name: str) -> tuple[Path, Path]:
    directory = preview_root / object_name
    return directory / "mesh_preview.png", directory / "mesh_preview.json"


def _write_mesh_preview(mesh_path: Path, preview_root: Path,
                        object_name: str) -> tuple[Path, dict[str, Any]]:
    """Cache a CPU-only three-view point rendering of the exact perception mesh.

    This intentionally renders the v8 ``raw_mesh`` passed to FoundPose rather
    than a planning mesh or an opaque representation archive.  A bounded vertex
    sample keeps the preview responsive even for scanned 40+ MB meshes.
    """
    import cv2
    import trimesh

    output_path, metadata_path = _mesh_preview_paths(preview_root, object_name)
    stat = mesh_path.stat()
    fingerprint = {"renderer_version": 2, "path": str(mesh_path),
                   "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    cached = _load_json_mapping(metadata_path)
    if output_path.is_file() and cached.get("fingerprint") == fingerprint:
        return output_path, cached.get("mesh", {})

    loaded = trimesh.load(str(mesh_path), process=False)
    mesh = loaded.dump(concatenate=True) if isinstance(loaded, trimesh.Scene) else loaded
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError(f"mesh contains no vertices: {mesh_path}")
    max_points = 45_000
    indices = (np.linspace(0, len(vertices) - 1, max_points, dtype=np.int64)
               if len(vertices) > max_points else np.arange(len(vertices)))
    points = vertices[indices]
    vertex_colours = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if vertex_colours is not None and len(vertex_colours) == len(vertices):
        colours = np.asarray(vertex_colours, dtype=np.uint8)[indices, :3][:, ::-1]  # RGB -> BGR
    else:
        # Stable axis-based colour preserves shape cues for untextured meshes.
        span = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-9)
        norm = (points - vertices.min(axis=0)) / span
        colours = np.stack([80 + 120 * norm[:, 2], 100 + 100 * norm[:, 1],
                            100 + 110 * norm[:, 0]], axis=1).astype(np.uint8)

    panel_w, panel_h, margin, header_h = 330, 340, 18, 54
    canvas = np.full((header_h + panel_h, panel_w * 3, 3), (18, 24, 38), dtype=np.uint8)
    # Object canonical axes are not guaranteed to align with a human notion of
    # front/up across the v8 pool, so label exact coordinate projections.
    views = ((0, 2, "X / Z projection"), (1, 2, "Y / Z projection"),
             (0, 1, "X / Y projection"))
    for panel, (axis_x, axis_y, label) in enumerate(views):
        x0 = panel * panel_w
        cv2.rectangle(canvas, (x0 + 6, header_h + 6),
                      (x0 + panel_w - 6, header_h + panel_h - 6), (55, 65, 81), 1)
        coords = points[:, [axis_x, axis_y]]
        low, high = vertices[:, [axis_x, axis_y]].min(axis=0), vertices[:, [axis_x, axis_y]].max(axis=0)
        span = np.maximum(high - low, 1e-9)
        scale = min((panel_w - 2 * margin) / span[0], (panel_h - 2 * margin) / span[1])
        mapped_x = x0 + panel_w / 2 + (coords[:, 0] - (low[0] + high[0]) / 2) * scale
        mapped_y = header_h + panel_h / 2 - (coords[:, 1] - (low[1] + high[1]) / 2) * scale
        for x, y, colour in zip(mapped_x.astype(np.int32), mapped_y.astype(np.int32), colours):
            if x0 + 6 <= x < x0 + panel_w - 6 and header_h + 6 <= y < header_h + panel_h - 6:
                canvas[y, x] = colour
        cv2.putText(canvas, label, (x0 + 12, header_h + 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (225, 231, 239), 1, cv2.LINE_AA)

    extents = np.asarray(mesh.extents, dtype=np.float64)
    title = f"{object_name} | perception raw mesh | {len(vertices):,} vertices / {len(mesh.faces):,} faces"
    subtitle = "extent: " + " x ".join(f"{value * 1000:.1f} mm" for value in extents)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (248, 250, 252), 1, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (12, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (156, 163, 175), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"could not write mesh preview: {output_path}")
    mesh_info = {
        "vertices": int(len(vertices)),
        "faces": int(len(mesh.faces)),
        "extents_m": extents.tolist(),
        "bounds_m": np.asarray(mesh.bounds, dtype=np.float64).tolist(),
        "sampled_vertices": int(len(points)),
    }
    _save_json(metadata_path, {"fingerprint": fingerprint, "mesh": mesh_info})
    return output_path, mesh_info


def _blend_mask(image: np.ndarray, mask: Optional[np.ndarray], color: tuple[int, int, int],
                alpha: float) -> np.ndarray:
    if mask is None:
        return image
    import cv2

    result = image.copy()
    value = np.asarray(mask, dtype=bool)
    if value.shape != result.shape[:2]:
        value = cv2.resize(value.astype(np.uint8), (result.shape[1], result.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
    if value.any():
        tint = np.empty_like(result)
        tint[:] = color
        result[value] = cv2.addWeighted(result, 1.0 - alpha, tint, alpha, 0)[value]
    return result


def _draw_tile_text(image: np.ndarray, first: str, second: str) -> None:
    import cv2

    for text, y in ((first, 23), (second, image.shape[0] - 10)):
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)


def _make_stage_grid(*, serials: Iterable[str], masks: dict[str, dict[str, Any]],
                     poses: dict[str, dict[str, Any]], capture_dir: Path,
                     output_path: Path, stage: str,
                     rendered_masks: Optional[dict[str, np.ndarray]] = None,
                     pose_color: tuple[int, int, int] = (0, 200, 0)) -> Optional[Path]:
    """Render one all-camera grid for a diagnostic perception stage.

    SAM3 masks are amber; a rendered mesh silhouette is stage-coloured.  Empty
    cameras remain visible as labelled tiles so a 0/N failure is not mistaken
    for an omitted view.
    """
    import cv2

    ordered = sorted(set(serials))
    if not ordered:
        return None
    tiles: list[np.ndarray] = []
    for serial in ordered:
        mask = (masks.get(serial) or {}).get("mask")
        pose = poses.get(serial) or {}
        image_path = capture_dir / "images" / f"{serial}.png"
        image = cv2.imread(str(image_path)) if image_path.is_file() else None
        if image is None:
            if mask is not None:
                image = np.zeros((*np.asarray(mask).shape[:2], 3), dtype=np.uint8)
            else:
                image = np.zeros((480, 640, 3), dtype=np.uint8)
        image = _blend_mask(image, mask, (0, 191, 255), 0.38)  # BGR amber
        if rendered_masks is not None:
            image = _blend_mask(image, rendered_masks.get(serial), pose_color, 0.50)

        fp_state = "FP OK" if pose.get("ok") else (
            "FP FAIL" if serial in poses else "pose missing")
        image_state = "image" if image_path.is_file() else "image missing"
        _draw_tile_text(image, f"{serial} | {fp_state}", image_state)
        tiles.append(cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA))

    ncols = min(5, len(tiles))
    nrows = math.ceil(len(tiles) / ncols)
    title_h = 34
    grid = np.full((title_h + nrows * 240, ncols * 320, 3), 32, dtype=np.uint8)
    title = f"{stage} | amber=SAM3 mask, coloured=rendered mesh"
    cv2.putText(grid, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (230, 230, 230), 1, cv2.LINE_AA)
    for index, tile in enumerate(tiles):
        row, col = divmod(index, ncols)
        y0 = title_h + row * 240
        grid[y0:y0 + 240, col * 320:(col + 1) * 320] = tile
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    return output_path


def _render_pose_masks(orch: Any, pose_world: np.ndarray, serials: Iterable[str],
                       image_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    """Project a world pose using the same silhouette renderer as refinement."""
    sil = getattr(orch, "_sil", None)
    if sil is None:
        raise RuntimeError("silhouette renderer is unavailable")
    rendered: dict[str, np.ndarray] = {}
    for serial in sorted(set(serials)):
        if serial not in orch.intrinsics_undist or serial not in orch.extrinsics:
            continue
        rendered[serial] = sil.render_silhouette_mask(
            pose_world=np.asarray(pose_world, dtype=np.float64),
            K=orch.intrinsics_undist[serial],
            extrinsic=orch.extrinsics[serial],
            image_hw=image_hw,
        )
    return rendered


def _save_per_view_foundpose(poses: dict[str, dict[str, Any]], path: Path) -> Path:
    """Persist per-camera pose results needed to explain a later grid."""
    view_data: dict[str, dict[str, Any]] = {}
    for serial, payload in poses.items():
        view_data[serial] = {
            "ok": bool(payload.get("ok", False)),
            "quality": payload.get("quality"),
            "inliers": payload.get("inliers"),
            "t_fp": payload.get("t_fp"),
        }
        if payload.get("pose_world") is not None:
            view_data[serial]["pose_world"] = np.asarray(
                payload["pose_world"], dtype=np.float64).reshape(4, 4)
    _save_json(path, view_data)
    return path


def _make_stage_visualizations(*, run_dir: Path, capture_dir: Path, serials: Iterable[str],
                               masks: dict[str, dict[str, Any]],
                               poses: dict[str, dict[str, Any]], orch: Any,
                               image_hw: tuple[int, int], timing: dict[str, Any],
                               pose_world: Optional[np.ndarray]) -> tuple[dict[str, Path], dict[str, str]]:
    """Write inspectable grids without changing the perception computation."""
    visual_dir = run_dir / "visualizations"
    assets: dict[str, Path] = {}
    errors: dict[str, str] = {}
    ordered = sorted(set(serials))

    def write(stage_key: str, title: str, rendered: Optional[dict[str, np.ndarray]],
              color: tuple[int, int, int]) -> None:
        try:
            output = visual_dir / f"{stage_key}.png"
            path = _make_stage_grid(serials=ordered, masks=masks, poses=poses,
                                    capture_dir=capture_dir, output_path=output,
                                    stage=title, rendered_masks=rendered,
                                    pose_color=color)
            if path is not None:
                assets[stage_key] = path
        except Exception as exc:
            errors[stage_key] = repr(exc)

    write("01_sam3_foundpose", "SAM3 + FoundPose availability", None, (0, 0, 0))

    try:
        individual: dict[str, np.ndarray] = {}
        for serial, payload in poses.items():
            if payload.get("ok") and payload.get("pose_world") is not None:
                individual.update(_render_pose_masks(
                    orch, np.asarray(payload["pose_world"]), [serial], image_hw))
        write("02_foundpose_candidates", "Per-view FoundPose candidate", individual,
              (180, 80, 180))  # BGR purple
    except Exception as exc:
        errors["02_foundpose_candidates"] = repr(exc)

    pre_sil = timing.get("pre_sil_pose")
    if pre_sil is not None:
        try:
            write("03_iou_selected", "Cross-view IoU selected pose",
                  _render_pose_masks(orch, np.asarray(pre_sil), ordered, image_hw),
                  (255, 150, 40))  # BGR blue
        except Exception as exc:
            errors["03_iou_selected"] = repr(exc)

    if pose_world is not None:
        try:
            write("04_silhouette_refined", "Silhouette refined final pose",
                  _render_pose_masks(orch, pose_world, ordered, image_hw),
                  (30, 210, 30))
        except Exception as exc:
            errors["04_silhouette_refined"] = repr(exc)
    return assets, errors


def _camera_summary(masks: dict[str, dict[str, Any]], poses: dict[str, dict[str, Any]],
                    expected_serials: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for serial in sorted(expected_serials):
        mask_payload = masks.get(serial) or {}
        pose_payload = poses.get(serial) or {}
        mask = mask_payload.get("mask")
        mask_pixels = int(np.asarray(mask, dtype=bool).sum()) if mask is not None else 0
        mask_total = int(np.asarray(mask).size) if mask is not None else 0
        rows.append({
            "serial": serial,
            "mask_received": serial in masks,
            "mask_pixels": mask_pixels,
            "mask_ratio": (mask_pixels / mask_total) if mask_total else 0.0,
            "sam3_s": mask_payload.get("t_sam3"),
            "pose_received": serial in poses,
            "foundpose_ok": bool(pose_payload.get("ok", False)),
            "foundpose_quality": pose_payload.get("quality"),
            "foundpose_inliers": pose_payload.get("inliers"),
            "foundpose_s": pose_payload.get("t_fp"),
        })
    return rows


def _print_camera_table(rows: list[dict[str, Any]]) -> None:
    print("  cameras:")
    print("    serial            mask%   SAM3     FoundPose  quality  inliers  FP")
    for row in rows:
        mask_text = f"{row['mask_ratio'] * 100:5.1f}" if row["mask_received"] else "  n/a"
        sam3_text = f"{float(row['sam3_s']):5.2f}s" if row["sam3_s"] is not None else "   n/a"
        fp_text = f"{float(row['foundpose_s']):5.2f}s" if row["foundpose_s"] is not None else "   n/a"
        quality = row["foundpose_quality"]
        quality_text = f"{float(quality):7.3f}" if quality is not None else "    n/a"
        inliers = row["foundpose_inliers"]
        inlier_text = f"{int(inliers):7d}" if inliers is not None else "    n/a"
        state = "OK" if row["foundpose_ok"] else ("FAIL" if row["pose_received"] else "missing")
        print(f"    {row['serial']:<16} {mask_text:>5}  {sam3_text:>6}  "
              f"{state:<9}  {quality_text}  {inlier_text}  {fp_text:>6}")


def _rotation_rpy_deg(rotation: np.ndarray) -> np.ndarray:
    """Return conventional xyz roll/pitch/yaw without adding scipy dependency."""
    value = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.hypot(value[0, 0], value[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(value[2, 1], value[2, 2])
        pitch = math.atan2(-value[2, 0], sy)
        yaw = math.atan2(value[1, 0], value[0, 0])
    else:
        roll = math.atan2(-value[1, 2], value[1, 1])
        pitch = math.atan2(-value[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _print_result(record: dict[str, Any]) -> None:
    timing = record.get("timing") or {}
    status = record.get("status", "FAIL")
    print("\n" + "=" * 78)
    print(f"[RESULT] {status}  |  {record['object']}")
    if record.get("reason"):
        print(f"reason       : {record['reason']}")
    print(f"views        : masks {record.get('n_masks_recv', 0)}/{record['n_expected']} | "
          f"poses {record.get('n_poses_recv', 0)}/{record['n_expected']} | "
          f"valid {record.get('n_candidates', 0)}")
    if record.get("missing_masks"):
        print("missing mask : " + ", ".join(record["missing_masks"]))
    if record.get("missing_poses"):
        print("missing pose : " + ", ".join(record["missing_poses"]))
    if timing.get("best_serial"):
        metric = (f"mean IoU={float(timing['best_iou']):.3f}"
                  if timing.get("best_iou") is not None else
                  f"quality={float(timing.get('best_quality', 0.0)):.3f}")
        print(f"selection    : serial={timing['best_serial']} | {metric}")
    if timing.get("sil_loss") is not None:
        print(f"silhouette   : loss={float(timing['sil_loss']):.6f} "
              f"(threshold {record['sil_loss_max']:.6f})")
    if record.get("pose_world") is not None:
        pose = np.asarray(record["pose_world"], dtype=np.float64).reshape(4, 4)
        xyz = ", ".join(f"{value:.4f}" for value in pose[:3, 3])
        rpy = ", ".join(f"{value:.1f}" for value in _rotation_rpy_deg(pose[:3, :3]))
        print(f"pose_world   : xyz=[{xyz}] m | rpy=[{rpy}] deg")
    print("timing       : "
          f"capture+FP={float(timing.get('dispatch_to_collected_s', 0.0)):.2f}s | "
          f"IoU={float(timing.get('iou_select_s', 0.0)):.2f}s | "
          f"silhouette={float(timing.get('sil_refine_s', 0.0)):.2f}s | "
          f"total={float(record.get('total_s', 0.0)):.2f}s")
    persist = (record.get("stage_timing") or {}).get("persist") or {}
    if persist.get("duration_s") is not None:
        print(f"artifact save: {float(persist['duration_s']):.2f}s (excluded from total)")
    _print_camera_table(record.get("cameras", []))
    print(f"artifacts    : {record['run_dir']}")
    print("=" * 78 + "\n")


def _prompt_object(previous: Optional[str], supported_names: list[str]) -> Optional[str]:
    prompt = f"Object [{previous}] > " if previous else "Object > "
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            print()
            return None
        if answer.lower() in {"q", "quit", "exit"}:
            return None
        if answer.lower() in {"list", "ls"}:
            _print_object_list(supported_names)
            continue
        if not answer:
            if previous:
                return previous
            print("[INPUT] no previous object; enter a v8 object name or 'list'.")
            continue
        return answer


def _report_progress(callback: Optional[Callable[[dict[str, Any]], None]], stage: str,
                     state: str, **details: Any) -> None:
    """Best-effort bridge from the checker to an optional UI observer."""
    if callback is None:
        return
    try:
        callback({"stage": stage, "state": state, **details})
    except Exception:
        # Status reporting must never turn a valid perception result into a
        # failure, particularly when a browser disconnects mid-run.
        pass


def _run_one(*, object_name: str, paths: dict[str, Path], args: argparse.Namespace,
             orch: Any, intrinsics: dict[str, dict[str, Any]],
             extrinsics: dict[str, np.ndarray], image_hw: tuple[int, int],
             pc_serials: dict[str, list[str]], initialized_object: Optional[str],
             calibration_dir: Optional[Path] = None,
             progress_callback: Optional[Callable[[dict[str, Any]], None]] = None) -> tuple[dict[str, Any], Optional[str]]:
    run_dir = _new_run_dir(args.output_root, object_name)
    capture_dir = run_dir / "capture"
    expected_serials = sorted(orch.intrinsics_undist) if initialized_object == object_name else sorted(intrinsics)
    record: dict[str, Any] = {
        "object": object_name,
        "run_dir": run_dir,
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "mesh_path": paths["mesh"],
        "foundpose_assets_root": paths["assets_root"],
        "representation_path": paths["representation"],
        "n_expected": len(expected_serials),
        "sil_loss_max": args.sil_loss_max,
        "prompt": args.prompt,
        "status": "FAIL",
        "resources": {
            "inputs": [
                _path_resource("mesh", "perception mesh", paths["mesh"]),
                _path_resource("foundpose_assets", "FoundPose object assets",
                               paths["assets_root"]),
                _path_resource("representation", "FoundPose representation",
                               paths["representation"]),
            ],
        },
    }
    if calibration_dir is not None:
        record["resources"]["inputs"].extend([
            _path_resource("calibration", "calibration directory", calibration_dir),
            _path_resource("intrinsics", "calibration intrinsics",
                           calibration_dir / "intrinsics.json"),
            _path_resource("extrinsics", "calibration extrinsics",
                           calibration_dir / "extrinsics.json"),
        ])
    started = time.perf_counter()
    masks: dict[str, dict[str, Any]] = {}
    poses: dict[str, dict[str, Any]] = {}
    timing: dict[str, Any] = {}
    pose_world: Optional[np.ndarray] = None
    stage_timing: dict[str, dict[str, Any]] = {}

    def report(stage: str, state: str, **details: Any) -> None:
        """Record a durable stage timeline and optionally update the web UI."""
        now_offset = time.perf_counter() - started
        entry = stage_timing.setdefault(stage, {})
        if "started_offset_s" not in entry and state in {"running", "done", "error", "skipped"}:
            entry["started_offset_s"] = now_offset
        entry["state"] = state
        entry["last_update_offset_s"] = now_offset
        entry.update(_jsonable(details))
        if state in {"done", "error", "skipped"}:
            entry["completed_offset_s"] = now_offset
            entry["duration_s"] = now_offset - float(entry.get("started_offset_s", now_offset))
        _report_progress(progress_callback, stage, state, **details)

    def report_from_orchestrator(event: dict[str, Any]) -> None:
        value = dict(event)
        stage = str(value.pop("stage", "unknown"))
        state = str(value.pop("state", "running"))
        report(stage, state, **value)

    try:
        print(f"\n[PERCEPTION CHECK] {object_name}")
        print("[1/4] Validate assets       PASS")
        report("validate", "done", object=object_name)
        if initialized_object != object_name:
            print("[2/4] Initialize object     loading FoundPose templates and silhouette mesh...")
            report("initialize", "running")
            init_started = time.perf_counter()
            orch.init_object(
                obj_name=object_name,
                mesh_path=str(paths["mesh"]),
                assets_root=str(paths["assets_root"]),
                intrinsics_full=intrinsics,
                extrinsics_full=extrinsics,
                image_hw=image_hw,
                mode="live",
                pc_serials=pc_serials,
            )
            record["object_init_s"] = time.perf_counter() - init_started
            initialized_object = object_name
            expected_serials = sorted(orch.intrinsics_undist)
            record["n_expected"] = len(expected_serials)
            print(f"[2/4] Initialize object     ready ({record['object_init_s']:.2f}s)")
            report("initialize", "done", elapsed_s=record["object_init_s"], reused=False)
        else:
            print("[2/4] Initialize object     reusing prior object state")
            report("initialize", "done", reused=True)

        print(f"[3/4] Collect observations  prompt={args.prompt!r}")
        report("collect", "running", n_masks_recv=0, n_poses_recv=0,
               n_expected=len(expected_serials))
        request_id = int(time.time() * 1000) & 0x7FFFFFFF
        masks, poses, capture_timing = orch.collect_payloads(
            prompt=args.prompt,
            request_id=request_id,
            n_expected_serials=len(expected_serials),
            timeout_s=args.timeout_s,
            save_capture_dir=str(capture_dir),
            progress_callback=report_from_orchestrator,
        )
        timing.update(capture_timing)
        print("[4/4] Refine pose           cross-view IoU selection -> silhouette refinement")
        report("iou", "queued")
        pose_world, refine_timing = orch.refine_from_payloads(
            masks,
            poses,
            sil_iters=args.sil_iters,
            sil_lr=args.sil_lr,
            sil_loss_threshold=args.sil_loss_max,
            save_capture_dir=str(capture_dir),
            sil_debug=args.sil_debug,
            selection_mode="iou",
            progress_callback=report_from_orchestrator,
        )
        timing.update(refine_timing)
        if pose_world is None:
            record["reason"] = timing.get("reason", "perception_failed")
        else:
            record["pose_world"] = np.asarray(pose_world, dtype=np.float64)
            missing = (set(expected_serials) - set(masks)) | (set(expected_serials) - set(poses))
            record["status"] = "WARN" if missing else "PASS"
    except KeyboardInterrupt:
        print("\n[perception] interrupted")
        record["reason"] = "interrupted"
    except Exception as exc:
        record["reason"] = "exception"
        record["exception"] = repr(exc)
        print(f"[perception] FAILED: {exc!r}")
    finally:
        # ``total_s`` is the actual perception wall time: object initialisation,
        # capture/SAM3/FoundPose, IoU, and silhouette refinement.  Persisting
        # images/JSON/diagnostic grids is deliberately outside that number.
        record["total_s"] = time.perf_counter() - started
        report("persist", "running", excluded_from_total=True)
        _save_masks(masks, run_dir / "masks")
        # JPEG/PNG writing on the capture PCs is deliberately asynchronous.
        # Give the local writer threads a short chance to finish before making
        # the overlay; the grid still falls back to a mask-only tile if needed.
        deadline = time.time() + 1.0
        while masks and time.time() < deadline:
            if all((capture_dir / "images" / f"{serial}.png").exists() for serial in masks):
                break
            time.sleep(0.05)
        overlay = _make_overlay_grid(masks, poses, capture_dir, run_dir / "mask_overlay_grid.png")
        record["overlay_grid"] = overlay
        per_view_path = _save_per_view_foundpose(poses, run_dir / "per_view_foundpose.json")
        if timing.get("pre_sil_pose") is not None:
            np.save(run_dir / "pose_pre_silhouette.npy",
                    np.asarray(timing["pre_sil_pose"], dtype=np.float64))
        record["timing"] = timing
        record["n_masks_recv"] = len(masks)
        record["n_poses_recv"] = len(poses)
        record["n_candidates"] = sum(
            bool(payload.get("ok")) and "pose_world" in payload for payload in poses.values())
        record["missing_masks"] = sorted(set(expected_serials) - set(masks))
        record["missing_poses"] = sorted(set(expected_serials) - set(poses))
        record["cameras"] = _camera_summary(masks, poses, expected_serials)
        if pose_world is not None:
            np.save(run_dir / "pose_world.npy", np.asarray(pose_world, dtype=np.float64))
        visual_started = time.perf_counter()
        visualizations, visualization_errors = _make_stage_visualizations(
            run_dir=run_dir, capture_dir=capture_dir, serials=expected_serials,
            masks=masks, poses=poses, orch=orch, image_hw=image_hw,
            timing=timing, pose_world=pose_world,
        )
        record["visualizations"] = visualizations
        if visualization_errors:
            record["visualization_errors"] = visualization_errors
        record["visualization_s"] = time.perf_counter() - visual_started
        artifact_resources = [
            _path_resource("per_view_foundpose", "per-view FoundPose data", per_view_path),
            _path_resource("capture_images", "captured images", capture_dir / "images"),
            _path_resource("saved_masks", "saved SAM3 masks", run_dir / "masks"),
        ]
        if pose_world is not None:
            artifact_resources.append(_path_resource(
                "final_pose", "final world pose", run_dir / "pose_world.npy"))
        if timing.get("pre_sil_pose") is not None:
            artifact_resources.append(_path_resource(
                "pre_silhouette_pose", "IoU-selected world pose",
                run_dir / "pose_pre_silhouette.npy"))
        artifact_resources.extend(
            _path_resource(f"visual_{key}", f"visualization: {key}", value)
            for key, value in visualizations.items()
        )
        record["resources"]["artifacts"] = artifact_resources
        record["resources"]["cameras"] = [
            {
                "serial": serial,
                "image_saved": (capture_dir / "images" / f"{serial}.png").is_file(),
                "mask_received": serial in masks,
                "mask_saved": (run_dir / "masks" / f"{serial}.png").is_file(),
                "foundpose_payload_received": serial in poses,
                "foundpose_ok": bool((poses.get(serial) or {}).get("ok", False)),
            }
            for serial in expected_serials
        ]
        stage_sources = {
            "per_view_foundpose": "collect",
            "capture_images": "collect",
            "saved_masks": "collect",
            "final_pose": "silhouette",
            "pre_silhouette_pose": "iou",
            "visual_01_sam3_foundpose": "collect",
            "visual_02_foundpose_candidates": "collect",
            "visual_03_iou_selected": "iou",
            "visual_04_silhouette_refined": "silhouette",
        }
        artifact_manifest = []
        for resource in artifact_resources:
            source_stage = stage_sources.get(resource["key"], "persist")
            source_info = stage_timing.get(source_stage, {})
            artifact_manifest.append({
                **resource,
                "source_stage": source_stage,
                "source_stage_completed_offset_s": source_info.get("completed_offset_s"),
                "saved_offset_s": time.perf_counter() - started,
                "saved_at": dt.datetime.now().isoformat(timespec="milliseconds"),
            })
        record["artifact_manifest"] = artifact_manifest
        report("persist", "done", artifact_count=len(artifact_manifest))
        record["stage_timing"] = stage_timing
        _save_json(run_dir / "summary.json", record)
        # Add the summary only after the first write so its resource state is
        # truthful in the JSON it contains.
        summary_resource = _path_resource("summary", "run summary", run_dir / "summary.json")
        record["resources"]["artifacts"].insert(0, summary_resource)
        record["artifact_manifest"].insert(0, {
            **summary_resource,
            "source_stage": "persist",
            "source_stage_completed_offset_s": stage_timing["persist"].get("completed_offset_s"),
            "saved_offset_s": time.perf_counter() - started,
            "saved_at": dt.datetime.now().isoformat(timespec="milliseconds"),
        })
        _save_json(run_dir / "summary.json", record)

    _print_result(record)
    _report_progress(progress_callback, "complete", "done", status=record["status"],
                     reason=record.get("reason"), run_dir=str(run_dir))
    return record, initialized_object


class PerceptionCheckWebController:
    """Thread-safe adapter exposing the unchanged checker through a web UI."""

    def __init__(self, *, args: argparse.Namespace, supported_names: list[str],
                 object_root: Path, calibration_dir: Path,
                 intrinsics: dict[str, dict[str, Any]],
                 extrinsics: dict[str, np.ndarray], image_hw: tuple[int, int],
                 pc_serials: dict[str, list[str]], orch: Any, rcc: Any):
        self.args = args
        self.supported_names = supported_names
        self.supported = set(supported_names)
        self.object_root = object_root
        self.calibration_dir = calibration_dir
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.image_hw = image_hw
        self.pc_serials = pc_serials
        self.orch = orch
        self.rcc = rcc
        self.initialized_object: Optional[str] = None
        self._lock = threading.RLock()
        self._preview_lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._runs: dict[str, dict[str, Any]] = {}
        self._latest_run_id: Optional[str] = None
        self._state: dict[str, Any] = {
            "state": "ready",
            "stage": "ready",
            "stages": {},
            "started_at": None,
            "object": None,
        }

    def _camera_status(self) -> dict[str, Any]:
        if self.rcc is None:
            return {"managed": False, "message": "existing stream; controller unchanged"}
        try:
            return {"managed": True, "status": self.rcc.get_status()}
        except Exception as exc:
            return {"managed": True, "error": repr(exc)}

    def session(self) -> dict[str, Any]:
        """A JSON-safe snapshot used by the dashboard's polling endpoint."""
        with self._lock:
            state = _jsonable(dict(self._state))
            latest = self._runs.get(self._latest_run_id or "")
            initialized = self.initialized_object
        resources = [
            _path_resource("v8_list", "v8 object allow-list", self.args.object_list),
            _path_resource("object_root", "v8 object root", self.object_root),
            _path_resource("foundpose_assets_root", "FoundPose assets root",
                           self.args.assets_root),
            _path_resource("calibration", "calibration directory", self.calibration_dir),
            _path_resource("intrinsics", "calibration intrinsics",
                           self.calibration_dir / "intrinsics.json"),
            _path_resource("extrinsics", "calibration extrinsics",
                           self.calibration_dir / "extrinsics.json"),
            {
                "key": "capture_pc_assets",
                "label": "capture-PC FoundPose asset mount",
                "path": "~/shared_data/AutoDex/foundpose_assets/<object>",
                "required": True,
                "exists": None,
                "kind": "remote",
                "note": ("init_daemon has no init acknowledgement; this becomes "
                         "confirmed only when a mask/pose payload arrives"),
            },
        ]
        return {
            "state": state,
            "initialized_object": initialized,
            "supported_object_count": len(self.supported_names),
            "active_camera_count": len(self.intrinsics),
            "image_hw": list(self.image_hw),
            "calibration_dir": str(self.calibration_dir),
            "pc_serials": self.pc_serials,
            "camera_controller": self._camera_status(),
            "resources": _jsonable(resources),
            "latest": _jsonable(latest) if latest is not None else None,
        }

    def inspect_object(self, value: str) -> dict[str, Any]:
        name = value.strip()
        paths, problems = _validate_object(name, self.supported, self.object_root,
                                            self.args.assets_root)
        suggestions = (difflib.get_close_matches(name, self.supported_names, n=3, cutoff=0.55)
                       if name and name not in self.supported else [])
        profile = _object_asset_profile(name, paths) if paths is not None else {}
        return {
            "object": name,
            "valid": paths is not None,
            "problems": problems,
            "suggestions": suggestions,
            "resources": _jsonable(_object_resources(
                name, self.args.object_list, self.object_root, self.args.assets_root)),
            "catalog_index": (self.supported_names.index(name) + 1
                              if name in self.supported else None),
            "asset_profile": _jsonable(profile),
        }

    def mesh_preview(self, value: str) -> Optional[dict[str, Any]]:
        """Generate/return a cached preview of the exact FoundPose mesh."""
        name = value.strip()
        paths, _ = _validate_object(name, self.supported, self.object_root,
                                    self.args.assets_root)
        if paths is None:
            return None
        preview_root = self.args.output_root / "_resource_previews"
        try:
            with self._preview_lock:
                path, mesh_info = _write_mesh_preview(paths["mesh"], preview_root, name)
        except Exception as exc:
            return {"error": repr(exc)}
        return {"path": path, "mesh": mesh_info}

    def _update_progress(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            if run_id != self._latest_run_id:
                return
            clean = _jsonable(event)
            stage = str(clean.get("stage", "unknown"))
            self._state["stage"] = stage
            self._state.setdefault("stages", {})[stage] = clean

    def start_run(self, value: str) -> dict[str, Any]:
        inspection = self.inspect_object(value)
        if not inspection["valid"]:
            return {"accepted": False, "inspection": inspection}
        object_name = inspection["object"]
        paths, _ = _validate_object(object_name, self.supported, self.object_root,
                                    self.args.assets_root)
        assert paths is not None
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return {"accepted": False, "busy": True,
                        "run_id": self._latest_run_id, "inspection": inspection}
            run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._latest_run_id = run_id
            self._state = {
                "state": "running",
                "stage": "validate",
                "stages": {"validate": {"stage": "validate", "state": "queued"}},
                "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                "object": object_name,
                "run_id": run_id,
            }

            def worker() -> None:
                try:
                    record, initialized = _run_one(
                        object_name=object_name,
                        paths=paths,
                        args=self.args,
                        orch=self.orch,
                        intrinsics=self.intrinsics,
                        extrinsics=self.extrinsics,
                        image_hw=self.image_hw,
                        pc_serials=self.pc_serials,
                        initialized_object=self.initialized_object,
                        calibration_dir=self.calibration_dir,
                        progress_callback=lambda event: self._update_progress(run_id, event),
                    )
                    with self._lock:
                        self.initialized_object = initialized
                        self._runs[run_id] = _jsonable(record)
                        self._state.update({
                            "state": "complete",
                            "stage": "complete",
                            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
                            "result": _jsonable(record),
                        })
                except Exception as exc:
                    with self._lock:
                        self._state.update({
                            "state": "error",
                            "stage": "complete",
                            "error": repr(exc),
                            "completed_at": dt.datetime.now().isoformat(timespec="seconds"),
                        })

            self._worker = threading.Thread(
                target=worker, name=f"perception-check-{object_name}", daemon=True)
            self._worker.start()
        return {"accepted": True, "run_id": run_id, "inspection": inspection}

    def run(self, run_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            if run_id == self._latest_run_id:
                state = dict(self._state)
                if run_id in self._runs:
                    state["result"] = self._runs[run_id]
                return _jsonable(state)
            value = self._runs.get(run_id)
        return _jsonable(value) if value is not None else None

    def history(self) -> list[dict[str, Any]]:
        with self._lock:
            values = list(self._runs.items())[-20:]
        return [{"run_id": run_id, "object": record.get("object"),
                 "status": record.get("status"), "run_dir": record.get("run_dir"),
                 "total_s": record.get("total_s")}
                for run_id, record in reversed(values)]


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    args.assets_root = args.assets_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.object_list = args.object_list.expanduser().resolve()
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")
    if args.sil_iters < 0:
        raise SystemExit("--sil-iters must be non-negative")
    if not (1 <= args.web_port <= 65535):
        raise SystemExit("--web-port must be in 1..65535")

    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.utils.path import get_obj_root
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.system import get_camera_list, get_pc_ip
    # Keep the hardware setup semantics in lockstep with the normal AutoDex
    # runners instead of maintaining local copies of these helpers.
    from src.demo.banana_test.run_demo import (
        CAM_PARAM_ROOT,
        _clear_camera_errors,
        _ensure_camera_lock,
        _load_calib,
        _rcc_start,
        _warn_if_not_streaming,
    )

    try:
        supported_names = _read_object_names(args.object_list)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"[setup] {exc}") from exc
    supported = set(supported_names)
    object_root = Path(get_obj_root("v8")).expanduser().resolve()
    if not object_root.is_dir():
        raise SystemExit(f"[setup] v8 object root missing: {object_root}")

    calib_dir = args.calib_dir.expanduser().resolve() if args.calib_dir else None
    if calib_dir is None:
        try:
            calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
        except (FileNotFoundError, IndexError) as exc:
            raise SystemExit(f"[setup] no calibration directories under {CAM_PARAM_ROOT}") from exc
    try:
        intrinsics, extrinsics, height, width = _load_calib(calib_dir)
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"[setup] invalid calibration {calib_dir}: {exc}") from exc

    try:
        pc_ips = [get_pc_ip(pc) for pc in args.pc_list]
        pc_serials = {pc: list(get_camera_list(pc)) for pc in args.pc_list}
    except Exception as exc:
        raise SystemExit(f"[setup] could not resolve capture PCs: {exc!r}") from exc
    active_serials = {serial for serials in pc_serials.values() for serial in serials}
    intrinsics = {serial: value for serial, value in intrinsics.items() if serial in active_serials}
    extrinsics = {serial: value for serial, value in extrinsics.items() if serial in active_serials}
    if not intrinsics or set(intrinsics) != set(extrinsics):
        raise SystemExit("[setup] active cameras and calibration do not have a common serial set")

    print("[setup] AutoDex live perception check")
    print(f"[setup] v8 list={args.object_list} ({len(supported_names)} objects)")
    print(f"[setup] object root={object_root}")
    print(f"[setup] calibration={calib_dir} | active cameras={len(intrinsics)}")
    print(f"[setup] artifacts={args.output_root}")

    rcc = None
    orch = None
    initialized_object: Optional[str] = None
    previous_object: Optional[str] = None
    try:
        if args.auto_start_stream:
            rcc = remote_camera_controller("perception_check", pc_list=args.pc_list,
                                           stall_timeout=15.0)
            if not _ensure_camera_lock(rcc):
                raise RuntimeError("camera daemons are controlled by another session")
            if not _clear_camera_errors(rcc):
                raise RuntimeError("capture cameras remain in an error state")
            print(f"[setup] starting camera stream @ {args.stream_fps} FPS...")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            if args.stream_warmup_s > 0:
                time.sleep(args.stream_warmup_s)
            if not _warn_if_not_streaming(rcc):
                raise RuntimeError("camera stream did not become healthy")
        else:
            print("[setup] using existing camera stream (controller unchanged)")

        orch = InitOrchestrator(
            pc_list=args.pc_list,
            capture_ips=pc_ips,
            port_mask=args.port_mask,
            port_pose=args.port_pose,
            port_cmd=args.port_cmd,
        )
        if args.web:
            from autodex.dashboard.perception_check import run_dashboard

            controller = PerceptionCheckWebController(
                args=args,
                supported_names=supported_names,
                object_root=object_root,
                calibration_dir=calib_dir,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_hw=(height, width),
                pc_serials=pc_serials,
                orch=orch,
                rcc=rcc,
            )
            print("[setup] web mode ready; select a v8 object in the browser.")
            print(f"[setup] open: http://{args.web_host}:{args.web_port}")
            run_dashboard(controller, args.web_host, args.web_port)
            return
        print("[setup] ready. Enter an object name, 'list', or 'q'.")
        while True:
            object_name = _prompt_object(previous_object, supported_names)
            if object_name is None:
                break
            paths, problems = _validate_object(object_name, supported, object_root,
                                                args.assets_root)
            if problems:
                for problem in problems:
                    print(f"[INVALID] {problem}")
                continue
            assert paths is not None
            record, initialized_object = _run_one(
                object_name=object_name,
                paths=paths,
                args=args,
                orch=orch,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                image_hw=(height, width),
                pc_serials=pc_serials,
                initialized_object=initialized_object,
                calibration_dir=calib_dir,
            )
            # A valid object becomes the repeat target even when its perception
            # failed. This lets the operator adjust the physical setup and hit
            # Enter without having to type the name again.
            previous_object = object_name
            del record
    except KeyboardInterrupt:
        print("\n[exit] interrupted")
    finally:
        if orch is not None:
            try:
                orch.close()
            except Exception as exc:
                print(f"[cleanup] orchestrator close failed: {exc!r}")
        if rcc is not None:
            try:
                rcc.stop()
            except Exception as exc:
                print(f"[cleanup] stream stop failed: {exc!r}")
            try:
                rcc.end()
            except Exception as exc:
                print(f"[cleanup] camera controller close failed: {exc!r}")
        print("[exit] perception check closed")


if __name__ == "__main__":
    main()
