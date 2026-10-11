#!/usr/bin/env python3
"""Replay stock AutoDex IoU selection and silhouette refinement, offline.

Consumes a hash-bound ``probe_foundpose_saved_views.py`` v2 report. It never
constructs the networked InitOrchestrator, captures a frame, or commands a
robot. The reported mask fit is not an independent metric pose measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from probe_foundpose_saved_views import _hash_file, _load_view  # noqa: E402
from precision_insertion.assets import AssetPaths  # noqa: E402
from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.geometry import validate_se3  # noqa: E402


def _load_bound_views(report: dict, *, shared_root: Path,
                      calib_dir: Path) -> tuple[Path, dict[str, dict], dict]:
    """Recheck every original source and only then expose saved candidates."""
    if (report.get("schema") != "precision_insertion_saved_foundpose_probe_v2" or
            report.get("robot_ready") is not False or
            report.get("part") not in {"key", "socket"}):
        raise ValueError("a non-ready saved FoundPose v2 report is required")
    mode = select_mode(report["family"], float(report["gap_mm"]))
    name = mode.key_object if report["part"] == "key" else mode.socket_object
    if name != report.get("object"):
        raise ValueError("report object differs from selected precision mode")
    assets = AssetPaths(shared_root, mode)
    mesh = assets.raw_mesh(name)
    repre = assets.foundpose_repre(name)
    if (_hash_file(mesh) != report.get("mesh_sha256") or
            _hash_file(repre) != report.get("repre_sha256")):
        raise ValueError("canonical mesh or representation changed since init")
    shot_dir = Path(report["shot_dir"]).expanduser().resolve()
    expected = report.get("views")
    if not isinstance(expected, list) or len(expected) < 2:
        raise ValueError("stock multi-view replay needs at least two views")
    views = {}
    for row in expected:
        if row.get("status") != "pose_estimated_not_validated":
            raise ValueError("all selected views need saved FoundPose candidates")
        camera = row["camera_id"]
        if camera in views:
            raise ValueError("duplicate camera in FoundPose source report")
        mask_path = Path(row["mask_path"]).expanduser().resolve()
        view = _load_view(shot_dir, calib_dir, camera, mask_path)
        for field in ("image_path", "image_sha256", "mask_path",
                      "mask_sha256", "shot_json_sha256", "intrinsics_sha256",
                      "extrinsics_sha256", "condition_id"):
            if view[field] != row.get(field):
                raise ValueError(f"saved FoundPose source changed: {field}")
        if view["condition_id"] != report.get("condition_id"):
            raise ValueError("view condition differs from report condition")
        view["pose_world"] = validate_se3(
            row["pose_world_m"], name=f"saved FoundPose {camera}")
        views[camera] = view
    return mesh, views, {"object": name, "mode": mode}


def _replay_stock(mesh: Path, views: dict[str, dict], *,
                  iterations: int, loss_threshold: float) -> tuple[dict, np.ndarray | None]:
    """Call unchanged InitOrchestrator refinement without its network constructor."""
    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.perception.pose_select import compute_cross_view_iou
    from precision_insertion.silhouette_compat import SolidMtlSilhouetteOptimizer

    if (type(iterations) is not int or iterations < 1 or
            not math.isfinite(loss_threshold) or loss_threshold <= 0):
        raise ValueError("stock silhouette settings must be positive")
    heights = {view["mask"].shape[0] for view in views.values()}
    widths = {view["mask"].shape[1] for view in views.values()}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("stock renderer requires one image size")
    # This object's only purpose is to reuse the exact stock method; calling
    # InitOrchestrator.__init__ here would contact capture PCs.
    offline = object.__new__(InitOrchestrator)
    offline.H, offline.W = heights.pop(), widths.pop()
    offline.intrinsics_undist = {s: v["K"] for s, v in views.items()}
    offline.extrinsics = {s: v["extrinsics"] for s, v in views.items()}
    offline._sil = SolidMtlSilhouetteOptimizer(str(mesh), device="cuda")
    masks = {s: {"mask": v["mask"]} for s, v in views.items()}
    poses = {s: {"ok": True, "pose_world": v["pose_world"]}
             for s, v in views.items()}
    refined, selection = offline.refine_from_payloads(
        masks, poses, subset_serials=list(views), sil_iters=iterations,
        sil_lr=0.002, sil_loss_threshold=loss_threshold,
        selection_mode="iou")
    intrinsics = offline.intrinsics_undist
    extrinsics = offline.extrinsics
    mask_arrays = {s: v["mask"] for s, v in views.items()}

    def score(pose: np.ndarray) -> dict:
        mean, per_view = compute_cross_view_iou(
            pose, mask_arrays, intrinsics, extrinsics,
            offline.H, offline.W, offline._sil.glctx,
            offline._sil.mesh_tensors)
        return {"mean_iou": mean, "per_view_iou": per_view}

    pre = validate_se3(selection["pre_sil_pose"], name="stock selected pose")
    diagnostics = {
        "stock_selection": selection,
        "pre_refinement": {"pose_world_m": pre.tolist(), **score(pre)},
        "post_refinement": None if refined is None else {
            "pose_world_m": validate_se3(refined, name="stock refined pose").tolist(),
            **score(refined)},
        "stock_accepts_silhouette": refined is not None,
    }
    return diagnostics, refined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--shared-root", type=Path, required=True)
    parser.add_argument("--calib-dir", type=Path, required=True)
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-loss-threshold", type=float, default=0.003)
    parser.add_argument("--output", type=Path, required=True,
                        help="new JSON file; refuses overwrite")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = args.source_report.expanduser().resolve()
    source_raw = source_path.read_bytes()
    source = json.loads(source_raw)
    mesh, views, selection = _load_bound_views(
        source, shared_root=args.shared_root.expanduser().resolve(),
        calib_dir=args.calib_dir.expanduser().resolve())
    result, _ = _replay_stock(
        mesh, views, iterations=args.sil_iters,
        loss_threshold=args.sil_loss_threshold)
    report = {
        "schema": "precision_insertion_saved_stock_refinement_v1",
        "object": selection["object"], "family": selection["mode"].family,
        "gap_mm": selection["mode"].gap_mm,
        "source_report": str(source_path),
        "source_report_sha256": hashlib.sha256(source_raw).hexdigest(),
        "source_camera_ids": list(views),
        "sil_iters": args.sil_iters,
        "sil_lr": 0.002,
        "sil_loss_threshold": args.sil_loss_threshold,
        **result,
        "mask_provenance": source["mask_provenance"],
        "camera_acquisition_times_verified": False,
        "scope": "stock_multiview_saved_mask_fit_not_metric_pose_ground_truth",
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
