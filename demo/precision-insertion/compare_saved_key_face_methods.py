#!/usr/bin/env python3
"""Compare saved SAM3 face-center candidates and render evidence-only figures.

This reads existing undistorted AutoDex captures and saved masks. It cannot
connect to the camera or robot, and its 3D points are review proxies only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from precision_insertion.assets import AssetPaths
from precision_insertion.config import select_mode
from precision_insertion.held_key_tip import padded_key_box_roi, triangulate_tip_proxy


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GREEN = "#42f58d"  # observed SAM ellipse/center
CYAN = "#00daf3"   # planned geometry or measured socket pose
PINK = "#f85dc7"   # Qwen box or multiview reprojection
YELLOW = "#ffe15b"  # SAM mask
METHOD_BY_PANEL = {
    "observed_ellipse": "sam_half_ellipse",
    "spatial_proxy": "sam_half_ellipse",
    "sam_boundary_end": "sam_boundary_end",
    "sam_rectangle_end": "sam_rectangle_end",
    "unconstrained_cv_ellipse": "unconstrained_cv_ellipse",
    "planned_projection": "planned_pose_projection_diagnostic",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def indexed(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {(row["run_id"], row["camera_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate camera view in saved evidence")
    return result


def camera_projection(run: Path, camera: str) -> tuple[np.ndarray, np.ndarray, dict]:
    calibration = read(run / "frozen_session_calibration.json")
    snapshot = calibration["camera_calibration"]
    K = np.asarray(snapshot["intrinsics_full"][camera]["K_undist"], dtype=float)
    E = np.asarray(snapshot["extrinsics_full"][camera], dtype=float)
    C = np.asarray(calibration["c2r"], dtype=float)
    socket = np.asarray(calibration["socket_pose_robot"], dtype=float)
    return K @ (E @ C @ socket)[:3], socket, calibration


def point(P: np.ndarray, xyz: np.ndarray | tuple[float, float, float]) -> list[float]:
    h = P @ np.r_[xyz, 1.]
    if h[2] <= 0:
        raise ValueError("saved point projects behind camera")
    return (h[:2] / h[2]).tolist()


def socket_ring(run: Path, camera: str,
                fixture_root: Path | None = None) -> list[list[float]]:
    P, _socket, calibration = camera_projection(run, camera)
    mode = select_mode("cylinder", float(calibration["mode"]["gap_mm"]))
    root = (fixture_root if fixture_root is not None else
            Path(read(run / "endpoint_catalog.json")["shared_root"]))
    geometry = read(AssetPaths(root, mode).task_geometry)
    radius = float(geometry["socket_bore_radius_m"])
    z = float(geometry["socket_entry_plane_z_m"])
    return [point(P, (radius * math.cos(a), radius * math.sin(a), z))
            for a in np.linspace(0., 2. * math.pi, 96, endpoint=False)]


def observed_points(row: dict, boundary: dict | None) -> dict[str, list[float] | None]:
    ellipse = row["sam_half_ellipse"]
    direct = row["unconstrained_cv_ellipse"]
    return {
        "sam_half_ellipse_center": None if ellipse is None else ellipse["center_full_px"],
        "sam_rectangle_end": row["sam_rectangle_tip_full_px"],
        "sam_boundary_end": None if boundary is None else boundary.get("tip_full_px"),
        "unconstrained_cv_ellipse_center": None if direct is None else direct["center_full_px"],
    }


def compare(rows: list[dict], boundaries: dict, runs_root: Path) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["run_id"], []).append(row)
    result = {}
    for run_id, views in grouped.items():
        run = runs_root / run_id
        projections = {r["camera_id"]: camera_projection(run, r["camera_id"])[0]
                       for r in views}
        candidates = {}
        for method in ("sam_half_ellipse_center", "sam_rectangle_end",
                       "sam_boundary_end", "unconstrained_cv_ellipse_center"):
            observations = []
            for r in views:
                key = (run_id, r["camera_id"])
                uv = observed_points(r, boundaries.get(key))[method]
                if uv is not None:
                    observations.append((r["camera_id"], projections[r["camera_id"]], uv))
            if len(observations) < 2:
                candidates[method] = {"status": "fewer_than_two_views"}
                continue
            proxy = triangulate_tip_proxy(observations)
            residuals = list(proxy["reprojection_error_px"].values())
            heldout = None
            if len(observations) >= 3:
                heldout = {}
                for i, (camera, P, uv) in enumerate(observations):
                    trained = triangulate_tip_proxy(observations[:i] + observations[i+1:])
                    projected = point(P, np.asarray(trained["point_socket_m"]))
                    heldout[camera] = float(np.linalg.norm(np.asarray(projected)-uv))
            candidates[method] = {
                "status": "estimated", "camera_count": len(observations),
                "point_socket_m": proxy["point_socket_m"],
                "median_reprojection_error_px": float(np.median(residuals)),
                "per_camera_reprojection_error_px": proxy["reprojection_error_px"],
                "median_leave_one_view_out_error_px": None if heldout is None else
                    float(np.median(list(heldout.values()))),
                "per_camera_leave_one_view_out_error_px": heldout,
                "per_camera_proxy_reprojection_px": {
                    camera: point(projections[camera], np.asarray(proxy["point_socket_m"]))
                    for camera, _P, _uv in observations},
            }
        result[run_id] = {"camera_ids": [r["camera_id"] for r in views],
                          "methods": candidates}
    return {"schema": "saved_key_face_method_comparison_v1",
            "scope": "read_only; visual socket-pose screen; no physical face-center ground truth",
            "metric_note": "Reprojection measures multiview consistency, not true face-center accuracy. Two-view leave-one-out is undefined. Planned physical circle is excluded because its position is a prior, not an observation.",
            "runs": result}


def _local(points: list[list[float]] | list[float], crop: list[int]):
    if len(points) == 2 and isinstance(points[0], (int, float)):
        return (float(points[0] - crop[0]), float(points[1] - crop[1]))
    return [_local(p, crop) for p in points]


def _dot(draw: ImageDraw.ImageDraw, xy: tuple[float, float], color: str, radius=6):
    x, y = xy
    draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                 fill=color, outline="#0b1421", width=2)


def _caption(panel: Image.Image, label: str) -> Image.Image:
    out = Image.new("RGB", (panel.width, panel.height+42), "#111927")
    out.paste(panel, (0, 42))
    ImageDraw.Draw(out).text((12, 8), label, font=ImageFont.truetype(FONT, 20),
                             fill="white")
    return out


def render_view(row: dict, bbox: dict, boundary: dict | None,
                comparison: dict, runs_root: Path,
                fixture_root: Path | None = None) -> tuple[dict[str, Image.Image], dict]:
    identity = (row["run_id"], row["camera_id"], row["frame_id"], row["source_sha256"])
    if identity != (bbox["run_id"], bbox["camera_id"], bbox["frame_id"],
                    bbox["source_sha256"]):
        raise ValueError("Qwen and SAM ellipse use different saved frames")
    source = Path(row["source"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != row["source_sha256"]:
        raise ValueError("source image hash changed")
    mask_file = Path(row["mask_path"])
    if hashlib.sha256(mask_file.read_bytes()).hexdigest() != row["mask_sha256"]:
        raise ValueError("SAM mask hash changed")
    crop = list(map(int, row["crop_xyxy"]))
    original = Image.open(source).convert("RGB").crop(tuple(crop))
    if original.size != (512, 512):
        raise ValueError("saved key crop was not 512x512")
    panels = {"original": original.copy()}

    qwen = original.copy()
    ImageDraw.Draw(qwen).rectangle(tuple(bbox["bbox_crop_px"]),
                                   outline=PINK, width=4)
    panels["qwen_box"] = qwen

    mask = np.asarray(Image.open(mask_file).convert("L")) > 0
    roi = boundary.get("roi_within_crop") if boundary else None
    roi = roi or padded_key_box_roi(bbox["bbox_crop_px"], canvas_size_wh=(512, 512))
    x0, y0, x1, y1 = map(int, roi)
    if mask.shape != (y1-y0, x1-x0):
        raise ValueError("saved SAM mask and Qwen ROI differ")
    rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    rgba[y0:y1, x0:x1][mask] = (255, 225, 91, 105)
    mask_panel = Image.alpha_composite(original.convert("RGBA"),
                                       Image.fromarray(rgba)).convert("RGB")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    draw = ImageDraw.Draw(mask_panel)
    for contour in contours:
        xy = [(float(p[0][0]+x0), float(p[0][1]+y0)) for p in contour]
        if len(xy) > 1:
            draw.line(xy + xy[:1], fill=YELLOW, width=2)
    panels["sam_mask"] = mask_panel

    ellipse = row["sam_half_ellipse"]
    face = mask_panel.copy()
    draw = ImageDraw.Draw(face)
    if ellipse is not None:
        draw.line(_local(ellipse["ellipse_full_px"], crop) +
                  [_local(ellipse["ellipse_full_px"][0], crop)],
                  fill=GREEN, width=4)
        for sample in ellipse["distal_boundary_samples_full_px"]:
            _dot(draw, _local(sample, crop), "#ffffff", radius=2)
        _dot(draw, _local(ellipse["center_full_px"], crop), GREEN, radius=8)
    panels["observed_ellipse"] = face

    rectangle = original.copy()
    if row["sam_rectangle_tip_full_px"] is not None:
        _dot(ImageDraw.Draw(rectangle),
             _local(row["sam_rectangle_tip_full_px"], crop), YELLOW, radius=8)
    panels["sam_rectangle_end"] = rectangle

    boundary_panel = original.copy()
    if boundary is not None and boundary.get("tip_full_px") is not None:
        draw = ImageDraw.Draw(boundary_panel)
        fit = boundary.get("boundary_fit")
        if fit is not None:
            draw.line(_local(fit["fitted_boundary_full_px"], crop),
                      fill=YELLOW, width=4)
        _dot(draw, _local(boundary["tip_full_px"], crop), YELLOW, radius=8)
    panels["sam_boundary_end"] = boundary_panel

    direct_panel = original.copy()
    direct = row["unconstrained_cv_ellipse"]
    if direct is not None:
        center = _local(direct["center_full_px"], crop)
        axes = direct["axes_px"]
        angle = math.radians(direct["angle_deg"])
        ca, sa = math.cos(angle), math.sin(angle)
        curve = []
        for theta in np.linspace(0., 2.*math.pi, 96, endpoint=False):
            u, v = axes[0]*.5*math.cos(theta), axes[1]*.5*math.sin(theta)
            curve.append((center[0] + ca*u - sa*v,
                          center[1] + sa*u + ca*v))
        draw = ImageDraw.Draw(direct_panel)
        draw.line(curve + curve[:1], fill=PINK, width=4)
        _dot(draw, center, PINK, radius=8)
    panels["unconstrained_cv_ellipse"] = direct_panel

    prior = original.copy()
    draw = ImageDraw.Draw(prior)
    predicted = row["planned_physical_circle"]
    draw.line(_local(predicted["ellipse_full_px"], crop) +
              [_local(predicted["ellipse_full_px"][0], crop)], fill=CYAN, width=4)
    _dot(draw, _local(predicted["center_full_px"], crop), CYAN, radius=8)
    panels["planned_projection"] = prior

    run = runs_root / row["run_id"]
    ring = socket_ring(run, row["camera_id"], fixture_root)
    final = original.copy()
    draw = ImageDraw.Draw(final)
    draw.line(_local(ring, crop) + [_local(ring[0], crop)], fill=CYAN, width=3)
    if ellipse is not None:
        draw.line(_local(ellipse["ellipse_full_px"], crop) +
                  [_local(ellipse["ellipse_full_px"][0], crop)],
                  fill=GREEN, width=3)
        _dot(draw, _local(ellipse["center_full_px"], crop), GREEN, radius=7)
    method = comparison["runs"][row["run_id"]]["methods"]["sam_half_ellipse_center"]
    reproj = method.get("per_camera_proxy_reprojection_px", {}).get(row["camera_id"])
    if reproj is not None:
        _dot(draw, _local(reproj, crop), PINK, radius=5)
    panels["spatial_proxy"] = final
    provenance = {key: row[key] for key in ("run_id", "camera_id", "frame_id",
                                              "timestamp_s", "source", "source_sha256",
                                              "mask_path", "mask_sha256")}
    provenance["qwen_prompt"] = row["qwen_prompt"]
    provenance["sam_prompt"] = row["sam3_prompt"]
    provenance["observed_center_full_px"] = None if ellipse is None else ellipse["center_full_px"]
    provenance["planned_center_full_px"] = predicted["center_full_px"]
    provenance["proxy_reprojection_full_px"] = reproj
    return panels, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ellipse-core", required=True, type=Path)
    parser.add_argument("--ellipse-manual", required=True, type=Path)
    parser.add_argument("--bbox-core", required=True, type=Path)
    parser.add_argument("--bbox-manual", required=True, type=Path)
    parser.add_argument("--boundary-core", required=True, type=Path)
    parser.add_argument("--boundary-manual", required=True, type=Path)
    parser.add_argument("--socket-qc", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fixture-root", type=Path,
                        help="Current location of the session fixture assets if moved")
    args = parser.parse_args()
    accepted = set(read(args.socket_qc)["accepted_runs"])
    # The 103429 manual five-view set supersedes its one-view core record.
    manual = read(args.ellipse_manual)["rows"]
    manual_runs = {row["run_id"] for row in manual}
    rows = [r for r in read(args.ellipse_core)["rows"]
            if r["run_id"] in accepted and r["run_id"] not in manual_runs]
    rows.extend(r for r in manual if r["run_id"] in accepted)
    boxes = indexed([r for r in read(args.bbox_core)["rows"]
                     if r["run_id"] not in manual_runs] +
                    read(args.bbox_manual)["rows"])
    boundary_rows = [r for r in read(args.boundary_core)["rows"]
                     if r["run_id"] not in manual_runs]
    boundary_rows.extend(read(args.boundary_manual)["rows"])
    boundaries = indexed(boundary_rows)
    root = args.runs_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison = compare(rows, boundaries, root)
    comparison["socket_pose_qc"] = str(args.socket_qc.resolve())
    (output / "method_comparison.json").write_text(
        json.dumps(comparison, indent=2)+"\n", encoding="utf-8")
    episode_panels: dict[str, dict[str, list[tuple[str, Image.Image]]]] = {}
    evidence = []
    for row in rows:
        key = (row["run_id"], row["camera_id"])
        panels, provenance = render_view(row, boxes[key], boundaries.get(key),
                                          comparison, root, args.fixture_root)
        per_view_files = {}
        for name, method_name in METHOD_BY_PANEL.items():
            path = (output / "methods" / method_name / "per_view" / name /
                    f"{row['run_id']}_{row['camera_id']}_frame{row['frame_id']}.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            panels[name].save(path)
            per_view_files[name] = str(path)
        provenance["per_view_files"] = per_view_files
        evidence.append(provenance)
        episode = episode_panels.setdefault(row["run_id"], {})
        for name, panel in panels.items():
            episode.setdefault(name, []).append((row["camera_id"], panel))
    for run_id, panel_groups in episode_panels.items():
        for name, group in panel_groups.items():
            # One figure has one visual question. Do not stack all overlays.
            cols = min(3, len(group))
            rows_count = math.ceil(len(group)/cols)
            canvas = Image.new("RGB", (512*cols, 554*rows_count), "#111927")
            for index, (camera, panel) in enumerate(group):
                canvas.paste(_caption(panel, f"{camera} | {name.replace('_', ' ')}"),
                             ((index % cols)*512, (index // cols)*554))
            folder = output / "role_images" / name
            folder.mkdir(parents=True, exist_ok=True)
            canvas.save(folder / f"{run_id}.png")

            method_name = METHOD_BY_PANEL.get(name)
            if method_name is not None:
                method_dir = output / "methods" / method_name / name
                method_dir.mkdir(parents=True, exist_ok=True)
                canvas.save(method_dir / f"{run_id}.png")

        representative = "24122734" if any(c == "24122734" for c, _ in
                                           panel_groups["original"]) else \
                         panel_groups["original"][0][0]
        stages = [("1  Original camera crop", "original"),
                  ("2  Qwen key box", "qwen_box"),
                  ("3  SAM3 key mask", "sam_mask"),
                  ("4  Fit observed face ellipse", "observed_ellipse"),
                  ("5  Multiview proxy and socket", "spatial_proxy")]
        frames = []
        stage_dir = output / "animation_stages" / run_id
        stage_dir.mkdir(parents=True, exist_ok=True)
        for number, (title, name) in enumerate(stages, 1):
            panel = next(p for c, p in panel_groups[name] if c == representative)
            frame = _caption(panel, title)
            frame.save(stage_dir / f"{number:02d}_{name}.png")
            frames.append(frame)
        gif_dir = output / "animations"
        gif_dir.mkdir(parents=True, exist_ok=True)
        frames[0].save(gif_dir / f"{run_id}.gif", save_all=True,
                       append_images=frames[1:], duration=[1200]*len(frames),
                       loop=0, optimize=False)
        chosen_gif = output / "methods" / "sam_half_ellipse" / "animations"
        chosen_gif.mkdir(parents=True, exist_ok=True)
        frames[0].save(chosen_gif / f"{run_id}.gif", save_all=True,
                       append_images=frames[1:], duration=[1200]*len(frames),
                       loop=0, optimize=False)

    descriptions = {
        "sam_half_ellipse": (
            "Observed SAM3 mask half ellipse. Use Qwen box only for ROI; take the "
            "largest connected SAM mask component, estimate its body width and "
            "long axis, sample the distal boundary, and robustly fit a half "
            "ellipse. The center is a 2D face-center proxy; triangulate those "
            "centers across saved undistorted cameras. The planned axis chooses "
            "only the distal sign. This is the strongest saved-data candidate, "
            "not physical ground truth."),
        "sam_boundary_end": (
            "SAM3 mask end-boundary center. Preserve the full rotated-box "
            "lateral center and robustly fit a line through central distal "
            "mask-boundary samples. This estimates the visible end edge, "
            "which differs from the circular face center."),
        "sam_rectangle_end": (
            "Midpoint of the distal short edge of the SAM3 mask's minimum-area "
            "rectangle. The planned axis chooses the distal edge. This is the "
            "simplest silhouette-end baseline."),
        "unconstrained_cv_ellipse": (
            "OpenCV ellipse fitted directly to the distal 30 percent of the "
            "SAM3 contour, with free axes and center. Truncated/occluded "
            "contours can shift the center; diagnostic baseline only."),
        "planned_pose_projection_diagnostic": (
            "The known 30 mm key circular face, transformed by the saved "
            "planned 6D key pose and frozen camera calibration. This is a "
            "prediction at planned position, not an observed landmark; it is "
            "excluded from triangulation and method ranking."),
    }
    procedures = {
        "sam_half_ellipse": [
            "1. Qwen whole-key box locates the search ROI; SAM3 returns the saved key mask.",
            "2. Keep the largest connected mask component. Use its minimum-area rectangle for observed long-axis orientation; the planned key axis selects only the distal sign.",
            "3. Estimate shaft radius from the 7th and 93rd lateral percentiles of the middle-to-distal body pixels. Sample the 99th axial percentile in 14 lateral bins spanning +/-90% of that radius.",
            "4. Fit a half ellipse with fixed lateral radius, free axial radius and axial center using robust least squares. Preserve the observed rectangle's lateral centerline.",
            "5. Triangulate the observed 2D ellipse centers with saved undistorted intrinsics, camera extrinsics and frozen socket pose. The 3D point and its per-view reprojections are visual proxies only.",
            "The circular face is partially hidden in some views. A silhouette fit cannot uniquely recover the true physical face center without a measured reference.",
        ],
        "sam_boundary_end": [
            "1. Read the same Qwen ROI and SAM3 mask. The planned axis selects the distal direction.",
            "2. Find the mask minimum-area rectangle and keep its lateral centerline. In eight central lateral strips, measure the 99th-percentile distal boundary.",
            "3. Robustly fit a line to the boundary samples and intersect it with the preserved centerline.",
            "4. Triangulate the per-camera 2D points. This is a visible silhouette-end proxy; it does not account for the projected circular face's interior center.",
        ],
        "sam_rectangle_end": [
            "1. Read the same Qwen ROI and SAM3 mask.",
            "2. Fit a minimum-area rotated rectangle to all mask pixels. The planned projected key axis selects the distal short edge.",
            "3. Use that edge's midpoint as the 2D point and triangulate points across cameras.",
            "The rectangle end can extend beyond the observed curved lower boundary and can shift with disconnected or occluded mask pixels.",
        ],
        "unconstrained_cv_ellipse": [
            "1. Read the SAM3 outer contour in original undistorted pixels.",
            "2. Keep the 30% most distal contour pixels along the planned projected key direction.",
            "3. Apply OpenCV fitEllipse with unrestricted center, two radii and angle; use its center as the 2D point.",
            "4. Triangulate centers from multiple cameras. A partial arc can admit many full ellipses, so the free center is unstable under occlusion.",
        ],
        "planned_pose_projection_diagnostic": [
            "1. Read frozen K_undist, camera extrinsic, c2r, the planned preinsert key transform and task geometry.",
            "2. Take the physical key radius (15 mm) and key-frame tip z. Sample 96 points around that circle and project each through the planned 6D transform.",
            "3. Render the predicted ellipse at the planned image location. It is not fitted to the image and is never scored as an observed method.",
        ],
    }
    for method_name, description in descriptions.items():
        method_dir = output / "methods" / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        metric_key = {
            "sam_half_ellipse": "sam_half_ellipse_center",
            "unconstrained_cv_ellipse": "unconstrained_cv_ellipse_center",
        }.get(method_name, method_name)
        details = {
            run_id: record["methods"][metric_key]
            for run_id, record in comparison["runs"].items()
        } if metric_key != "planned_pose_projection_diagnostic" else {}
        (method_dir / "results.json").write_text(
            json.dumps({"method": method_name, "description": description,
                        "runs": details}, indent=2)+"\n", encoding="utf-8")
        image_rows = sorted(method_dir.rglob("*.png"))
        gif_rows = sorted(method_dir.rglob("*.gif"))
        lines = [f"# {method_name}", "", description, "",
                 "## Inputs and procedure", ""]
        lines.extend(f"- {step}" for step in procedures[method_name])
        lines.extend(["", "## Evidence and interpretation", "",
                 "All images are overlays on saved real undistorted AutoDex frames.",
                 "`per_view/` contains one PNG per camera frame; the other",
                 "image folders contain episode-level multiview grids.",
                 "Source IDs, timestamps, original hashes, and mask hashes are in",
                 "`../../evidence_manifest.json`. Reprojection error measures",
                 "cross-camera consistency, not physical center accuracy.", "",
                 "`../../method_comparison.json` shows this method against",
                 "the other candidates on the same accepted episodes; the",
                 "`../../socket_pose_visual_qc.json` records exclusions.", "",
                 "## Outputs", ""])
        lines.extend(
            f"- [{p.relative_to(method_dir)}]({p.relative_to(method_dir)})"
            for p in image_rows + gif_rows + [method_dir / "results.json"])
        (method_dir / "README.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output / "evidence_manifest.json").write_text(
        json.dumps({"schema": "saved_key_face_separate_roles_v1",
                    "scope": "read_only_real_undistorted_frames; derived overlays",
                    "accepted_runs": sorted(episode_panels), "views": evidence},
                   indent=2)+"\n", encoding="utf-8")
    print(output)
    for run_id, record in comparison["runs"].items():
        print(run_id, len(record["camera_ids"]),
              {name: (round(v["median_reprojection_error_px"], 2),
                      None if v["median_leave_one_view_out_error_px"] is None
                      else round(v["median_leave_one_view_out_error_px"], 2))
               for name, v in record["methods"].items()
               if v["status"] == "estimated"})


if __name__ == "__main__":
    main()
