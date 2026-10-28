#!/usr/bin/env python3
"""Compare read-only cylinder-axis estimates on saved preinsert SAM3 masks.

The script never connects to a camera or robot.  It reuses the exact masks,
tip centres, undistorted calibration, and frozen socket frame from the saved
key-face review.  Its purpose is to decide how to extend the existing XY-only
proxy to the two observable cylinder-tilt degrees of freedom.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


COLORS = {
    "min_area_rectangle": "#ff4d4d",
    "partial_face_ellipse_minor_axis": "#ffe15b",
    "bilateral_strip_midline": "#7fa8ff",
    "tip": "#42f58d",
}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_saved_path(value: str, evidence_root: Path) -> Path:
    old = "/home/robot/shared_data/hyunsu/tmp/20261024_key_face_center_methods/"
    return evidence_root / value[len(old):] if value.startswith(old) else Path(value)


def _unit(value) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    return vector / np.linalg.norm(vector)


def _directed(direction, planned) -> np.ndarray:
    direction = _unit(direction)
    return -direction if direction @ planned < 0 else direction


def _angle_deg(first, second) -> float:
    return math.degrees(math.acos(float(np.clip(
        _unit(first) @ _unit(second), -1., 1.))))


def _largest_component_points(row: dict, qwen: dict,
                              evidence_root: Path) -> np.ndarray:
    mask = np.asarray(Image.open(_resolve_saved_path(
        row["mask_path"], evidence_root)).convert("L")) > 0
    box = qwen["bbox_crop_px"]
    x0, y0 = max(0, int(box[0])-25), max(0, int(box[1])-25)
    x1, y1 = min(512, int(box[2])+25), min(512, int(box[3])+25)
    if mask.shape != (y1-y0, x1-x0):
        raise ValueError("saved Qwen ROI and SAM3 mask dimensions differ")
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        raise ValueError("saved SAM3 mask has no foreground component")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == component)
    crop = row["crop_xyxy"]
    return np.column_stack((xs+x0+crop[0], ys+y0+crop[1])).astype(float)


def _minimum_rectangle(points: np.ndarray, planned: np.ndarray):
    rectangle = cv2.minAreaRect(points.astype(np.float32))
    corners = cv2.boxPoints(rectangle).astype(float)
    edges = [corners[(index+1) % 4]-corners[index] for index in range(4)]
    direction = _directed(max(edges, key=np.linalg.norm), planned)
    return np.asarray(rectangle[0], dtype=float), direction


def _trimmed_pca(points: np.ndarray, planned: np.ndarray):
    _centre, initial = _minimum_rectangle(points, planned)
    axial = points @ initial
    selected = points[(axial >= np.quantile(axial, .10)) &
                      (axial <= np.quantile(axial, .80))]
    centre = selected.mean(axis=0)
    _u, _s, vh = np.linalg.svd(selected-centre)
    return centre, _directed(vh[0], planned)


def _strip_midline(points: np.ndarray, planned: np.ndarray):
    _centre, initial = _minimum_rectangle(points, planned)
    lateral = np.array([-initial[1], initial[0]])
    axial_values, lateral_values = points @ initial, points @ lateral
    low, high = np.quantile(axial_values, (.12, .78))
    centres = []
    edges = np.linspace(low, high, 13)
    for start, end in zip(edges[:-1], edges[1:]):
        values = lateral_values[(axial_values >= start) &
                                (axial_values < end)]
        if len(values) < 20:
            continue
        middle = (np.quantile(values, .08)+np.quantile(values, .92))/2
        centres.append(initial*((start+end)/2)+lateral*middle)
    centres = np.asarray(centres)
    centre = np.median(centres, axis=0)
    _u, _s, vh = np.linalg.svd(centres-centre)
    return centre, _directed(vh[0], planned)


def _image_line(centre: np.ndarray, direction: np.ndarray) -> np.ndarray:
    first = np.r_[centre-100.*direction, 1.]
    second = np.r_[centre+100.*direction, 1.]
    line = np.cross(first, second)
    return line / np.linalg.norm(line[:2])


def _axis_from_lines(lines: dict[str, np.ndarray], views: dict[str, dict]):
    normals = []
    for camera, line in lines.items():
        view = views[camera]
        intrinsics = np.asarray(view["intrinsics"], dtype=float)
        T_camera_socket = np.asarray(view["T_camera_socket"], dtype=float)
        normal = T_camera_socket[:3, :3].T @ (intrinsics.T @ line)
        normals.append(_unit(normal))
    _u, singular_values, vh = np.linalg.svd(np.vstack(normals))
    axis = vh[-1]
    if axis[2] > 0:
        axis *= -1
    return _unit(axis), singular_values


def _line_errors(axis: np.ndarray, tip: np.ndarray,
                 lines: dict[str, np.ndarray], views: dict[str, dict]):
    result = {}
    for camera, line in lines.items():
        view = views[camera]
        K = np.asarray(view["intrinsics"], dtype=float)
        E = np.asarray(view["T_camera_socket"], dtype=float)
        errors = []
        for point in (tip, tip-.08*axis):
            projected = K @ (E[:3, :3] @ point + E[:3, 3])
            uv = projected[:2]/projected[2]
            errors.append(abs(float(line @ np.r_[uv, 1.])))
        result[camera] = errors
    return result


def _alignment_rotvec_deg(axis: np.ndarray) -> list[float]:
    target = np.array([0., 0., -1.])
    angle = math.acos(float(np.clip(axis @ target, -1., 1.)))
    rotation_axis = np.cross(axis, target)
    if np.linalg.norm(rotation_axis) < 1e-12:
        return [0., 0., 0.]
    return np.degrees(_unit(rotation_axis)*angle).tolist()


def _load_rows(root: Path):
    rows, qwen = {}, {}
    for name in ("sam_ellipse_core.json", "sam_ellipse_manual.json"):
        for row in _read(root/name)["rows"]:
            rows[(row["run_id"], row["camera_id"])] = row
    for name in ("inputs/qwen_sam_key_axis/results.json",
                 "inputs/qwen_sam_manual_103429/qwen.json"):
        for row in _read(root/name)["rows"]:
            qwen[(row["run_id"], row["camera_id"])] = row
    return rows, qwen


def _render_overlay(root: Path, output: Path, run_id: str,
                    cameras: list[str], method_records: dict[str, dict],
                    rows: dict[tuple[str, str], dict]) -> None:
    source = root/"role_images"/"original"/f"{run_id}.png"
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    columns = min(3, len(cameras))
    for index, camera in enumerate(cameras):
        panel_origin = np.array([(index % columns)*512,
                                 (index // columns)*554+42], dtype=float)
        crop_origin = np.asarray(rows[(run_id, camera)]["crop_xyxy"][:2],
                                 dtype=float)
        for method in ("min_area_rectangle",
                       "partial_face_ellipse_minor_axis",
                       "bilateral_strip_midline"):
            record = method_records[method]["per_view"][camera]
            centre = np.asarray(record["centre_full_px"])-crop_origin+panel_origin
            direction = np.asarray(record["direction_full_px"])
            first, second = centre-150*direction, centre+150*direction
            draw.line((*first, *second), fill=COLORS[method], width=4)
        tip = np.asarray(rows[(run_id, camera)]["sam_half_ellipse"]
                         ["center_full_px"])-crop_origin+panel_origin
        draw.ellipse((tip[0]-6, tip[1]-6, tip[0]+6, tip[1]+6),
                     fill=COLORS["tip"], outline="#0a0f14", width=2)
    image.save(output/f"{run_id}_axis_method_overlay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True,
                        help="Parent of precision_xy_review_<timestamp> folders")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, review, output = (p.expanduser().resolve() for p in
                            (args.evidence_root, args.review_root, args.output))
    output.mkdir(parents=True, exist_ok=True)
    rows, qwen = _load_rows(root)
    comparison = _read(root/"method_comparison.json")
    results = {"schema": "saved_key_axis_method_comparison_v1",
               "scope": "read_only_saved_frames_not_robot_command",
               "target_axis_socket": [0., 0., -1.], "runs": {}}
    run_ids = sorted({run_id for run_id, _camera in rows})
    for run_id in run_ids:
        matches = list(review.glob(
            f"precision_xy_review_{run_id[:15]}*/manifest.json"))
        if not matches:
            continue
        manifest = _read(matches[0])
        views = {view["camera_id"]: view for view in manifest["views"]}
        cameras = [camera for candidate_run, camera in rows
                   if candidate_run == run_id and camera in views]
        if len(cameras) < 2:
            continue
        tip = np.asarray(comparison["runs"][run_id]["methods"]
                         ["sam_half_ellipse_center"]["point_socket_m"],
                         dtype=float)
        run_record = {"camera_ids": cameras,
                      "tip_socket_m": tip.tolist(), "methods": {}}
        for method in ("min_area_rectangle",
                       "partial_face_ellipse_minor_axis",
                       "trimmed_mask_pca", "bilateral_strip_midline"):
            lines, per_view = {}, {}
            for camera in cameras:
                row, box = rows[(run_id, camera)], qwen[(run_id, camera)]
                points = _largest_component_points(row, box, root)
                planned = (np.asarray(box["nominal_tip_px"])-
                           np.asarray(box["planned_rear_full_px"]))
                if method == "min_area_rectangle":
                    centre, direction = _minimum_rectangle(points, planned)
                elif method == "trimmed_mask_pca":
                    centre, direction = _trimmed_pca(points, planned)
                elif method == "bilateral_strip_midline":
                    centre, direction = _strip_midline(points, planned)
                else:
                    ellipse = row["unconstrained_cv_ellipse"]
                    angle = math.radians(float(ellipse["angle_deg"]))
                    direction = _directed((math.cos(angle), math.sin(angle)),
                                          planned)
                    centre = np.asarray(row["sam_half_ellipse"]
                                        ["center_full_px"], dtype=float)
                # All infinite lines are anchored to the independently fitted
                # face centre. Moving along the line does not change it.
                face = np.asarray(row["sam_half_ellipse"]["center_full_px"],
                                  dtype=float)
                lines[camera] = _image_line(face, direction)
                per_view[camera] = {
                    "centre_full_px": face.tolist(),
                    "direction_full_px": direction.tolist(),
                    "angle_from_planned_projection_deg":
                        _angle_deg(direction, planned),
                }
            axis, singular = _axis_from_lines(lines, views)
            errors = _line_errors(axis, tip, lines, views)
            leave_one_out = []
            if len(cameras) >= 3:
                for removed in cameras:
                    subset = {camera: line for camera, line in lines.items()
                              if camera != removed}
                    candidate, _singular = _axis_from_lines(subset, views)
                    leave_one_out.append(_angle_deg(axis, candidate))
            flattened = [value for pair in errors.values() for value in pair]
            run_record["methods"][method] = {
                "axis_socket": axis.tolist(),
                "tilt_to_socket_axis_deg": _angle_deg(axis, (0, 0, -1)),
                "minimal_alignment_rotvec_socket_deg":
                    _alignment_rotvec_deg(axis),
                "normal_matrix_singular_values": singular.tolist(),
                "median_projected_axis_line_error_px":
                    float(np.median(flattened)),
                "max_projected_axis_line_error_px": float(max(flattened)),
                "median_leave_one_view_out_axis_change_deg":
                    None if not leave_one_out else
                    float(np.median(leave_one_out)),
                "max_leave_one_view_out_axis_change_deg":
                    None if not leave_one_out else float(max(leave_one_out)),
                "per_view": per_view,
                "per_view_axis_line_error_px": errors,
            }
        results["runs"][run_id] = run_record
        _render_overlay(root, output, run_id, cameras,
                        run_record["methods"], rows)
    (output/"axis_method_comparison.json").write_text(
        json.dumps(results, indent=2)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
