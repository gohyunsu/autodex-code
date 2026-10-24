#!/usr/bin/env python3
"""Turn saved per-view key-silhouette ends into a read-only socket-frame proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from precision_insertion.held_key_tip import triangulate_tip_proxy


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tip_results", type=Path)
    parser.add_argument("--bbox-results", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tip_rows = _read(args.tip_results)["rows"]
    bbox_rows = _read(args.bbox_results)["rows"]
    bbox_by_view = {(row["run_id"], row["camera_id"]): row
                    for row in bbox_rows}
    if len(bbox_by_view) != len(bbox_rows):
        raise ValueError("duplicate saved Qwen camera view")
    grouped: dict[str, list] = {}
    for row in tip_rows:
        grouped.setdefault(row["run_id"], []).append(row)
    output = {
        "schema": "precision_insertion_saved_key_tip_proxy_replay_v1",
        "scope": "read_only_saved_undistorted_images_no_robot_command",
        "runs": {},
    }
    for run_id, rows in grouped.items():
        run = args.runs_root.expanduser().resolve() / run_id
        calibration = _read(run / "frozen_session_calibration.json")
        snapshot = calibration["camera_calibration"]
        C = np.asarray(calibration["c2r"], dtype=float)
        socket = np.asarray(calibration["socket_pose_robot"], dtype=float)
        projections = {}
        observations = []
        provenance = []
        for row in rows:
            camera = row["camera_id"]
            bbox_row = bbox_by_view[(run_id, camera)]
            source = Path(bbox_row["source"]).resolve()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if (digest != row["source_sha256"] or
                    digest != bbox_row["source_sha256"] or
                    row["frame_id"] != bbox_row["frame_id"]):
                raise ValueError(f"saved key-tip image provenance changed: {camera}")
            K = np.asarray(snapshot["intrinsics_full"][camera]["K_undist"],
                           dtype=float)
            E = np.asarray(snapshot["extrinsics_full"][camera], dtype=float)
            P = K @ (E @ C @ socket)[:3]
            projections[camera] = P
            point = row.get("tip_full_px")
            method = "sam3_mask_distal_edge" if point is not None else None
            if point is None and bbox_row.get("bbox_crop_px") is not None:
                x1, _y1, x2, y2 = bbox_row["bbox_crop_px"]
                x0, y0 = bbox_row["crop_xyxy"][:2]
                point = [x0+(x1+x2)/2, y0+y2]
                method = "qwen_bbox_bottom_midpoint_fallback"
            if point is not None:
                observations.append((camera, P, point))
            provenance.append({
                "camera_id": camera, "frame_id": row["frame_id"],
                "timestamp_s": row["timestamp_s"],
                "source_image": str(source), "source_sha256": digest,
                "observed_tip_px": point, "point_source": method,
                "sam3_status": row["status"],
                "qwen_answer": bbox_row["answer"],
            })
        if len(observations) < 2:
            output["runs"][run_id] = {
                "status": "insufficient_observed_views",
                "proxy": None, "views": provenance,
            }
            continue
        proxy = triangulate_tip_proxy(observations)
        xyz = np.asarray(proxy["point_socket_m"], dtype=float)
        robot = socket @ np.r_[xyz, 1.]
        proxy["point_robot_m"] = robot[:3].tolist()
        proxy["xy_from_socket_axis_mm"] = (xyz[:2]*1000.).tolist()
        for row in provenance:
            P = projections[row["camera_id"]]
            h = P @ np.r_[xyz, 1.]
            row["proxy_reprojection_px"] = (h[:2]/h[2]).tolist()
            row["proxy_reprojection_error_px"] = proxy[
                "reprojection_error_px"].get(row["camera_id"])
        output["runs"][run_id] = {
            "status": "spatial_proxy_estimated", "proxy": proxy,
            "frozen_socket_pose_robot": socket.tolist(),
            "socket_pose_source": calibration["socket_pose_source"],
            "views": provenance,
        }
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    print(target)
    for run_id, run in output["runs"].items():
        print(run_id, run["status"], None if run["proxy"] is None else
              run["proxy"]["point_socket_m"])


if __name__ == "__main__":
    main()
