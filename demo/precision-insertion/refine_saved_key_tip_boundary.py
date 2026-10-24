#!/usr/bin/env python3
"""Refine saved SAM3 key-end observations from actual mask boundary pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from precision_insertion.held_key_tip import (
    distal_key_mask_boundary_fit, padded_key_box_roi,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tip_results", type=Path)
    parser.add_argument("--bbox-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    boxes = {(row["run_id"], row["camera_id"]): row
             for row in _read(args.bbox_results)["rows"]}
    rows = []
    for source_row in _read(args.tip_results)["rows"]:
        row = dict(source_row)
        box = boxes[(row["run_id"], row["camera_id"])]
        source = Path(box["source"]).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if (digest != row["source_sha256"] or
                digest != box["source_sha256"] or
                row["frame_id"] != box["frame_id"]):
            raise ValueError("saved image hash or frame ID changed")
        row["previous_tip_full_px"] = row.get("tip_full_px")
        row["boundary_fit"] = None
        if row.get("mask_path"):
            mask_path = Path(row["mask_path"]).resolve()
            mask = np.asarray(Image.open(mask_path).convert("L")) > 0
            roi = row.get("roi_within_crop") or padded_key_box_roi(
                box["bbox_crop_px"], canvas_size_wh=(512, 512))
            x0, y0, x1, y1 = map(int, roi)
            if mask.shape != (y1-y0, x1-x0):
                raise ValueError("SAM3 mask does not match its Qwen box ROI")
            origin = (box["crop_xyxy"][0]+x0, box["crop_xyxy"][1]+y0)
            fit = distal_key_mask_boundary_fit(
                mask, planned_rear_px=box["planned_rear_full_px"],
                planned_tip_px=box["nominal_tip_px"],
                mask_origin_px=origin)
            row["mask_sha256"] = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            row["boundary_fit"] = fit
            if fit is not None:
                row["tip_full_px"] = fit["center_full_px"]
                row["status"] = "observed_mask_distal_boundary_center"
        if (row.get("human_approx_tip_full_px") is not None and
                row.get("tip_full_px") is not None):
            row["boundary_delta_from_human_approx_px"] = float(np.linalg.norm(
                np.asarray(row["tip_full_px"]) -
                np.asarray(row["human_approx_tip_full_px"])))
        rows.append(row)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema": "precision_saved_sam3_boundary_refinement_v1",
        "scope": "read_only_existing_masks_no_model_or_robot_call",
        "rows": rows,
    }, indent=2)+"\n", encoding="utf-8")
    print(output)
    for row in rows:
        if row["previous_tip_full_px"] is not None and row["tip_full_px"] is not None:
            shift = np.linalg.norm(np.asarray(row["tip_full_px"])-
                                   np.asarray(row["previous_tip_full_px"]))
            print(row["run_id"], row["camera_id"], f"boundary shift {shift:.1f} px")


if __name__ == "__main__":
    main()
