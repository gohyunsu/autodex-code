#!/usr/bin/env python3
"""Export the exact saved VLM pixels for offline checkpoint review.

This reads completed checkpoint reports and PNGs only. It never opens a camera,
loads a model, or connects to a robot. Hashes in the report bind every exported
input to the pixels that the model received.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def _rgb_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _stage(report: dict) -> str:
    schema = report.get("schema")
    if schema == "precision_insertion_raw_lift_checkpoint_v1":
        return "lift"
    if schema == "precision_insertion_preinsert_checkpoint_v1":
        return "preinsert"
    if schema == "precision_insertion_released_checkpoint_v1":
        return "final"
    raise ValueError("report has no supported saved VLM input policy")


def _requests(report: dict, stage: str):
    if stage == "lift":
        for row in report["visual"]["aggregation"]["per_view"]:
            camera = row["camera_id"]
            yield camera, row, [("after_lift",
                Path(report["after_bundle"]) / "images" / f"{camera}.png",
                row["crop"]["source_roi_xyxy"], None)]
    elif stage == "preinsert":
        crops = report["visual"].get("input_crops", {}).get("per_camera", {})
        for row in report["visual"]["per_view"]:
            camera = row["image_order"][0].split("/", 1)[1].split("@", 1)[0]
            inputs = []
            for phase, bundle_key in (("after_lift", "after_lift_raw_bundle"),
                                      ("preinsert", "raw_bundle")):
                crop = crops.get(camera, {}).get(phase, {})
                inputs.append((phase, Path(report[bundle_key]) / "images" /
                               f"{camera}.png", crop.get("roi_xyxy_full_px"),
                               crop.get("crop_rgb_sha256")))
            yield camera, row, inputs
    else:
        frames = {(row["phase"], row["camera_id"]): Path(row["path"])
                  for row in report["frames"]}
        for row in report["visual"]["aggregation"]["per_view"]:
            camera = row["image_order"][0].split("/", 1)[1].split("@", 1)[0]
            phases = (["initial_tabletop", "final_clear_view"]
                      if len(row["image_order"]) == 2 else
                      ["final_clear_view"])
            sources = []
            for phase in phases:
                crop = report.get(
                    "initial_visual_crops" if phase == "initial_tabletop"
                    else "final_visual_crops", {}).get(camera, {})
                sources.append((phase, frames[phase, camera],
                                crop.get("roi_xyxy_full_px"),
                                crop.get("crop_rgb_sha256")))
            yield camera, row, sources


def _input_image(path: Path, roi, crop_hash, transform: dict) -> Image.Image:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
    if roi is not None:
        if (len(roi) != 4 or not all(type(v) is int for v in roi) or
                not (0 <= roi[0] < roi[2] <= rgb.width and
                     0 <= roi[1] < roi[3] <= rgb.height)):
            raise ValueError(f"invalid saved VLM crop: {path}")
        rgb = rgb.crop(tuple(roi))
    if crop_hash is not None and _rgb_hash(rgb) != crop_hash:
        raise ValueError(f"saved VLM crop pixels changed: {path}")
    if (list(rgb.size) != transform["source_size"] or
            _rgb_hash(rgb) != transform["source_rgb_sha256"]):
        raise ValueError(f"saved VLM source pixels changed: {path}")
    size = tuple(transform["input_size"])
    prepared = (rgb if rgb.size == size else
                rgb.resize(size, Image.Resampling.LANCZOS))
    if _rgb_hash(prepared) != transform["input_rgb_sha256"]:
        raise ValueError(f"saved VLM resized pixels changed: {path}")
    return prepared


def export(report_path: Path, output_dir: Path) -> Path:
    report_path = report_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stage = _stage(report)
    rows, tiles = [], []
    for camera, answer, sources in _requests(report, stage):
        transforms = answer["input_transform"]["images"]
        if len(sources) != len(transforms):
            raise ValueError("saved VLM image order and transforms disagree")
        for (phase, path, roi, crop_hash), transform in zip(sources, transforms):
            prepared = _input_image(path, roi, crop_hash, transform)
            name = f"{camera}_{phase}.png"
            tiles.append((name, prepared))
            rows.append({
                "camera_id": camera, "phase": phase, "source": str(path),
                "roi_xyxy_full_px": roi,
                "model_input_rgb_sha256": _rgb_hash(prepared),
                "model_input_size_wh": list(prepared.size),
                "raw_answer": answer["raw_answer"],
                "parsed": answer.get("parsed", answer.get("yes")),
                "output": name,
            })
    output_dir.mkdir(parents=True)
    for name, prepared in tiles:
        prepared.save(output_dir / name)
    width = max(image.width for _, image in tiles)
    height = max(image.height for _, image in tiles)
    columns = 2 if stage == "preinsert" or any(
        row["phase"] == "initial_tabletop" for row in rows) else 1
    cameras = len(tiles) // columns
    sheet = Image.new("RGB", (columns * width, cameras * (height + 28)),
                      "#14202a")
    draw = ImageDraw.Draw(sheet)
    for index, (name, prepared) in enumerate(tiles):
        column = index % columns
        row = index // columns
        x, y = column * width, row * (height + 28)
        draw.text((x + 8, y + 6), name.removesuffix(".png"), fill="white")
        sheet.paste(prepared, (x, y + 28))
    sheet.save(output_dir / "contact_sheet.png")
    record = {
        "schema": "precision_insertion_saved_vlm_input_export_v1",
        "stage": stage, "attempt_id": report["attempt_id"],
        "report": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "saved_label": (report.get("vlm_label") or
                        report.get("visual", {}).get("status")),
        "images": rows,
        "scope": "exact_saved_model_pixels_for_offline_review",
    }
    target = output_dir / "manifest.json"
    target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(export(args.report, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
