"""Replay a SAM3 text prompt on saved AutoDex cylinder-key camera shots.

This is an image-only diagnostic: no camera, FoundPose, or robot is started.
Each condition produces all-view full-frame and zoomed mask-overlay grids.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_EVAL = Path(
    "/mnt/paradex2/hyunsu/autodex_precision_insertion_perception_eval_20261010_652ac909"
)
DEFAULT_CKPT = Path(
    "/mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909"
    "/weights/sam3/sam3.pt"
)
SAM3_SOURCE = Path(__file__).resolve().parents[2] / "autodex/perception/thirdparty/sam3"
TILE_W, TILE_H = 480, 337
GRID_W, GRID_H = TILE_W * 5, TILE_H * 4  # 2400 x 1348, near 16:9.


def _load_segmentor(checkpoint: Path):
    if str(SAM3_SOURCE) not in sys.path:
        sys.path.insert(0, str(SAM3_SOURCE))
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 inference needs CUDA; use --saved for NAS masks")
    model = build_sam3_image_model(
        device="cuda", checkpoint_path=str(checkpoint), load_from_HF=False
    )
    return Sam3Processor(model, device="cuda", confidence_threshold=0.5)


def _infer(processor, rgb: Image.Image, prompt: str):
    import torch

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(rgb)
        state = processor.set_text_prompt(state=state, prompt=prompt)
        if "masks" not in state or len(state["masks"]) == 0:
            return None, None
        scores = state["scores"]
        index = int(scores.argmax().item())
        mask = state["masks"][index].squeeze().cpu().numpy().astype(bool)
        return mask, float(scores[index].item())


def _overlay(image: Image.Image, mask: np.ndarray | None) -> Image.Image:
    base = image.convert("RGB")
    if mask is None or not mask.any():
        return base
    rgb = np.asarray(base).copy()
    rgb[mask] = (0.46 * rgb[mask] + 0.54 * np.array([0, 222, 255])).astype(
        np.uint8
    )
    return Image.fromarray(rgb)


def _zoom_box(mask: np.ndarray | None, reference: np.ndarray | None, size):
    w, h = size
    union = np.zeros((h, w), dtype=bool)
    if mask is not None:
        union |= mask
    if reference is not None:
        union |= reference
    ys, xs = np.nonzero(union)
    if len(xs):
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        # Include enough board/socket context to expose wrong-object masks.
        half_w = max(250, int((x1 - x0 + 1) * 1.15))
        half_h = max(188, int((y1 - y0 + 1) * 1.15))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    else:
        cx, cy, half_w, half_h = w // 2, int(h * 0.72), 460, 345
    half_w = max(half_w, int(half_h * 4 / 3))
    half_h = max(half_h, int(half_w * 3 / 4))
    return max(0, cx - half_w), max(0, cy - half_h), min(w, cx + half_w), min(h, cy + half_h)


def _contact_sheet(rows, zoom: bool, title: str, output: Path):
    canvas = Image.new("RGB", (GRID_W, GRID_H + 28), (35, 39, 43))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 7), title, fill="white")
    for n, item in enumerate(rows):
        x, y = (n % 5) * TILE_W, 28 + (n // 5) * TILE_H
        picture = item["overlay"].copy()
        if zoom:
            picture = picture.crop(item["zoom_box"])
        picture.thumbnail((TILE_W - 12, TILE_H - 31), Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (TILE_W - picture.width) // 2, y + 26))
        score = item["score"]
        caption = f'{item["serial"]}  mask={item["pixels"]} px'
        if score is not None:
            caption += f"  score={score:.2f}"
        draw.text((x + 8, y + 4), caption, fill=(230, 238, 240))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def _case(report_path: Path, output_root: Path, prompt: str, processor, mode: str):
    report = json.loads(report_path.read_text())
    if mode == "saved":
        recorded = {row.get("prompt") for row in report["per_view_status"].values()}
        if len(recorded) != 1 or not next(iter(recorded)):
            raise ValueError("saved SAM3 masks need one recorded prompt")
        prompt = next(iter(recorded))
    shot_dir = Path(report["shot_dir"])
    shot = json.loads((shot_dir / "shot.json").read_text())
    case_id = report["condition_id"]
    case_out = output_root / case_id.replace("/with_key/", "/")
    case_out.mkdir(parents=True, exist_ok=True)
    rows, statuses = [], {}
    for serial in sorted(shot["images"]):
        image = Image.open(shot_dir / shot["images"][serial]["file"]).convert("RGB")
        old_path = report_path.parent / f"{serial}_mask.png"
        reference = (
            np.asarray(Image.open(old_path).convert("L")) > 0 if old_path.exists() else None
        )
        begin = time.monotonic()
        if processor is None:
            mask, score = reference, None
        else:
            mask, score = _infer(processor, image, prompt)
        duration = time.monotonic() - begin
        if mask is not None and mask.shape != (image.height, image.width):
            raise ValueError(f"{serial}: unexpected mask size {mask.shape}")
        pixels = int(mask.sum()) if mask is not None else 0
        Image.fromarray(
            (mask.astype(np.uint8) * 255)
            if mask is not None else np.zeros((image.height, image.width), np.uint8)
        ).save(case_out / f"{serial}_mask.png")
        rows.append(
            dict(
                serial=serial, overlay=_overlay(image, mask),
                zoom_box=_zoom_box(mask, reference, image.size),
                pixels=pixels, score=score,
            )
        )
        statuses[serial] = dict(mask_pixels=pixels, confidence=score, seconds=duration)
        print(f"{case_id} {serial}: {pixels} pixels", flush=True)
    label = "NAS saved SAM3" if processor is None else "SAM3 new inference"
    title = f"{label} | {case_id} | prompt: {prompt} | cyan = key mask"
    _contact_sheet(rows, False, title, case_out / "all_views_overlay.png")
    _contact_sheet(rows, True, title, case_out / "all_views_zoom.png")
    (case_out / "probe.json").write_text(
        json.dumps(
            dict(
                condition_id=case_id, prompt=prompt, mode=mode,
                shot_dir=str(shot_dir), original_report=str(report_path), views=statuses,
            ),
            indent=2,
        ) + "\n"
    )
    return case_id, len(rows), sum(v["mask_pixels"] == 0 for v in statuses.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="white plastic cylindrical peg")
    parser.add_argument("--mode", choices=("infer", "saved"), default="infer")
    parser.add_argument("--case", action="append", help="Substring of condition ID; repeatable")
    args = parser.parse_args()
    base = args.evaluation_root / "results/key_direct_sam3_cylinder/cylinder"
    reports = sorted(base.glob("*/with_key/pose_*/report.json"))
    if args.case:
        reports = [p for p in reports if any(s in str(p) for s in args.case)]
    if not reports:
        raise SystemExit("No cylinder-key reports matched")
    processor = _load_segmentor(args.checkpoint) if args.mode == "infer" else None
    summary = [_case(p, args.output, args.prompt, processor, args.mode)
               for p in reports]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
