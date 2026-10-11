"""Lossless pixel crop for read-only local-VLM metric grounding.

The raw undistorted pixels are neither resized nor warped. Padding is only
appended to the right and bottom, so a crop-local mark maps to original
camera pixels by one exact integer translation. Padded-region marks are
invalid. The current Qwen3-VL processor uses a 32-pixel patch/merge factor;
the backend still checks its own actual processor geometry and fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class NativeGroundingCrop:
    image: Image.Image
    source_size_px: tuple[int, int]
    roi_xyxy_full_px: tuple[int, int, int, int]
    content_size_px: tuple[int, int]
    patch_factor_px: int
    minimum_side_px: int

    @property
    def model_size_px(self) -> tuple[int, int]:
        return self.image.size

    def to_full_pixel(self, point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        width, height = self.content_size_px
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("grounded pixel is in padding or outside crop")
        x0, y0, _, _ = self.roi_xyxy_full_px
        return float(x + x0), float(y + y0)

    def mapping_record(self) -> dict:
        return {
            "schema": "precision_insertion_lossless_grounding_crop_v1",
            "source_size_px": list(self.source_size_px),
            "roi_xyxy_full_px": list(self.roi_xyxy_full_px),
            "content_size_px": list(self.content_size_px),
            "model_size_px": list(self.model_size_px),
            "crop_to_full_translation_px": list(self.roi_xyxy_full_px[:2]),
            "patch_factor_px": self.patch_factor_px,
            "minimum_side_px": self.minimum_side_px,
            "padding_rgb": [127, 127, 127],
            "resampled": False,
            "padding_right_bottom_only": True,
        }


def make_native_grounding_crop(
    raw: Image.Image, roi_xyxy: tuple[int, int, int, int], *,
    patch_factor_px: int = 32, minimum_side_px: int = 256,
) -> NativeGroundingCrop:
    """Crop raw RGB pixels and pad without modifying any source pixel."""
    if not isinstance(raw, Image.Image) or raw.mode != "RGB":
        raise ValueError("metric ROI requires an undistorted RGB source image")
    if (not isinstance(roi_xyxy, (tuple, list)) or len(roi_xyxy) != 4 or
            any(type(value) is not int for value in roi_xyxy)):
        raise ValueError("metric ROI needs integer x0,y0,x1,y1")
    x0, y0, x1, y1 = roi_xyxy
    if not (0 <= x0 < x1 <= raw.width and 0 <= y0 < y1 <= raw.height):
        raise ValueError("metric ROI lies outside the original image")
    if (type(patch_factor_px) is not int or patch_factor_px < 1 or
            type(minimum_side_px) is not int or minimum_side_px < 1):
        raise ValueError("positive native VLM patch and side sizes are required")
    width, height = x1 - x0, y1 - y0
    padded_width = max(minimum_side_px, width)
    padded_height = max(minimum_side_px, height)
    padded_width = ((padded_width + patch_factor_px - 1) //
                    patch_factor_px) * patch_factor_px
    padded_height = ((padded_height + patch_factor_px - 1) //
                     patch_factor_px) * patch_factor_px
    crop = raw.crop((x0, y0, x1, y1))
    image = Image.new("RGB", (padded_width, padded_height),
                      color=(127, 127, 127))
    image.paste(crop, (0, 0))
    return NativeGroundingCrop(
        image, raw.size, (x0, y0, x1, y1), (width, height),
        patch_factor_px, minimum_side_px)
