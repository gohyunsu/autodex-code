"""ZeroDex-only multi-camera SAM3 inference with per-view output provenance.

The vendored ``Sam3Processor.set_image_batch`` computes a batched backbone,
but its ``set_text_prompt`` postprocessor assumes a single image.  This module
uses the same grounding model and score rule as ``Sam3ImageSegmentor.segment``
while retaining the batch dimension through mask selection.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


@torch.inference_mode()
def segment_batch(segmentor, rgbs: Sequence[np.ndarray], prompt: str) -> list[np.ndarray | None]:
    """Return one uint8 0/255 mask (or None) per input image, in input order."""
    if not rgbs:
        return []

    from PIL import Image
    from sam3.model.data_misc import FindStage, interpolate

    processor = segmentor.processor
    images = [Image.fromarray(rgb) for rgb in rgbs]
    state = processor.set_image_batch(images)
    count = len(images)
    device = processor.device
    backbone_out = state["backbone_out"]
    backbone_out.update(processor.model.backbone.forward_text([prompt], device=device))
    find_input = FindStage(
        img_ids=torch.arange(count, device=device, dtype=torch.long),
        text_ids=torch.zeros(count, device=device, dtype=torch.long),
        input_boxes=None,
        input_boxes_mask=None,
        input_boxes_label=None,
        input_points=None,
        input_points_mask=None,
    )
    outputs = processor.model.forward_grounding(
        backbone_out=backbone_out,
        find_input=find_input,
        geometric_prompt=processor.model._get_dummy_prompt(num_prompts=count),
        find_target=None,
    )

    scores = (outputs["pred_logits"].sigmoid() *
              outputs["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
    masks = outputs["pred_masks"]
    if scores.shape[0] != count or masks.shape[0] != count:
        raise ValueError("SAM3 batch output does not match input image count")

    result: list[np.ndarray | None] = []
    for index, image in enumerate(images):
        confidence, best_index = scores[index].max(dim=0)
        if confidence <= processor.confidence_threshold:
            result.append(None)
            continue
        logits = masks[index, best_index].unsqueeze(0).unsqueeze(0)
        full_size = interpolate(
            logits, (image.height, image.width), mode="bilinear",
            align_corners=False,
        ).sigmoid()
        result.append((full_size[0, 0] > 0.5).to(torch.uint8).cpu().numpy() * 255)
    return result
