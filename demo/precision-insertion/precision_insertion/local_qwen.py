"""First-party, offline-only Qwen3-VL inference for AutoDex image checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from PIL import Image


@dataclass(frozen=True)
class LocalQwenReply:
    answer: str


class LocalQwenModel:
    """Load a Qwen3-VL snapshot with Transformers, without ZeroDex imports."""

    family = "qwen"

    def __init__(self, *, model_id: str, max_input_size: tuple[int, int],
                 require_cuda: bool = True):
        if (not isinstance(model_id, str) or not model_id.strip() or
                not isinstance(max_input_size, tuple) or
                len(max_input_size) != 2 or
                any(type(value) is not int or value <= 0
                    for value in max_input_size)):
            raise ValueError("local Qwen needs a model and image-size ceiling")
        if Path(model_id).is_absolute() and not Path(model_id).is_dir():
            raise FileNotFoundError(f"local Qwen model directory is missing: {model_id}")
        import torch
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; local Qwen was not loaded")
        try:
            from transformers import (
                AutoProcessor, Qwen3VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError("local Qwen needs Transformers Qwen3-VL support") from exc
        self.model_id = model_id
        self.max_input_size = max_input_size
        self.processor = AutoProcessor.from_pretrained(
            model_id, local_files_only=True)
        self.device = torch.device("cuda" if require_cuda else "cpu")
        dtype = torch.float16 if require_cuda else torch.float32
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, local_files_only=True, dtype=dtype).to(self.device).eval()

    def infer_images_prompt(
        self, images: list[Image.Image], prompt: str, *, max_new_tokens: int,
    ) -> LocalQwenReply:
        if (not images or any(not isinstance(image, Image.Image)
                              for image in images) or
                not isinstance(prompt, str) or not prompt.strip() or
                type(max_new_tokens) is not int or max_new_tokens <= 0):
            raise ValueError("Qwen inference needs ordered images, text and token limit")
        import torch
        content = ([{"type": "image", "image": image} for image in images] +
                   [{"type": "text", "text": prompt}])
        encoded = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True,
            add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = {name: value.to(self.device) if hasattr(value, "to") else value
                  for name, value in encoded.items()}
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False)
        prompt_tokens = inputs["input_ids"].shape[1]
        answer = self.processor.batch_decode(
            output[:, prompt_tokens:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        return LocalQwenReply(answer=answer)
