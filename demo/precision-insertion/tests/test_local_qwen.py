"""The first-party Qwen adapter preserves image order and loads offline."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.local_qwen import LocalQwenModel  # noqa: E402


def test_local_qwen_uses_offline_snapshot_and_decodes_only_new_tokens(
        monkeypatch, tmp_path):
    calls = []

    class Processor:
        image_processor = SimpleNamespace(
            patch_size=16, merge_size=2,
            size={"shortest_edge": 65536, "longest_edge": 16777216})

        def apply_chat_template(self, messages, **kwargs):
            calls.append(("messages", messages))
            assert kwargs == {
                "tokenize": True, "add_generation_prompt": True,
                "return_dict": True, "return_tensors": "pt"}
            return {"input_ids": torch.tensor([[1, 2]])}

        def batch_decode(self, ids, **kwargs):
            assert ids.tolist() == [[7, 8]]
            assert kwargs == {"skip_special_tokens": True,
                              "clean_up_tokenization_spaces": False}
            return ['{"class":"unobservable"}']

    class Model:
        def to(self, device):
            calls.append(("device", str(device)))
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["input_ids"].tolist() == [[1, 2]]
            assert kwargs["max_new_tokens"] == 24
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 7, 8]])

    transformer_stub = ModuleType("transformers")
    transformer_stub.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda model_id, **kwargs:
        (calls.append(("processor", model_id, kwargs)) or Processor()))
    transformer_stub.Qwen3VLForConditionalGeneration = SimpleNamespace(
        from_pretrained=lambda model_id, **kwargs:
        (calls.append(("model", model_id, kwargs)) or Model()))
    monkeypatch.setitem(sys.modules, "transformers", transformer_stub)
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    loaded = LocalQwenModel(
        model_id=str(model_dir), max_input_size=(640, 480),
        require_cuda=False)
    first = Image.new("RGB", (224, 224), "white")
    second = Image.new("RGB", (224, 224), "white")
    assert loaded.infer_images_prompt(
        [first, second], "Describe visible white objects",
        max_new_tokens=24).answer == '{"class":"unobservable"}'
    assert calls[0] == ("processor", str(model_dir),
                        {"local_files_only": True})
    assert calls[1][0:2] == ("model", str(model_dir))
    assert calls[1][2]["local_files_only"] is True
    assert calls[1][2]["dtype"] is torch.float32
    content = next(row[1][0]["content"] for row in calls if row[0] == "messages")
    assert content == [
        {"type": "image", "image": first},
        {"type": "image", "image": second},
        {"type": "text", "text": "Describe visible white objects"},
    ]


def test_local_qwen_batches_independent_camera_prompts(monkeypatch, tmp_path):
    calls = []

    class Processor:
        def __init__(self):
            self.tokenizer = SimpleNamespace(padding_side="right")

        def apply_chat_template(self, conversations, **kwargs):
            calls.append((conversations, kwargs, self.tokenizer.padding_side))
            return {"input_ids": torch.tensor([[1, 2], [3, 4]])}

        def batch_decode(self, tokens, **_kwargs):
            assert tokens.tolist() == [[7], [8]]
            return ["first", "second"]

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 12
            return torch.tensor([[1, 2, 7], [3, 4, 8]])

    processor = Processor()
    transformer_stub = ModuleType("transformers")
    transformer_stub.AutoProcessor = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: processor)
    transformer_stub.Qwen3VLForConditionalGeneration = SimpleNamespace(
        from_pretrained=lambda *_args, **_kwargs: Model())
    monkeypatch.setitem(sys.modules, "transformers", transformer_stub)
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    model = LocalQwenModel(
        model_id=str(model_dir), max_input_size=(640, 480), require_cuda=False)
    frames = [Image.new("RGB", (8, 8), color)
              for color in ("red", "green", "blue", "white")]
    replies = model.infer_batch_images_prompt(
        [([frames[0], frames[1]], "camera one"),
         ([frames[2], frames[3]], "camera two")], max_new_tokens=12)
    assert [reply.answer for reply in replies] == ["first", "second"]
    conversations, kwargs, padding_side = calls[0]
    assert padding_side == "left"
    assert kwargs["processor_kwargs"] == {"padding": True}
    assert conversations[0][0]["content"][0]["image"] is frames[0]
    assert conversations[1][0]["content"][-1]["text"] == "camera two"
