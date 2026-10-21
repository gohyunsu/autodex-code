"""Read-only, event-driven VLM observations over AutoDex camera frames.

This module never commands a robot. It makes image order explicit, validates
closed-set responses, and fails closed on malformed output. A visual verdict
does not measure millimetres or authorize a retry or insertion contact.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
import re
import time
from typing import Protocol, Sequence

from PIL import Image

from .xy_voting import ViewVote, XYChoice, validate_choices
from .timing_profile import timed
from .view_parallel import map_view_calls, map_view_calls_until


class ImageVLM(Protocol):
    def infer(self, images: list[Image.Image], prompt: str) -> str: ...


DEFAULT_LOCAL_VLM_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


_CAMERA_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class LabeledFrame:
    camera_id: str
    phase: str
    timestamp_s: float
    image: Image.Image


@dataclass(frozen=True)
class XYView:
    camera_id: str
    timestamp_s: float
    raw: Image.Image
    overlay: Image.Image


@dataclass(frozen=True)
class HeldSceneView:
    """Same-exposure raw frame and full-scene predicted-mesh overlay.

    These pixels must be paired and source-bound by the caller. The overlay
    is a hypothesis, not a measured key pose, even when generated from a
    grasp-specific calibration.
    """

    camera_id: str
    timestamp_s: float
    raw: Image.Image
    predicted_overlay: Image.Image


@dataclass(frozen=True)
class VLMObservation:
    stage: str
    parsed: dict
    raw_answer: str
    prompt: str
    image_order: tuple[str, ...]
    parse_error: str | None
    backend_model: str
    latency_s: float
    pixel_mapping: dict | None = None
    aggregation: dict | None = None
    input_transform: dict | None = None

    def to_record(self) -> dict:
        record = {
            "stage": self.stage,
            "parsed": self.parsed,
            "raw_answer": self.raw_answer,
            "prompt": self.prompt,
            "image_order": list(self.image_order),
            "parse_error": self.parse_error,
            "backend_model": self.backend_model,
            "latency_s": self.latency_s,
            "scope": "read_only_vlm_observation_not_motion_authorization",
        }
        if self.pixel_mapping is not None:
            record["pixel_mapping"] = self.pixel_mapping
        if self.aggregation is not None:
            record["aggregation"] = self.aggregation
        if self.input_transform is not None:
            record["input_transform"] = self.input_transform
        return record


def resize_classification_images(
    images: list[Image.Image], max_size: tuple[int, int],
) -> tuple[list[Image.Image], dict]:
    """Derive small VLM classification inputs and bind them to raw pixels."""
    if (len(max_size) != 2 or
            any(type(side) is not int or side <= 0 for side in max_size)):
        raise ValueError("classification image size must be positive width and height")
    prepared, rows = [], []
    for image in images:
        if image.mode != "RGB":
            raise ValueError("classification resize needs RGB source images")
        width, height = image.size
        scale = min(1.0, max_size[0] / width, max_size[1] / height)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        result = (image if target == image.size else
                  image.resize(target, Image.Resampling.LANCZOS))
        prepared.append(result)
        rows.append({
            "source_size": list(image.size), "input_size": list(result.size),
            "source_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "input_rgb_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        })
    return prepared, {
        "schema": "precision_insertion_vlm_resize_fit_v1",
        "max_size": list(max_size), "resampling": "PIL_LANCZOS",
        "images": rows,
    }


@dataclass(frozen=True)
class PreinsertVisualAssessment:
    status: str
    supporting_cameras: tuple[str, ...]
    per_view: tuple[VLMObservation, ...]

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_preinsert_visual_v1",
            "status": self.status,
            "supporting_cameras": list(self.supporting_cameras),
            "per_view": [row.to_record() for row in self.per_view],
            "scope": "coarse_visual_check_not_metric_alignment_or_arrival_label",
            "robot_ready": False,
        }


class ZeroDexGeminiBackend:
    """ZeroDex-style ordered multi-image Gemini request without its CLI imports.

    Only inference is reused conceptually. Importing ZeroDex's full voting
    script also imports local-model/vision dependencies unrelated to Gemini;
    this narrow adapter preserves its PNG-part ordering and one-turn prompt.
    The key is never persisted or included in observation records.
    """

    def __init__(self, client, model: str, *, thinking_level: str = "LOW",
                 parallel_views: int = 1, client_factory=None):
        if (client is None or not isinstance(model, str) or not model.strip() or
                thinking_level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"} or
                type(parallel_views) is not int or not 1 <= parallel_views <= 8 or
                (parallel_views > 1 and not callable(client_factory))):
            raise ValueError("Gemini needs a client, model and valid thinking level")
        self.client = client
        self.model = model
        self.thinking_level = thinking_level
        self.max_parallel_views = parallel_views
        self.preferred_image_batch_size = 2 if parallel_views > 1 else None
        self._client_factory = client_factory
        # The API may transform image resolution internally; no reversible
        # original-pixel mapping is exposed by this narrow adapter.
        self.native_pixel_coordinates = False

    @classmethod
    def from_env(cls, *, model: str, env_name: str = "GEMINI_API_KEY",
                 thinking_level: str = "LOW",
                 parallel_views: int = 1) -> "ZeroDexGeminiBackend":
        """Explicit opt-in; fail closed if SDK/key is absent (no API call)."""
        if env_name != "GEMINI_API_KEY":
            raise ValueError("precision Gemini key must use GEMINI_API_KEY")
        key = os.environ.get(env_name, "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install google-genai for Gemini inference") from exc
        return cls(genai.Client(api_key=key), model,
                   thinking_level=thinking_level, parallel_views=parallel_views,
                   client_factory=(lambda: genai.Client(api_key=key))
                   if parallel_views > 1 else None)

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        if (not isinstance(prompt, str) or not prompt.strip() or
                not images or any(not isinstance(img, Image.Image)
                                  for img in images)):
            raise ValueError("Gemini needs ordered PIL images and a prompt")
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Install google-genai for Gemini inference") from exc
        parts = []
        for img in images:
            stream = io.BytesIO()
            img.save(stream, format="PNG")
            parts.append(types.Part.from_bytes(
                data=stream.getvalue(), mime_type="image/png"))
        parts.append(types.Part.from_text(text=prompt))
        client = (self._client_factory() if self._client_factory is not None
                  else self.client)
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=parts)],
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.thinking_level)),
            )
        finally:
            if client is not self.client and callable(getattr(client, "close", None)):
                client.close()
        return (response.text or "").strip()


class LocalQwenBackend:
    """Use local Transformers Qwen3-VL, guarding original pixel geometry."""

    def __init__(self, vlm, *, max_new_tokens: int = 512,
                 require_native_pixels: bool = True,
                 classification_image_size: tuple[int, int] | None = None):
        if (not hasattr(vlm, "infer_images_prompt") or
                type(max_new_tokens) is not int or max_new_tokens <= 0):
            raise ValueError("local Qwen needs an image adapter and token limit")
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens
        self.native_pixel_coordinates = bool(require_native_pixels)
        self.classification_image_size = classification_image_size
        # A physical 20-camera checkpoint must not become one 40-image Qwen
        # request. The observer asks for one same-camera before/after pair at
        # a time and conservatively combines independent labels.
        self.preferred_image_batch_size = 2

    @classmethod
    def from_transformers(
        cls, *, model_id: str = DEFAULT_LOCAL_VLM_MODEL_ID,
        max_input_size: tuple[int, int],
        max_new_tokens: int = 512, require_cuda: bool = True,
        require_native_pixels: bool = True,
        classification_image_size: tuple[int, int] | None = None,
    ) -> "LocalQwenBackend":
        """Load an offline Qwen model; fail before any model load if no GPU.

        Set ``require_cuda=False`` only for a slow offline CPU smoke test.
        ``max_input_size`` is a native-image ceiling, not a target resize.
        """
        if (not isinstance(model_id, str) or not model_id.strip() or
                not isinstance(max_input_size, tuple) or
                len(max_input_size) != 2 or
                any(type(v) is not int or v <= 0 for v in max_input_size)):
            raise ValueError("local Qwen needs a model and native size ceiling")
        if require_cuda:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is unavailable; local Qwen model was not loaded")
        from .local_qwen import LocalQwenModel
        vlm = LocalQwenModel(
            model_id=model_id, max_input_size=max_input_size,
            require_cuda=require_cuda)
        return cls(vlm, max_new_tokens=max_new_tokens,
                   require_native_pixels=require_native_pixels,
                   classification_image_size=classification_image_size)

    def prepare_classification_images(
        self, images: list[Image.Image],
    ) -> tuple[list[Image.Image], dict | None]:
        if self.classification_image_size is None:
            return images, None
        return resize_classification_images(images, self.classification_image_size)

    def _native_processor_geometry(self) -> tuple[int, int, int]:
        """Return Qwen's exact patch factor and accepted pixel-area range.

        The Qwen image processor may resize independently of the caller.
        We pad only right and bottom so the original pixel origin and every
        source pixel are unchanged, then require the processor to leave the
        padded image untouched. Unknown processors still fail closed.
        """
        if getattr(self.vlm, "family", None) != "qwen":
            raise ValueError(
                "native-pixel grounding requires a verified Qwen processor")
        processor = getattr(getattr(self.vlm, "processor", None),
                            "image_processor", None)
        try:
            patch = processor.patch_size
            merge = processor.merge_size
            minimum = processor.size["shortest_edge"]
            maximum = processor.size["longest_edge"]
        except (AttributeError, KeyError, TypeError) as exc:
            raise ValueError(
                "native-pixel grounding needs Qwen processor geometry") from exc
        if (any(type(value) is not int or value <= 0
                for value in (patch, merge, minimum, maximum)) or
                minimum > maximum):
            raise ValueError("invalid Qwen processor geometry")
        return patch * merge, minimum, maximum

    def _native_images(self, images: list[Image.Image]) -> list[Image.Image]:
        factor, minimum, maximum = self._native_processor_geometry()
        ceiling = getattr(self.vlm, "max_input_size", None)
        if (not isinstance(ceiling, (tuple, list)) or len(ceiling) != 2):
            raise ValueError("native-pixel VLM needs an image-size ceiling")
        prepared = []
        for frame in images:
            if frame.mode != "RGB":
                raise ValueError("native-pixel VLM needs RGB source images")
            width = ((frame.width + factor - 1) // factor) * factor
            height = ((frame.height + factor - 1) // factor) * factor
            if (width > ceiling[0] or height > ceiling[1] or
                    not minimum <= width * height <= maximum):
                raise ValueError(
                    "local VLM processor would resize an original grounding "
                    f"image {frame.width}x{frame.height}; use a verified "
                    "smaller pixel-preserving crop")
            if (width, height) == frame.size:
                prepared.append(frame)
            else:
                padded = Image.new("RGB", (width, height), (127, 127, 127))
                padded.paste(frame, (0, 0))
                prepared.append(padded)
        return prepared

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        if (not images or any(not isinstance(img, Image.Image)
                              for img in images) or
                not isinstance(prompt, str) or not prompt.strip()):
            raise ValueError("local VLM needs ordered PIL images and a prompt")
        if self.native_pixel_coordinates:
            images = self._native_images(images)
        result = self.vlm.infer_images_prompt(
            images, prompt, max_new_tokens=self.max_new_tokens)
        return result.answer


# Older demo imports can still construct the same image adapter; the active
# loader never imports or instantiates a ZeroDex module.
ZeroDexLocalBackend = LocalQwenBackend


def load_vlm_backend(
    *, mode: str = "local", model_id: str | None = None,
    max_input_size: tuple[int, int] | None = None,
    max_new_tokens: int = 512, require_cuda: bool = True,
    require_native_pixels: bool = True, parallel_views: int = 1,
    classification_image_size: tuple[int, int] | None = None,
) -> ImageVLM:
    """Choose one VLM for existing read-only checkpoint/grounding calls.

    Neither branch acquires cameras or authorizes a robot action. Gemini is
    explicit external-API opt-in; local uses first-party offline Qwen loading.
    """
    if mode == "gemini":
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Gemini needs an explicit model ID")
        return ZeroDexGeminiBackend.from_env(
            model=model_id, parallel_views=parallel_views)
    if mode == "local":
        if parallel_views != 1:
            raise ValueError("single-GPU local Qwen does not support parallel views")
        selected_model = (DEFAULT_LOCAL_VLM_MODEL_ID if model_id is None
                          else model_id)
        if max_input_size is None:
            raise ValueError("local mode needs original image-size ceiling")
        return LocalQwenBackend.from_transformers(
            model_id=selected_model, max_input_size=max_input_size,
            max_new_tokens=max_new_tokens, require_cuda=require_cuda,
            require_native_pixels=require_native_pixels,
            classification_image_size=classification_image_size)
    raise ValueError("VLM mode must be gemini or local")


def _frame_order(frames: Sequence[LabeledFrame]) -> tuple[str, ...]:
    if not frames:
        raise ValueError("at least one image is required")
    order = []
    for frame in frames:
        if not frame.camera_id or not frame.phase:
            raise ValueError("every frame needs a camera ID and phase")
        if not math.isfinite(float(frame.timestamp_s)):
            raise ValueError("frame timestamps must be finite")
        if not isinstance(frame.image, Image.Image):
            raise TypeError("frame image must be a PIL image")
        order.append(f"{frame.phase}/{frame.camera_id}@{frame.timestamp_s:.6f}")
    if len(order) != len(set(order)):
        raise ValueError("duplicate labeled frame")
    return tuple(order)


def _require_temporal_camera_pairs(
    frames: Sequence[LabeledFrame], *, before: str, after: str,
) -> None:
    """A temporal VLM comparison needs the *same* camera at both phases."""
    grouped: dict[str, dict[str, float]] = {before: {}, after: {}}
    for frame in frames:
        if frame.phase not in grouped:
            raise ValueError("unexpected image phase in temporal comparison")
        if frame.camera_id in grouped[frame.phase]:
            raise ValueError("duplicate camera in one observation phase")
        grouped[frame.phase][frame.camera_id] = float(frame.timestamp_s)
    if (not grouped[before] or
            set(grouped[before]) != set(grouped[after])):
        raise ValueError(f"{before} and {after} need paired camera views")
    if any(grouped[before][camera] >= grouped[after][camera]
           for camera in grouped[before]):
        raise ValueError("temporal camera pairs are not time ordered")


def _parse_object(answer: str) -> dict:
    if not isinstance(answer, str):
        raise ValueError("VLM answer must be text")
    source = answer.strip()
    # Small local VLMs often wrap otherwise valid JSON in one Markdown fence.
    # Accept only a complete JSON fence, never extra prose or a JSON fragment.
    fenced = re.fullmatch(r"```(?:json)?\s*\n([\s\S]*?)\n```", source,
                          flags=re.IGNORECASE)
    if fenced is not None:
        source = fenced.group(1).strip()
    value = json.loads(source)
    if not isinstance(value, dict):
        raise ValueError("VLM answer must be a JSON object")
    return value


def _backend_model(backend: ImageVLM) -> str:
    model = getattr(backend, "model", None)
    if model is None:
        model = getattr(getattr(backend, "vlm", None), "model_id", None)
    return str(model) if model is not None else type(backend).__name__


def _infer_closed_set(
    backend: ImageVLM,
    *,
    stage: str,
    frames: Sequence[LabeledFrame],
    prompt_body: str,
    field: str,
    allowed: frozenset[str],
    fallback: str,
) -> VLMObservation:
    order = _frame_order(frames)
    prompt = (
        "Images are supplied in this exact order:\n"
        + "\n".join(f"{i + 1}: {label}" for i, label in enumerate(order))
        + "\n\nValid camera IDs: "
        + ", ".join(sorted({frame.camera_id for frame in frames}))
        + ". In evidence_views use camera IDs only, never image numbers "
        "or phase names. Use [] if nothing is visually established. "
        "Return exactly one compact JSON object with no Markdown or text "
        "outside it. Keep evidence to one short visual phrase (at most "
        "12 words); never repeat a sentence.\n\n"
        + prompt_body
        + f"\nThe JSON object must have exactly these keys: {field}, "
        "evidence_views, evidence. Set "
        + field
        + " to exactly ONE of these strings: "
        + ", ".join(json.dumps(value) for value in sorted(allowed))
        + ". Do not copy the entire option list as a value. "
        "evidence_views must be an array of cited camera ID strings; "
        "evidence must be one short string."
    )
    images = [frame.image for frame in frames]
    prepare = getattr(backend, "prepare_classification_images", None)
    images, input_transform = prepare(images) if callable(prepare) else (images, None)
    started = time.perf_counter()
    camera_ids = sorted({frame.camera_id for frame in frames})
    with timed("vlm.infer", phase="vlm", kind="inference",
               stage=stage, camera_id=(camera_ids[0] if len(camera_ids) == 1
                                      else "multiple"), camera_ids=camera_ids,
               image_count=len(frames), image_sizes=[list(image.size) for image in images],
               model=_backend_model(backend)):
        answer = backend.infer(images, prompt)
    latency = time.perf_counter() - started
    if not isinstance(answer, str):
        raise TypeError("VLM backend must return text")
    try:
        parsed = _parse_object(answer)
        if parsed.get(field) not in allowed:
            raise ValueError(f"invalid {field}")
        if not isinstance(parsed.get("evidence"), str):
            raise ValueError("missing visual evidence")
        evidence_views = parsed.get("evidence_views")
        available = {frame.camera_id for frame in frames}
        if (not isinstance(evidence_views, list) or
                not all(isinstance(v, str) and v in available
                        for v in evidence_views) or
                len(evidence_views) != len(set(evidence_views)) or
                (parsed[field] != fallback and not evidence_views)):
            raise ValueError("invalid evidence_views")
        error = None
    except (json.JSONDecodeError, ValueError) as exc:
        parsed = {field: fallback, "evidence": "", "evidence_views": []}
        error = str(exc)
    return VLMObservation(
        stage, parsed, answer, prompt, order, error,
        _backend_model(backend), latency, input_transform=input_transform)


def _per_view_consensus(
    backend: ImageVLM, *, stage: str, frames: Sequence[LabeledFrame],
    prompt_body: str, field: str, allowed: frozenset[str], fallback: str,
    before: str, after: str, majority: bool = False,
) -> VLMObservation:
    """Bound local VLM memory to a temporal pair, then require agreement.

    Abstaining/invalid views do not vote. Most stages abstain on conflicting
    decisive cameras. The released-key stage uses a majority of configured
    cameras, so one false negative cannot veto three agreeing views.
    The aggregate is explicitly synthetic; exact model replies remain in
    ``aggregation.per_view`` for replay and inspection.
    """
    _require_temporal_camera_pairs(frames, before=before, after=after)
    order = _frame_order(frames)
    cameras = sorted({frame.camera_id for frame in frames})
    minimum_support = (len(cameras) // 2 + 1) if majority else 2
    def observe_camera(camera: str) -> VLMObservation:
        pair = [next(frame for frame in frames
                     if frame.camera_id == camera and frame.phase == phase)
                for phase in (before, after)]
        return _infer_closed_set(
            backend, stage=stage, frames=pair, prompt_body=prompt_body,
            field=field, allowed=allowed, fallback=fallback)
    early = getattr(backend, "early_quorum_views", None)
    replay_cameras = getattr(backend, "replay_completed_camera_ids", None)
    if replay_cameras is not None:
        # Saved checkpoints replay the exact workers that answered before
        # quorum; their completion order need not match sorted camera order.
        voted_cameras = sorted(replay_cameras)
        if (early is None or not voted_cameras or
                not set(voted_cameras) <= set(cameras)):
            raise ValueError("retry replay camera set differs from saved views")
        observations = [observe_camera(camera) for camera in voted_cameras]
    elif early is not None:
        def sufficient(done: dict[str, VLMObservation]) -> bool:
            if len(done) < early:
                return False
            selected = sorted(done)
            result = _consensus_parsed(
                [done[camera].to_record() for camera in selected],
                field=field, fallback=fallback, cameras=selected,
                majority=majority, minimum_support=minimum_support)
            return result[field] != fallback
        completed = map_view_calls_until(
            backend, cameras, observe_camera, sufficient)
        voted_cameras = sorted(completed)
        observations = [completed[camera] for camera in voted_cameras]
    else:
        voted_cameras = cameras
        observations = map_view_calls(backend, cameras, observe_camera)
    parsed = _consensus_parsed(
        [row.to_record() for row in observations], field=field,
        fallback=fallback, cameras=voted_cameras, majority=majority,
        minimum_support=minimum_support)
    prompt = (
        "Images are supplied in this exact order:\n" +
        "\n".join(f"{index + 1}: {label}" for index, label in
                  enumerate(order)) +
        "\n\nEach camera's temporal pair was sent in a separate local VLM "
        "request. The aggregate below is a deterministic consensus, not "
        "an additional model reply.")
    return VLMObservation(
        stage, parsed, json.dumps(parsed, sort_keys=True), prompt, order,
        None, _backend_model(backend),
        sum(row.latency_s for row in observations), aggregation={
            "schema": "precision_insertion_per_camera_vlm_consensus_v1",
            "per_view": [row.to_record() for row in observations],
            **({"vote_rule": "configured_camera_majority"} if majority else {}),
            **({"completed_camera_ids": voted_cameras,
                "requested_camera_ids": cameras} if early is not None else {}),
        })


def _consensus_parsed(
    per_view: Sequence[dict], *, field: str, fallback: str,
    cameras: Sequence[str], majority: bool = False,
    minimum_support: int = 2,
) -> dict:
    votes: dict[str, list[str]] = {}
    for camera, row in zip(cameras, per_view):
        answer = row.get("parsed", {})
        if (row.get("parse_error") is None and
                answer.get(field) != fallback and
                answer.get("evidence_views") == [camera]):
            votes.setdefault(answer[field], []).append(camera)
    if votes:
        label, supporters = max(votes.items(), key=lambda item: len(item[1]))
        dissent = sum(len(rows) for other, rows in votes.items()
                      if other != label)
        if (len(supporters) >= minimum_support and
                (dissent == 0 or majority and len(supporters) > dissent)):
            return {field: label, "evidence": "Independent same-camera votes",
                    "evidence_views": supporters}
    return {field: fallback, "evidence": "Insufficient or conflicting views",
            "evidence_views": []}


def verify_per_view_consensus(
    visual: dict, *, field: str, fallback: str,
    cameras: set[str], before: str, after: str,
) -> None:
    """Replay a local checkpoint's deterministic consensus and raw answers."""
    aggregation = visual.get("aggregation")
    if aggregation is None:
        return
    if (not isinstance(aggregation, dict) or aggregation.get("schema") !=
            "precision_insertion_per_camera_vlm_consensus_v1" or
            not isinstance(aggregation.get("per_view"), list) or
            aggregation.get("vote_rule") not in {
                None, "configured_camera_majority"} or
            (aggregation.get("vote_rule") is not None and
             visual.get("stage") != "released_insertion_visual")):
        raise ValueError("invalid per-camera VLM aggregation")
    ordered = aggregation.get("completed_camera_ids", sorted(cameras))
    if (not isinstance(ordered, list) or
            len(ordered) != len(aggregation["per_view"]) or
            ordered != sorted(set(ordered)) or
            not set(ordered) <= cameras or
            aggregation.get("requested_camera_ids", sorted(cameras)) !=
            sorted(cameras)):
        raise ValueError("invalid completed per-camera VLM views")
    global_order = visual.get("image_order")
    if (not isinstance(global_order, list) or
            len(global_order) != 2 * len(cameras)):
        raise ValueError("per-camera VLM aggregate image order is invalid")
    for camera, row in zip(ordered, aggregation["per_view"]):
        expected_order = [next((label for label in global_order
                                if isinstance(label, str) and
                                label.startswith(f"{phase}/{camera}@")), None)
                          for phase in (before, after)]
        if (not isinstance(row, dict) or row.get("aggregation") is not None or
                not isinstance(row.get("image_order"), list) or
                any(item is None for item in expected_order) or
                row["image_order"] != expected_order or
                not str(row.get("prompt", "")).startswith(
                    "Images are supplied in this exact order:\n" +
                    "\n".join(f"{index + 1}: {label}" for index, label in
                              enumerate(expected_order)) + "\n\n")):
            raise ValueError("per-camera VLM pair differs from checkpoint views")
        if row.get("parse_error") is None:
            try:
                raw = _parse_object(row.get("raw_answer", ""))
            except (TypeError, ValueError) as exc:
                raise ValueError("per-camera VLM raw answer is invalid") from exc
            if raw != row.get("parsed"):
                raise ValueError("per-camera VLM raw answer changed")
    expected = _consensus_parsed(
        aggregation["per_view"], field=field, fallback=fallback,
        cameras=ordered,
        majority=aggregation.get("vote_rule") ==
        "configured_camera_majority",
        minimum_support=(len(cameras) // 2 + 1) if aggregation.get(
            "vote_rule") == "configured_camera_majority" else 2)
    if (visual.get("parsed") != expected or
            visual.get("raw_answer") != json.dumps(expected, sort_keys=True) or
            visual.get("parse_error") is not None):
        raise ValueError("per-camera VLM consensus changed")


def observe_lift(backend: ImageVLM, frames: Sequence[LabeledFrame]) -> VLMObservation:
    """Classify whether the key follows the held hand across lift frames."""
    phases = {frame.phase for frame in frames}
    if not {"before_grasp", "after_lift"} <= phases:
        raise ValueError("lift observation needs before_grasp and after_lift")
    _require_temporal_camera_pairs(
        frames, before="before_grasp", after="after_lift")
    infer = (_per_view_consensus if
             getattr(backend, "preferred_image_batch_size", None) == 2
             else _infer_closed_set)
    return infer(
        backend, stage="post_lift", frames=frames,
        prompt_body=(
            "The key is the small white cylinder; the socket is the large "
            "white cup. Inspect image 2 first. Is the key on the checkerboard "
            "or tabletop? If yes, class MUST be miss, even if it moved or the "
            "fingers closed. Say held only if the key itself is visibly "
            "between the fingers and absent from the board. Falling from "
            "the hand is slip. Hidden is not held: use unobservable. State "
            "the key's location in image 2 as evidence."
        ),
        field="class", allowed=frozenset({"held", "miss", "slip", "unobservable"}),
        fallback="unobservable",
        **({"before": "before_grasp", "after": "after_lift"}
           if infer is _per_view_consensus else {}),
    )


def observe_insertion_visual(
    backend: ImageVLM, frames: Sequence[LabeledFrame],
) -> VLMObservation:
    """Classify visible insertion state independently of numeric depth/force."""
    phases = {frame.phase for frame in frames}
    if not {"preinsert", "final_or_abort"} <= phases:
        raise ValueError("insertion observation needs preinsert and final_or_abort")
    _require_temporal_camera_pairs(
        frames, before="preinsert", after="final_or_abort")
    infer = (_per_view_consensus if
             getattr(backend, "preferred_image_batch_size", None) == 2
             else _infer_closed_set)
    return infer(
        backend, stage="insertion_visual", frames=frames,
        prompt_body=(
            "The key and socket are both white; use shape and location. "
            "Compare the time-ordered RAW key and socket pixels. CAD overlays "
            "are predictions, not measurements. A hidden key is not proof "
            "of insertion. Do not claim numerical depth from pixels. "
            "Describe only what the raw images visibly establish."
        ),
        field="visual_class",
        allowed=frozenset({
            "normal_appearance", "partial", "rim_jam", "slip", "unobservable",
        }),
        fallback="unobservable",
        **({"before": "preinsert", "after": "final_or_abort"}
           if infer is _per_view_consensus else {}),
    )


def observe_released_insertion_visual(
    backend: ImageVLM, frames: Sequence[LabeledFrame],
) -> VLMObservation:
    """Judge a released key from paired initial/final clear-view images.

    The first image identifies the loose key before pickup.  The second image
    is acquired only after the 20 mm stroke, full hand opening and robot
    clear-view exit.  A missing/occluded key is deliberately not positive
    evidence that it is inside the socket.
    """
    phases = {frame.phase for frame in frames}
    if not {"initial_tabletop", "final_clear_view"} <= phases:
        raise ValueError(
            "released insertion needs initial_tabletop and final_clear_view")
    _require_temporal_camera_pairs(
        frames, before="initial_tabletop", after="final_clear_view")
    return _per_view_consensus(
        backend, stage="released_insertion_visual", frames=frames,
        prompt_body=(
            "The key and socket are both white; distinguish them by shape "
            "and location. Image 1 shows the loose target key during the initial tabletop "
            "perception step. Image 2 shows the same workspace after the "
            "20 mm insertion attempt, after the hand has fully opened and "
            "the robot has moved to clear view. Decide from visible pixels "
            "whether that key is now physically inside the socket. Classify "
            "inside_socket only when the final image contains positive visual "
            "evidence that the key occupies the socket opening/interior. "
            "Classify outside_socket when the key is visibly outside the "
            "socket. Occlusion, disappearance, the empty hand, or the "
            "commanded motion is not evidence of insertion; use unobservable "
            "when the final relation cannot be established. Do not infer a "
            "numerical insertion depth from pixels."
        ),
        field="visual_class",
        allowed=frozenset({
            "inside_socket", "outside_socket", "unobservable",
        }),
        fallback="unobservable",
        before="initial_tabletop", after="final_clear_view",
        majority=True,
    )


def observe_preinsert_hold_views(
    backend: ImageVLM, views: Sequence[HeldSceneView], *,
    max_capture_skew_s: float, minimum_agreeing_views: int = 2,
) -> PreinsertVisualAssessment:
    """Compare raw and *predicted* meshes per camera, then demand consensus.

    This deliberately does not estimate XY in millimetres. A coarse match is
    only visual support: measured wrist/hand feedback, a validated key-hand
    relation, collision-free arrival and calibrated socket-frame residuals
    must still be checked before labelling ``preinsert_reached=True``.
    """
    if (type(minimum_agreeing_views) is not int or
            minimum_agreeing_views < 2 or
            not math.isfinite(float(max_capture_skew_s)) or
            max_capture_skew_s <= 0 or len(views) < minimum_agreeing_views):
        raise ValueError("preinsert check needs multiple synchronized views")
    cameras = [view.camera_id for view in views]
    if (any(not isinstance(camera, str) or not _CAMERA_ID.fullmatch(camera)
            for camera in cameras) or
            len(cameras) != len(set(cameras))):
        raise ValueError("preinsert views need unique camera IDs with safe camera names")
    times = [float(view.timestamp_s) for view in views]
    if (not all(math.isfinite(value) and value > 0 for value in times) or
            max(times) - min(times) > max_capture_skew_s):
        raise ValueError("preinsert cameras are stale or asynchronous")
    observations: list[VLMObservation] = []
    for view in views:
        if (not isinstance(view.raw, Image.Image) or
                not isinstance(view.predicted_overlay, Image.Image) or
                view.raw.size != view.predicted_overlay.size or
                view.raw.mode != "RGB" or
                view.predicted_overlay.mode != "RGB"):
            raise ValueError("preinsert needs same-size RGB raw/overlay pairs")
        frames = (
            LabeledFrame(view.camera_id, "raw_preinsert", view.timestamp_s,
                         view.raw),
            LabeledFrame(view.camera_id, "predicted_scene", view.timestamp_s,
                         view.predicted_overlay),
        )
        observations.append(_infer_closed_set(
            backend, stage="preinsert_visual", frames=frames,
            prompt_body=(
                "The key and socket are both white; use shape and location. "
                "Image 1 is the RAW camera frame. Image 2 is the same frame "
                "with a translucent robot/key/socket CAD prediction. The CAD "
                "key is a hypothesis, not observed evidence; synthetic "
                "occlusions can be wrong. Inspect the RAW pixels first. "
                "Choose coarse_match only if visible key pixels are still "
                "carried by the hand and are grossly consistent with the "
                "socket approach. Choose gross_misalignment if the visible "
                "key/socket axes are plainly inconsistent, or slip_or_miss "
                "if visible key pixels show the key is no longer held. "
                "If the key/rim is occluded, pixels do not resolve the state, "
                "or the overlay alone suggests success, choose unobservable. "
                "Do not estimate millimetres, depth, or contact from images. "
                f"Use only camera ID {view.camera_id}."
            ),
            field="class", allowed=frozenset({
                "coarse_match", "gross_misalignment", "slip_or_miss",
                "unobservable",
            }), fallback="unobservable",
        ))
    decisive: dict[str, list[str]] = {}
    for view, observed in zip(views, observations):
        category = observed.parsed["class"]
        if (observed.parse_error is None and
                category != "unobservable" and
                observed.parsed["evidence"].strip() and
                observed.parsed["evidence_views"] == [view.camera_id]):
            decisive.setdefault(category, []).append(view.camera_id)
    if (len(decisive) == 1 and
            len(next(iter(decisive.values()))) >= minimum_agreeing_views):
        status = next(iter(decisive))
        support = tuple(sorted(decisive[status]))
    else:
        status, support = "unknown", ()
    return PreinsertVisualAssessment(status, support, tuple(observations))


def observe_preinsert_raw_views(
    backend: ImageVLM, views: Sequence[HeldSceneView], *,
    max_capture_skew_s: float, minimum_agreeing_views: int = 2,
) -> PreinsertVisualAssessment:
    """Legacy single-preinsert-image diagnostic; not the live checkpoint.

    ``HeldSceneView`` is retained as the report container because the
    deterministic geometry/renderer audit still stores a predicted scene for
    humans.  The VLM receives only ``view.raw``; the CAD rendering is never an
    inference input.
    """
    if (type(minimum_agreeing_views) is not int or
            minimum_agreeing_views < 2 or
            not math.isfinite(float(max_capture_skew_s)) or
            max_capture_skew_s <= 0 or len(views) < minimum_agreeing_views):
        raise ValueError("preinsert check needs multiple synchronized views")
    cameras = [view.camera_id for view in views]
    if (any(not isinstance(camera, str) or not _CAMERA_ID.fullmatch(camera)
            for camera in cameras) or len(cameras) != len(set(cameras))):
        raise ValueError("preinsert views need unique safe camera IDs")
    times = [float(view.timestamp_s) for view in views]
    if (not all(math.isfinite(value) and value > 0 for value in times) or
            max(times) - min(times) > max_capture_skew_s):
        raise ValueError("preinsert cameras are stale or asynchronous")
    observations: list[VLMObservation] = []
    for view in views:
        if not isinstance(view.raw, Image.Image) or view.raw.mode != "RGB":
            raise ValueError("preinsert needs raw RGB camera images")
        frames = (LabeledFrame(
            view.camera_id, "raw_preinsert", view.timestamp_s, view.raw),)
        observations.append(_infer_closed_set(
            backend, stage="preinsert_visual", frames=frames,
            prompt_body=(
                "This is one RAW camera image at the pre-insertion hold. "
                "Both key and socket are white; use shape and location. "
                "Inspect only visible pixels. Choose coarse_match only when "
                "the target key is visibly still carried by the hand above "
                "the socket and its insertion axis is grossly compatible "
                "with the socket axis. Choose gross_misalignment when the "
                "visible key and socket are plainly displaced or tilted, "
                "or slip_or_miss when visible pixels show that the key is no "
                "longer held. If the key or socket rim is occluded, choose "
                "unobservable. Do not use an expected CAD pose, infer hidden "
                "geometry, estimate millimetres, or infer contact/depth. "
                f"Use only camera ID {view.camera_id}."
            ),
            field="class", allowed=frozenset({
                "coarse_match", "gross_misalignment", "slip_or_miss",
                "unobservable",
            }), fallback="unobservable",
        ))
    decisive: dict[str, list[str]] = {}
    for view, observed in zip(views, observations):
        category = observed.parsed["class"]
        if (observed.parse_error is None and category != "unobservable" and
                observed.parsed["evidence"].strip() and
                observed.parsed["evidence_views"] == [view.camera_id]):
            decisive.setdefault(category, []).append(view.camera_id)
    if (len(decisive) == 1 and
            len(next(iter(decisive.values()))) >= minimum_agreeing_views):
        status = next(iter(decisive))
        support = tuple(sorted(decisive[status]))
    else:
        status, support = "unknown", ()
    return PreinsertVisualAssessment(status, support, tuple(observations))


def observe_preinsert_temporal_views(
    backend: ImageVLM, frames: Sequence[LabeledFrame], *,
    max_capture_skew_s: float, minimum_agreeing_views: int = 2,
) -> PreinsertVisualAssessment:
    """Compare each camera's raw after-lift and preinsert pixels in order.

    This is a coarse continuity/arrival label, never an XY or depth measure.
    CAD overlays are deliberately absent from the inference input.
    """
    if (type(minimum_agreeing_views) is not int or
            minimum_agreeing_views < 2 or
            not math.isfinite(float(max_capture_skew_s)) or
            max_capture_skew_s <= 0):
        raise ValueError("preinsert pair needs positive multiview limits")
    _require_temporal_camera_pairs(
        frames, before="after_lift", after="preinsert")
    grouped = {phase: {} for phase in ("after_lift", "preinsert")}
    for frame in frames:
        if (not _CAMERA_ID.fullmatch(frame.camera_id) or
                not isinstance(frame.image, Image.Image) or
                frame.image.mode != "RGB" or
                not math.isfinite(float(frame.timestamp_s)) or
                frame.timestamp_s <= 0):
            raise ValueError("preinsert pair needs identified raw RGB exposures")
        grouped[frame.phase][frame.camera_id] = frame
    cameras = sorted(grouped["after_lift"])
    if len(cameras) < minimum_agreeing_views:
        raise ValueError("preinsert pair needs multiple camera views")
    for phase in grouped:
        times = [frame.timestamp_s for frame in grouped[phase].values()]
        if max(times) - min(times) > max_capture_skew_s:
            raise ValueError("preinsert pair has asynchronous camera views")
    def observe_camera(camera: str) -> VLMObservation:
        before = grouped["after_lift"][camera]
        after = grouped["preinsert"][camera]
        return _infer_closed_set(
            backend, stage="preinsert_visual", frames=(before, after),
            prompt_body=(
                "Image 1 is the RAW after-lift view; image 2 is the RAW "
                "preinsert view from the same camera. Both peg and socket "
                "are white; use shape and location. Is the same white peg "
                "still visibly held after transfer and roughly aligned "
                "above the socket? Choose coarse_match only from visible "
                "key pixels in both images. Choose gross_misalignment for "
                "a visibly displaced or incompatible arrival, slip_or_miss "
                "when the key is visibly no longer held, and unobservable "
                "when occlusion prevents a comparison. Do not infer metric "
                "XY, contact, insertion success or depth. "
                f"Use only camera ID {camera}."
            ),
            field="class", allowed=frozenset({
                "coarse_match", "gross_misalignment", "slip_or_miss",
                "unobservable",
            }), fallback="unobservable",
        )
    early = getattr(backend, "early_quorum_views", None)
    if early is not None:
        def sufficient(done: dict[str, VLMObservation]) -> bool:
            if len(done) < early:
                return False
            votes = {row.parsed["class"] for row in done.values()
                     if row.parse_error is None and
                     row.parsed["class"] != "unobservable" and
                     row.parsed["evidence"].strip()}
            if len(votes) != 1:
                return False
            label = next(iter(votes))
            return sum(row.parsed["class"] == label and
                       row.parsed["evidence_views"] == [camera]
                       for camera, row in done.items()) >= minimum_agreeing_views
        completed = map_view_calls_until(
            backend, cameras, observe_camera, sufficient)
        observed_cameras = sorted(completed)
        observations = [completed[camera] for camera in observed_cameras]
    else:
        observed_cameras = cameras
        observations = map_view_calls(backend, cameras, observe_camera)
    decisive: dict[str, list[str]] = {}
    for camera, observed in zip(observed_cameras, observations):
        category = observed.parsed["class"]
        if (observed.parse_error is None and category != "unobservable" and
                observed.parsed["evidence"].strip() and
                observed.parsed["evidence_views"] == [camera]):
            decisive.setdefault(category, []).append(camera)
    if len(decisive) == 1 and len(next(iter(decisive.values()))) >= minimum_agreeing_views:
        status = next(iter(decisive))
        support = tuple(sorted(decisive[status]))
    else:
        status, support = "unknown", ()
    return PreinsertVisualAssessment(status, support, tuple(observations))


def observe_xy_views(
    backend: ImageVLM,
    views: Sequence[XYView],
    choices: Sequence[XYChoice],
) -> tuple[list[ViewVote], list[VLMObservation]]:
    """Ask each camera separately for an ID; no pixel/mm coordinates accepted.

    The caller must prepare raw images and CAD overlays in the same calibrated
    image frame. A vote is only a proposal to ``resolve_multiview_choice``;
    it is neither independent metric pose estimation nor motion permission.
    """
    catalog = validate_choices(choices)
    if not views or len({view.camera_id for view in views}) != len(views):
        raise ValueError("XY voting needs unique nonempty camera views")
    def observe_view(view: XYView) -> tuple[ViewVote, VLMObservation]:
        frames = [
            LabeledFrame(view.camera_id, "raw_hold", view.timestamp_s, view.raw),
            LabeledFrame(view.camera_id, "candidate_overlay", view.timestamp_s,
                         view.overlay),
        ]
        order = _frame_order(frames)
        ids = ", ".join(catalog)
        prompt = (
            "Both key and socket are white; use shape and location. "
            "Image 1 is the RAW camera crop; image 2 marks calibrated "
            "candidate center anchors on the same crop. Markers are target "
            "projections, not observed key poses or measured millimetres. "
            "Choose the ID that reduces visible key/socket centerline "
            "misalignment, or abstain if "
            "occluded, indistinguishable, tilted, or uncertain. Never invent "
            "coordinates or a new ID. "
            f"Allowed IDs: [{ids}]. Camera: {view.camera_id}. "
            "Return one compact JSON object with exactly the keys visible, "
            "choice_id, failure_class, evidence. Set visible to a JSON "
            "boolean. Set choice_id to exactly one allowed ID or abstain. "
            "Set failure_class to exactly one of these strings: "
            '"misaligned", "rim_jam", "roughly_aligned", "slip", '
            '"occluded", "unknown". Do not copy an entire option list as '
            "one value. Set evidence to one short visual phrase."
        )
        started = time.perf_counter()
        with timed("vlm.infer", phase="vlm", kind="inference",
                   stage="legacy_xy_choice", camera_id=view.camera_id,
                   image_count=2, model=_backend_model(backend)):
            answer = backend.infer([view.raw, view.overlay], prompt)
        latency = time.perf_counter() - started
        if not isinstance(answer, str):
            raise TypeError("VLM backend must return text")
        try:
            parsed = _parse_object(answer)
            visible = parsed.get("visible")
            choice_id = parsed.get("choice_id")
            failure = parsed.get("failure_class")
            if (type(visible) is not bool or
                    choice_id not in catalog.keys() | {"abstain", None} or
                    (visible is False and choice_id not in {"abstain", None}) or
                    failure not in {
                        "misaligned", "rim_jam", "roughly_aligned",
                        "slip", "occluded", "unknown",
                    } or not isinstance(parsed.get("evidence"), str)):
                raise ValueError("invalid or inconsistent XY response")
            error = None
        except (json.JSONDecodeError, ValueError) as exc:
            parsed = {
                "visible": False, "choice_id": "abstain",
                "failure_class": "unknown", "evidence": "",
            }
            error = str(exc)
        vote = ViewVote(
            view.camera_id, float(view.timestamp_s),
            parsed["visible"], parsed["choice_id"], parsed["failure_class"],
        )
        record = VLMObservation(
            "preinsert_xy", parsed, answer, prompt, order, error,
            _backend_model(backend), latency)
        return vote, record

    results = map_view_calls(backend, views, observe_view)
    return [vote for vote, _ in results], [record for _, record in results]
