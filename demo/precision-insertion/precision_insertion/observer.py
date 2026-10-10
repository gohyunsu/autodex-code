"""Read-only, event-driven VLM observations using ZeroDex backend contracts.

This module never commands a robot. It makes image order explicit, validates
closed-set responses, and fails closed on malformed output. A visual verdict
does not measure millimetres or authorize a retry or insertion contact.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
import time
from typing import Protocol, Sequence

from PIL import Image

from .xy_voting import ViewVote, XYChoice, validate_choices


class ImageVLM(Protocol):
    def infer(self, images: list[Image.Image], prompt: str) -> str: ...


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

    def to_record(self) -> dict:
        return {
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
    """Reuse ZeroDex's Gemini multi-image request; client/key are injected."""

    def __init__(self, client, model: str, *, thinking_level: str = "LOW"):
        self.client = client
        self.model = model
        self.thinking_level = thinking_level

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        try:
            from mv_grounding_voting.mv_grounding_depth_voting import infer_gemini_multi
        except ImportError as exc:
            raise RuntimeError(
                "Install the ZeroDex realtime_vlm package on PYTHONPATH "
                "to use its Gemini inference helper") from exc
        answer, _elapsed = infer_gemini_multi(
            self.client, self.model, images, prompt,
            thinking_level=self.thinking_level)
        return answer


class ZeroDexLocalBackend:
    """Reuse an already-loaded ZeroDex BaseVLM (Qwen/Gemma)."""

    def __init__(self, vlm, *, max_new_tokens: int = 512):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    def infer(self, images: list[Image.Image], prompt: str) -> str:
        result = self.vlm.infer_images_prompt(
            images, prompt, max_new_tokens=self.max_new_tokens)
        return result.answer


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
    value = json.loads(answer)
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
        + "\n\n" + prompt_body
    )
    started = time.perf_counter()
    answer = backend.infer([frame.image for frame in frames], prompt)
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
        _backend_model(backend), latency)


def observe_lift(backend: ImageVLM, frames: Sequence[LabeledFrame]) -> VLMObservation:
    """Classify whether the key follows the held hand across lift frames."""
    phases = {frame.phase for frame in frames}
    if not {"before_grasp", "after_lift"} <= phases:
        raise ValueError("lift observation needs before_grasp and after_lift")
    _require_temporal_camera_pairs(
        frames, before="before_grasp", after="after_lift")
    return _infer_closed_set(
        backend, stage="post_lift", frames=frames,
        prompt_body=(
            "Track the loose key relative to its tabletop location and the "
            "robot hand. Classify held only when distinct key pixels visibly "
            "move with the hand in the cited camera views; closed fingers, "
            "a hidden key, or a commanded lift are not held evidence. "
            "Classify only visible evidence. Hidden is not held; classify it "
            "as unobservable. "
            "Return JSON only: "
            '{"class":"held|miss|slip|unobservable","evidence_views":'
            '["camera_id"],"evidence":"..."}. '
            "Do not infer a grasp from the commanded hand closure."
        ),
        field="class", allowed=frozenset({"held", "miss", "slip", "unobservable"}),
        fallback="unobservable",
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
    return _infer_closed_set(
        backend, stage="insertion_visual", frames=frames,
        prompt_body=(
            "Compare the time-ordered RAW key and socket pixels. CAD overlays "
            "are predictions, not measurements. A hidden key is not proof "
            "of insertion. Do not claim numerical depth from pixels. "
            "Return JSON only: "
            '{"visual_class":"normal_appearance|partial|rim_jam|slip|'
            'unobservable","evidence_views":["camera_id"],"evidence":"..."}.'
        ),
        field="visual_class",
        allowed=frozenset({
            "normal_appearance", "partial", "rim_jam", "slip", "unobservable",
        }),
        fallback="unobservable",
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
                'Return JSON only: {"class":"coarse_match|gross_misalignment|'
                'slip_or_miss|unobservable","evidence_views":'
                '["camera_id"],"evidence":"..."}. '
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
    votes = []
    records = []
    for view in views:
        frames = [
            LabeledFrame(view.camera_id, "raw_hold", view.timestamp_s, view.raw),
            LabeledFrame(view.camera_id, "candidate_overlay", view.timestamp_s,
                         view.overlay),
        ]
        order = _frame_order(frames)
        ids = ", ".join(catalog)
        prompt = (
            "Image 1 is the RAW camera crop; image 2 marks calibrated "
            "candidate center anchors on the same crop. Markers are target "
            "projections, not observed key poses or measured millimetres. "
            "Choose the ID that reduces visible key/socket centerline "
            "misalignment, or abstain if "
            "occluded, indistinguishable, tilted, or uncertain. Never invent "
            "coordinates or a new ID. "
            f"Allowed IDs: [{ids}]. Camera: {view.camera_id}. "
            "Return JSON only: "
            '{"visible":true|false,"choice_id":"ID|abstain",'
            '"failure_class":"misaligned|rim_jam|roughly_aligned|slip|'
            'occluded|unknown","evidence":"..."}.'
        )
        started = time.perf_counter()
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
        votes.append(ViewVote(
            view.camera_id, float(view.timestamp_s),
            parsed["visible"], parsed["choice_id"], parsed["failure_class"],
        ))
        records.append(VLMObservation(
            "preinsert_xy", parsed, answer, prompt, order, error,
            _backend_model(backend), latency))
    return votes, records
