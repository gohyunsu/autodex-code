"""Read-only, checkpoint-scoped visual observer for insertion trials.

The VLM classifies evidence; it never supplies metric offsets, depth, force,
or motor commands.  Those fields must come from calibrated sensing.
"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any, Mapping


CHECKPOINTS = ("post_lift", "pre_insert", "insertion_abort", "insertion_hold", "finish")
CLASSES = ("aligned", "misaligned", "rim_jam", "partial_insertion",
           "seated", "slip", "occluded", "unknown")


def prompt_for(checkpoint: str, measurements: Mapping[str, Any]) -> str:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint: {checkpoint}")
    return (
        "You are a read-only visual observer for a Franka/Inspire key insertion trial. "
        "Use the labeled synchronized camera views and the provided sensor context. "
        "Do not infer millimeters, force, or motor commands from images. "
        "Do not claim insertion success from occlusion or commanded motion. "
        "If the key or rim cannot be seen clearly, answer unknown or occluded. "
        "Return one JSON object only with keys: checkpoint, class, confidence "
        "(0..1), visible_evidence (short string), cameras_used (array of labels). "
        f"Checkpoint: {checkpoint}. Sensor context: "
        f"{json.dumps(dict(measurements), ensure_ascii=False, sort_keys=True)}"
    )


def parse_answer(raw: str, checkpoint: str, available_views: set[str]) -> dict[str, Any]:
    if checkpoint not in CHECKPOINTS:
        raise ValueError(f"unknown checkpoint: {checkpoint}")
    try:
        payload = json.loads(raw.strip())
        label = payload["class"]
        confidence = float(payload["confidence"])
        views = payload["cameras_used"]
        if not isinstance(payload, dict) or payload.get("checkpoint") != checkpoint:
            raise ValueError("checkpoint mismatch")
        if label not in CLASSES or not 0 <= confidence <= 1:
            raise ValueError("invalid class/confidence")
        if not isinstance(views, list) or not views or not set(views) <= available_views:
            raise ValueError("invalid camera references")
        if not isinstance(payload["visible_evidence"], str):
            raise ValueError("evidence must be a string")
        return {"checkpoint": checkpoint, "class": label,
                "confidence": confidence,
                "visible_evidence": payload["visible_evidence"][:500],
                "cameras_used": views}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"checkpoint": checkpoint, "class": "unknown", "confidence": 0.0,
                "visible_evidence": "unparseable_or_unverifiable_vlm_response",
                "cameras_used": []}


def observe_with_gemini(*, checkpoint: str, image_paths: Mapping[str, Path],
                        measurements: Mapping[str, Any], client: Any,
                        model: str) -> dict[str, Any]:
    """Send saved AutoDex checkpoint frames to Gemini on explicit invocation.

    This deliberately imports the SDK lazily so offline replay/tests run with
    no network access or API credentials.  The caller owns the image capture.
    """
    from google.genai import types  # type: ignore

    if not image_paths:
        raise ValueError("at least one saved camera image is required")
    parts = []
    for label, path in sorted(image_paths.items()):
        path = Path(path)
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in ("image/png", "image/jpeg"):
            raise ValueError(f"expected PNG/JPEG checkpoint frame: {path}")
        parts.append(types.Part.from_text(text=f"[Camera {label}]"))
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
    parts.append(types.Part.from_text(text=prompt_for(checkpoint, measurements)))
    response = client.models.generate_content(
        model=model, contents=[types.Content(role="user", parts=parts)],
        config=types.GenerateContentConfig(response_mime_type="application/json"))
    return parse_answer(response.text or "", checkpoint, set(image_paths))
