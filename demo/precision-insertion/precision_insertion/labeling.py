"""Replayable task-label policy; motion and geometric preflight stay separate.

``vlm_only`` is an experimental *visual label*, not proof of 20 mm physical
penetration. ``manual`` records an explicit human annotation of the same saved
camera checkpoint. The older metric-fused resolver remains available for
comparison and for callers that do not opt into either new mode.
"""

from __future__ import annotations

from typing import Mapping
from pathlib import Path
from typing import Callable


LABEL_MODES = frozenset({"evidence_fused", "vlm_only", "manual"})
STAGES = frozenset({"grasp_success", "preinsert_reached", "insertion_success"})


def validate_annotation(stage: str, annotation: Mapping | None) -> dict | None:
    if annotation is None:
        return None
    if (stage not in STAGES or not isinstance(annotation, Mapping) or
            set(annotation) != {"stage", "value", "annotator", "note"} or
            annotation["stage"] != stage or
            annotation["value"] is not None and
            type(annotation["value"]) is not bool or
            not isinstance(annotation["annotator"], str) or
            not annotation["annotator"].strip() or
            not isinstance(annotation["note"], str)):
        raise ValueError("manual label needs stage, tri-state value, annotator and note")
    return dict(annotation)


def resolve_label(
    *, stage: str, mode: str, visual_class: str,
    fused_value: bool | None, fused_reason: str,
    annotation: Mapping | None = None,
) -> tuple[bool | None, str]:
    """Resolve one checkpoint from its saved VLM class or human annotation.

    Unknown/occluded is always an abstention in visual-only mode. The class
    must already have passed the checkpoint's multiview consensus check.
    """
    if stage not in STAGES or mode not in LABEL_MODES:
        raise ValueError("unknown task-label stage or mode")
    manual = validate_annotation(stage, annotation)
    if (mode == "manual") != (manual is not None):
        raise ValueError("manual mode requires exactly one human annotation")
    if mode == "manual":
        return manual["value"], "manual_" + (
            "positive" if manual["value"] is True else
            "negative" if manual["value"] is False else "unknown")
    if mode == "evidence_fused":
        return fused_value, fused_reason
    classes = {
        "grasp_success": ({"held"}, {"miss", "slip"}),
        "preinsert_reached": ({"coarse_match"},
                              {"gross_misalignment", "slip_or_miss"}),
        "insertion_success": ({"normal_appearance", "normal_20mm"},
                              {"partial", "rim_jam", "slip"}),
    }
    positive, negative = classes[stage]
    if visual_class in positive:
        return True, "vlm_visual_positive"
    if visual_class in negative:
        return False, "vlm_visual_negative"
    return None, "vlm_visual_abstained"


def independent_vlm_label(stage: str, visual_class: str) -> dict:
    """Keep the VLM's own verdict even when a human overrides the stage."""
    value, reason = resolve_label(
        stage=stage, mode="vlm_only", visual_class=visual_class,
        fused_value=None, fused_reason="not_used")
    return {"value": value, "reason": reason, "class": visual_class}


def manual_annotation(
    stage: str, value: bool | None, annotator: str, note: str = "",
) -> dict:
    return validate_annotation(stage, {
        "stage": stage, "value": value,
        "annotator": annotator, "note": note,
    })


def request_annotation(
    *, mode: str, provider: Callable | None, stage: str,
    evidence_dirs: tuple[Path, ...], attempt_id: str, candidate_id: str,
) -> dict | None:
    if mode not in LABEL_MODES:
        raise ValueError("unknown label mode")
    if mode != "manual":
        return None
    if not callable(provider):
        raise ValueError("manual mode needs a human label provider")
    return validate_annotation(stage, provider(
        stage, evidence_dirs, attempt_id, candidate_id))


def terminal_label_provider(annotator: str) -> Callable:
    """A minimal blocking reviewer for saved images in a development session."""
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError("manual labeling needs --annotator")

    def review(stage, evidence_dirs, attempt_id, candidate_id):
        print(f"\nManual {stage}: {attempt_id} / {candidate_id}", flush=True)
        for directory in evidence_dirs:
            print(f"Inspect saved PNGs in {directory}", flush=True)
        while True:
            answer = input("Label [y=yes, n=no, u=unknown]: ").strip().lower()
            if answer in {"y", "n", "u"}:
                break
            print("Enter y, n, or u.", flush=True)
        note = input("Note (optional): ").strip()
        return manual_annotation(
            stage, {"y": True, "n": False, "u": None}[answer],
            annotator, note)

    return review
