"""VLM-led, sensor-vetoed tri-state insertion outcome contract.

This read-only resolver does not measure depth, alignment, or force itself.
It combines time-aligned, calibrated evidence without letting a wrist command
or VLM-only occlusion inference become a physical 20 mm success label.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


VLM_CLASSES = frozenset({
    # New observers emit normal_appearance: 20 mm is measured independently.
    # Keep normal_20mm as an older recorded-assessment alias.
    "normal_appearance", "normal_20mm", "partial", "rim_jam", "slip",
    "unobservable",
})
KEY_DEPTH_SOURCES = frozenset({
    "key_pose_multiview", "exposed_length_cad", "crosschecked_wrist_key",
})


@dataclass(frozen=True)
class InsertionEvidence:
    vlm_class: str
    # Conservative lower/upper bounds on KEY penetration below socket rim.
    key_depth_interval_m: tuple[float, float] | None
    key_depth_source: str | None
    alignment_within_limits: bool | None
    safety_abort: bool | None
    grasp_held: bool | None


@dataclass(frozen=True)
class InsertionOutcome:
    insertion_success: bool | None
    reason: str
    vlm_assessment: str
    depth_lower_m: float | None
    depth_upper_m: float | None

    def to_record(self) -> dict:
        return {
            "insertion_success": self.insertion_success,
            "reason": self.reason,
            "vlm_insertion_assessment": self.vlm_assessment,
            "key_depth_interval_m": (
                [self.depth_lower_m, self.depth_upper_m]
                if self.depth_lower_m is not None else None
            ),
            "scope": "vlm_led_sensor_vetoed_task_label_not_contact_control",
        }


def judge_insertion(
    evidence: InsertionEvidence, *, target_depth_m: float = 0.020,
) -> InsertionOutcome:
    """Return true/false/unknown with VLM and independent key-depth evidence.

    A visually normal VLM verdict is required for true. A robust lower depth
    bound must reach the target; the bound's source cannot be a commanded or
    observed wrist stroke without an independent key/grasp cross-check.
    """
    if evidence.vlm_class not in VLM_CLASSES:
        raise ValueError("unknown VLM insertion class")
    if not math.isfinite(target_depth_m) or target_depth_m <= 0:
        raise ValueError("target_depth_m must be positive")
    if evidence.key_depth_source not in KEY_DEPTH_SOURCES | {None}:
        raise ValueError("unrecognized key-depth source")
    depth = evidence.key_depth_interval_m
    if depth is not None:
        if len(depth) != 2:
            raise ValueError("key depth interval must contain lower and upper")
        lower, upper = (float(depth[0]), float(depth[1]))
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise ValueError("invalid key depth interval")
    else:
        lower = upper = None
    if depth is not None and evidence.key_depth_source is None:
        raise ValueError("key depth interval requires a validated source")

    def outcome(value: bool | None, reason: str) -> InsertionOutcome:
        return InsertionOutcome(value, reason, evidence.vlm_class, lower, upper)

    if evidence.safety_abort is True:
        return outcome(False, "safety_abort")
    if evidence.grasp_held is False:
        return outcome(False, "grasp_lost")
    if evidence.alignment_within_limits is False:
        return outcome(False, "alignment_outside_commissioned_limits")
    if upper is not None and upper < target_depth_m:
        return outcome(False, "key_depth_definitely_short")
    if evidence.vlm_class in {"partial", "rim_jam", "slip"}:
        if lower is not None and lower >= target_depth_m:
            return outcome(None, "vlm_and_key_depth_conflict")
        if evidence.vlm_class == "slip" and evidence.grasp_held is True:
            return outcome(None, "vlm_and_grasp_state_conflict")
        return outcome(False, "vlm_observed_failure")
    if evidence.vlm_class == "unobservable":
        return outcome(None, "vlm_abstained")
    if (evidence.grasp_held is not True or
            evidence.alignment_within_limits is not True or
            evidence.safety_abort is not False or
            lower is None or lower < target_depth_m):
        return outcome(None, "success_evidence_incomplete")
    return outcome(True, "vlm_normal_and_verified_key_depth")


def resolve_insertion_outcome(
    evidence: InsertionEvidence, *, target_depth_m: float,
    label_mode: str = "evidence_fused", manual_annotation: dict | None = None,
) -> dict:
    """Return a task label while retaining independent metric diagnostics.

    In ``vlm_only`` and ``manual`` modes the result is an experimental label,
    not a claim that measured key penetration reached 20 mm. A controller
    abort still stops motion through its own execution path.
    """
    from .labeling import resolve_label

    fused = judge_insertion(evidence, target_depth_m=target_depth_m)
    value, reason = resolve_label(
        stage="insertion_success", mode=label_mode,
        visual_class=evidence.vlm_class,
        fused_value=fused.insertion_success, fused_reason=fused.reason,
        annotation=manual_annotation)
    record = fused.to_record()
    record["insertion_success"] = value
    record["reason"] = reason
    if label_mode != "evidence_fused":
        record["scope"] = ("visual_task_label_not_verified_20mm_depth"
                           if label_mode == "vlm_only" else
                           "human_task_label_not_verified_20mm_depth")
    return record
