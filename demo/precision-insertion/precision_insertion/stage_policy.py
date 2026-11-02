"""Versioned execution-policy values for one precision-insertion rig.

The live handlers own validation and behavior.  This module only separates
commissioned/provisional rig values from their wiring so changing a study
parameter does not require editing action code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping

from .grounded_alignment import AlignmentLimits
from .lift_execution import LiftExecutionLimits
from .path_audit import PathAuditLimits
from .retry_session import RetrySessionLimits


SCHEMA = "precision_insertion_zerodex_stage_policy_v1"
DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1] /
    "config/zerodex_capture4_stage_policy.json")

_TOP_LEVEL = frozenset({
    "schema", "key_prompt", "tabletop_pose_error_deg",
    "post_lift_settle_s", "axial_waypoint_step_m", "path_limits",
    "motion_limits", "key_capture", "stationary", "reset_geometry",
    "landing", "return_parameters", "retry_limits", "alignment_limits",
    "pickup_limits", "lift_assessment", "preinsert", "final", "repose",
})

_SECTION_KEYS = {
    "key_capture": frozenset({
        "maximum_multiview_center_error_mm",
        "maximum_multiview_angle_error_deg",
        "maximum_socket_mask_overlap_fraction",
        "socket_projection_dilation_px", "minimum_refinement_iou",
        "max_key_state_skew_s", "capture_timeout_s",
        "image_write_timeout_s",
    }),
    "stationary": frozenset({
        "max_arm_hand_skew_s", "max_hand_command_error_raw",
        "max_arm_velocity_rad_s",
    }),
    "reset_geometry": frozenset({
        "minimum_rest_socket_clearance_m",
        "minimum_board_edge_clearance_m",
    }),
    "landing": frozenset({
        "max_return_center_shift_m", "support_tolerance_m",
    }),
    "return_parameters": frozenset({
        "release_height_m", "minimum_release_key_clearance_m",
        "max_withdrawal_state_gap_s",
    }),
    "retry_limits": frozenset({
        "max_state_skew_s", "max_grasp_translation_drift_m",
        "max_grasp_rotation_drift_deg", "minimum_anchor_separation_px",
        "crop_width_px", "max_frame_age_s", "max_capture_skew_s",
        "max_total_offset_m",
    }),
    "pickup_limits": frozenset({
        "max_start_age_s", "max_start_joint_error_rad",
    }),
    "lift_assessment": frozenset({
        "max_phase_skew_s", "minimum_visual_views",
    }),
    "preinsert": frozenset({
        "max_capture_skew_s", "max_joint_frame_skew_s",
        "render_audit_overlay",
    }),
    "final": frozenset({"max_phase_skew_s", "minimum_visual_views"}),
    "repose": frozenset({"max_state_skew_s"}),
}


@dataclass(frozen=True)
class ZeroDexStagePolicy:
    source: Path
    key_prompt: str
    tabletop_pose_error_deg: float
    post_lift_settle_s: float
    axial_waypoint_step_m: float
    path_limits: PathAuditLimits
    motion_limits: LiftExecutionLimits
    key_capture: Mapping[str, object]
    stationary: Mapping[str, float]
    reset_geometry: Mapping[str, float]
    landing: Mapping[str, float]
    return_parameters: Mapping[str, float]
    retry_limits: RetrySessionLimits
    alignment_limits: AlignmentLimits
    pickup_limits: Mapping[str, float]
    lift_assessment: Mapping[str, object]
    preinsert: Mapping[str, object]
    final: Mapping[str, object]
    repose: Mapping[str, float]


def _section(record: Mapping, name: str) -> dict:
    value = record.get(name)
    if not isinstance(value, dict) or set(value) != _SECTION_KEYS[name]:
        raise ValueError(f"stage policy has invalid {name} fields")
    return dict(value)


def _finite(name: str, value, *, allow_zero: bool = False) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or
            not math.isfinite(float(value)) or
            (float(value) < 0 if allow_zero else float(value) <= 0)):
        raise ValueError(f"stage policy {name} must be finite")
    return float(value)


def load_zerodex_stage_policy(path: str | Path | None = None) -> ZeroDexStagePolicy:
    """Load one exact stage-policy document without opening hardware."""
    selected = path
    if selected is None:
        selected = os.environ.get("PRECISION_STAGE_POLICY", "").strip()
    source = Path(selected or DEFAULT_POLICY_PATH).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or set(record) != _TOP_LEVEL or
            record.get("schema") != SCHEMA):
        raise ValueError("invalid ZeroDex stage-policy document")
    prompt = record["key_prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("stage policy key prompt must be nonempty")
    tabletop = _finite("tabletop_pose_error_deg",
                        record["tabletop_pose_error_deg"])
    settle = _finite("post_lift_settle_s", record["post_lift_settle_s"],
                     allow_zero=True)
    axial = _finite("axial_waypoint_step_m",
                    record["axial_waypoint_step_m"])

    path_limits = PathAuditLimits(**record["path_limits"])
    path_limits.validate()
    motion_limits = LiftExecutionLimits(**record["motion_limits"])
    motion_limits.validate()
    stationary = _section(record, "stationary")
    retry_limits = RetrySessionLimits(
        **_section(record, "retry_limits"), **stationary,
        axial_waypoint_step_m=axial, path_limits=path_limits)
    retry_limits.validate()
    alignment_limits = AlignmentLimits(**record["alignment_limits"])
    alignment_limits.validate()

    return ZeroDexStagePolicy(
        source=source, key_prompt=prompt.strip(),
        tabletop_pose_error_deg=tabletop,
        post_lift_settle_s=settle, axial_waypoint_step_m=axial,
        path_limits=path_limits, motion_limits=motion_limits,
        key_capture=_section(record, "key_capture"),
        stationary=stationary,
        reset_geometry=_section(record, "reset_geometry"),
        landing=_section(record, "landing"),
        return_parameters=_section(record, "return_parameters"),
        retry_limits=retry_limits, alignment_limits=alignment_limits,
        pickup_limits=_section(record, "pickup_limits"),
        lift_assessment=_section(record, "lift_assessment"),
        preinsert=_section(record, "preinsert"),
        final=_section(record, "final"), repose=_section(record, "repose"))
