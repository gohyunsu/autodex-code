"""Append-only precision-insertion attempt labels and evidence references.

This records *observed decisions*, not robot commands. Unknown is retained
as ``None`` and a later stage is never silently marked false when an earlier
stage fails. The insertion verdict must pass through ``judge_insertion``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from .config import TaskMode
from .outcome import InsertionEvidence, judge_insertion
from .xy_voting import ChoiceDecision, VLM_XY_STEP_M


STAGES = (
    "grasp_success", "preinsert_reached", "insertion_success",
    "release_success", "retreat_success", "reset_success",
    "reorient_success",
)
PREREQUISITE = {
    "preinsert_reached": "grasp_success",
    "insertion_success": "preinsert_reached",
    "release_success": "insertion_success",
    "retreat_success": "release_success",
}
TRUE_EVIDENCE = {
    "grasp_success": frozenset({"vlm_observation", "key_wrist_check"}),
    "preinsert_reached": frozenset({
        "trajectory", "key_socket_pose", "grasp_state"}),
    "insertion_success": frozenset({
        "vlm_observation", "key_depth", "alignment", "force_trace"}),
    "release_success": frozenset({"release_observation", "key_pose"}),
    "retreat_success": frozenset({"trajectory", "hand_pose"}),
    "reset_success": frozenset({"key_pose"}),
    "reorient_success": frozenset({"key_pose"}),
}
FAILURE_CODES = frozenset({
    "perception_unreliable", "grasp_miss", "slip",
    "transfer_unreachable", "transfer_collision", "preinsert_misaligned",
    "rim_jam", "depth_shortfall", "force_abort", "reset_failed",
})


def _finite_time(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("event timestamp must be finite")
    return result


def _refs(value: Mapping[str, str], stage: str, status: bool | None) -> dict:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("each stage decision needs evidence references")
    refs = dict(value)
    if not all(isinstance(key, str) and key and
               isinstance(ref, str) and ref for key, ref in refs.items()):
        raise ValueError("evidence references need nonempty string keys and paths")
    if status is True and not TRUE_EVIDENCE[stage] <= refs.keys():
        missing = sorted(TRUE_EVIDENCE[stage] - refs.keys())
        raise ValueError(f"true {stage} missing evidence references: {missing}")
    return refs


@dataclass
class AttemptRecord:
    attempt_id: str
    mode: TaskMode
    session_calibration_sha256: str
    candidate_id: str | None
    tabletop_pose_stem: str | None
    xy_offset_socket_m: tuple[float, float]
    started_at_s: float
    labels: dict[str, bool | None] = field(
        default_factory=lambda: {name: None for name in STAGES})
    events: list[dict] = field(default_factory=list)
    failure_code: str | None = None
    _pending_retry: bool = False

    def _append_stage(
        self, stage: str, status: bool | None, *, timestamp_s: float,
        evidence_refs: Mapping[str, str], detail: dict | None = None,
    ) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown attempt stage: {stage}")
        if status is not None and type(status) is not bool:
            raise ValueError("stage status must be true, false, or unknown")
        already_recorded = any(event["stage"] == stage for event in self.events)
        if already_recorded and not (
                stage == "insertion_success" and self._pending_retry and
                self.labels[stage] is False):
            raise ValueError(f"stage already recorded: {stage}")
        required = PREREQUISITE.get(stage)
        if required is not None and self.labels[required] is not True:
            raise ValueError(f"{stage} requires observed {required}=true")
        if stage == "reset_success" and not any(
                event["stage"] in STAGES[:3] for event in self.events):
            raise ValueError("reset outcome needs a preceding physical attempt")
        refs = _refs(evidence_refs, stage, status)
        timestamp = _finite_time(timestamp_s)
        if timestamp < self.started_at_s or (
                self.events and timestamp < self.events[-1]["timestamp_s"]):
            raise ValueError("attempt events must be in nondecreasing time order")
        self.labels[stage] = status
        self.events.append({
            "event_index": len(self.events),
            "timestamp_s": timestamp,
            "stage": stage,
            "value": status,
            "evidence_refs": refs,
            "detail": detail or {},
        })
        if stage == "insertion_success":
            self._pending_retry = False

    def record_stage(
        self, stage: str, status: bool | None, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> None:
        if stage == "insertion_success":
            raise ValueError("use record_insertion_evidence for insertion")
        self._append_stage(stage, status, timestamp_s=timestamp_s,
                           evidence_refs=evidence_refs)

    def record_insertion_evidence(
        self, evidence: InsertionEvidence, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> dict:
        outcome = judge_insertion(evidence, target_depth_m=self.mode.target_depth_m)
        self._append_stage(
            "insertion_success", outcome.insertion_success,
            timestamp_s=timestamp_s, evidence_refs=evidence_refs,
            detail={"input": asdict(evidence), "outcome": outcome.to_record()},
        )
        return outcome.to_record()

    def record_retry(
        self, decision: ChoiceDecision, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> None:
        """Log a separately preflighted 1 mm retry after an insertion failure.

        This is not a robot-motion method. ``axial_withdrawal`` and
        ``live_preflight`` must refer to evidence from completed safe checks;
        a VLM vote alone is insufficient to register a retry.
        """
        if not any(event["stage"] == "insertion_success" for event in self.events):
            raise ValueError("retry needs a preceding insertion observation")
        if self.labels["insertion_success"] is not False or self._pending_retry:
            raise ValueError("retry requires an observed failure and no pending retry")
        last_insertion = next(
            event for event in reversed(self.events)
            if event["stage"] == "insertion_success")
        insertion_input = last_insertion["detail"]["input"]
        if (insertion_input["grasp_held"] is not True or
                insertion_input["safety_abort"] is not False or
                self.failure_code in {"force_abort", "slip", "reset_failed"}):
            raise ValueError("retry requires held grasp without safety abort")
        if (not isinstance(decision, ChoiceDecision) or
                decision.status != "propose" or
                decision.offset_socket_m is None or
                len(set(decision.supporting_cameras)) < 2 or
                not all(isinstance(camera, str) and camera
                        for camera in decision.supporting_cameras)):
            raise ValueError("retry needs a two-view proposed XY choice")
        target = decision.offset_socket_m
        if len(target) != 2 or not all(math.isfinite(float(v)) for v in target):
            raise ValueError("retry target needs two finite socket-frame offsets")
        dx = target[0] - self.xy_offset_socket_m[0]
        dy = target[1] - self.xy_offset_socket_m[1]
        if (abs(math.hypot(dx, dy) - VLM_XY_STEP_M) > 1e-12 or
                min(abs(dx), abs(dy)) > 1e-12):
            raise ValueError("retry must be exactly one 1 mm cardinal step")
        expected_id = (
            "x_plus_1mm" if dx > 0 and abs(dy) <= 1e-12 else
            "x_minus_1mm" if dx < 0 and abs(dy) <= 1e-12 else
            "y_plus_1mm" if dy > 0 and abs(dx) <= 1e-12 else
            "y_minus_1mm"
        )
        if decision.choice_id != expected_id:
            raise ValueError("retry ID does not describe the 1 mm socket-frame move")
        refs = _refs(evidence_refs, "preinsert_reached", False)
        required = {"axial_withdrawal", "live_preflight", "xy_vlm_vote"}
        if not required <= refs.keys():
            raise ValueError("retry needs withdrawal, preflight, and VLM refs")
        timestamp = _finite_time(timestamp_s)
        if timestamp < self.events[-1]["timestamp_s"]:
            raise ValueError("attempt events must be in nondecreasing time order")
        self.xy_offset_socket_m = tuple(target)
        self._pending_retry = True
        self.events.append({
            "event_index": len(self.events), "timestamp_s": timestamp,
            "stage": "xy_retry", "value": decision.choice_id,
            "offset_socket_m": list(target),
            "choice_decision": decision.to_record(),
            "evidence_refs": refs,
            "scope": "observed_safe_preflight_not_motion_authorization",
        })

    def record_failure(
        self, code: str, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> None:
        if code not in FAILURE_CODES:
            raise ValueError(f"unknown failure code: {code}")
        if self.failure_code is not None:
            raise ValueError("attempt failure code is already set")
        refs = _refs(evidence_refs, "grasp_success", False)
        timestamp = _finite_time(timestamp_s)
        if timestamp < self.started_at_s or (
                self.events and timestamp < self.events[-1]["timestamp_s"]):
            raise ValueError("attempt events must be in nondecreasing time order")
        self.failure_code = code
        self.events.append({
            "event_index": len(self.events), "timestamp_s": timestamp,
            "stage": "failure_code", "value": code,
            "evidence_refs": refs,
        })

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_attempt_v1",
            "attempt_id": self.attempt_id,
            "mode": self.mode.family,
            "gap_mm": self.mode.gap_mm,
            "key_object": self.mode.key_object,
            "socket_object": self.mode.socket_object,
            "session_calibration_sha256": self.session_calibration_sha256,
            "candidate_id": self.candidate_id,
            "tabletop_pose_stem": self.tabletop_pose_stem,
            "xy_offset_socket_m": list(self.xy_offset_socket_m),
            **self.labels,
            "failure_code": self.failure_code,
            "events": list(self.events),
            "scope": "evidence_record_not_motion_authorization",
        }

    def write_new(self, path: Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(self.to_record(), stream, indent=2)
            stream.write("\n")
        return target


def begin_attempt(
    *, attempt_id: str, mode: TaskMode, session_record: Mapping,
    candidate_id: str | None, tabletop_pose_stem: str | None,
    xy_offset_socket_m: tuple[float, float], started_at_s: float,
) -> AttemptRecord:
    """Bind a new record to a frozen session without claiming readiness."""
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("attempt_id must be a nonempty string")
    if (not isinstance(session_record, Mapping) or
            session_record.get("schema") !=
            "precision_insertion_session_calibration_v1"):
        raise ValueError("attempt needs a session calibration record")
    if session_record.get("mode") != {
            "family": mode.family, "gap_mm": mode.gap_mm,
            "key_object": mode.key_object,
            "socket_object": mode.socket_object}:
        raise ValueError("attempt mode does not match frozen socket session")
    if candidate_id is not None and (
            not isinstance(candidate_id, str) or
            len(candidate_id.split("/")) != 3 or
            any(not part for part in candidate_id.split("/"))):
        raise ValueError("candidate_id must be type/scene/grasp or null")
    if tabletop_pose_stem is not None and (
            not isinstance(tabletop_pose_stem, str) or
            not tabletop_pose_stem.isdigit()):
        raise ValueError("tabletop_pose_stem must be numeric or null")
    if candidate_id is not None and tabletop_pose_stem is None:
        raise ValueError("a grasp attempt needs a tabletop pose stem")
    if len(xy_offset_socket_m) != 2 or not all(
            math.isfinite(float(v)) for v in xy_offset_socket_m):
        raise ValueError("XY offset must be two finite socket-frame values")
    canonical_session = json.dumps(
        dict(session_record), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return AttemptRecord(
        attempt_id=attempt_id, mode=mode,
        session_calibration_sha256=hashlib.sha256(canonical_session).hexdigest(),
        candidate_id=candidate_id, tabletop_pose_stem=tabletop_pose_stem,
        xy_offset_socket_m=tuple(float(v) for v in xy_offset_socket_m),
        started_at_s=_finite_time(started_at_s),
    )
