"""Evidence-only session supervisor for the independent insertion demo.

The supervisor connects the existing fresh-key v8 planner, observed attempt
labels, and next-step policy. It deliberately has no camera or motor adapter:
none of its decisions authorizes robot motion. Every attempt state is saved as
an immutable snapshot so a failed process cannot silently erase an observation.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

from .calibration import SessionCalibration
from .candidates import validate_catalog_session
from .config import TaskMode
from .key_perception import (
    KeyPoseObservation, verify_key_capture_artifacts,
)
from .records import AttemptRecord, begin_attempt
from .retry_preflight import XYRetryPreflight
from .session_policy import (
    SessionDecision, decide_after_attempt, decide_after_trial_preflight,
)
from .trial_preflight import (
    TrialPreflight, plan_admitted_key_trial, write_trial_preflight_artifacts,
)
from .xy_retry import XYRetryAssessment


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _digest(value: Mapping) -> str:
    return hashlib.sha256(json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


class SessionRunner:
    """Plan and record one frozen-socket session without executing motion.

    A physical caller must separately provide commissioned camera acquisition,
    robot/force control, post-lift measurement, and observed outcomes. In
    particular, a planning pass never creates a positive attempt label.
    """

    def __init__(
        self, *, mode: TaskMode, calibration: SessionCalibration,
        catalog: dict, shared_root: Path, output_dir: Path,
        max_xy_retries: int,
    ) -> None:
        if type(max_xy_retries) is not int or max_xy_retries < 0:
            raise ValueError("max_xy_retries must be a nonnegative integer")
        if not isinstance(calibration, SessionCalibration):
            raise TypeError("session needs a frozen SessionCalibration")
        validate_catalog_session(catalog, mode=mode,
                                 session_record=calibration.record)
        root = Path(shared_root).expanduser().resolve()
        if Path(catalog["shared_root"]).expanduser().resolve() != root:
            raise ValueError("catalogue and session shared roots differ")
        if not catalog.get("complete_scan"):
            raise ValueError("session needs a complete offline endpoint catalogue")
        target = Path(output_dir).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        self.mode = mode
        self.calibration = calibration
        self.catalog = catalog
        self.shared_root = root
        self.output_dir = target
        self.max_xy_retries = max_xy_retries
        self.session_sha256 = _digest(calibration.record)
        self.catalog_sha256 = _digest(catalog)
        self._preflight: TrialPreflight | None = None
        self._preflight_report_path: Path | None = None
        self._attempt: AttemptRecord | None = None
        self._attempted: set[tuple[str, str, str]] = set()
        self._same_capture_planning_rejects: set[tuple[str, str, str]] = set()
        self._capture_id: str | None = None
        self._capture_timestamp_s = -math.inf
        self._capture_interval_end_s = -math.inf
        self._capture_observation_sha256: str | None = None
        self._used_attempt_ids: set[str] = set()
        self._preflight_index = 0
        self._attempt_index = 0
        self._attempt_dir: Path | None = None
        self._write_exclusive(
            target / "frozen_session_calibration.json", calibration.record)
        self._write_exclusive(target / "endpoint_catalog.json", catalog)
        self._write_exclusive(target / "session_run.json", {
            "schema": "precision_insertion_session_runner_v1",
            "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                     "key_object": mode.key_object,
                     "socket_object": mode.socket_object},
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "max_xy_retries": max_xy_retries,
            "scope": "evidence_and_preflight_only_not_robot_motion_authorization",
            "robot_ready": False,
        })

    @staticmethod
    def _write_exclusive(path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")

    @property
    def attempted_candidates(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(self._attempted))

    @property
    def active_attempt(self) -> AttemptRecord | None:
        """Return a copy; observations must enter through the snapshot methods."""
        return deepcopy(self._attempt)

    def current_decision(self) -> SessionDecision:
        if self._attempt is not None:
            return decide_after_attempt(
                self._attempt, max_xy_retries=self.max_xy_retries)
        if self._preflight is not None:
            return decide_after_trial_preflight(self._preflight.to_record())
        return SessionDecision("capture_fresh_key", "frozen_socket_session_ready",
                               None, ("fresh_multiview_key_pose",
                                      "measured_franka_inspire_state"))

    def _can_plan_key(self, observation: KeyPoseObservation) -> None:
        if self._attempt is not None:
            action = self.current_decision().action
            if action != "reobserve_key_and_preflight":
                raise ValueError(
                    f"previous attempt has no verified return to key observation: {action}")
        if not isinstance(observation, KeyPoseObservation):
            raise TypeError("next trial needs an admitted multi-view key pose")
        timestamp = float(observation.selected_acquisition_timestamp_s)
        if not math.isfinite(timestamp):
            raise ValueError("key exposure timestamp must be finite")
        same_capture = observation.capture_id == self._capture_id
        if same_capture:
            if (self._preflight is None or self._attempt is not None or
                    self.current_decision().action !=
                    "continue_candidate_preflight" or
                    timestamp != self._capture_timestamp_s or
                    _digest(observation.to_record()) !=
                    self._capture_observation_sha256):
                raise ValueError("reusing a key frame is only allowed for a pilot prefix")
        elif timestamp <= self._capture_timestamp_s:
            raise ValueError("next trial requires a newer key camera acquisition")
        if (self._attempt is not None and self._attempt.events and
                observation.acquisition_interval_s[0] <=
                self._attempt.events[-1]["timestamp_s"]):
            raise ValueError("new key image predates the previous attempt outcome")

    def preflight_next_key(
        self, *, planner, key_observation: KeyPoseObservation,
        key_evidence_dir: Path,
        live_start_q, start_q_acquisition_timestamp_s: float,
        max_key_state_skew_s: float, limits, max_pose_error_deg: float,
        axial_waypoint_step_m: float, max_candidate_attempts: int | None = None,
        **optional_planning_kwargs,
    ) -> TrialPreflight:
        """Run the existing v8 pickup/20 mm preflight for a fresh key pose.

        Failed planning candidates are skipped only while continuing the same
        capture's bounded pilot. A new key pose/state may make them feasible.
        Physical attempts are excluded for the remainder of this session.
        """
        self._can_plan_key(key_observation)
        evidence_dir = Path(key_evidence_dir).expanduser().resolve()
        manifest = verify_key_capture_artifacts(evidence_dir)
        saved_observation = json.loads((evidence_dir / "key_observation.json")
                                       .read_text(encoding="utf-8"))
        if (saved_observation != key_observation.to_record() or
                manifest["capture_id"] != key_observation.capture_id or
                manifest["request_id"] != key_observation.request_id):
            raise ValueError("saved key evidence does not match admitted pose")
        validate_catalog_session(self.catalog, mode=self.mode,
                                 session_record=self.calibration.record)
        if (_digest(self.calibration.record) != self.session_sha256 or
                _digest(self.catalog) != self.catalog_sha256):
            raise ValueError("frozen session or endpoint catalogue changed")
        new_capture = key_observation.capture_id != self._capture_id
        exclusions = set(self._attempted)
        if not new_capture:
            exclusions.update(self._same_capture_planning_rejects)
        if {"attempted", "covered_scenes", "mode", "calibration", "catalog",
            "shared_root", "key_pose_world", "key_observation_id",
            "key_capture_timestamp_s"} & optional_planning_kwargs.keys():
            raise ValueError("runner-owned planning inputs cannot be overridden")
        result = plan_admitted_key_trial(
            planner=planner, mode=self.mode, shared_root=self.shared_root,
            calibration=self.calibration, catalog=self.catalog,
            key_observation=key_observation, live_start_q=live_start_q,
            start_q_acquisition_timestamp_s=start_q_acquisition_timestamp_s,
            max_key_state_skew_s=max_key_state_skew_s, limits=limits,
            max_pose_error_deg=max_pose_error_deg,
            axial_waypoint_step_m=axial_waypoint_step_m,
            max_candidate_attempts=max_candidate_attempts,
            attempted=tuple(sorted(exclusions)), covered_scenes=(),
            **optional_planning_kwargs)
        if (result.session_calibration_sha256 != self.session_sha256 or
                result.catalog_sha256 != self.catalog_sha256 or
                result.key_observation_id != key_observation.capture_id):
            raise ValueError("trial preflight is not bound to this session and key")
        decide_after_trial_preflight(result.to_record())
        path = (self.output_dir / "trial_preflights" /
                f"{self._preflight_index:04d}")
        write_trial_preflight_artifacts(result, path)
        report_path = path / "report.json"
        if not report_path.is_file():
            raise ValueError("trial preflight artifact lacks its report")
        self._write_exclusive(path / "key_evidence_binding.json", {
            "schema": "precision_insertion_trial_key_binding_v1",
            "key_capture_id": key_observation.capture_id,
            "key_evidence_dir": str(evidence_dir),
            "key_evidence_manifest_sha256": hashlib.sha256(
                (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest(),
            "key_observation_sha256": hashlib.sha256(
                (evidence_dir / "key_observation.json").read_bytes()).hexdigest(),
            "preflight_report_sha256": hashlib.sha256(
                report_path.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "robot_ready": False,
        })
        self._preflight_index += 1
        self._preflight = result
        self._preflight_report_path = report_path
        self._attempt = None
        self._attempt_dir = None
        self._capture_id = key_observation.capture_id
        self._capture_timestamp_s = float(
            key_observation.selected_acquisition_timestamp_s)
        self._capture_interval_end_s = float(
            key_observation.acquisition_interval_s[1])
        self._capture_observation_sha256 = _digest(key_observation.to_record())
        if new_capture:
            self._same_capture_planning_rejects.clear()
        for row in result.attempted_candidates:
            key = tuple(row["key"])
            if key != result.selected_candidate_key:
                self._same_capture_planning_rejects.add(key)
        return result

    def begin_selected_attempt(
        self, *, attempt_id: str, started_at_s: float,
    ) -> AttemptRecord:
        """Start an evidence record, not an execution command."""
        if (not isinstance(attempt_id, str) or
                not _SAFE_ID.fullmatch(attempt_id) or
                attempt_id in self._used_attempt_ids):
            raise ValueError("attempt ID must be new and file-safe")
        if (self._attempt is not None or self._preflight is None or
                self._preflight_report_path is None or
                self.current_decision().action != "execution_gate_required"):
            raise ValueError("no uniquely selected, passing trial preflight")
        if (not math.isfinite(float(started_at_s)) or
                float(started_at_s) < max(
                    self._capture_interval_end_s,
                    self._preflight.start_q_acquisition_timestamp_s)):
            raise ValueError("physical attempt cannot start before key/state acquisition")
        key = self._preflight.selected_candidate_key
        assert key is not None
        attempt = begin_attempt(
            attempt_id=attempt_id, mode=self.mode,
            session_record=self.calibration.record, candidate_id="/".join(key),
            tabletop_pose_stem=self._preflight.pose_class["stem"],
            xy_offset_socket_m=(0.0, 0.0), started_at_s=started_at_s)
        attempt_dir = self.output_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        self._write_exclusive(attempt_dir / "preflight_binding.json", {
            "schema": "precision_insertion_attempt_preflight_binding_v1",
            "candidate_id": "/".join(key),
            "key_capture_id": self._preflight.key_observation_id,
            "preflight_report": str(self._preflight_report_path),
            "preflight_report_sha256": hashlib.sha256(
                self._preflight_report_path.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "robot_ready": False,
        })
        attempt.write_new(attempt_dir / "state_000.json")
        self._used_attempt_ids.add(attempt_id)
        self._attempt = attempt
        self._attempt_dir = attempt_dir
        self._attempt_index = 0
        return deepcopy(attempt)

    def _record(self, mutation) -> AttemptRecord:
        if self._attempt is None or self._attempt_dir is None:
            raise ValueError("no active attempt to receive an observation")
        updated = deepcopy(self._attempt)
        mutation(updated)
        index = self._attempt_index + 1
        updated.write_new(self._attempt_dir / f"state_{index:03d}.json")
        self._attempt = updated
        self._attempt_index = index
        if updated.events:
            assert updated.candidate_id is not None
            self._attempted.add(tuple(updated.candidate_id.split("/")))
        return deepcopy(updated)

    def observe_stage(
        self, stage: str, status: bool | None, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        return self._record(lambda row: row.record_stage(
            stage, status, timestamp_s=timestamp_s,
            evidence_refs=evidence_refs))

    def observe_insertion(
        self, evidence, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        return self._record(lambda row: row.record_insertion_evidence(
            evidence, timestamp_s=timestamp_s, evidence_refs=evidence_refs))

    def record_retry(
        self, assessment: XYRetryAssessment,
        preflight: XYRetryPreflight, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        if self.current_decision().action != "guarded_withdrawal_then_xy_assessment":
            raise ValueError("retry is not the next evidence gate")
        if (not isinstance(assessment, XYRetryAssessment) or
                assessment.status != "proposal_requires_live_preflight" or
                assessment.decision is None or
                not isinstance(preflight, XYRetryPreflight) or
                preflight.status != "sampled_retry_preflight_pass" or
                preflight.planning.sampled_planning_pass is not True or
                preflight.endpoint_screen.get("endpoint_pass") is not True):
            raise ValueError("retry needs a voted proposal and passing fresh preflight")
        assert self._attempt is not None
        if (tuple(preflight.candidate_key) !=
                tuple(self._attempt.candidate_id.split("/")) or
                preflight.choice_id != assessment.decision.choice_id or
                tuple(preflight.targets.xy_offset_socket_m) !=
                tuple(assessment.decision.offset_socket_m or ())):
            raise ValueError("retry preflight does not match the held grasp/XY vote")
        return self._record(lambda row: row.record_retry(
            assessment.decision, timestamp_s=timestamp_s,
            evidence_refs=evidence_refs))

    def record_failure(
        self, code: str, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        return self._record(lambda row: row.record_failure(
            code, timestamp_s=timestamp_s, evidence_refs=evidence_refs))
