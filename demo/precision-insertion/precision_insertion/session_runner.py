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
import time
from typing import Mapping

import numpy as np

from .assets import AssetPaths
from .calibration import SessionCalibration
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .key_perception import (
    KeyPoseObservation, verify_key_capture_artifacts,
)
from .geometry import validate_se3
from .endpoint import _load_mesh
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .pose_selection import classify_key_tabletop_pose
from .postlift_preflight import (
    PostLiftPreflight, plan_postlift_observed_transfer,
    write_postlift_preflight,
)
from .records import AttemptRecord, begin_attempt
from .retry_session import (
    RetrySessionLimits, RetrySessionResult,
    assess_and_plan_observed_xy_retry, write_retry_session_artifacts,
)
from .repose_artifacts import write_repose_preflight_artifacts
from .repose_preflight import validate_repose_rest_target
from .repose_transition import (
    ReposeTransitionPreflight, preflight_v8_repose_transition,
)
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
        self._repose_preflight: ReposeTransitionPreflight | None = None
        self._repose_report_path: Path | None = None
        self._repose_index = 0
        self._same_capture_reset_rejects: set[tuple[int, str, str]] = set()
        self._key_evidence_dir: Path | None = None
        self._key_evidence_manifest_sha256: str | None = None
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
        self._postlift_preflight: PostLiftPreflight | None = None
        self._postlift_report_path: Path | None = None
        self._postlift_report_sha256: str | None = None
        self._postlift_index = 0
        self._retry_assessment_index = 0
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
            if (self._postlift_preflight is not None and
                    self._postlift_preflight.status ==
                    "sampled_postlift_preflight_pass" and
                    self._attempt.labels["grasp_success"] is True and
                    self._attempt.labels["preinsert_reached"] is None):
                if (self._postlift_report_path is None or
                        not self._postlift_report_path.is_file() or
                        self._postlift_report_sha256 is None or
                        hashlib.sha256(
                            self._postlift_report_path.read_bytes()).hexdigest() !=
                        self._postlift_report_sha256):
                    return SessionDecision(
                        "stop_for_review", "postlift_report_changed", None)
                return SessionDecision(
                    "transfer_execution_gate_required",
                    "observed_postlift_path_planned_arrival_unobserved",
                    self._attempt.candidate_id,
                    ("commissioned_robot_and_force_limits",
                     "independent_transfer_execution_evidence",
                     "fresh_preinsert_key_and_grip_observation"))
            return decide_after_attempt(
                self._attempt, max_xy_retries=self.max_xy_retries)
        if self._repose_preflight is not None:
            status = self._repose_preflight.status
            if status == "nominal_reset_preflight_pass_drop_unobserved":
                return SessionDecision(
                    "repose_execution_gate_required",
                    "nominal_repose_pick_place_only_landing_unobserved", None,
                    ("commissioned_robot_and_force_limits",
                     "fresh_key_and_joint_observation",
                     "measured_post_lift_key_hand_relation",
                     "observed_release_and_landing_pose"))
            if status == "reset_seed_budget_exhausted":
                return SessionDecision(
                    "continue_repose_seed_preflight",
                    "pilot_prefix_is_not_reset_seed_exhaustion", None)
            if status == "held_reset_path_available_release_unplanned":
                return SessionDecision(
                    "preflight_repose_release",
                    "held_path_has_no_opening_or_retreat_preflight", None)
            if status in {"no_verified_reset_seed", "no_held_reset_path",
                          "no_nominal_reset_path"}:
                return SessionDecision(
                    "preflight_alternative_repose_cell", status, None)
            return SessionDecision(
                "stop_for_review", status, None)
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
        if observation.phase != "tabletop":
            raise ValueError("new tabletop trial cannot use a held-key observation")
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
        self._repose_preflight = None
        self._repose_report_path = None
        self._same_capture_reset_rejects.clear()
        self._key_evidence_dir = evidence_dir
        self._key_evidence_manifest_sha256 = hashlib.sha256(
            (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest()
        self._attempt = None
        self._attempt_dir = None
        self._postlift_preflight = None
        self._postlift_report_path = None
        self._postlift_report_sha256 = None
        self._postlift_index = 0
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

    def preflight_repose(
        self, *, planner, key_observation: KeyPoseObservation,
        to_pose_stem: str, height_cm: int,
        release_xy_robot_m: tuple[float, float], live_start_q,
        start_q_acquisition_timestamp_s: float, max_state_skew_s: float,
        max_pose_error_deg: float, max_center_in_hand_drift_m: float,
        max_symmetry_axis_tilt_deg: float,
        minimum_rest_socket_clearance_m: float,
        minimum_board_edge_clearance_m: float, limits,
        reset_candidate_root: Path | None = None,
        max_seed_attempts: int | None = None,
        retreat_goal_arm_q=None,
        minimum_release_key_clearance_m: float | None = None,
    ) -> ReposeTransitionPreflight:
        """Connect a pose-exhausted trial to the existing directed v8 reset.

        The same saved key exposure may be reused only because no grasp or
        robot motion has occurred since the preceding insertion preflight.
        A result remains a nominal plan: landing must be observed separately.
        """
        if self.current_decision().action not in {
                "preflight_repose", "continue_repose_seed_preflight",
                "preflight_repose_release",
                "preflight_alternative_repose_cell"}:
            raise ValueError("repose requires observed current-pose candidate exhaustion")
        if self._preflight is None or self._key_evidence_dir is None or (
                self._key_evidence_manifest_sha256 is None):
            raise ValueError("repose lacks a bound fresh-key trial")
        if (not isinstance(key_observation, KeyPoseObservation) or
                key_observation.capture_id != self._capture_id or
                _digest(key_observation.to_record()) !=
                self._capture_observation_sha256):
            raise ValueError("repose key pose differs from exhausted trial")
        if to_pose_stem not in self._preflight.repose_target_stems:
            raise ValueError("repose target has no eligible insertion grasp")
        key_observation.require_state_alignment(
            state_timestamp_s=start_q_acquisition_timestamp_s,
            maximum_skew_s=max_state_skew_s)
        verify_key_capture_artifacts(self._key_evidence_dir)
        if hashlib.sha256(
                (self._key_evidence_dir / "evidence_manifest.json")
                .read_bytes()).hexdigest() != self._key_evidence_manifest_sha256:
            raise ValueError("saved key capture changed after trial preflight")
        if (_digest(self.calibration.record) != self.session_sha256 or
                _digest(self.catalog) != self.catalog_sha256):
            raise ValueError("frozen session or endpoint catalogue changed")
        limits.validate()
        result = preflight_v8_repose_transition(
            planner=planner, shared_root=self.shared_root, mode=self.mode,
            calibration=self.calibration, trial_scene=self._preflight.trial_scene,
            catalog=self.catalog,
            from_pose_stem=self._preflight.pose_class["stem"],
            to_pose_stem=to_pose_stem, height_cm=height_cm,
            release_xy_robot_m=release_xy_robot_m,
            live_start_q=live_start_q,
            observation_id=key_observation.capture_id,
            key_capture_timestamp_s=(
                key_observation.selected_acquisition_timestamp_s),
            start_q_acquisition_timestamp_s=start_q_acquisition_timestamp_s,
            max_state_skew_s=max_state_skew_s,
            max_pose_error_deg=max_pose_error_deg,
            max_center_in_hand_drift_m=max_center_in_hand_drift_m,
            max_symmetry_axis_tilt_deg=max_symmetry_axis_tilt_deg,
            minimum_rest_socket_clearance_m=minimum_rest_socket_clearance_m,
            minimum_board_edge_clearance_m=minimum_board_edge_clearance_m,
            limits=limits, attempted_insertion=self.attempted_candidates,
            covered_scenes=(),
            attempted_reset_ids=tuple(sorted(
                seed_id for row_height, row_target, seed_id in
                self._same_capture_reset_rejects
                if row_height == height_cm and row_target == to_pose_stem)),
            reset_candidate_root=reset_candidate_root,
            max_seed_attempts=max_seed_attempts,
            retreat_goal_arm_q=retreat_goal_arm_q,
            minimum_release_key_clearance_m=(
                minimum_release_key_clearance_m))
        if (result.observation_id != key_observation.capture_id or
                result.from_pose_stem != self._preflight.pose_class["stem"] or
                result.to_pose_stem != to_pose_stem or
                result.height_cm != height_cm):
            raise ValueError("directed reset result does not match the frozen trial")
        if result.status == "nominal_reset_preflight_pass_drop_unobserved" and (
                result.selected_seed is None or
                result.pickup_plan is None or
                result.held_plan is None or
                result.held_plan.status !=
                "sampled_held_path_pass_release_unplanned" or
                result.release_plan is None or
                result.release_plan.status !=
                "nominal_release_exit_path_pass_drop_unobserved"):
            raise ValueError("nominal reset pass lacks pickup/held/release paths")
        index = self._repose_index
        inputs = self.output_dir / "repose_inputs" / f"{index:04d}"
        inputs.mkdir(parents=True, exist_ok=False)
        key_pose_file = inputs / "key_pose_world.npy"
        state_file = inputs / "live_start_q.npy"
        np.save(key_pose_file, key_observation.pose_world)
        np.save(state_file, np.asarray(live_start_q, dtype=np.float64))
        limits_file = inputs / "limits.json"
        self._write_exclusive(limits_file, vars(limits).copy())
        output = self.output_dir / "repose_preflights" / f"{index:04d}"
        write_repose_preflight_artifacts(
            result=result, trial_scene=self._preflight.trial_scene,
            output_dir=output,
            source_files={
                "session": self.output_dir / "frozen_session_calibration.json",
                "catalog": self.output_dir / "endpoint_catalog.json",
                "key_pose_world": key_pose_file,
                "live_start_q": state_file,
                "limits": limits_file,
            })
        report_path = output / "report.json"
        if not report_path.is_file():
            raise ValueError("directed reset preflight artifact lacks its report")
        self._write_exclusive(output / "key_evidence_binding.json", {
            "schema": "precision_insertion_repose_key_binding_v1",
            "key_capture_id": key_observation.capture_id,
            "key_evidence_dir": str(self._key_evidence_dir),
            "key_evidence_manifest_sha256": self._key_evidence_manifest_sha256,
            "repose_report_sha256": hashlib.sha256(
                report_path.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "robot_ready": False,
        })
        self._repose_index += 1
        self._repose_preflight = result
        self._repose_report_path = report_path
        for row in result.attempted_seeds:
            seed_id = row.get("seed_id")
            if isinstance(seed_id, str) and seed_id != (
                    None if result.selected_seed is None
                    else result.selected_seed.get("seed_id")):
                self._same_capture_reset_rejects.add(
                    (height_cm, to_pose_stem, seed_id))
        return result

    def begin_repose_attempt(
        self, *, attempt_id: str, started_at_s: float,
    ) -> AttemptRecord:
        """Open a separate observed reorientation attempt, without moving."""
        if (not isinstance(attempt_id, str) or
                not _SAFE_ID.fullmatch(attempt_id) or
                attempt_id in self._used_attempt_ids):
            raise ValueError("repose attempt ID must be new and file-safe")
        if (self._attempt is not None or self._repose_preflight is None or
                self._repose_report_path is None or
                self.current_decision().action !=
                "repose_execution_gate_required"):
            raise ValueError("no complete nominal reset release/exit preflight")
        if (not math.isfinite(float(started_at_s)) or
                float(started_at_s) < max(
                    self._capture_interval_end_s,
                    self._repose_preflight.start_q_acquisition_timestamp_s)):
            raise ValueError("repose cannot start before key/state acquisition")
        attempt = begin_attempt(
            attempt_id=attempt_id, mode=self.mode,
            session_record=self.calibration.record, candidate_id=None,
            tabletop_pose_stem=self._repose_preflight.from_pose_stem,
            xy_offset_socket_m=(0.0, 0.0), started_at_s=started_at_s)
        attempt_dir = self.output_dir / "attempts" / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=False)
        self._write_exclusive(attempt_dir / "repose_preflight_binding.json", {
            "schema": "precision_insertion_repose_attempt_binding_v1",
            "from_pose_stem": self._repose_preflight.from_pose_stem,
            "to_pose_stem": self._repose_preflight.to_pose_stem,
            "repose_report": str(self._repose_report_path),
            "repose_report_sha256": hashlib.sha256(
                self._repose_report_path.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "robot_ready": False,
        })
        attempt.write_new(attempt_dir / "state_000.json")
        self._used_attempt_ids.add(attempt_id)
        self._attempt = attempt
        self._attempt_dir = attempt_dir
        self._postlift_preflight = None
        self._postlift_report_path = None
        self._postlift_report_sha256 = None
        self._postlift_index = 0
        self._attempt_index = 0
        self._retry_assessment_index = 0
        return deepcopy(attempt)

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
        self._postlift_preflight = None
        self._postlift_report_path = None
        self._postlift_report_sha256 = None
        self._postlift_index = 0
        self._attempt_index = 0
        self._retry_assessment_index = 0
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
        if updated.events and updated.candidate_id is not None:
            self._attempted.add(tuple(updated.candidate_id.split("/")))
        return deepcopy(updated)

    def postlift_candidate_pose_prior(
        self, *, planner, joint_sample: LiveRobotState,
        max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
        max_arm_velocity_rad_s: float,
    ) -> np.ndarray:
        """Provide a loose first-lift FoundPose prior from measured wrist + v8.

        This is a search prior, never an achieved hand/key relation. The
        separately admitted post-lift image must replace it before transfer.
        """
        if (self.current_decision().action !=
                "postlift_observed_preflight_required" or
                self._attempt is None or self._preflight is None or
                self._preflight.selected_candidate_key is None):
            raise ValueError("candidate pose prior requires observed lift success")
        if not isinstance(joint_sample, LiveRobotState):
            raise TypeError("post-lift prior needs measured robot feedback")
        joint_sample.validate(
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s)
        last_grasp_time = self._attempt.events[-1]["timestamp_s"]
        if joint_sample.sample_timestamp_s <= last_grasp_time:
            raise ValueError("post-lift joint sample predates observed grasp")
        if (_digest(self.calibration.record) != self.session_sha256 or
                _digest(self.catalog) != self.catalog_sha256):
            raise ValueError("frozen session or endpoint catalogue changed")
        selected = select_pose_candidates(
            self.catalog, expected_mode=self.mode,
            tabletop_pose_stem=self._attempt.tabletop_pose_stem)
        matches = [row for row in selected["candidates"]
                   if tuple(row["key"]) ==
                   self._preflight.selected_candidate_key]
        if selected["status"] != "candidates_available" or len(matches) != 1:
            raise ValueError("selected v8 grasp is no longer endpoint-eligible")
        candidate_dir = Path(matches[0]["candidate_dir"]).expanduser().resolve()
        nominal = validate_se3(np.load(
            candidate_dir / "wrist_se3.npy", allow_pickle=False),
            name="selected candidate T_key_hand")
        wrist = validate_se3(planner.fk_wrist(joint_sample.full_q),
                             name="measured post-lift wrist FK")
        c2r = validate_se3(self.calibration.record.get("c2r"),
                           name="session C2R")
        return validate_se3(c2r @ wrist @ np.linalg.inv(nominal),
                            name="first post-lift candidate pose prior")

    def prepare_postlift_transfer(
        self, *, planner, key_observation: KeyPoseObservation,
        key_evidence_dir: Path, joint_sample: LiveRobotState,
        max_state_skew_s: float, max_arm_hand_skew_s: float,
        max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
        max_grasp_translation_drift_m: float,
        max_grasp_rotation_drift_deg: float,
        limits: PathAuditLimits, axial_waypoint_step_m: float,
    ) -> PostLiftPreflight:
        """Bind an observed post-lift key/hand relation to a fresh path plan.

        Saves both rejected and passing plans. No command is sent and a pass
        never labels pre-insertion arrival; that still needs observation.
        """
        if ((self._postlift_preflight is not None and
             self._postlift_preflight.status ==
             "sampled_postlift_preflight_pass") or
                self.current_decision().action !=
                "postlift_observed_preflight_required" or
                self._attempt_dir is None or
                self._attempt is None or self._preflight is None or
                not isinstance(key_observation, KeyPoseObservation) or
                not isinstance(joint_sample, LiveRobotState) or
                key_observation.phase != "held_postlift"):
            raise ValueError("post-lift replan needs one first held-key observation")
        if self._postlift_preflight is not None and (
                key_observation.selected_acquisition_timestamp_s <=
                self._postlift_preflight.key_capture_timestamp_s or
                key_observation.capture_id ==
                self._postlift_preflight.key_observation_id):
            raise ValueError("post-lift re-observation must use newer frames")
        evidence_dir = Path(key_evidence_dir).expanduser().resolve()
        manifest = verify_key_capture_artifacts(evidence_dir)
        saved = json.loads((evidence_dir / "key_observation.json")
                           .read_text(encoding="utf-8"))
        if (saved != key_observation.to_record() or
                manifest["capture_id"] != key_observation.capture_id or
                manifest["request_id"] != key_observation.request_id or
                key_observation.acquisition_interval_s[0] <=
                self._attempt.events[-1]["timestamp_s"] or
                key_observation.capture_id == self._capture_id):
            raise ValueError("first held-key frames are stale or unbound")
        key_observation.require_state_alignment(
            state_timestamp_s=joint_sample.sample_timestamp_s,
            maximum_skew_s=max_state_skew_s)
        prior = key_observation.consistency.get("held_pose_prior")
        if (not isinstance(prior, dict) or
                prior.get("source") !=
                "measured_wrist_plus_candidate_grasp" or
                prior.get("timestamp_s") != joint_sample.sample_timestamp_s):
            raise ValueError("held-key admission lacks the selected candidate prior")
        expected_prior = self.postlift_candidate_pose_prior(
            planner=planner, joint_sample=joint_sample,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s)
        if not np.allclose(validate_se3(
                prior.get("pose_world"), name="saved post-lift prior"),
                expected_prior, atol=1e-8, rtol=0):
            raise ValueError("held-key admission prior differs from selected v8 grasp")
        result = plan_postlift_observed_transfer(
            planner=planner, trial=self._preflight, attempt=self._attempt,
            calibration=self.calibration, catalog=self.catalog,
            mode=self.mode, shared_root=self.shared_root,
            key_pose_world=key_observation.pose_world,
            key_observation_id=key_observation.capture_id,
            key_capture_timestamp_s=(
                key_observation.selected_acquisition_timestamp_s),
            key_pose_source="multiview_foundpose", joint_sample=joint_sample,
            max_state_skew_s=max_state_skew_s,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s,
            max_grasp_translation_drift_m=max_grasp_translation_drift_m,
            max_grasp_rotation_drift_deg=max_grasp_rotation_drift_deg,
            limits=limits, axial_waypoint_step_m=axial_waypoint_step_m)
        output = (self._attempt_dir / "postlift_preflights" /
                  f"{self._postlift_index:03d}")
        write_postlift_preflight(result, output)
        report = output / "report.json"
        self._write_exclusive(output / "key_evidence_binding.json", {
            "schema": "precision_insertion_postlift_key_binding_v1",
            "key_capture_id": key_observation.capture_id,
            "key_evidence_dir": str(evidence_dir),
            "key_evidence_manifest_sha256": hashlib.sha256(
                (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest(),
            "key_observation_sha256": hashlib.sha256(
                (evidence_dir / "key_observation.json").read_bytes()).hexdigest(),
            "postlift_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "robot_ready": False,
        })
        self._postlift_preflight = result
        self._postlift_report_path = report
        self._postlift_report_sha256 = hashlib.sha256(report.read_bytes()).hexdigest()
        self._postlift_index += 1
        return result

    def observe_stage(
        self, stage: str, status: bool | None, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        if stage == "reorient_success":
            raise ValueError("use observe_repose_landing for the target-pose label")
        if stage == "reset_success" and status is True:
            raise ValueError("use observe_reset_landing for verified reset success")
        if self._attempt is not None and self._attempt.candidate_id is None:
            raise ValueError("repose and insertion attempt labels must stay separate")
        if stage == "preinsert_reached" and status is True:
            if (self._postlift_preflight is None or
                    self._postlift_preflight.status !=
                    "sampled_postlift_preflight_pass" or
                    self._postlift_report_path is None or
                    not isinstance(evidence_refs, Mapping) or
                    evidence_refs.get("postlift_preflight") !=
                    str(self._postlift_report_path) or
                    not self._postlift_report_path.is_file() or
                    self._postlift_report_sha256 is None or
                    hashlib.sha256(
                        self._postlift_report_path.read_bytes()).hexdigest() !=
                    self._postlift_report_sha256):
                raise ValueError("preinsert arrival needs this attempt's passing post-lift plan")
            if float(timestamp_s) < max(
                    self._postlift_preflight.key_capture_timestamp_s,
                    self._postlift_preflight.joint_timestamp_s):
                raise ValueError("preinsert observation predates post-lift replan")
        return self._record(lambda row: row.record_stage(
            stage, status, timestamp_s=timestamp_s,
            evidence_refs=evidence_refs))

    def observe_reset_landing(
        self, *, key_observation: KeyPoseObservation,
        key_evidence_dir: Path, recovery_log_path: Path,
        timestamp_s: float, max_pose_error_deg: float,
        max_return_center_shift_m: float, support_tolerance_m: float,
        minimum_rest_socket_clearance_m: float,
        minimum_board_edge_clearance_m: float,
    ) -> AttemptRecord:
        """Verify a supervised return from a new supported tabletop key pose.

        This does not extract the key or execute reset motion. In particular,
        a successful insertion still needs an externally logged supervised
        extraction before this method can make another trial eligible.
        """
        if (self._attempt is None or self._attempt.candidate_id is None or
                self._attempt_dir is None or self._preflight is None or
                not self._attempt.events or
                any(event["stage"] == "reset_success"
                    for event in self._attempt.events) or
                not isinstance(key_observation, KeyPoseObservation) or
                key_observation.phase != "tabletop"):
            raise ValueError("no insertion attempt awaits a tabletop reset check")
        maximum_shift = float(max_return_center_shift_m)
        if not math.isfinite(maximum_shift) or maximum_shift <= 0:
            raise ValueError("return center-shift limit must be positive")
        recovery_file = Path(recovery_log_path).expanduser().resolve()
        recovery_bytes = recovery_file.read_bytes()
        recovery = json.loads(recovery_bytes)
        if not isinstance(recovery, dict):
            raise ValueError("supervised recovery log must be a JSON object")
        try:
            completed = float(recovery.get("completed_at_s", float("nan")))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid supervised recovery completion time") from exc
        if (recovery.get("schema") !=
                "precision_insertion_supervised_reset_v1" or
                recovery.get("attempt_id") != self._attempt.attempt_id or
                recovery.get("method") != "supervised_manual_return" or
                not isinstance(recovery.get("reviewed_by"), str) or
                not recovery["reviewed_by"].strip() or
                recovery.get("socket_clear") is not True or
                recovery.get("hand_open") is not True or
                not math.isfinite(completed) or
                completed <= self._attempt.events[-1]["timestamp_s"] or
                (self._attempt.labels["insertion_success"] is not None and
                 recovery.get("key_removed_from_socket") is not True)):
            raise ValueError("reset needs a later supervised recovery log")
        observation_time = float(timestamp_s)
        if (not math.isfinite(observation_time) or
                key_observation.capture_id == self._capture_id or
                key_observation.acquisition_interval_s[0] <= completed or
                key_observation.acquisition_interval_s[0] <=
                self._attempt.events[-1]["timestamp_s"] or
                observation_time < key_observation.acquisition_interval_s[1]):
            raise ValueError("reset landing needs fresh frames after recovery")
        evidence_dir = Path(key_evidence_dir).expanduser().resolve()
        manifest = verify_key_capture_artifacts(evidence_dir)
        saved = json.loads((evidence_dir / "key_observation.json")
                           .read_text(encoding="utf-8"))
        if (saved != key_observation.to_record() or
                manifest["capture_id"] != key_observation.capture_id or
                manifest["request_id"] != key_observation.request_id):
            raise ValueError("reset landing differs from saved key frames")
        if (_digest(self.calibration.record) != self.session_sha256 or
                _digest(self.catalog) != self.catalog_sha256):
            raise ValueError("frozen session or endpoint catalogue changed")
        c2r = validate_se3(self.calibration.record.get("c2r"),
                           name="session C2R")
        original = validate_se3(
            np.linalg.inv(c2r) @ self._preflight.key_pose_world,
            name="trial-start T_robot_key")
        landed = validate_se3(
            np.linalg.inv(c2r) @ key_observation.pose_world,
            name="reset-landed T_robot_key")
        classification = classify_key_tabletop_pose(
            mode=self.mode, shared_root=self.shared_root,
            pose_robot_key=landed,
            max_rotation_error_deg=max_pose_error_deg)
        support = validate_repose_rest_target(
            shared_root=self.shared_root, mode=self.mode,
            calibration=self.calibration, T_robot_key_rest=landed,
            support_tolerance_m=support_tolerance_m,
            minimum_rest_socket_clearance_m=(
                minimum_rest_socket_clearance_m),
            minimum_board_edge_clearance_m=minimum_board_edge_clearance_m)
        mesh = _load_mesh(AssetPaths(self.shared_root, self.mode).raw_mesh(
            self.mode.key_object))
        center_local = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
        if center_local.shape != (3,) or not np.all(np.isfinite(center_local)):
            raise ValueError("key CAD has no finite physical center")
        original_xy = (original[:3, :3] @ center_local + original[:3, 3])[:2]
        landed_xy = (landed[:3, :3] @ center_local + landed[:3, 3])[:2]
        shift = float(np.linalg.norm(landed_xy - original_xy))
        success = (classification["stem"] ==
                   self._preflight.pose_class["stem"] and
                   shift <= maximum_shift)
        report_path = self._attempt_dir / "reset_landing.json"
        self._write_exclusive(report_path, {
            "schema": "precision_insertion_reset_landing_v1",
            "attempt_id": self._attempt.attempt_id,
            "key_capture_id": key_observation.capture_id,
            "key_evidence_dir": str(evidence_dir),
            "key_evidence_manifest_sha256": hashlib.sha256(
                (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest(),
            "recovery_log": str(recovery_file),
            "recovery_log_sha256": hashlib.sha256(
                recovery_bytes).hexdigest(),
            "target_pose_stem": self._preflight.pose_class["stem"],
            "observed_pose_class": classification,
            "observed_support": support,
            "center_xy_displacement_m": shift,
            "maximum_center_xy_displacement_m": maximum_shift,
            "reset_success": success,
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "scope": "observed_tabletop_return_after_supervised_recovery_not_automatic_extraction",
            "robot_ready": False,
        })
        return self._record(lambda row: row.record_stage(
            "reset_success", success, timestamp_s=timestamp_s,
            evidence_refs={
                "key_pose": str(evidence_dir / "key_observation.json"),
                "reset_assessment": str(report_path),
                "recovery_log": str(recovery_file),
            }))

    def observe_repose_landing(
        self, *, key_observation: KeyPoseObservation,
        key_evidence_dir: Path, timestamp_s: float,
        release_completed_at_s: float, release_evidence_path: Path,
        max_pose_error_deg: float, support_tolerance_m: float,
        minimum_rest_socket_clearance_m: float,
        minimum_board_edge_clearance_m: float,
    ) -> AttemptRecord:
        """Label repose only from a new supported, in-board key observation.

        A missing or ambiguous observation raises and keeps the label unknown.
        The release file/time are caller assertions until a commissioned
        executor supplies them. This is not proof of physical repeatability.
        """
        if (self._attempt is None or self._attempt.candidate_id is not None or
                self._repose_preflight is None or
                self.current_decision().action != "await_repose_observation"):
            raise ValueError("no pending directed repose landing to observe")
        if not isinstance(key_observation, KeyPoseObservation):
            raise TypeError("landing needs an admitted fresh multi-view key pose")
        if key_observation.phase != "tabletop":
            raise ValueError("repose landing needs a tabletop key observation")
        release_time = float(release_completed_at_s)
        release_file = Path(release_evidence_path).expanduser().resolve()
        if (not math.isfinite(release_time) or
                release_time < self._attempt.started_at_s or
                not release_file.is_file() or
                key_observation.acquisition_interval_s[0] <= release_time):
            raise ValueError("landing frames need a prior logged physical release")
        evidence_dir = Path(key_evidence_dir).expanduser().resolve()
        manifest = verify_key_capture_artifacts(evidence_dir)
        saved_observation = json.loads((evidence_dir / "key_observation.json")
                                       .read_text(encoding="utf-8"))
        if (saved_observation != key_observation.to_record() or
                manifest["capture_id"] != key_observation.capture_id or
                key_observation.capture_id == self._capture_id or
                key_observation.acquisition_interval_s[0] <=
                self._attempt.started_at_s or
                float(timestamp_s) < key_observation.acquisition_interval_s[1]):
            raise ValueError("landing evidence predates repose or differs from saved frames")
        c2r = validate_se3(self.calibration.record.get("c2r"),
                           name="session C2R")
        T_robot_key = validate_se3(
            np.linalg.inv(c2r) @ key_observation.pose_world,
            name="observed landed T_robot_key")
        classification = classify_key_tabletop_pose(
            mode=self.mode, shared_root=self.shared_root,
            pose_robot_key=T_robot_key,
            max_rotation_error_deg=max_pose_error_deg)
        support = validate_repose_rest_target(
            shared_root=self.shared_root, mode=self.mode,
            calibration=self.calibration, T_robot_key_rest=T_robot_key,
            support_tolerance_m=support_tolerance_m,
            minimum_rest_socket_clearance_m=(
                minimum_rest_socket_clearance_m),
            minimum_board_edge_clearance_m=minimum_board_edge_clearance_m)
        success = classification["stem"] == self._repose_preflight.to_pose_stem
        assert self._attempt_dir is not None
        report_path = self._attempt_dir / "repose_landing.json"
        self._write_exclusive(report_path, {
            "schema": "precision_insertion_repose_landing_v1",
            "capture_id": key_observation.capture_id,
            "key_evidence_dir": str(evidence_dir),
            "key_evidence_manifest_sha256": hashlib.sha256(
                (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest(),
            "release_completed_at_s": release_time,
            "release_evidence_path": str(release_file),
            "release_evidence_sha256": hashlib.sha256(
                release_file.read_bytes()).hexdigest(),
            "target_pose_stem": self._repose_preflight.to_pose_stem,
            "observed_pose_class": classification,
            "observed_support": support,
            "reorient_success": success,
            "robot_ready": False,
        })
        return self._record(lambda row: row.record_stage(
            "reorient_success", success, timestamp_s=timestamp_s,
            evidence_refs={
                "key_pose": str(evidence_dir / "key_observation.json"),
                "tabletop_classification": str(report_path),
                "release_execution": str(release_file),
            }))

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

    def prepare_observed_xy_retry(
        self, *, planner, held_key_observation: KeyPoseObservation,
        held_key_evidence_dir: Path, joint_sample, frames,
        intrinsics_full: Mapping, extrinsics_full: Mapping,
        frame_request_id: int, frame_ids: Mapping[str, int],
        acquisition_metadata: Mapping, backend,
        withdrawal_completed_at_s: float,
        withdrawal_evidence_path: Path,
        postlift_preflight_report_path: Path,
        decision_timestamp_s: float,
        limits: RetrySessionLimits,
    ) -> RetrySessionResult:
        """Persist a same-frame VLM assessment and fresh held-state replan.

        This consumes a caller-logged guarded withdrawal and never sends a
        robot command. Only a passing result records a pending 1 mm retry.
        """
        if (self.current_decision().action !=
                "guarded_withdrawal_then_xy_assessment" or
                self._attempt is None or self._preflight is None or
                self._attempt_dir is None):
            raise ValueError("retry requires an observed failed insertion")
        result = assess_and_plan_observed_xy_retry(
            planner=planner, mode=self.mode, shared_root=self.shared_root,
            calibration=self.calibration, catalog=self.catalog,
            trial=self._preflight, attempt=self._attempt,
            held_key_observation=held_key_observation,
            held_key_evidence_dir=held_key_evidence_dir,
            joint_sample=joint_sample, frames=frames,
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full,
            frame_request_id=frame_request_id, frame_ids=frame_ids,
            acquisition_metadata=acquisition_metadata, backend=backend,
            withdrawal_completed_at_s=withdrawal_completed_at_s,
            withdrawal_evidence_path=withdrawal_evidence_path,
            postlift_preflight_report_path=postlift_preflight_report_path,
            decision_timestamp_s=decision_timestamp_s, limits=limits)
        output = (self._attempt_dir / "xy_retry_assessments" /
                  f"{self._retry_assessment_index:03d}")
        write_retry_session_artifacts(result, frames, output)
        self._retry_assessment_index += 1
        if result.status == "ready_to_record_pending_retry":
            assert result.assessment is not None and result.preflight is not None
            completed_at_s = time.time()
            if completed_at_s < decision_timestamp_s:
                raise ValueError("retry planning completion predates VLM decision")
            self.record_retry(
                result.assessment, result.preflight,
                timestamp_s=completed_at_s,
                evidence_refs={
                    "axial_withdrawal": str(result.withdrawal_evidence_path),
                    "live_preflight": str(output / "preflight" / "report.json"),
                    "xy_vlm_vote": str(output / "report.json"),
                })
        return result

    def record_failure(
        self, code: str, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        return self._record(lambda row: row.record_failure(
            code, timestamp_s=timestamp_s, evidence_refs=evidence_refs))
