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
from .bounded_postlift import (
    BoundedPostLiftPreflight, plan_bounded_postlift_transfer,
    verify_bounded_postlift_preflight, write_bounded_postlift_preflight,
)
from .calibration import SessionCalibration
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .key_perception import (
    KeyPoseObservation, verify_key_capture_artifacts,
)
from .lift_checkpoint import (
    LiftCheckpoint, assess_lift_checkpoint, assess_raw_lift_checkpoint,
    verify_lift_checkpoint, write_lift_checkpoint,
)
from .geometry import validate_se3
from .insertion_checkpoint import (
    assess_insertion_checkpoint, verify_insertion_checkpoint,
    write_insertion_checkpoint,
)
from .endpoint import _load_mesh
from .live_robot_state import LiveRobotState
from .observer import ImageVLM
from .raw_camera_capture import verify_raw_camera_capture
from .outcome import InsertionEvidence
from .path_audit import PathAuditLimits
from .pose_selection import classify_key_tabletop_pose
from .postlift_preflight import (
    PostLiftPreflight, plan_postlift_observed_transfer,
    write_postlift_preflight,
)
from .preinsert_checkpoint import (
    PreinsertCheckpoint, assess_preinsert_checkpoint,
    verify_preinsert_checkpoint, write_preinsert_checkpoint,
)
from .records import AttemptRecord, begin_attempt
from .retry_session import (
    RetrySessionLimits, RetrySessionResult, UnobservedXYDiagnostic,
    GroundedXYDiagnostic, assess_grounded_xy_diagnostic,
    assess_and_plan_observed_xy_retry, assess_unobserved_xy_diagnostic,
    write_retry_session_artifacts,
)
from .grounded_alignment import AlignmentLimits
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
from .uncertainty_margin import SurfaceDeviationBounds


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
        self._preflight_report_sha256: str | None = None
        self._preflight_binding_path: Path | None = None
        self._preflight_binding_sha256: str | None = None
        self._measured_start_state_sha256: str | None = None
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
        self._postlift_preflight: (
            PostLiftPreflight | BoundedPostLiftPreflight | None) = None
        self._postlift_report_path: Path | None = None
        self._postlift_report_sha256: str | None = None
        self._postlift_index = 0
        self._measured_lift_preflight = None
        self._measured_lift_report_path: Path | None = None
        self._measured_lift_report_sha256: str | None = None
        self._measured_lift_index = 0
        self._preinsert_checkpoint: PreinsertCheckpoint | None = None
        self._preinsert_report_path: Path | None = None
        self._preinsert_report_sha256: str | None = None
        self._preinsert_assessment_index = 0
        self._preinsert_frame_signatures: set[tuple] = set()
        self._preinsert_latest_exposure_s = -math.inf
        self._lift_checkpoint: LiftCheckpoint | None = None
        self._lift_report_path: Path | None = None
        self._lift_report_sha256: str | None = None
        self._lift_execution_path: Path | None = None
        self._lift_execution_sha256: str | None = None
        self._lift_assessment_index = 0
        self._lift_assessed_capture_ids: set[str] = set()
        self._insertion_assessment_index = 0
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
                bounded = isinstance(
                    self._postlift_preflight, BoundedPostLiftPreflight)
                if bounded:
                    candidate_dir = (
                        AssetPaths(self.shared_root, self.mode).candidate_dir /
                        Path(*self._postlift_preflight.candidate_key))
                    try:
                        verify_bounded_postlift_preflight(
                            self._postlift_report_path, mode=self.mode,
                            shared_root=self.shared_root,
                            candidate_dir=candidate_dir)
                    except (FileNotFoundError, KeyError, OSError, TypeError,
                            ValueError):
                        return SessionDecision(
                            "stop_for_review",
                            "bounded_postlift_evidence_changed", None)
                return SessionDecision(
                    "transfer_execution_gate_required",
                    ("bounded_postlift_path_planned_arrival_unobserved"
                     if bounded else
                     "observed_postlift_path_planned_arrival_unobserved"),
                    self._attempt.candidate_id,
                    (("commissioned_robot_and_force_limits",) +
                     (("verified_future_trial_surface_bounds",)
                      if bounded else ()) +
                     ("independent_transfer_execution_evidence",
                      "fresh_preinsert_key_and_grip_observation")))
            if (isinstance(self._lift_checkpoint, LiftCheckpoint) and
                    self._lift_checkpoint.evidence_kind == "raw_visual" and
                    self._attempt.labels["grasp_success"] is True and
                    self._attempt.labels["preinsert_reached"] is None):
                return SessionDecision(
                    "held_relation_evidence_required",
                    "visible_grasp_but_key_hand_pose_unknown",
                    self._attempt.candidate_id,
                    ("fresh_visible_key_pose_or_commissioned_grasp_relation",
                     "uncertainty_bounded_socket_endpoint_and_path_preflight"))
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
        measured_start_state: LiveRobotState | None = None,
        measured_state_limits: Mapping[str, float] | None = None,
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
        state_record = None
        if (measured_start_state is None) != (measured_state_limits is None):
            raise ValueError("measured state and validation limits must be paired")
        if measured_start_state is not None:
            if not isinstance(measured_start_state, LiveRobotState):
                raise TypeError("live preflight needs measured robot feedback")
            required = {"max_arm_hand_skew_s", "max_hand_command_error_raw",
                        "max_arm_velocity_rad_s"}
            if (not isinstance(measured_state_limits, Mapping) or
                    set(measured_state_limits) != required):
                raise ValueError("measured state validation limits are incomplete")
            measured_start_state.validate(**dict(measured_state_limits))
            state_record = measured_start_state.to_record()
            if (not np.array_equal(np.asarray(live_start_q, dtype=float),
                                   np.asarray(state_record["full_q"])) or
                    float(start_q_acquisition_timestamp_s) !=
                    state_record["sample_timestamp_s"]):
                raise ValueError("planner start joints differ from measured feedback")
            key_observation.require_state_alignment(
                state_timestamp_s=state_record["sample_timestamp_s"],
                maximum_skew_s=max_key_state_skew_s)
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
        state_sha256 = None
        if state_record is not None:
            state_path = path / "measured_start_state.json"
            self._write_exclusive(state_path, {
                **state_record,
                "validation_limits": dict(measured_state_limits),
                "key_capture_id": key_observation.capture_id,
                "key_acquisition_interval_s": list(
                    key_observation.acquisition_interval_s),
            })
            state_sha256 = hashlib.sha256(state_path.read_bytes()).hexdigest()
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
        binding_path = path / "key_evidence_binding.json"
        self._write_exclusive(binding_path, {
            "schema": "precision_insertion_trial_key_binding_v1",
            "key_capture_id": key_observation.capture_id,
            "key_evidence_dir": str(evidence_dir),
            "key_evidence_manifest_sha256": hashlib.sha256(
                (evidence_dir / "evidence_manifest.json").read_bytes()).hexdigest(),
            "key_observation_sha256": hashlib.sha256(
                (evidence_dir / "key_observation.json").read_bytes()).hexdigest(),
            "preflight_report_sha256": report_sha256,
            "measured_start_state_sha256": state_sha256,
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "robot_ready": False,
        })
        self._preflight_index += 1
        self._preflight = result
        self._preflight_report_path = report_path
        self._preflight_report_sha256 = report_sha256
        self._preflight_binding_path = binding_path
        self._preflight_binding_sha256 = hashlib.sha256(
            binding_path.read_bytes()).hexdigest()
        self._measured_start_state_sha256 = state_sha256
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
        self._measured_lift_preflight = None
        self._measured_lift_report_path = None
        self._measured_lift_report_sha256 = None
        self._measured_lift_index = 0
        self._preinsert_checkpoint = None
        self._preinsert_report_path = None
        self._preinsert_report_sha256 = None
        self._preinsert_assessment_index = 0
        self._preinsert_frame_signatures.clear()
        self._preinsert_latest_exposure_s = -math.inf
        self._lift_checkpoint = None
        self._lift_report_path = None
        self._lift_report_sha256 = None
        self._lift_execution_path = None
        self._lift_execution_sha256 = None
        self._lift_assessment_index = 0
        self._lift_assessed_capture_ids.clear()
        self._insertion_assessment_index = 0
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

    def verify_current_preflight_evidence(self) -> dict:
        """Recheck the exact saved plan, camera bundle and measured state.

        This is still not motion authorization. It closes the gap where an
        altered report could otherwise be rehashed when an attempt begins.
        """
        if (self._preflight is None or self._preflight_report_path is None or
                self._preflight_report_sha256 is None or
                self._preflight_binding_path is None or
                self._preflight_binding_sha256 is None or
                self._key_evidence_dir is None or
                self._key_evidence_manifest_sha256 is None):
            raise ValueError("no bound current trial preflight exists")
        report_path = self._preflight_report_path
        binding_path = self._preflight_binding_path
        if (not report_path.is_file() or
                hashlib.sha256(report_path.read_bytes()).hexdigest() !=
                self._preflight_report_sha256):
            raise ValueError("saved trial preflight report changed")
        if (not binding_path.is_file() or
                hashlib.sha256(binding_path.read_bytes()).hexdigest() !=
                self._preflight_binding_sha256):
            raise ValueError("saved key/preflight binding changed")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if (binding.get("preflight_report_sha256") !=
                self._preflight_report_sha256 or
                binding.get("measured_start_state_sha256") !=
                self._measured_start_state_sha256 or
                binding.get("session_calibration_sha256") !=
                self.session_sha256 or
                binding.get("catalog_sha256") != self.catalog_sha256 or
                binding.get("key_capture_id") !=
                self._preflight.key_observation_id or
                Path(binding.get("key_evidence_dir", "")).resolve() !=
                self._key_evidence_dir):
            raise ValueError("saved trial binding differs from its frozen inputs")
        if (_digest(self.calibration.record) != self.session_sha256 or
                _digest(self.catalog) != self.catalog_sha256):
            raise ValueError("frozen session or endpoint catalogue changed")
        verify_key_capture_artifacts(self._key_evidence_dir)
        manifest_path = self._key_evidence_dir / "evidence_manifest.json"
        if (not manifest_path.is_file() or
                hashlib.sha256(manifest_path.read_bytes()).hexdigest() !=
                self._key_evidence_manifest_sha256 or
                binding.get("key_evidence_manifest_sha256") !=
                self._key_evidence_manifest_sha256):
            raise ValueError("saved key capture changed after trial preflight")
        state_hash = self._measured_start_state_sha256
        if state_hash is not None:
            state_path = report_path.parent / "measured_start_state.json"
            if (not state_path.is_file() or
                    hashlib.sha256(state_path.read_bytes()).hexdigest() !=
                    state_hash):
                raise ValueError("saved measured start state changed")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(self._preflight, TrialPreflight):
            artifacts = report.get("artifacts")
            if (not isinstance(artifacts, dict) or
                    not isinstance(artifacts.get("trial_scene"), str) or
                    not isinstance(artifacts.get("trial_scene_sha256"), str)):
                raise ValueError("trial preflight has no hashed scene artifact")
            if (self._preflight.insertion_plan is not None and
                    not isinstance(artifacts.get("planned_trajectories"), str)):
                raise ValueError("selected insertion plan has no hashed trajectories")
            for name in ("trial_scene", "planned_trajectories"):
                relative = artifacts.get(name)
                if relative is None:
                    continue
                artifact = (report_path.parent / relative).resolve()
                if (not artifact.is_relative_to(report_path.parent) or
                        not artifact.is_file() or
                        hashlib.sha256(artifact.read_bytes()).hexdigest() !=
                        artifacts.get(f"{name}_sha256")):
                    raise ValueError(f"trial preflight {name} artifact changed")
            expected = self._preflight.to_record()
            if {key: value for key, value in report.items()
                    if key != "artifacts"} != expected:
                raise ValueError("saved preflight differs from selected candidate")
        return {
            "report_sha256": self._preflight_report_sha256,
            "binding_sha256": self._preflight_binding_sha256,
            "measured_start_state_sha256": state_hash,
            "key_evidence_manifest_sha256": self._key_evidence_manifest_sha256,
        }

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
        self.verify_current_preflight_evidence()
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
        self._measured_lift_preflight = None
        self._measured_lift_report_path = None
        self._measured_lift_report_sha256 = None
        self._measured_lift_index = 0
        self._preinsert_checkpoint = None
        self._preinsert_report_path = None
        self._preinsert_report_sha256 = None
        self._preinsert_assessment_index = 0
        self._preinsert_frame_signatures.clear()
        self._preinsert_latest_exposure_s = -math.inf
        self._lift_checkpoint = None
        self._lift_report_path = None
        self._lift_report_sha256 = None
        self._lift_execution_path = None
        self._lift_execution_sha256 = None
        self._lift_assessment_index = 0
        self._lift_assessed_capture_ids.clear()
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
        verified = self.verify_current_preflight_evidence()
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
            "preflight_report_sha256": verified["report_sha256"],
            "key_evidence_binding_sha256": verified["binding_sha256"],
            "measured_start_state_sha256": (
                verified["measured_start_state_sha256"]),
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
        self._measured_lift_preflight = None
        self._measured_lift_report_path = None
        self._measured_lift_report_sha256 = None
        self._measured_lift_index = 0
        self._preinsert_checkpoint = None
        self._preinsert_report_path = None
        self._preinsert_report_sha256 = None
        self._preinsert_assessment_index = 0
        self._preinsert_frame_signatures.clear()
        self._preinsert_latest_exposure_s = -math.inf
        self._lift_checkpoint = None
        self._lift_report_path = None
        self._lift_report_sha256 = None
        self._lift_execution_path = None
        self._lift_execution_sha256 = None
        self._lift_assessment_index = 0
        self._lift_assessed_capture_ids.clear()
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

    def _read_lift_execution(self, path: Path) -> tuple[Path, bytes, float]:
        if self._attempt is None:
            raise ValueError("lift needs an active physical attempt")
        lift_file = Path(path).expanduser().resolve()
        lift_bytes = lift_file.read_bytes()
        execution = json.loads(lift_bytes)
        if not isinstance(execution, dict):
            raise ValueError("lift execution log must be a JSON object")
        try:
            completed = float(execution.get("completed_at_s", float("nan")))
        except (TypeError, ValueError) as exc:
            raise ValueError("lift completion time is invalid") from exc
        if (execution.get("schema") !=
                "precision_insertion_lift_execution_v1" or
                execution.get("attempt_id") != self._attempt.attempt_id or
                execution.get("candidate_id") != self._attempt.candidate_id or
                execution.get("trajectory_complete") is not True or
                execution.get("force_abort") is not False or
                not math.isfinite(completed) or
                completed <= self._attempt.started_at_s):
            raise ValueError("lift needs a completed non-aborted execution log")
        return lift_file, lift_bytes, completed

    def _save_lift_assessment(
        self, *, result: LiftCheckpoint, lift_file: Path,
        lift_bytes: bytes, completed: float,
    ) -> LiftCheckpoint:
        if (self._attempt is None or self._attempt_dir is None or
                result.attempt_id != self._attempt.attempt_id or
                result.candidate_id != self._attempt.candidate_id or
                result.lift_completed_at_s != completed):
            raise ValueError("lift checkpoint does not match this physical attempt")
        output = (self._attempt_dir / "lift_assessments" /
                  f"{self._lift_assessment_index:03d}")
        write_lift_checkpoint(result, output)
        report_path = output / "report.json"
        verify_lift_checkpoint(report_path)
        self._write_exclusive(output / "execution_binding.json", {
            "schema": "precision_insertion_lift_execution_binding_v1",
            "lift_execution_log": str(lift_file),
            "lift_execution_log_sha256": hashlib.sha256(
                lift_bytes).hexdigest(),
            "lift_checkpoint_report_sha256": hashlib.sha256(
                report_path.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "robot_ready": False,
        })
        self._lift_assessment_index += 1
        self._lift_assessed_capture_ids.add(result.after_capture_id)
        if result.grasp_success is not None:
            self._lift_checkpoint = result
            self._lift_report_path = report_path
            self._lift_report_sha256 = hashlib.sha256(
                report_path.read_bytes()).hexdigest()
            self._lift_execution_path = lift_file
            self._lift_execution_sha256 = hashlib.sha256(
                lift_bytes).hexdigest()
            refs = {
                "vlm_observation": str(report_path),
                ("raw_lift_visual" if result.evidence_kind == "raw_visual"
                 else "key_wrist_check"): str(report_path),
                "lift_execution": str(lift_file),
            }
            self.observe_stage(
                "grasp_success", result.grasp_success,
                timestamp_s=completed, evidence_refs=refs)
        return result

    def prepare_measured_lift_chain(
        self, *, planner, pickup_execution_log: Path,
        joint_sample: LiveRobotState, bounds: SurfaceDeviationBounds,
        limits: PathAuditLimits, max_state_age_s: float,
        max_post_squeeze_arm_drift_rad: float,
        max_post_squeeze_hand_drift_raw: float,
        max_arm_hand_skew_s: float,
        max_hand_command_error_raw: float,
        max_arm_velocity_rad_s: float,
        axial_waypoint_step_m: float,
    ):
        """Save one full-chain replan from measured squeeze, without motion.

        A passing result remains a geometric hypothesis until a commissioned
        held-lift controller executes it and cameras assess the key after lift.
        """
        from .measured_lift_preflight import (
            plan_measured_lift_chain, verify_measured_lift_chain,
            write_measured_lift_chain,
        )

        if (self._attempt_dir is None or
                (self._measured_lift_preflight is not None and
                 self._measured_lift_preflight.status ==
                 "sampled_measured_chain_pass")):
            raise ValueError("measured lift is absent or already preflighted")
        result = plan_measured_lift_chain(
            runner=self, planner=planner,
            pickup_execution_log=pickup_execution_log,
            joint_sample=joint_sample, bounds=bounds, limits=limits,
            max_state_age_s=max_state_age_s,
            max_post_squeeze_arm_drift_rad=max_post_squeeze_arm_drift_rad,
            max_post_squeeze_hand_drift_raw=max_post_squeeze_hand_drift_raw,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s,
            axial_waypoint_step_m=axial_waypoint_step_m)
        output = (self._attempt_dir / "measured_lift_preflights" /
                  f"{self._measured_lift_index:03d}")
        write_measured_lift_chain(result, output)
        report = output / "report.json"
        verify_measured_lift_chain(report, expected=result)
        self._measured_lift_preflight = result
        self._measured_lift_report_path = report
        self._measured_lift_report_sha256 = hashlib.sha256(
            report.read_bytes()).hexdigest()
        self._measured_lift_index += 1
        return result

    def prepare_observed_lift_label(
        self, *, planner, after_observation: KeyPoseObservation,
        after_key_evidence_dir: Path, joint_sample: LiveRobotState,
        backend: ImageVLM, lift_execution_log_path: Path,
        decision_timestamp_s: float, max_state_skew_s: float,
        max_phase_skew_s: float, max_lift_observation_gap_s: float,
        min_center_rise_m: float, max_arm_hand_skew_s: float,
        max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
        minimum_visual_views: int = 2,
    ) -> LiftCheckpoint:
        """Persist paired lift VLM evidence and record only a decisive label.

        The execution log supplies the physical completion time; its content
        remains an external controller assertion. Unknown visual/pose
        evidence is saved but leaves the attempt unlabelled for re-observation.
        """
        if (self.current_decision().action != "await_lift_observation" or
                self._attempt is None or self._attempt_dir is None or
                self._preflight is None or self._key_evidence_dir is None or
                after_observation.capture_id in self._lift_assessed_capture_ids):
            raise ValueError("lift assessment needs a new held-key capture for this attempt")
        lift_file, lift_bytes, completed = self._read_lift_execution(
            lift_execution_log_path)
        prior = self.postlift_candidate_pose_prior(
            planner=planner, joint_sample=joint_sample,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s)
        result = assess_lift_checkpoint(
            mode=self.mode, shared_root=self.shared_root,
            calibration=self.calibration,
            attempt_id=self._attempt.attempt_id,
            candidate_id=self._attempt.candidate_id,
            attempt_started_at_s=self._attempt.started_at_s,
            lift_completed_at_s=completed,
            decision_timestamp_s=decision_timestamp_s,
            before_capture_id=self._preflight.key_observation_id,
            before_pose_world=self._preflight.key_pose_world,
            before_bundle=self._key_evidence_dir,
            after_observation=after_observation,
            after_bundle=after_key_evidence_dir,
            joint_sample=joint_sample,
            expected_candidate_prior_world=prior,
            backend=backend, max_state_skew_s=max_state_skew_s,
            max_phase_skew_s=max_phase_skew_s,
            max_lift_observation_gap_s=max_lift_observation_gap_s,
            min_center_rise_m=min_center_rise_m,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s,
            minimum_visual_views=minimum_visual_views)
        return self._save_lift_assessment(
            result=result, lift_file=lift_file,
            lift_bytes=lift_bytes, completed=completed)

    def prepare_raw_lift_label(
        self, *, after_raw_evidence_dir: Path,
        joint_sample: LiveRobotState, backend: ImageVLM,
        lift_execution_log_path: Path, decision_timestamp_s: float,
        max_phase_skew_s: float, max_lift_observation_gap_s: float,
        max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
        max_arm_velocity_rad_s: float,
        minimum_visual_views: int = 2,
    ) -> LiftCheckpoint:
        """Observe visible held/miss/slip without requiring post-lift FoundPose.

        A raw visual success unlocks *relation calibration review*, not a
        nominal BODex transfer. The existing transfer planner still needs an
        independently bounded key–hand relation.
        """
        if (self.current_decision().action != "await_lift_observation" or
                self._attempt is None or self._attempt_dir is None or
                self._preflight is None or self._key_evidence_dir is None):
            raise ValueError("raw lift needs an active selected grasp attempt")
        after_root = Path(after_raw_evidence_dir).expanduser().resolve()
        after = verify_raw_camera_capture(after_root, phase="after_lift")
        if (after["capture_id"] in self._lift_assessed_capture_ids or
                after["capture_id"] == self._preflight.key_observation_id):
            raise ValueError("raw lift needs a new post-attempt camera capture")
        lift_file, lift_bytes, completed = self._read_lift_execution(
            lift_execution_log_path)
        result = assess_raw_lift_checkpoint(
            mode=self.mode, attempt_id=self._attempt.attempt_id,
            candidate_id=self._attempt.candidate_id,
            attempt_started_at_s=self._attempt.started_at_s,
            lift_completed_at_s=completed,
            decision_timestamp_s=decision_timestamp_s,
            before_capture_id=self._preflight.key_observation_id,
            before_bundle=self._key_evidence_dir, after_bundle=after_root,
            joint_sample=joint_sample, backend=backend,
            max_phase_skew_s=max_phase_skew_s,
            max_lift_observation_gap_s=max_lift_observation_gap_s,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s,
            minimum_visual_views=minimum_visual_views)
        return self._save_lift_assessment(
            result=result, lift_file=lift_file,
            lift_bytes=lift_bytes, completed=completed)

    def postlift_candidate_pose_prior(
        self, *, planner, joint_sample: LiveRobotState,
        max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
        max_arm_velocity_rad_s: float,
    ) -> np.ndarray:
        """Provide a loose first-lift FoundPose prior from measured wrist + v8.

        This is a search prior, never an achieved hand/key relation. The
        separately admitted post-lift image must replace it before transfer.
        """
        if (self._attempt is None or self._preflight is None or
                self._preflight.selected_candidate_key is None or
                self._attempt.candidate_id !=
                "/".join(self._preflight.selected_candidate_key) or
                self._attempt.failure_code is not None or
                self._attempt.labels["grasp_success"] is False or
                self._attempt.labels["preinsert_reached"] is not None or
                self.current_decision().action not in {
                    "await_lift_observation",
                    "postlift_observed_preflight_required",
                    "held_relation_evidence_required"}):
            raise ValueError("candidate pose prior needs an active selected lift")
        if not isinstance(joint_sample, LiveRobotState):
            raise TypeError("post-lift prior needs measured robot feedback")
        joint_sample.validate(
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s)
        earliest = (self._attempt.events[-1]["timestamp_s"]
                    if self._attempt.events else self._attempt.started_at_s)
        if joint_sample.sample_timestamp_s <= max(
                earliest, self._capture_interval_end_s):
            raise ValueError("candidate pose prior predates this physical attempt")
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
                self.current_decision().action not in {
                    "postlift_observed_preflight_required",
                    "held_relation_evidence_required"} or
                self._attempt_dir is None or
                self._attempt is None or self._preflight is None or
                not isinstance(key_observation, KeyPoseObservation) or
                not isinstance(joint_sample, LiveRobotState) or
                key_observation.phase != "held_postlift"):
            raise ValueError("post-lift replan needs one first held-key observation")
        if (self._lift_report_path is None or
                not self._lift_report_path.is_file() or
                self._lift_report_sha256 is None or
                hashlib.sha256(
                    self._lift_report_path.read_bytes()).hexdigest() !=
                self._lift_report_sha256):
            raise ValueError("post-lift replan needs unchanged lift evidence")
        if isinstance(self._lift_checkpoint, LiftCheckpoint):
            verify_lift_checkpoint(self._lift_report_path)
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

    def prepare_bounded_postlift_transfer(
        self, *, planner, physical_calibration_path: Path,
        joint_sample: LiveRobotState, bounds: SurfaceDeviationBounds,
        max_state_age_s: float, max_arm_hand_skew_s: float,
        max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
        max_postlift_arm_drift_rad: float,
        max_postlift_hand_drift_raw: float,
        max_calibration_hand_excess_rad: float,
        limits: PathAuditLimits, axial_waypoint_step_m: float,
    ) -> BoundedPostLiftPreflight:
        """Replan with an independently calibrated grasp when the key is hidden.

        This is a sampled geometry preflight. It does not authorize transfer;
        physical calibration authenticity, future bounds and the external
        controller must be commissioned separately.
        """
        if (self.current_decision().action !=
                "held_relation_evidence_required" or
                self._attempt is None or self._attempt_dir is None or
                self._preflight is None or
                not isinstance(self._lift_checkpoint, LiftCheckpoint) or
                self._lift_checkpoint.evidence_kind != "raw_visual" or
                self._lift_checkpoint.grasp_success is not True or
                self._lift_report_path is None or
                self._lift_report_sha256 is None or
                not self._lift_report_path.is_file() or
                hashlib.sha256(self._lift_report_path.read_bytes()).hexdigest()
                != self._lift_report_sha256):
            raise ValueError("bounded post-lift plan needs this raw lift checkpoint")
        result = plan_bounded_postlift_transfer(
            planner=planner, trial=self._preflight, attempt=self._attempt,
            calibration=self.calibration, catalog=self.catalog,
            mode=self.mode, shared_root=self.shared_root,
            raw_lift=self._lift_checkpoint,
            raw_lift_report_path=self._lift_report_path,
            physical_calibration_path=physical_calibration_path,
            joint_sample=joint_sample, bounds=bounds,
            max_state_age_s=max_state_age_s,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s,
            max_postlift_arm_drift_rad=max_postlift_arm_drift_rad,
            max_postlift_hand_drift_raw=max_postlift_hand_drift_raw,
            max_calibration_hand_excess_rad=
            max_calibration_hand_excess_rad,
            limits=limits, axial_waypoint_step_m=axial_waypoint_step_m)
        output = (self._attempt_dir / "postlift_preflights" /
                  f"{self._postlift_index:03d}")
        write_bounded_postlift_preflight(result, output)
        report = output / "report.json"
        selected = select_pose_candidates(
            self.catalog, expected_mode=self.mode,
            tabletop_pose_stem=self._attempt.tabletop_pose_stem)
        matches = [row for row in selected["candidates"]
                   if tuple(row["key"]) == result.candidate_key]
        if len(matches) != 1:
            raise ValueError("selected bounded grasp is no longer in the catalogue")
        verify_bounded_postlift_preflight(
            report, mode=self.mode, shared_root=self.shared_root,
            candidate_dir=matches[0]["candidate_dir"])
        self._write_exclusive(output / "physical_relation_binding.json", {
            "schema": "precision_insertion_bounded_relation_binding_v1",
            "physical_calibration_path": str(result.relation.calibration_path),
            "physical_calibration_sha256": (
                result.relation.calibration_sha256),
            "raw_lift_checkpoint_path": str(result.lift_checkpoint_path),
            "raw_lift_checkpoint_sha256": result.lift_checkpoint_sha256,
            "postlift_report_sha256": hashlib.sha256(
                report.read_bytes()).hexdigest(),
            "session_calibration_sha256": self.session_sha256,
            "catalog_sha256": self.catalog_sha256,
            "robot_ready": False,
        })
        self._postlift_preflight = result
        self._postlift_report_path = report
        self._postlift_report_sha256 = hashlib.sha256(
            report.read_bytes()).hexdigest()
        self._postlift_index += 1
        return result

    def prepare_observed_preinsert_label(
        self, *, raw_bundle: Path, transfer_execution_path: Path,
        joint_sample: LiveRobotState, backend: ImageVLM,
        max_capture_skew_s: float, max_joint_frame_skew_s: float,
        max_transfer_observation_gap_s: float,
        max_hand_translation_error_m: float,
        max_hand_rotation_error_deg: float,
        max_arm_hand_skew_s: float,
        max_hand_command_error_raw: float,
        max_arm_velocity_rad_s: float,
        renderer_factory=None, robot_loader=None,
    ) -> PreinsertCheckpoint:
        """Save an observed arrival assessment; never command the transfer.

        A positive checkpoint is required by ``observe_stage`` before an
        insertion attempt can carry ``preinsert_reached=True``. It remains an
        external-evidence result, not a commissioned motion or contact gate.
        """
        if (self._attempt is None or self._attempt_dir is None or
                self._attempt.labels["grasp_success"] is not True or
                self._attempt.labels["preinsert_reached"] is not None or
                self._postlift_preflight is None or
                self._postlift_report_path is None or
                self._postlift_report_sha256 is None or
                self._postlift_preflight.status !=
                "sampled_postlift_preflight_pass" or
                not self._postlift_report_path.is_file() or
                hashlib.sha256(self._postlift_report_path.read_bytes()).hexdigest()
                != self._postlift_report_sha256 or
                _digest(self.calibration.record) != self.session_sha256):
            raise ValueError("preinsert assessment needs the same passing post-lift plan")
        raw_root = Path(raw_bundle).expanduser().resolve()
        raw_capture = verify_raw_camera_capture(raw_root, phase="preinsert")
        rows = raw_capture["frame_evidence"]
        signature = tuple(sorted((serial, row["frame_id"],
                                  row["image_sha256"])
                                 for serial, row in rows.items()))
        first_exposure = min(row["timestamp_s"] - row["max_error_s"]
                             for row in rows.values())
        last_exposure = max(row["timestamp_s"] + row["max_error_s"]
                            for row in rows.values())
        if (signature in self._preinsert_frame_signatures or
                first_exposure <= self._preinsert_latest_exposure_s):
            raise ValueError("preinsert reassessment needs newer camera exposures")
        kwargs = {}
        if renderer_factory is not None:
            kwargs["renderer_factory"] = renderer_factory
        if robot_loader is not None:
            kwargs["robot_loader"] = robot_loader
        result = assess_preinsert_checkpoint(
            attempt=self._attempt, postlift=self._postlift_preflight,
            postlift_report_path=self._postlift_report_path,
            calibration=self.calibration, shared_root=self.shared_root,
            mode=self.mode, raw_bundle=raw_root,
            transfer_execution_path=transfer_execution_path,
            joint_sample=joint_sample, backend=backend,
            max_capture_skew_s=max_capture_skew_s,
            max_joint_frame_skew_s=max_joint_frame_skew_s,
            max_transfer_observation_gap_s=max_transfer_observation_gap_s,
            max_hand_translation_error_m=max_hand_translation_error_m,
            max_hand_rotation_error_deg=max_hand_rotation_error_deg,
            max_arm_hand_skew_s=max_arm_hand_skew_s,
            max_hand_command_error_raw=max_hand_command_error_raw,
            max_arm_velocity_rad_s=max_arm_velocity_rad_s,
            **kwargs)
        output = (self._attempt_dir / "preinsert_assessments" /
                  f"{self._preinsert_assessment_index:03d}")
        report_path = write_preinsert_checkpoint(result, output)
        verify_preinsert_checkpoint(report_path)
        self._preinsert_checkpoint = result
        self._preinsert_report_path = report_path
        self._preinsert_report_sha256 = hashlib.sha256(
            report_path.read_bytes()).hexdigest()
        self._preinsert_assessment_index += 1
        self._preinsert_frame_signatures.add(signature)
        self._preinsert_latest_exposure_s = last_exposure
        return result

    def observe_stage(
        self, stage: str, status: bool | None, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        if stage == "reorient_success":
            raise ValueError("use observe_repose_landing for the target-pose label")
        if self._attempt is not None and self._attempt.candidate_id is None:
            raise ValueError("repose and insertion attempt labels must stay separate")
        if stage == "grasp_success" and status is True:
            if (self._lift_checkpoint is None or
                    self._lift_checkpoint.grasp_success is not True or
                    self._attempt is None or
                    self._lift_checkpoint.attempt_id !=
                    self._attempt.attempt_id or
                    self._lift_checkpoint.candidate_id !=
                    self._attempt.candidate_id or
                    float(timestamp_s) !=
                    self._lift_checkpoint.lift_completed_at_s or
                    self._lift_report_path is None or
                    not isinstance(evidence_refs, Mapping) or
                    evidence_refs.get("vlm_observation") !=
                    str(self._lift_report_path) or
                    evidence_refs.get(
                        "raw_lift_visual" if
                        getattr(self._lift_checkpoint, "evidence_kind", None) ==
                        "raw_visual"
                        else "key_wrist_check") !=
                    str(self._lift_report_path) or
                    self._lift_execution_path is None or
                    evidence_refs.get("lift_execution") !=
                    str(self._lift_execution_path) or
                    not self._lift_execution_path.is_file() or
                    self._lift_execution_sha256 is None or
                    hashlib.sha256(
                        self._lift_execution_path.read_bytes()).hexdigest() !=
                    self._lift_execution_sha256 or
                    not self._lift_report_path.is_file() or
                    self._lift_report_sha256 is None or
                    hashlib.sha256(
                        self._lift_report_path.read_bytes()).hexdigest() !=
                    self._lift_report_sha256):
                raise ValueError("grasp success needs this attempt's bound lift checkpoint")
            if isinstance(self._lift_checkpoint, LiftCheckpoint):
                verify_lift_checkpoint(self._lift_report_path)
        if stage == "reset_success" and status is True:
            raise ValueError("use observe_reset_landing for verified reset success")
        if stage == "preinsert_reached" and status is True:
            if (self._attempt is None or
                    self._postlift_preflight is None or
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
            if (self._preinsert_checkpoint is None or
                    self._preinsert_checkpoint.preinsert_reached is not True or
                    self._preinsert_checkpoint.attempt_id !=
                    self._attempt.attempt_id or
                    self._preinsert_checkpoint.candidate_id !=
                    self._attempt.candidate_id or
                    self._preinsert_report_path is None or
                    evidence_refs.get("preinsert_checkpoint") !=
                    str(self._preinsert_report_path) or
                    not self._preinsert_report_path.is_file() or
                    self._preinsert_report_sha256 is None or
                    hashlib.sha256(
                        self._preinsert_report_path.read_bytes()).hexdigest() !=
                        self._preinsert_report_sha256):
                raise ValueError("preinsert arrival needs this attempt's observed checkpoint")
            checkpoint_record = verify_preinsert_checkpoint(
                self._preinsert_report_path)
            if (checkpoint_record["preinsert_reached"] is not True or
                    float(timestamp_s) <
                    checkpoint_record["observation_completed_at_s"] or
                    evidence_refs.get("trajectory") !=
                    checkpoint_record["transfer_execution_path"] or
                    evidence_refs.get("grasp_state") !=
                    checkpoint_record["transfer_execution_path"] or
                    evidence_refs.get("key_socket_pose") !=
                    str(self._preinsert_report_path) or
                    evidence_refs.get("preinsert_image") !=
                    str(Path(checkpoint_record["raw_bundle"]) /
                        "manifest.json")):
                raise ValueError("preinsert stage contradicts verified checkpoint")
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
        evidence_refs: Mapping[str, str], checkpoint_path: Path,
    ) -> AttemptRecord:
        """Record only an insertion result backed by this attempt's saved VLM."""
        path = Path(checkpoint_path).expanduser().resolve()
        report = verify_insertion_checkpoint(path)
        sources = json.loads(Path(report["metric_record_path"]).read_text(
            encoding="utf-8"))["source_records"]
        expected_refs = {
            "vlm_observation": str(path),
            "vlm_observation_sha256": hashlib.sha256(
                path.read_bytes()).hexdigest(),
            "guarded_execution": report["metric_record_path"],
            **{name: sources[name]["path"] for name in (
                "key_depth", "alignment", "force_trace", "grasp_state")},
        }
        if (self._attempt is None or
                report["attempt_id"] != self._attempt.attempt_id or
                report["candidate_id"] != self._attempt.candidate_id or
                report["session_calibration_sha256"] != self.session_sha256 or
                report["target_depth_m"] != self.mode.target_depth_m or
                report["decision_timestamp_s"] != float(timestamp_s) or
                not isinstance(evidence, InsertionEvidence) or
                json.loads(json.dumps(evidence.__dict__)) !=
                report["evidence"] or
                not isinstance(evidence_refs, Mapping) or
                any(evidence_refs.get(name) != value
                    for name, value in expected_refs.items())):
            raise ValueError("insertion label needs this attempt's bound checkpoint")
        preinsert_events = [event for event in self._attempt.events
                            if event["stage"] == "preinsert_reached" and
                            event["value"] is True]
        retry_events = [event for event in self._attempt.events
                        if event["stage"] == "xy_retry"]
        if len(preinsert_events) != 1:
            raise ValueError("insertion checkpoint has no observed pre-insertion hold")
        if not retry_events:
            before_root = Path(report["preinsert_bundle"])
            reference = (
                "key_socket_pose", before_root / "key_observation.json"
            ) if report["preinsert_capture_kind"] == "held_key_pose" else (
                "preinsert_image", before_root / "manifest.json")
            if preinsert_events[0]["evidence_refs"].get(
                    reference[0]) != str(reference[1]):
                raise ValueError("insertion checkpoint used another pre-insertion image")
        if retry_events and (
                report["preinsert_reached_at_s"] !=
                retry_events[-1]["timestamp_s"]):
            raise ValueError("retry insertion checkpoint predates the XY choice")
        return self._record(lambda row: row.record_insertion_evidence(
            evidence, timestamp_s=timestamp_s, evidence_refs=evidence_refs))

    def prepare_observed_insertion_label(
        self, *, preinsert_bundle: Path, final_bundle: Path,
        metric_record_path: Path, backend: ImageVLM,
        decision_timestamp_s: float, max_phase_skew_s: float,
        max_preinsert_age_s: float, max_final_observation_gap_s: float,
        minimum_visual_views: int = 2,
    ) -> dict:
        """Persist paired VLM frames and external metric sources, then label.

        This is a read-only observation path. Guarded motion, calibrated key
        depth and force samples must come from separately commissioned producers.
        Unknown visual/metric evidence is recorded as unknown, not success.
        """
        if (self._attempt is None or self._attempt_dir is None or
                self._attempt.candidate_id is None or
                self.current_decision().action not in {
                    "await_guarded_insertion_and_observation",
                    "await_retry_execution_and_observation"}):
            raise ValueError("no held-key insertion attempt awaits observation")
        preinsert_events = [event for event in self._attempt.events
                            if event["stage"] == "preinsert_reached" and
                            event["value"] is True]
        if len(preinsert_events) != 1:
            raise ValueError("insertion needs one observed pre-insertion hold")
        before_root = Path(preinsert_bundle).expanduser().resolve()
        is_key_capture = (before_root / "evidence_manifest.json").is_file()
        before_file = (before_root / "key_observation.json" if is_key_capture
                       else before_root / "manifest.json")
        before_record = json.loads(before_file.read_text(encoding="utf-8"))
        if is_key_capture and (
                before_record.get("key_object") != self.mode.key_object or
                before_record.get("family") != self.mode.family):
            raise ValueError("pre-insertion capture is for another key")
        retry_events = [event for event in self._attempt.events
                        if event["stage"] == "xy_retry"]
        if retry_events:
            latest_retry = retry_events[-1]["timestamp_s"]
            before_lower = min(
                row["timestamp_s"] - row["max_error_s"]
                for row in before_record["frame_evidence"].values())
            if before_lower <= latest_retry:
                raise ValueError("retry needs a fresh post-choice held-key frame")
        elif (preinsert_events[0]["evidence_refs"].get(
                "key_socket_pose" if is_key_capture else "preinsert_image") !=
              str(before_file)):
            raise ValueError("pre-insertion VLM image differs from arrival observation")
        reached_at = max(preinsert_events[0]["timestamp_s"],
                         retry_events[-1]["timestamp_s"] if retry_events else
                         preinsert_events[0]["timestamp_s"])
        report = assess_insertion_checkpoint(
            attempt_id=self._attempt.attempt_id,
            candidate_id=self._attempt.candidate_id,
            target_depth_m=self.mode.target_depth_m,
            session_calibration_sha256=self.session_sha256,
            preinsert_bundle=before_root, final_bundle=final_bundle,
            metric_record_path=metric_record_path,
            preinsert_reached_at_s=reached_at,
            decision_timestamp_s=decision_timestamp_s, backend=backend,
            max_phase_skew_s=max_phase_skew_s,
            max_preinsert_age_s=max_preinsert_age_s,
            max_final_observation_gap_s=max_final_observation_gap_s,
            minimum_visual_views=minimum_visual_views)
        output = (self._attempt_dir / "insertion_assessments" /
                  f"{self._insertion_assessment_index:03d}")
        path = write_insertion_checkpoint(report, output)
        verify_insertion_checkpoint(path)
        metric = json.loads(Path(metric_record_path).read_text(encoding="utf-8"))
        sources = metric["source_records"]
        refs = {
            "vlm_observation": str(path),
            "vlm_observation_sha256": hashlib.sha256(
                path.read_bytes()).hexdigest(),
            "key_depth": sources["key_depth"]["path"],
            "alignment": sources["alignment"]["path"],
            "force_trace": sources["force_trace"]["path"],
            "grasp_state": sources["grasp_state"]["path"],
            "guarded_execution": str(Path(metric_record_path).resolve()),
        }
        self.observe_insertion(
            InsertionEvidence(**report["evidence"]),
            timestamp_s=decision_timestamp_s, evidence_refs=refs,
            checkpoint_path=path)
        self._insertion_assessment_index += 1
        return report

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

    def prepare_unobserved_xy_diagnostic(
        self, *, joint_sample: LiveRobotState, frames,
        intrinsics_full: Mapping, extrinsics_full: Mapping,
        frame_request_id: int, frame_ids: Mapping[str, int],
        acquisition_metadata: Mapping, backend: ImageVLM,
        withdrawal_completed_at_s: float,
        withdrawal_evidence_path: Path,
        postlift_preflight_report_path: Path,
        decision_timestamp_s: float,
        limits: RetrySessionLimits,
    ) -> UnobservedXYDiagnostic:
        """Save unobserved-key VLM advice without recording an XY retry."""
        if (self.current_decision().action !=
                "guarded_withdrawal_then_xy_assessment" or
                self._attempt is None or self._preflight is None or
                self._attempt_dir is None):
            raise ValueError("XY diagnostic requires an observed failed insertion")
        result = assess_unobserved_xy_diagnostic(
            mode=self.mode, shared_root=self.shared_root,
            calibration=self.calibration, catalog=self.catalog,
            trial=self._preflight, attempt=self._attempt,
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
        return result

    def prepare_grounded_xy_diagnostic(
        self, *, joint_sample: LiveRobotState, frames,
        intrinsics_full: Mapping, extrinsics_full: Mapping,
        frame_request_id: int, frame_ids: Mapping[str, int],
        acquisition_metadata: Mapping, backend: ImageVLM,
        withdrawal_completed_at_s: float,
        withdrawal_evidence_path: Path,
        postlift_preflight_report_path: Path,
        decision_timestamp_s: float, limits: RetrySessionLimits,
        alignment_limits: AlignmentLimits,
        axis_reference_key_z_m: float | None = None,
    ) -> GroundedXYDiagnostic:
        """Save camera-grounded metric 1 mm advice without scheduling motion."""
        if (self.current_decision().action !=
                "guarded_withdrawal_then_xy_assessment" or
                self._attempt is None or self._preflight is None or
                self._attempt_dir is None):
            raise ValueError("grounded XY needs an observed failed insertion")
        result = assess_grounded_xy_diagnostic(
            mode=self.mode, shared_root=self.shared_root,
            calibration=self.calibration, catalog=self.catalog,
            trial=self._preflight, attempt=self._attempt,
            joint_sample=joint_sample, frames=frames,
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full,
            frame_request_id=frame_request_id, frame_ids=frame_ids,
            acquisition_metadata=acquisition_metadata, backend=backend,
            withdrawal_completed_at_s=withdrawal_completed_at_s,
            withdrawal_evidence_path=withdrawal_evidence_path,
            postlift_preflight_report_path=postlift_preflight_report_path,
            decision_timestamp_s=decision_timestamp_s, limits=limits,
            alignment_limits=alignment_limits,
            axis_reference_key_z_m=axis_reference_key_z_m)
        output = (self._attempt_dir / "xy_retry_assessments" /
                  f"{self._retry_assessment_index:03d}")
        write_retry_session_artifacts(result, frames, output)
        self._retry_assessment_index += 1
        return result

    def record_failure(
        self, code: str, *, timestamp_s: float,
        evidence_refs: Mapping[str, str],
    ) -> AttemptRecord:
        return self._record(lambda row: row.record_failure(
            code, timestamp_s=timestamp_s, evidence_refs=evidence_refs))
