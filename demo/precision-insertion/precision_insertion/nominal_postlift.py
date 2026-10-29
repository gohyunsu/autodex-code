"""Experimental post-lift replan using the v8 grasp relation as a hypothesis.

The raw lift label does not measure the occluded key's six-DoF pose.  This
route deliberately assumes the selected v8 key-to-hand transform and records
that assumption separately from observed and physically calibrated routes.
It is useful for studying repeatable squeeze bias with next-trial XY updates;
its sampled CAD/path pass is not evidence of the actual held-key transform.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Callable

import numpy as np

from .assets import AssetPaths
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import screen_grasp_endpoint
from .geometry import validate_se3
from .lift_checkpoint import LiftCheckpoint, verify_lift_checkpoint
from .live_robot_state import LiveRobotState
from .next_trial_xy import compensated_key_hand_relation
from .path_audit import PathAuditLimits
from .preflight import InsertionPreflight, plan_held_transfer_and_axial
from .records import AttemptRecord
from .targets import InsertionTargets, build_rigid_insertion_targets
from .trial_preflight import TrialPreflight, _canonical_sha256
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_CANDIDATE_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class NominalHeldRelation:
    T_key_hand: np.ndarray
    T_key_hand_command: np.ndarray
    wrist_source_path: Path
    wrist_source_sha256: str
    wrist_correction_socket_m: tuple[float, float]

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_nominal_held_relation_v1",
            "T_key_hand": self.T_key_hand.tolist(),
            "T_key_hand_command": self.T_key_hand_command.tolist(),
            "wrist_source_path": str(self.wrist_source_path),
            "wrist_source_sha256": self.wrist_source_sha256,
            "wrist_correction_socket_m": list(self.wrist_correction_socket_m),
            "source": "selected_v8_wrist_hypothesis_not_observed_key_pose",
            "scope": "experimental_repeatable_grasp_bias_not_physical_calibration",
        }


@dataclass(frozen=True)
class NominalPostLiftPreflight:
    status: str
    attempt_id: str
    candidate_key: tuple[str, str, str]
    session_calibration_sha256: str
    catalog_sha256: str
    trial_key_observation_id: str
    key_observation_id: str  # raw after-lift capture ID, not a 6D key pose
    key_capture_timestamp_s: float
    joint_timestamp_s: float
    live_start_q: np.ndarray
    joint_feedback: LiveRobotState
    lift_checkpoint_path: Path
    lift_checkpoint_sha256: str
    relation: NominalHeldRelation
    endpoint_screen: dict | None
    targets: InsertionTargets | None
    planning: InsertionPreflight | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_nominal_postlift_preflight_v1",
            "status": self.status,
            "attempt_id": self.attempt_id,
            "candidate_key": list(self.candidate_key),
            "session_calibration_sha256": self.session_calibration_sha256,
            "catalog_sha256": self.catalog_sha256,
            "trial_key_observation_id": self.trial_key_observation_id,
            "key_observation_id": self.key_observation_id,
            "key_capture_timestamp_s": self.key_capture_timestamp_s,
            "joint_timestamp_s": self.joint_timestamp_s,
            "live_start_q": self.live_start_q.tolist(),
            "joint_feedback": self.joint_feedback.to_record(),
            "raw_lift_checkpoint_path": str(self.lift_checkpoint_path),
            "raw_lift_checkpoint_sha256": self.lift_checkpoint_sha256,
            "nominal_held_relation": self.relation.to_record(),
            "endpoint_screen": self.endpoint_screen,
            "targets": None if self.targets is None else self.targets.to_record(),
            "planning": None if self.planning is None else self.planning.to_record(),
            "scope": "experimental_v8_grasp_hypothesis_not_observed_runtime_6d",
            "robot_ready": False,
        }


def plan_nominal_postlift_transfer(
    *, planner, trial: TrialPreflight, attempt: AttemptRecord,
    calibration, catalog: dict, mode: TaskMode, shared_root: Path,
    raw_lift: LiftCheckpoint, raw_lift_report_path: Path,
    joint_sample: LiveRobotState,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float, limits: PathAuditLimits,
    axial_waypoint_step_m: float,
    screen: Callable = screen_grasp_endpoint,
) -> NominalPostLiftPreflight:
    """Replan from measured joints while *assuming* the selected v8 relation.

    No physical grasp calibration or post-lift key-pose observation is inferred.
    The selected candidate's offline 20 mm endpoint screen is reused. A
    pending XY correction is screened against the candidate's nominal hold
    poses. Measured hand joints still seed the actual transfer planner, but
    they are not treated as a newly observed key-in-hand transform.
    """
    thresholds = (max_arm_hand_skew_s, max_hand_command_error_raw,
                  max_arm_velocity_rad_s, axial_waypoint_step_m)
    if (not all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in thresholds) or axial_waypoint_step_m > .005):
        raise ValueError("nominal transfer needs positive supplied limits")
    limits.validate()
    root = Path(shared_root).expanduser().resolve()
    if (mode.family != "cylinder" or
            not isinstance(trial, TrialPreflight) or
            trial.status != "sampled_planning_pass" or
            trial.selected_candidate_key is None or
            trial.insertion_plan is None or
            not trial.insertion_plan.sampled_planning_pass or
            trial.pickup_plan is None or
            not isinstance(attempt, AttemptRecord) or attempt.mode != mode or
            attempt.candidate_id != "/".join(trial.selected_candidate_key) or
            attempt.tabletop_pose_stem != trial.pose_class["stem"] or
            attempt.labels["grasp_success"] is not True or
            attempt.labels["preinsert_reached"] is not None or
            attempt.labels["insertion_success"] is not None or
            attempt.failure_code is not None or
            not isinstance(raw_lift, LiftCheckpoint) or
            raw_lift.evidence_kind != "raw_visual" or
            raw_lift.grasp_success is not True or
            raw_lift.attempt_id != attempt.attempt_id or
            raw_lift.candidate_id != attempt.candidate_id or
            raw_lift.after_capture_id == trial.key_observation_id or
            not any(event["stage"] == "grasp_success" and
                    event["value"] is True and
                    event["timestamp_s"] == raw_lift.lift_completed_at_s
                    for event in attempt.events)):
        raise ValueError("nominal transfer needs this cylinder's positive raw lift")
    lift_file = Path(raw_lift_report_path).expanduser().resolve()
    if verify_lift_checkpoint(lift_file) != raw_lift.to_record():
        raise ValueError("raw lift checkpoint differs from selected evidence")
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if (Path(catalog["shared_root"]).expanduser().resolve() != root or
            trial.catalog_sha256 != _canonical_sha256(catalog) or
            trial.session_calibration_sha256 !=
            _canonical_sha256(calibration.record) or
            attempt.session_calibration_sha256 !=
            trial.session_calibration_sha256):
        raise ValueError("nominal transfer has a changed catalogue or session")
    validated_frozen_socket_pose(
        mode=mode, shared_root=root, calibration=calibration)
    build_held_scene_from_trial(
        trial_scene=trial.trial_scene, calibration=calibration)
    selected = select_pose_candidates(
        catalog, expected_mode=mode,
        tabletop_pose_stem=attempt.tabletop_pose_stem)
    key = trial.selected_candidate_key
    matches = [row for row in selected["candidates"]
               if tuple(row["key"]) == key]
    if selected["status"] != "candidates_available" or len(matches) != 1:
        raise ValueError("selected v8 grasp is no longer endpoint-eligible")
    candidate_dir = Path(matches[0]["candidate_dir"]).expanduser().resolve()
    canonical_dir = (AssetPaths(root, mode).candidate_dir /
                     Path(*key)).resolve()
    if candidate_dir != canonical_dir:
        raise ValueError("selected nominal grasp differs from canonical v8 pool")
    wrist_file = (candidate_dir / "wrist_se3.npy").resolve()
    relation = validate_se3(np.load(wrist_file, allow_pickle=False),
                            name="selected nominal v8 T_key_hand")
    command = compensated_key_hand_relation(
        mode=mode, shared_root=root, T_key_hand=relation,
        wrist_correction_socket_m=attempt.wrist_correction_socket_m)
    nominal = NominalHeldRelation(
        relation, command, wrist_file, _sha(wrist_file),
        tuple(attempt.wrist_correction_socket_m))
    if (not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("nominal transfer needs measured post-lift joints")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    if joint_sample.sample_timestamp_s <= raw_lift.decision_timestamp_s:
        raise ValueError("post-lift measured hold predates the lift observation")
    if not any(attempt.wrist_correction_socket_m):
        endpoint = matches[0]["endpoint_report"]
    else:
        endpoint = screen(
            shared_root=root, mode=mode, candidate_dir=candidate_dir,
            minimum_hand_clearance_m=catalog["minimum_hand_clearance_m"],
            T_key_hand_override=command,
            override_source="nominal_v8_relation_with_pending_xy")
    if (not np.allclose(validate_se3(endpoint.get("T_key_hand"),
                                    name="nominal endpoint relation"),
                        command, atol=1e-8, rtol=0) or
            not np.allclose(endpoint.get("xy_offset_socket_m"),
                            [0., 0.], atol=1e-12, rtol=0)):
        raise ValueError("nominal endpoint used another grasp or XY target")

    def result(status: str, *, targets=None,
               planning=None) -> NominalPostLiftPreflight:
        return NominalPostLiftPreflight(
            status, attempt.attempt_id, key,
            trial.session_calibration_sha256, trial.catalog_sha256,
            trial.key_observation_id, raw_lift.after_capture_id,
            raw_lift.decision_timestamp_s, joint_sample.sample_timestamp_s,
            joint_sample.full_q.copy(), joint_sample,
            lift_file, _sha(lift_file), nominal, endpoint, targets, planning)

    if endpoint.get("endpoint_pass") is not True:
        return result("nominal_20mm_endpoint_rejected")
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=command)
    planning = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial.trial_scene,
        shared_root=root, calibration=calibration,
        targets=targets, start_q=joint_sample.full_q,
        held_hand_q=joint_sample.full_q[7:],
        held_hand_source="measured", limits=limits,
        axial_waypoint_step_m=axial_waypoint_step_m,
        exact_distances=False)
    return result(
        "sampled_postlift_preflight_pass" if planning.sampled_planning_pass
        else planning.status, targets=targets, planning=planning)


def write_nominal_postlift_preflight(
    result: NominalPostLiftPreflight, output_dir: Path,
) -> Path:
    """Save the nominal hypothesis and dense path with explicit provenance."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    record = result.to_record()
    if (result.planning is not None and
            result.planning.transfer_trajectory is not None):
        path = target / "planned_trajectories.npz"
        arrays = {"transfer": result.planning.transfer_trajectory}
        if result.planning.axial_trajectory is not None:
            arrays["axial"] = result.planning.axial_trajectory
        np.savez_compressed(path, **arrays)
        record["planned_trajectories"] = path.name
        record["planned_trajectories_sha256"] = _sha(path)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


def verify_nominal_postlift_preflight(
    report_path: Path, *, mode: TaskMode, shared_root: Path,
    candidate_dir: Path,
) -> dict:
    """Check source bytes/relations without treating the hypothesis as a pose."""
    path = Path(report_path).expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or
            record.get("schema") !=
            "precision_insertion_nominal_postlift_preflight_v1" or
            record.get("robot_ready") is not False or
            not isinstance(record.get("status"), str) or
            not record["status"] or
            not isinstance(record.get("candidate_key"), list) or
            len(record["candidate_key"]) != 3 or
            any(not isinstance(part, str) or
                _CANDIDATE_PART.fullmatch(part) is None or
                part in {".", ".."} for part in record["candidate_key"])):
        raise ValueError("invalid nominal post-lift report")
    key = tuple(record["candidate_key"])
    expected_dir = (AssetPaths(Path(shared_root).expanduser().resolve(), mode)
                    .candidate_dir / Path(*key)).resolve()
    if Path(candidate_dir).expanduser().resolve() != expected_dir:
        raise ValueError("nominal grasp differs from canonical v8 candidate")
    source = (expected_dir / "wrist_se3.npy").resolve()
    relation_record = record.get("nominal_held_relation")
    if (not isinstance(relation_record, dict) or
            relation_record.get("schema") !=
            "precision_insertion_nominal_held_relation_v1" or
            relation_record.get("source") !=
            "selected_v8_wrist_hypothesis_not_observed_key_pose" or
            relation_record.get("wrist_source_path") != str(source) or
            relation_record.get("wrist_source_sha256") != _sha(source)):
        raise ValueError("nominal grasp source changed")
    nominal = validate_se3(np.load(source, allow_pickle=False),
                           name="saved v8 grasp relation")
    if not np.allclose(validate_se3(relation_record.get("T_key_hand")),
                       nominal, atol=1e-8, rtol=0):
        raise ValueError("nominal key-hand relation differs from v8 candidate")
    command = compensated_key_hand_relation(
        mode=mode, shared_root=shared_root, T_key_hand=nominal,
        wrist_correction_socket_m=relation_record.get(
            "wrist_correction_socket_m"))
    if not np.allclose(validate_se3(
            relation_record.get("T_key_hand_command")),
            command, atol=1e-8, rtol=0):
        raise ValueError("nominal wrist correction differs from socket geometry")
    lift_path = Path(record["raw_lift_checkpoint_path"]).expanduser().resolve()
    lift = verify_lift_checkpoint(lift_path)
    if (_sha(lift_path) != record["raw_lift_checkpoint_sha256"] or
            lift.get("evidence_kind") != "raw_visual" or
            lift.get("grasp_success") is not True or
            lift.get("attempt_id") != record["attempt_id"] or
            lift.get("candidate_id") != "/".join(key) or
            lift.get("after_capture_id") != record["key_observation_id"] or
            lift.get("decision_timestamp_s") !=
            record["key_capture_timestamp_s"]):
        raise ValueError("nominal post-lift raw lift source changed")
    if (not np.allclose(record["live_start_q"],
                        record["joint_feedback"]["full_q"], atol=1e-8, rtol=0)
            or record["joint_timestamp_s"] !=
            record["joint_feedback"]["sample_timestamp_s"] or
            record["joint_timestamp_s"] <=
            record["key_capture_timestamp_s"]):
        raise ValueError("nominal post-lift measured start changed")
    planned = record.get("planned_trajectories")
    if planned is not None:
        archive = path.parent / planned
        if (planned != "planned_trajectories.npz" or
                _sha(archive) != record.get("planned_trajectories_sha256")):
            raise ValueError("nominal planned trajectory bytes changed")
        with np.load(archive, allow_pickle=False) as arrays:
            planning = record["planning"]
            for stage in ("transfer", "axial"):
                count = planning["sample_counts"][stage]
                if stage not in arrays:
                    if count is not None:
                        raise ValueError("nominal planned trajectory is missing")
                    continue
                values = arrays[stage]
                if (values.ndim != 2 or values.shape != (count, 13) or
                        not np.all(np.isfinite(values)) or
                        not np.allclose(values[:, 7:],
                                        planning["held_hand_q"],
                                        atol=1e-4, rtol=0)):
                    raise ValueError("nominal trajectory differs from report")
    if record.get("status") == "sampled_postlift_preflight_pass":
        if (planned is None or
                record.get("endpoint_screen", {}).get("endpoint_pass") is not True or
                record.get("planning", {}).get("sampled_planning_pass") is not True or
                not np.allclose(validate_se3(record["targets"]["T_key_hand"]),
                                command, atol=1e-8, rtol=0) or
                not np.allclose(validate_se3(record["endpoint_screen"][
                    "T_key_hand"]), command, atol=1e-8, rtol=0)):
            raise ValueError("nominal passing plan lacks its assumed CAD screens")
    return record
