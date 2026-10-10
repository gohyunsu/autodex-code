"""Occlusion-tolerant post-lift transfer from physical grasp calibration.

When the key is hidden by Inspire, a raw multi-view lift label can confirm
that it is held without yielding a 6D pose. Reuse a grasp-specific relation
measured during *independent physical pickups*, but only with a separately
commissioned future-trial surface bound and a new measured robot state.
Nothing here commands transfer or certifies insertion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np

from .assets import AssetPaths
from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import _load_mesh, screen_grasp_endpoint
from .geometry import validate_se3
from .lift_checkpoint import LiftCheckpoint, verify_lift_checkpoint
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .physical_grasp_calibration import verify_physical_held_relation
from .preflight import InsertionPreflight, plan_held_transfer_and_axial
from .records import AttemptRecord
from .targets import InsertionTargets, build_rigid_insertion_targets
from .trial_preflight import TrialPreflight, _canonical_sha256
from .uncertainty_margin import (
    SurfaceDeviationBounds, audit_sampled_uncertainty_margins,
    relation_rotation_surface_bound,
)
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wall_time() -> float:
    return time.time()


_CANDIDATE_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class BoundedHeldRelation:
    T_key_hand: np.ndarray
    calibration_path: Path
    calibration_sha256: str
    medoid_trial_id: str
    required_descriptive_key_surface_m: float
    hand_calibration_excess_rad: float
    maximum_hand_calibration_excess_rad: float
    surface_bounds: SurfaceDeviationBounds

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_bounded_held_relation_v1",
            "T_key_hand": self.T_key_hand.tolist(),
            "source": "physical_grasp_calibration_medoid_not_runtime_key_pose",
            "physical_calibration_path": str(self.calibration_path),
            "physical_calibration_sha256": self.calibration_sha256,
            "medoid_trial_id": self.medoid_trial_id,
            "required_descriptive_key_surface_m": (
                self.required_descriptive_key_surface_m),
            "hand_calibration_excess_rad": self.hand_calibration_excess_rad,
            "maximum_hand_calibration_excess_rad": (
                self.maximum_hand_calibration_excess_rad),
            "surface_bounds": {
                "key_surface_m": self.surface_bounds.key_surface_m,
                "hand_surface_m": self.surface_bounds.hand_surface_m,
                "source": self.surface_bounds.source,
            },
            "scope": "physical_pickup_medoid_plus_commissioned_bound_not_online_pose",
        }


@dataclass(frozen=True)
class BoundedPostLiftPreflight:
    status: str
    attempt_id: str
    candidate_key: tuple[str, str, str]
    session_calibration_sha256: str
    catalog_sha256: str
    trial_key_observation_id: str
    key_observation_id: str  # raw after-lift capture ID, not a 6D observation
    key_capture_timestamp_s: float  # raw lift decision, no key pose timestamp
    joint_timestamp_s: float
    live_start_q: np.ndarray
    joint_feedback: LiveRobotState
    lift_checkpoint_path: Path
    lift_checkpoint_sha256: str
    relation: BoundedHeldRelation
    endpoint_screen: dict | None
    targets: InsertionTargets | None
    planning: InsertionPreflight | None
    uncertainty_margin: dict | None

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_bounded_postlift_preflight_v1",
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
            "bounded_held_relation": self.relation.to_record(),
            "endpoint_screen": self.endpoint_screen,
            "targets": None if self.targets is None else self.targets.to_record(),
            "planning": None if self.planning is None else self.planning.to_record(),
            "uncertainty_margin": self.uncertainty_margin,
            "scope": "occluded_key_physical_relation_prior_not_observed_runtime_6d",
            "robot_ready": False,
        }


def plan_bounded_postlift_transfer(
    *, planner, trial: TrialPreflight, attempt: AttemptRecord,
    calibration, catalog: dict, mode: TaskMode, shared_root: Path,
    raw_lift: LiftCheckpoint, raw_lift_report_path: Path,
    physical_calibration_path: Path,
    joint_sample: LiveRobotState, bounds: SurfaceDeviationBounds,
    max_state_age_s: float, max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
    max_postlift_arm_drift_rad: float, max_postlift_hand_drift_raw: float,
    max_calibration_hand_excess_rad: float,
    limits: PathAuditLimits, axial_waypoint_step_m: float,
    screen: Callable = screen_grasp_endpoint,
) -> BoundedPostLiftPreflight:
    """Replan transfer without claiming a newly observed key pose.

    The physical calibration medoid is only a relation hypothesis. Its
    descriptive sample envelope is a necessary *lower* bound on the supplied
    future surface error; it is never itself a sufficient worst-case bound.
    """
    thresholds = (
        max_state_age_s, max_arm_hand_skew_s,
        max_hand_command_error_raw, max_arm_velocity_rad_s,
        max_postlift_arm_drift_rad, max_postlift_hand_drift_raw,
        max_calibration_hand_excess_rad, axial_waypoint_step_m)
    if (not all(type(v) in (int, float) and math.isfinite(v) and v > 0
                for v in thresholds) or axial_waypoint_step_m > .005):
        raise ValueError("bounded transfer needs positive commissioned limits")
    bounds.validate()
    limits.validate()
    root = Path(shared_root).expanduser().resolve()
    if (not isinstance(trial, TrialPreflight) or
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
        raise ValueError("bounded transfer needs this grasp's positive raw lift")
    lift_file = Path(raw_lift_report_path).expanduser().resolve()
    lift_saved = verify_lift_checkpoint(lift_file)
    if lift_saved != raw_lift.to_record():
        raise ValueError("raw lift checkpoint differs from selected evidence")
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if (Path(catalog["shared_root"]).expanduser().resolve() != root or
            trial.catalog_sha256 != _canonical_sha256(catalog) or
            trial.session_calibration_sha256 !=
            _canonical_sha256(calibration.record) or
            attempt.session_calibration_sha256 !=
            trial.session_calibration_sha256):
        raise ValueError("bounded transfer has a changed catalogue or session")
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
    calibration_file = Path(physical_calibration_path).expanduser().resolve()
    physical = json.loads(calibration_file.read_text(encoding="utf-8"))
    verify_physical_held_relation(
        record=physical, mode=mode, shared_root=root,
        candidate_key=key, candidate_dir=candidate_dir)
    relation = validate_se3(physical["T_key_hand_medoid"],
                            name="physically calibrated T_key_hand medoid")
    key_mesh = _load_mesh(AssetPaths(root, mode).raw_mesh(mode.key_object))
    descriptive = relation_rotation_surface_bound(
        T_key_hand=relation, key_vertices=np.asarray(key_mesh.vertices),
        relation_translation_bound_m=physical[
            "descriptive_translation_envelope_m"],
        relation_rotation_bound_deg=physical[
            "descriptive_rotation_envelope_deg"])
    if bounds.key_surface_m < descriptive:
        raise ValueError("future key surface bound is smaller than observed scatter")
    if (not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("bounded transfer needs measured post-lift joints")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    hand_range = physical["hand_q_range"]
    lower = np.asarray(hand_range["minimum"], dtype=np.float64)
    upper = np.asarray(hand_range["maximum"], dtype=np.float64)
    if (lower.shape != (6,) or upper.shape != (6,) or
            not np.all(np.isfinite(lower)) or
            not np.all(np.isfinite(upper)) or
            np.any(lower > upper)):
        raise ValueError("physical grasp calibration hand range is invalid")
    held = joint_sample.full_q[7:]
    hand_excess = float(np.max(np.maximum(
        np.maximum(lower - held, held - upper), 0.0)))
    if hand_excess > max_calibration_hand_excess_rad:
        raise ValueError("measured Inspire pose is outside calibrated grasp range")
    bounded_relation = BoundedHeldRelation(
        relation, calibration_file, _sha(calibration_file),
        physical["medoid_trial_id"], descriptive, hand_excess,
        max_calibration_hand_excess_rad, bounds)
    now = _wall_time()
    if (joint_sample.sample_timestamp_s <=
            raw_lift.decision_timestamp_s or
            joint_sample.sample_timestamp_s > now or
            now - joint_sample.sample_timestamp_s > max_state_age_s or
            float(np.max(np.abs(
                joint_sample.full_q[:7] -
                raw_lift.joint_sample.full_q[:7]))) >
            max_postlift_arm_drift_rad or
            float(np.max(np.abs(
                joint_sample.hand_raw_measured -
                raw_lift.joint_sample.hand_raw_measured))) >
            max_postlift_hand_drift_raw):
        raise ValueError("post-lift measured hold moved or became stale")
    endpoint = screen(
        shared_root=root, mode=mode, candidate_dir=candidate_dir,
        minimum_hand_clearance_m=catalog["minimum_hand_clearance_m"],
        T_key_hand_override=relation,
        hand_poses_override={"measured_post_lift": joint_sample.full_q[7:]},
        override_source="physical_grasp_medoid_plus_measured_Inspire")
    if (not np.allclose(validate_se3(endpoint.get("T_key_hand"),
                                    name="bounded endpoint relation"),
                        relation, atol=1e-8, rtol=0) or
            not np.allclose(endpoint.get("xy_offset_socket_m"),
                            [0., 0.], atol=1e-12, rtol=0) or
            not np.allclose(endpoint.get("hold_pose_screens", {}).get(
                "measured_post_lift", {}).get("hand_q"),
                joint_sample.full_q[7:], atol=1e-8, rtol=0)):
        raise ValueError("bounded endpoint used another grasp or finger state")

    def result(status: str, *, targets=None, planning=None,
               margin=None) -> BoundedPostLiftPreflight:
        return BoundedPostLiftPreflight(
            status, attempt.attempt_id, key,
            trial.session_calibration_sha256, trial.catalog_sha256,
            trial.key_observation_id, raw_lift.after_capture_id,
            raw_lift.decision_timestamp_s, joint_sample.sample_timestamp_s,
            joint_sample.full_q.copy(), joint_sample,
            lift_file, _sha(lift_file), bounded_relation,
            endpoint, targets, planning, margin)

    if endpoint.get("endpoint_pass") is not True:
        return result("bounded_20mm_endpoint_rejected")
    targets = build_rigid_insertion_targets(
        mode=mode, shared_root=root, calibration=calibration,
        T_key_hand=relation)
    planning = plan_held_transfer_and_axial(
        planner=planner, trial_scene=trial.trial_scene,
        shared_root=root, calibration=calibration,
        targets=targets, start_q=joint_sample.full_q,
        held_hand_q=joint_sample.full_q[7:],
        held_hand_source="measured", limits=limits,
        axial_waypoint_step_m=axial_waypoint_step_m)
    if not planning.sampled_planning_pass:
        return result(planning.status, targets=targets, planning=planning)
    margin = audit_sampled_uncertainty_margins(
        mode=mode, endpoint=endpoint, targets=targets,
        planning=planning, bounds=bounds)
    return result(
        "sampled_postlift_preflight_pass" if margin[
            "sampled_margin_pass"] else "bounded_uncertainty_margin_rejected",
        targets=targets, planning=planning, margin=margin)


def write_bounded_postlift_preflight(
    result: BoundedPostLiftPreflight, output_dir: Path,
) -> Path:
    """Save an occluded-key plan without presenting it as observed 6D."""
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


def verify_bounded_postlift_preflight(
    report_path: Path, *, mode: TaskMode, shared_root: Path,
    candidate_dir: Path,
) -> dict:
    """Recheck the saved relation, exact inputs and planned trajectory bytes.

    This validates internal provenance, not authenticity of a physical
    calibration source or the claimed future-trial error bound.
    """
    path = Path(report_path).expanduser().resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or
            record.get("schema") != "precision_insertion_bounded_postlift_preflight_v1" or
            record.get("robot_ready") is not False or
            not isinstance(record.get("candidate_key"), list) or
            len(record["candidate_key"]) != 3 or
            any(not isinstance(part, str) or
                not _CANDIDATE_PART.fullmatch(part) or
                part in {".", ".."} for part in record["candidate_key"])):
        raise ValueError("invalid bounded post-lift report")
    key = tuple(record["candidate_key"])
    lift_source = Path(record["raw_lift_checkpoint_path"]).expanduser()
    if not lift_source.is_absolute():
        raise ValueError("bounded raw lift source must be absolute")
    lift_path = lift_source.resolve()
    lift = verify_lift_checkpoint(lift_path)
    if (_sha(lift_path) != record["raw_lift_checkpoint_sha256"] or
            lift.get("evidence_kind") != "raw_visual" or
            lift.get("grasp_success") is not True or
            lift.get("attempt_id") != record["attempt_id"] or
            lift.get("candidate_id") != "/".join(key) or
            lift.get("after_capture_id") != record["key_observation_id"] or
            lift.get("decision_timestamp_s") !=
            record["key_capture_timestamp_s"]):
        raise ValueError("bounded post-lift raw lift source changed")
    relation_record = record.get("bounded_held_relation")
    if (not isinstance(relation_record, dict) or
            relation_record.get("schema") !=
            "precision_insertion_bounded_held_relation_v1" or
            relation_record.get("source") !=
            "physical_grasp_calibration_medoid_not_runtime_key_pose"):
        raise ValueError("bounded post-lift relation provenance changed")
    calibration_source = Path(
        relation_record["physical_calibration_path"]).expanduser()
    if not calibration_source.is_absolute():
        raise ValueError("physical grasp calibration source must be absolute")
    calibration_path = calibration_source.resolve()
    if _sha(calibration_path) != relation_record["physical_calibration_sha256"]:
        raise ValueError("physical grasp calibration file changed")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    verify_physical_held_relation(
        record=calibration, mode=mode,
        shared_root=Path(shared_root).expanduser().resolve(),
        candidate_key=key, candidate_dir=candidate_dir)
    relation = validate_se3(relation_record.get("T_key_hand"),
                            name="saved physical medoid relation")
    if (not np.allclose(relation, calibration["T_key_hand_medoid"],
                        atol=1e-8, rtol=0) or
            relation_record.get("medoid_trial_id") !=
            calibration["medoid_trial_id"]):
        raise ValueError("bounded post-lift medoid changed")
    mesh = _load_mesh(AssetPaths(Path(shared_root), mode).raw_mesh(
        mode.key_object))
    descriptive = relation_rotation_surface_bound(
        T_key_hand=relation, key_vertices=np.asarray(mesh.vertices),
        relation_translation_bound_m=calibration[
            "descriptive_translation_envelope_m"],
        relation_rotation_bound_deg=calibration[
            "descriptive_rotation_envelope_deg"])
    bound_record = relation_record.get("surface_bounds", {})
    bounds = SurfaceDeviationBounds(
        bound_record.get("key_surface_m"),
        bound_record.get("hand_surface_m"), bound_record.get("source"))
    bounds.validate()
    maximum = float(relation_record["maximum_hand_calibration_excess_rad"])
    hand = np.asarray(record["joint_feedback"]["full_q"], dtype=np.float64)[7:]
    lower = np.asarray(calibration["hand_q_range"]["minimum"])
    upper = np.asarray(calibration["hand_q_range"]["maximum"])
    excess = float(np.max(np.maximum(np.maximum(lower - hand, hand - upper), 0.0)))
    if (not math.isfinite(maximum) or maximum <= 0 or
            not math.isclose(excess, relation_record[
                "hand_calibration_excess_rad"], abs_tol=1e-9) or
            excess > maximum or
            not math.isclose(descriptive, relation_record[
                "required_descriptive_key_surface_m"], abs_tol=1e-9) or
            bounds.key_surface_m < descriptive or
            not np.allclose(record["live_start_q"],
                            record["joint_feedback"]["full_q"],
                            atol=1e-8, rtol=0) or
            record["joint_timestamp_s"] !=
            record["joint_feedback"]["sample_timestamp_s"] or
            record["joint_timestamp_s"] <=
            record["key_capture_timestamp_s"]):
        raise ValueError("bounded post-lift hand or uncertainty record changed")
    planned = record.get("planned_trajectories")
    if planned is not None:
        if (planned != "planned_trajectories.npz" or
                _sha(path.parent / planned) !=
                record.get("planned_trajectories_sha256")):
            raise ValueError("bounded planned trajectory bytes changed")
        with np.load(path.parent / planned, allow_pickle=False) as trajectories:
            planning = record["planning"]
            for stage in ("transfer", "axial"):
                count = planning["sample_counts"][stage]
                if stage not in trajectories:
                    if count is not None:
                        raise ValueError("bounded planned trajectory is missing")
                    continue
                values = trajectories[stage]
                if (values.ndim != 2 or values.shape != (count, 13) or
                        not np.all(np.isfinite(values)) or
                        not np.allclose(values[:, 7:],
                                        planning["held_hand_q"],
                                        atol=1e-4, rtol=0)):
                    raise ValueError("bounded planned trajectory differs from report")
    if record.get("status") == "sampled_postlift_preflight_pass":
        if (planned is None or
                record.get("endpoint_screen", {}).get("endpoint_pass") is not True or
                record.get("planning", {}).get("sampled_planning_pass") is not True or
                record.get("uncertainty_margin", {}).get(
                    "sampled_margin_pass") is not True or
                not np.allclose(validate_se3(record["targets"]["T_key_hand"]),
                                relation, atol=1e-8, rtol=0) or
                not np.allclose(validate_se3(record["endpoint_screen"][
                    "T_key_hand"]), relation, atol=1e-8, rtol=0)):
            raise ValueError("bounded passing plan lacks its verified screens")
    return record
