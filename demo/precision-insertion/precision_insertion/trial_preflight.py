"""One fresh-key, pose-conditioned, candidate-by-candidate planning trial.

This is a read-only planning entry point. It neither captures the cameras nor
drives Franka/Inspire. It preserves AutoDex v8 grasp selection, reuses its
pickup planner, and requires the demo's insertion preflight for the chosen
candidate. No planning result is a physical grasp or insertion success label.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from autodex.utils.conversion import cart2se3

from .candidates import (planner_candidate_override, select_pose_candidates,
                         validate_catalog_session)
from .curobo_compat import install_curobo_planner_compat
from .config import TaskMode
from .endpoint import nominal_inspire_hold_poses
from .geometry import validate_se3
from .key_perception import KeyPoseObservation
from .path_audit import PathAuditLimits
from .planner_mode import cartesian_mode_from_planner
from .pose_selection import classify_key_tabletop_pose
from .preflight import InsertionPreflight, plan_insertion_after_pickup
from .repose_policy import assess_repose_options
from .targets import build_rigid_insertion_targets
from .world import build_trial_scene_from_session


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrialPreflight:
    status: str
    pose_class: dict
    attempted_candidates: tuple[dict, ...]
    selected_candidate_key: tuple[str, str, str] | None
    repose_target_stems: tuple[str, ...]
    insertion_plan: InsertionPreflight | None
    pickup_plan: Any | None
    trial_scene: dict
    key_observation_id: str
    key_capture_timestamp_s: float
    start_q_acquisition_timestamp_s: float
    max_key_state_skew_s: float
    key_pose_world: np.ndarray
    live_start_q: np.ndarray
    attempted_before_trial: tuple[tuple[str, str, str], ...]
    covered_scenes: tuple[int, ...]
    limits: PathAuditLimits
    max_pose_error_deg: float
    axial_waypoint_step_m: float
    max_candidate_attempts: int | None
    session_calibration_sha256: str
    catalog_sha256: str
    repose_assessment: dict | None = None
    cartesian_planner_mode: str = "default"

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_trial_preflight_v2",
            "status": self.status,
            "pose_class": self.pose_class,
            "attempted_candidates": list(self.attempted_candidates),
            "selected_candidate_key": (
                None if self.selected_candidate_key is None
                else list(self.selected_candidate_key)),
            "repose_target_stems": list(self.repose_target_stems),
            "repose_assessment": self.repose_assessment,
            "insertion_plan": (None if self.insertion_plan is None
                               else self.insertion_plan.to_record()),
            "key_observation_id": self.key_observation_id,
            "key_capture_timestamp_s": self.key_capture_timestamp_s,
            "start_q_acquisition_timestamp_s": self.start_q_acquisition_timestamp_s,
            "max_key_state_skew_s": self.max_key_state_skew_s,
            "key_pose_world": self.key_pose_world.tolist(),
            "live_start_q": self.live_start_q.tolist(),
            "attempted_before_trial": [list(key)
                                       for key in self.attempted_before_trial],
            "covered_scenes": list(self.covered_scenes),
            "limits": vars(self.limits).copy(),
            "max_pose_error_deg": self.max_pose_error_deg,
            "axial_waypoint_step_m": self.axial_waypoint_step_m,
            "max_candidate_attempts": self.max_candidate_attempts,
            "session_calibration_sha256": self.session_calibration_sha256,
            "catalog_sha256": self.catalog_sha256,
            "cartesian_planner_mode": self.cartesian_planner_mode,
            "scope": "read_only_planning_not_camera_or_robot_execution",
            "robot_ready": False,
        }


def write_trial_preflight_artifacts(
    result: TrialPreflight, output_dir: Path,
) -> Path:
    """Save a new planning report, frozen trial scene and any dense paths.

    The directory is created exclusively; no previous planning run is
    overwritten. A partial directory after an I/O error remains for audit.
    Saving trajectories does not authorize their physical replay.
    """
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    scene_bytes = (json.dumps(result.trial_scene, indent=2,
                              sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    (target / "trial_scene.json").write_bytes(scene_bytes)
    report = result.to_record()
    report["artifacts"] = {
        "trial_scene": "trial_scene.json",
        "trial_scene_sha256": hashlib.sha256(scene_bytes).hexdigest(),
    }
    if result.insertion_plan is not None:
        if result.pickup_plan is None:
            raise ValueError("insertion plan has no original AutoDex pickup plan")
        plan = result.insertion_plan
        if any(path is None for path in (
                plan.lift_trajectory, plan.transfer_trajectory,
                plan.axial_trajectory)):
            raise ValueError("selected insertion plan has missing trajectories")
        pickup = result.pickup_plan
        if (not pickup.success or result.selected_candidate_key is None or
                tuple(pickup.scene_info) != result.selected_candidate_key):
            raise ValueError("selected pickup plan differs from selected v8 grasp")
        pregrasp = np.asarray(pickup.pregrasp_pose, dtype=np.float64)
        grasp = np.asarray(pickup.grasp_pose, dtype=np.float64)
        wrist = np.asarray(pickup.wrist_se3, dtype=np.float64)
        if (pregrasp.shape != (6,) or grasp.shape != (6,) or
                wrist.shape != (4, 4) or
                not all(np.all(np.isfinite(array)) for array in
                        (pregrasp, grasp, wrist))):
            raise ValueError("selected pickup hand/wrist command is invalid")
        filename = "planned_trajectories.npz"
        path = target / filename
        np.savez_compressed(
            path,
            pickup_approach=np.asarray(pickup.traj, dtype=np.float64),
            pickup_pregrasp=pregrasp, pickup_grasp=grasp,
            pickup_wrist=wrist,
            held_lift=np.asarray(plan.lift_trajectory, dtype=np.float64),
            transfer=np.asarray(plan.transfer_trajectory, dtype=np.float64),
            axial=np.asarray(plan.axial_trajectory, dtype=np.float64),
            held_hand_q=np.asarray(plan.held_hand_q, dtype=np.float64),
        )
        report["artifacts"]["planned_trajectories"] = filename
        report["artifacts"]["planned_trajectories_sha256"] = (
            hashlib.sha256(path.read_bytes()).hexdigest())
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


def plan_fresh_key_trial(
    *, planner, mode: TaskMode, shared_root: Path, calibration,
    catalog: dict, key_pose_world: np.ndarray,
    key_observation_id: str, key_capture_timestamp_s: float,
    live_start_q: np.ndarray, start_q_acquisition_timestamp_s: float,
    max_key_state_skew_s: float, limits: PathAuditLimits,
    max_pose_error_deg: float, axial_waypoint_step_m: float,
    attempted: tuple[tuple[str, str, str], ...] = (),
    covered_scenes: tuple[int, ...] = (),
    max_candidate_attempts: int | None = None,
    max_reset_center_drift_m: float | None = None,
    max_reset_axis_tilt_deg: float | None = None,
    reset_candidate_root: Path | None = None,
    attempted_reset: tuple[tuple[int, str, str], ...] = (),
) -> TrialPreflight:
    """Plan one scene from an explicitly fresh pose, trying grasps in order.

    Candidate exhaustion can suggest other tabletop stems, but never executes
    repose/reorientation. A stale or incomplete catalogue cannot trigger
    those transitions. CUDA faults and contradictory planner outputs propagate
    instead of being treated as ordinary candidate failures.
    """
    root = Path(shared_root).expanduser().resolve()
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    if Path(catalog["shared_root"]).expanduser().resolve() != root:
        raise ValueError("catalog shared root differs from trial shared root")
    pose_world = validate_se3(key_pose_world, name="fresh key pose_world")
    if not isinstance(key_observation_id, str) or not key_observation_id:
        raise ValueError("fresh key observation ID is required")
    timestamp = float(key_capture_timestamp_s)
    if not math.isfinite(timestamp):
        raise ValueError("key acquisition timestamp must be finite")
    observed_socket = calibration.record.get("socket_observations")
    if not isinstance(observed_socket, list) or not observed_socket:
        raise ValueError("session has no timestamped socket observations")
    last_socket_time = max(float(row["timestamp_s"])
                           for row in observed_socket)
    if timestamp <= last_socket_time:
        raise ValueError("fresh key exposure must follow frozen socket measurement")
    state_time = float(start_q_acquisition_timestamp_s)
    skew_limit = float(max_key_state_skew_s)
    if (not math.isfinite(state_time) or not math.isfinite(skew_limit) or
            skew_limit <= 0 or state_time <= last_socket_time or
            abs(state_time - timestamp) > skew_limit):
        raise ValueError("start joints and key exposure are not time-aligned")
    start = np.asarray(live_start_q, dtype=np.float64)
    if start.shape != (13,) or not np.all(np.isfinite(start)):
        raise ValueError("live FR3/Inspire start state must be 13 finite joints")
    if max_candidate_attempts is not None and max_candidate_attempts < 1:
        raise ValueError("max_candidate_attempts must be positive")
    if ((max_reset_center_drift_m is None) !=
            (max_reset_axis_tilt_deg is None)):
        raise ValueError("both reset pose-fidelity limits must be supplied")
    if reset_candidate_root is not None and max_reset_center_drift_m is None:
        raise ValueError("reset candidate root requires pose-fidelity limits")
    if max_reset_center_drift_m is not None and (
            not math.isfinite(float(max_reset_center_drift_m)) or
            float(max_reset_center_drift_m) <= 0 or
            not math.isfinite(float(max_reset_axis_tilt_deg)) or
            float(max_reset_axis_tilt_deg) <= 0):
        raise ValueError("reset pose-fidelity limits must be finite and positive")
    if (not math.isfinite(float(axial_waypoint_step_m)) or
            not 0 < float(axial_waypoint_step_m) <= 0.005):
        raise ValueError("axial waypoint step must be finite and <= 5 mm")
    limits.validate()
    attempted_keys = tuple(tuple(str(part) for part in key)
                           for key in attempted)
    covered_ids = tuple(int(value) for value in covered_scenes)

    scene = build_trial_scene_from_session(
        mode=mode, shared_root=root, calibration=calibration,
        key_pose_world=pose_world, max_axis_error_deg=max_pose_error_deg)
    pose_robot = validate_se3(cart2se3(np.asarray(
        scene["mesh"]["target"]["pose"], dtype=float)),
        name="fresh trial T_robot_key")
    pose_class = classify_key_tabletop_pose(
        mode=mode, shared_root=root, pose_robot_key=pose_robot,
        max_rotation_error_deg=max_pose_error_deg)
    stem = pose_class["stem"]
    selected = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem=stem,
        attempted=attempted_keys, covered_scenes=covered_ids)

    def alternative_stems() -> tuple[str, ...]:
        candidates = sorted({row["tabletop_pose_stem"]
                             for row in catalog["candidates"]
                             if row.get("eligible") and
                             row.get("tabletop_pose_stem") != stem})
        available = []
        for candidate_stem in candidates:
            other = select_pose_candidates(
                catalog, expected_mode=mode,
                tabletop_pose_stem=candidate_stem,
                attempted=attempted_keys, covered_scenes=covered_ids)
            if other["status"] == "catalog_stale":
                raise ValueError("catalog became stale while checking repose targets")
            if other["status"] == "candidates_available":
                available.append(candidate_stem)
        return tuple(available)

    rows: list[dict] = []
    selected_key = None
    pickup_plan = None
    insertion_plan = None
    status = ""
    repose = ()
    repose_assessment = None
    if selected["status"] in {"catalog_incomplete", "catalog_stale"}:
        status = "catalog_unavailable"
    elif selected["status"] == "no_eligible_in_screened_pool":
        repose = alternative_stems()
        status = ("repose_required_unplanned" if repose
                  else "no_eligible_pose_in_catalog")
    elif selected["status"] != "candidates_available":
        raise ValueError(f"unexpected candidate selection: {selected['status']}")
    else:
        candidates = selected["candidates"]
        budgeted = (candidates if max_candidate_attempts is None
                    else candidates[:max_candidate_attempts])
        for candidate in budgeted:
            key = tuple(candidate["key"])
            override = planner_candidate_override(
                catalog=catalog, selected=[candidate], mode=mode,
                session_record=calibration.record, pose_robot_key=pose_robot,
                tabletop_pose_stem=stem)
            if len(override[0]) != 1 or tuple(override[3][0]) != key:
                raise ValueError("v8 loader did not return exactly the selected grasp")
            install_curobo_planner_compat()
            planner.set_start_state(start)
            pickup = planner.plan(
                scene, mode.key_object, "v8", hand="fr3_inspire",
                skip_done=False, success_only=False,
                candidate_override=override)
            row = {"key": list(key), "pickup_preflight_pass": bool(pickup.success),
                   "insertion_preflight_status": None}
            if not pickup.success:
                rows.append(row)
                continue
            if tuple(pickup.scene_info) != key:
                raise ValueError("AutoDex selected a grasp outside the one-grasp override")
            hand_in_key = validate_se3(np.load(
                Path(candidate["candidate_dir"]) / "wrist_se3.npy",
                allow_pickle=False), name="v8 T_key_hand")
            targets = build_rigid_insertion_targets(
                mode=mode, shared_root=root, calibration=calibration,
                T_key_hand=hand_in_key)
            hold = nominal_inspire_hold_poses(
                pickup.pregrasp_pose, pickup.grasp_pose,
            )["autodex_default_controller_hold"]
            trial = plan_insertion_after_pickup(
                planner=planner, pickup_plan=pickup, trial_scene=scene,
                shared_root=root, calibration=calibration, targets=targets,
                held_hand_q=hold, held_hand_source="commanded_nominal",
                limits=limits, axial_waypoint_step_m=axial_waypoint_step_m)
            row["insertion_preflight_status"] = trial.status
            rows.append(row)
            if trial.sampled_planning_pass:
                selected_key = key
                pickup_plan = pickup
                insertion_plan = trial
                status = "sampled_planning_pass"
                break
        if not status:
            if len(budgeted) < len(candidates):
                status = "candidate_budget_exhausted"
            else:
                repose = alternative_stems()
                status = ("repose_required_unplanned" if repose
                          else "planning_exhausted_current_pose")

    if repose and max_reset_center_drift_m is not None:
        repose_assessment = assess_repose_options(
            shared_root=root, mode=mode, catalog=catalog,
            current_pose_stem=stem, target_stems=repose,
            T_robot_key=pose_robot,
            max_center_in_hand_drift_m=max_reset_center_drift_m,
            max_symmetry_axis_tilt_deg=max_reset_axis_tilt_deg,
            attempted_insertion=attempted_keys,
            covered_scenes=covered_ids,
            attempted_reset=attempted_reset,
            candidate_root=reset_candidate_root)
        if repose_assessment["status"] == "catalog_unavailable":
            status = "catalog_unavailable"
        elif repose_assessment["status"] == "no_executable_repose_path":
            status = "repose_assets_unavailable"
        elif repose_assessment["status"] == (
                "staged_reset_seed_requires_install_and_full_chain_preflight"):
            status = "repose_staged_only_unplanned"

    return TrialPreflight(
        status=status, pose_class=pose_class,
        attempted_candidates=tuple(rows), selected_candidate_key=selected_key,
        repose_target_stems=repose, insertion_plan=insertion_plan,
        pickup_plan=pickup_plan, trial_scene=scene,
        key_observation_id=key_observation_id,
        key_capture_timestamp_s=timestamp,
        start_q_acquisition_timestamp_s=state_time,
        max_key_state_skew_s=skew_limit,
        key_pose_world=pose_world,
        live_start_q=start.copy(),
        attempted_before_trial=attempted_keys,
        covered_scenes=covered_ids,
        limits=limits, max_pose_error_deg=float(max_pose_error_deg),
        axial_waypoint_step_m=float(axial_waypoint_step_m),
        max_candidate_attempts=max_candidate_attempts,
        session_calibration_sha256=_canonical_sha256(calibration.record),
        catalog_sha256=_canonical_sha256(catalog),
        repose_assessment=repose_assessment,
        cartesian_planner_mode=cartesian_mode_from_planner(planner),
    )


def plan_admitted_key_trial(
    *, key_observation: KeyPoseObservation,
    start_q_acquisition_timestamp_s: float,
    max_key_state_skew_s: float,
    **planning_kwargs,
) -> TrialPreflight:
    """Feed a bound, multi-view key measurement into the existing v8 trial.

    The older ``plan_fresh_key_trial`` also serves saved offline replays and
    accepts one asserted timestamp. A live caller should enter here so the
    measured robot state is checked against the *entire* multi-camera time
    interval, including each camera's stated timing uncertainty.
    """
    if not isinstance(key_observation, KeyPoseObservation):
        raise TypeError("live trial needs an admitted KeyPoseObservation")
    mode = planning_kwargs.get("mode")
    if (not isinstance(mode, TaskMode) or
            mode.key_object != key_observation.key_object or
            mode.family != key_observation.family):
        raise ValueError("observed key identity differs from selected task mode")
    key_observation.require_state_alignment(
        state_timestamp_s=start_q_acquisition_timestamp_s,
        maximum_skew_s=max_key_state_skew_s)
    return plan_fresh_key_trial(
        key_pose_world=key_observation.pose_world,
        key_observation_id=key_observation.capture_id,
        key_capture_timestamp_s=(
            key_observation.selected_acquisition_timestamp_s),
        start_q_acquisition_timestamp_s=start_q_acquisition_timestamp_s,
        max_key_state_skew_s=max_key_state_skew_s,
        **planning_kwargs)
