"""Read-only v8 reset-seed → AutoDex pickup → socket-aware held-path search.

This is a planner preflight, not a reset executor. Optional nominal opening
and retreat planning does not command the hand, predict a dynamic drop,
observe the landed key, or certify physical reorientation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from autodex.utils.conversion import cart2se3

from .candidates import select_pose_candidates, validate_catalog_session
from .config import TaskMode
from .endpoint import nominal_inspire_hold_poses
from .geometry import validate_se3
from .path_audit import PathAuditLimits
from .pose_selection import classify_key_tabletop_pose
from .repose_preflight import (
    ReposeHeldPreflight, build_v8_repose_rest_pose, plan_repose_held_chain,
)
from .repose_release import ReposeReleasePreflight, plan_repose_release_exit
from .reset_candidates import load_v8_reset_seeds
from .world import build_held_scene_from_trial, validated_frozen_socket_pose


@dataclass(frozen=True)
class ReposeTransitionPreflight:
    status: str
    from_pose_stem: str
    to_pose_stem: str
    height_cm: int
    T_robot_key_rest: np.ndarray | None
    selected_seed: dict | None
    attempted_seeds: tuple[dict[str, Any], ...]
    pickup_plan: Any | None
    held_plan: ReposeHeldPreflight | None
    release_plan: ReposeReleasePreflight | None
    observation_id: str
    key_capture_timestamp_s: float
    start_q_acquisition_timestamp_s: float
    max_state_skew_s: float

    def to_record(self) -> dict:
        return {
            "schema": "precision_insertion_v8_repose_transition_preflight_v1",
            "status": self.status,
            "from_pose_stem": self.from_pose_stem,
            "to_pose_stem": self.to_pose_stem,
            "height_cm": self.height_cm,
            "T_robot_key_rest": (None if self.T_robot_key_rest is None
                                 else self.T_robot_key_rest.tolist()),
            "selected_seed": self.selected_seed,
            "attempted_seeds": list(self.attempted_seeds),
            "held_plan": (None if self.held_plan is None
                          else self.held_plan.to_record()),
            "release_plan": (None if self.release_plan is None
                             else self.release_plan.to_record()),
            "observation_id": self.observation_id,
            "key_capture_timestamp_s": self.key_capture_timestamp_s,
            "start_q_acquisition_timestamp_s": self.start_q_acquisition_timestamp_s,
            "max_state_skew_s": self.max_state_skew_s,
            "not_validated": [
                "live pickup/grasp and observed post-lift key-in-hand relation",
                "physical hand opening, key detachment and dynamic drop",
                *(["release and retreat planning absent"]
                   if self.release_plan is None else []),
                "actual landing pose and tabletop reclassification",
                "robot execution, physical reset or insertion success",
            ],
            "robot_ready": False,
        }


def preflight_v8_repose_transition(
    *, planner, shared_root: Path, mode: TaskMode, calibration,
    trial_scene: dict, catalog: dict, from_pose_stem: str,
    to_pose_stem: str, height_cm: int,
    release_xy_robot_m: tuple[float, float], live_start_q: np.ndarray,
    observation_id: str, key_capture_timestamp_s: float,
    start_q_acquisition_timestamp_s: float, max_state_skew_s: float,
    max_pose_error_deg: float, max_center_in_hand_drift_m: float,
    max_symmetry_axis_tilt_deg: float,
    minimum_rest_socket_clearance_m: float,
    minimum_board_edge_clearance_m: float, limits: PathAuditLimits,
    attempted_insertion: Iterable[tuple[str, str, str]] = (),
    covered_scenes: Iterable[int] = (),
    attempted_reset_ids: tuple[str, ...] = (),
    reset_candidate_root: Path | None = None,
    max_seed_attempts: int | None = None,
    retreat_goal_arm_q: np.ndarray | None = None,
    minimum_release_key_clearance_m: float | None = None,
) -> ReposeTransitionPreflight:
    """Try each stable directed v8 reset seed without driving the robot.

    The caller supplies a freshly observed key and synchronized measured
    robot start state, as for insertion trial preflight. A target pose is
    considered only if its own insertion endpoint catalogue has an eligible
    grasp. The explicit release XY must be commissioned inside the board's
    usable area; this routine checks support height and socket clearance but
    cannot infer a safe drop location from the broad AutoDex table cuboid.
    ``reset_candidate_root``, if given, is the exact ``reset_<height>``
    candidate directory accepted by ``load_v8_reset_seeds`` (unlike the
    read-only assessment CLI's parent-of-heights override).
    """
    root = Path(shared_root).expanduser().resolve()
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ValueError("fresh key observation ID is required")
    stamps = tuple(float(v) for v in (
        key_capture_timestamp_s, start_q_acquisition_timestamp_s,
        max_state_skew_s))
    if (not all(math.isfinite(v) for v in stamps) or stamps[2] <= 0 or
            abs(stamps[0] - stamps[1]) > stamps[2]):
        raise ValueError("key image and robot start state are not synchronized")
    start = np.asarray(live_start_q, dtype=np.float64)
    if start.shape != (13,) or not np.all(np.isfinite(start)):
        raise ValueError("measured Franka/Inspire start must have 13 finite joints")
    if max_seed_attempts is not None and (type(max_seed_attempts) is not int or
                                          max_seed_attempts <= 0):
        raise ValueError("max_seed_attempts must be a positive integer")
    if (retreat_goal_arm_q is None) != (
            minimum_release_key_clearance_m is None):
        raise ValueError("release preflight needs both retreat goal and key clearance")
    full_release = retreat_goal_arm_q is not None
    if full_release:
        retreat = np.asarray(retreat_goal_arm_q, dtype=np.float64)
        release_clearance = float(minimum_release_key_clearance_m)
        if (retreat.shape != (7,) or not np.all(np.isfinite(retreat)) or
                not math.isfinite(release_clearance) or release_clearance <= 0):
            raise ValueError("invalid release retreat goal or key clearance")
    if (type(height_cm) is not int or height_cm <= 0 or
            height_cm not in (4, 8, 12)):
        raise ValueError("held release preflight supports v8 heights 4, 8, 12 cm")
    source = str(from_pose_stem)
    target = str(to_pose_stem)
    if (not source.isdigit() or source != f"{int(source):03d}" or
            not target.isdigit() or target != f"{int(target):03d}" or
            source == target):
        raise ValueError("directed v8 reset requires distinct three-digit stems")
    limits.validate()
    validated_frozen_socket_pose(mode=mode, shared_root=root,
                                 calibration=calibration)
    build_held_scene_from_trial(trial_scene=trial_scene,
                                calibration=calibration)
    validate_catalog_session(catalog, mode=mode,
                             session_record=calibration.record)
    selected = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem=target,
        attempted=attempted_insertion, covered_scenes=covered_scenes)
    if selected["status"] in {"catalog_stale", "catalog_incomplete"}:
        raise ValueError(f"insertion target catalog unavailable: {selected['reason']}")

    def result(status: str, *, rest=None, seed=None, attempts=(),
               pickup=None, held=None, release=None) -> ReposeTransitionPreflight:
        return ReposeTransitionPreflight(
            status, source, target, height_cm, rest, seed, tuple(attempts),
            pickup, held, release, observation_id, stamps[0], stamps[1],
            stamps[2])

    if selected["status"] != "candidates_available":
        return result("target_without_insertable_grasp")
    initial = validate_se3(cart2se3(np.asarray(
        trial_scene["mesh"]["target"]["pose"], dtype=float)),
        name="fresh reset T_robot_key")
    classified = classify_key_tabletop_pose(
        mode=mode, shared_root=root, pose_robot_key=initial,
        max_rotation_error_deg=max_pose_error_deg)
    if classified["stem"] != source:
        raise ValueError("fresh key tabletop class differs from reset source cell")
    rest = build_v8_repose_rest_pose(
        shared_root=root, mode=mode, calibration=calibration,
        target_pose_stem=target, release_xy_robot_m=release_xy_robot_m,
        asset_support_tolerance_m=limits.goal_position_tolerance_m)
    seeds = load_v8_reset_seeds(
        shared_root=root, mode=mode, height_cm=height_cm,
        from_pose_stem=source, to_pose_stem=target,
        T_robot_key=initial,
        max_center_in_hand_drift_m=max_center_in_hand_drift_m,
        max_symmetry_axis_tilt_deg=max_symmetry_axis_tilt_deg,
        attempted_ids=attempted_reset_ids,
        candidate_root=reset_candidate_root)
    if seeds is None:
        return result("no_verified_reset_seed", rest=rest)
    count = seeds["n_total"] if max_seed_attempts is None else min(
        seeds["n_total"], max_seed_attempts)
    attempts = []
    for i in range(count):
        info = seeds["scene_info"][i]
        seed_id = str(info["grasp_idx"])
        seed_record = {"seed_id": seed_id, "source": info["source"],
                       "height_cm": height_cm, "cell": info["cell"]}
        # Every candidate is planned from the same *measured*, still-unmoved
        # start state. A failed planning query does not move the robot.
        planner.set_start_state(start)
        override = (
            seeds["wrist_se3"][i:i + 1],
            seeds["pregrasp"][i:i + 1],
            seeds["grasp"][i:i + 1],
            [("reset", info["cell"], seed_id)],
            [seeds["openpose_start"][i]],
        )
        pickup = planner.plan(
            trial_scene, obj_name=mode.key_object, grasp_version="v8",
            hand="inspire", candidate_override=override,
            skip_done=False, success_only=False)
        if not pickup.success or pickup.lift_preflight is None:
            attempts.append({**seed_record, "status": "pickup_unreachable"})
            continue
        hold = nominal_inspire_hold_poses(
            seeds["pregrasp"][i], seeds["grasp"][i]
        )["autodex_default_controller_hold"]
        planned = plan_repose_held_chain(
            planner=planner, pickup_plan=pickup, trial_scene=trial_scene,
            shared_root=root, calibration=calibration, mode=mode,
            T_key_hand=np.linalg.inv(initial) @ seeds["wrist_se3"][i],
            T_robot_key_rest=rest, release_height_m=height_cm / 100.0,
            minimum_rest_socket_clearance_m=minimum_rest_socket_clearance_m,
            minimum_board_edge_clearance_m=minimum_board_edge_clearance_m,
            held_hand_q=hold, held_hand_source="commanded_nominal",
            limits=limits)
        attempts.append({
            **seed_record, "status": planned.status,
            "held_planner_queries": list(planned.planner_query_records),
            "held_sampled_failures": (
                None if planned.sampled_held_path_audit is None else
                planned.sampled_held_path_audit.get("failures")),
        })
        if planned.status == "sampled_held_path_pass_release_unplanned":
            if full_release:
                release = plan_repose_release_exit(
                    planner=planner, trial_scene=trial_scene,
                    shared_root=root, calibration=calibration, mode=mode,
                    held_plan=planned,
                    release_hand_q=np.asarray(
                        pickup.pregrasp_pose, dtype=np.float64),
                    retreat_goal_arm_q=retreat,
                    minimum_release_key_clearance_m=release_clearance,
                    limits=limits)
                attempts[-1]["release_status"] = release.status
                attempts[-1]["release_planner_queries"] = list(
                    release.planner_query_records)
                attempts[-1]["release_sampled_failures"] = (
                    None if release.release_geometry_audit is None else
                    release.release_geometry_audit.get("failures"))
                if release.status != (
                        "nominal_release_exit_path_pass_drop_unobserved"):
                    continue
                return result("nominal_reset_preflight_pass_drop_unobserved",
                              rest=rest, seed=seed_record, attempts=attempts,
                              pickup=pickup, held=planned, release=release)
            return result("held_reset_path_available_release_unplanned",
                          rest=rest, seed=seed_record, attempts=attempts,
                          pickup=pickup, held=planned)
    return result(
        "reset_seed_budget_exhausted" if count < seeds["n_total"]
        else ("no_nominal_reset_path" if full_release
              else "no_held_reset_path"),
        rest=rest, attempts=attempts)
