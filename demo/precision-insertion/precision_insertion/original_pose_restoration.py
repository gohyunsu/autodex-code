"""Plan a fresh-key pickup and placement back to the trial's original pose.

The first failed-insertion return may not land exactly where it started.  This
module reuses the v8 AutoDex pickup and the existing socket-aware held/reset
planner for a *second* placement.  It never infers a post-squeeze key pose or
commands a robot.  The caller must execute, clear the view and recapture.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from autodex.utils.conversion import cart2se3
from autodex.utils.robot_config import INSPIRE_INIT

from .analytic_gripper import (
    analytic_franka_gripper_grasp_for_candidate,
    match_cylinder_tabletop_stem,
)
from .candidates import planner_candidate_override, select_pose_candidates
from .assets import AssetPaths
from .curobo_compat import install_curobo_planner_compat
from .end_effector import (full_dof_end_effector_spec,
                           planner_end_effector_spec)
from .endpoint import nominal_hold_poses
from .geometry import validate_se3
from .pose_selection import classify_key_tabletop_pose
from .repose_preflight import (
    UPRIGHT_REPOSE_CLEARANCE_M, build_v8_repose_rest_pose,
    plan_repose_held_chain,
)
from .repose_release import plan_repose_release_exit
from .repose_transition import (
    ReposeTransitionPreflight, preflight_v8_repose_transition,
)
from .table_surface import session_support_plane_robot
from .world import build_trial_scene_from_session


def original_center_release_xy(*, shared_root: Path, mode,
                               target_pose_stem: str,
                               original_center_xy_robot_m: tuple[float, float],
                               key_center_local_m: tuple[float, float, float]
                               ) -> tuple[float, float]:
    """Choose a v8 rest-pose origin that preserves the original CAD center."""
    canonical = validate_se3(np.load(
        AssetPaths(shared_root, mode).key_tabletop_dir /
        f"{target_pose_stem}.npy", allow_pickle=False),
        name="target v8 tabletop pose")
    offset = (canonical[:3, :3] @ np.asarray(key_center_local_m, dtype=float))[:2]
    xy = np.asarray(original_center_xy_robot_m, dtype=float) - offset
    return float(xy[0]), float(xy[1])


def plan_original_pose_restoration(
    *, planner, shared_root: Path, mode, calibration, catalog: dict,
    key_observation, measured_start_q: np.ndarray,
    start_q_timestamp_s: float, max_state_skew_s: float,
    target_pose_stem: str, target_xy_robot_m: tuple[float, float],
    release_height_cm: int, max_pose_error_deg: float,
    minimum_rest_socket_clearance_m: float,
    minimum_board_edge_clearance_m: float,
    minimum_release_key_clearance_m: float,
    retreat_goal_arm_q: np.ndarray, limits,
    preferred_candidate_key: tuple[str, str, str] | None = None,
) -> tuple[ReposeTransitionPreflight, dict]:
    """Return a full pickup→place→release/exit preflight and its frozen scene.

    Directed reset seeds are tried first when the landing orientation differs.
    A same-pose relocation uses an eligible v8 insertion grasp at the freshly
    observed source pose; the endpoint gate is conservative for placement.
    """
    q = np.asarray(measured_start_q, dtype=float)
    timestamp = float(start_q_timestamp_s)
    capture_time = float(key_observation.selected_acquisition_timestamp_s)
    if (q.ndim != 1 or not np.all(np.isfinite(q)) or
            not math.isfinite(timestamp) or
            not math.isfinite(max_state_skew_s) or max_state_skew_s <= 0 or
            abs(timestamp - capture_time) > max_state_skew_s):
        raise ValueError("restoration needs camera-synchronized measured joints")
    try:
        state_spec = full_dof_end_effector_spec(len(q))
    except ValueError as exc:
        raise ValueError(
            "restoration needs camera-synchronized measured joints") from exc
    spec = planner_end_effector_spec(planner)
    if state_spec.name != spec.name:
        raise ValueError("restoration planner and measured end effector differ")
    scene = build_trial_scene_from_session(
        mode=mode, shared_root=shared_root, calibration=calibration,
        key_pose_world=key_observation.pose_world,
        max_axis_error_deg=max_pose_error_deg)
    observed = validate_se3(cart2se3(np.asarray(
        scene["mesh"]["target"]["pose"], dtype=float)),
        name="fresh tabletop T_robot_key")
    support_normal_robot = None
    if spec.is_parallel_gripper:
        if mode.family != "cylinder":
            raise ValueError(
                "deterministic Franka-gripper restoration supports cylinder mode")
        support_normal_robot, _ = session_support_plane_robot(calibration)
        source = match_cylinder_tabletop_stem(
            observed,
            tabletop_dir=AssetPaths(shared_root, mode).key_tabletop_dir,
            table_normal_robot=support_normal_robot)["stem"]
    else:
        source = classify_key_tabletop_pose(
            mode=mode, shared_root=shared_root, pose_robot_key=observed,
            max_rotation_error_deg=max_pose_error_deg)["stem"]
    target = str(target_pose_stem)
    common = dict(
        planner=planner, shared_root=shared_root, mode=mode,
        calibration=calibration, trial_scene=scene, catalog=catalog,
        from_pose_stem=source, to_pose_stem=target,
        height_cm=release_height_cm,
        release_xy_robot_m=target_xy_robot_m, live_start_q=q,
        observation_id=key_observation.capture_id,
        key_capture_timestamp_s=capture_time,
        start_q_acquisition_timestamp_s=timestamp,
        max_state_skew_s=max_state_skew_s,
        max_pose_error_deg=max_pose_error_deg,
        minimum_rest_socket_clearance_m=minimum_rest_socket_clearance_m,
        minimum_board_edge_clearance_m=minimum_board_edge_clearance_m,
        limits=limits,
        retreat_goal_arm_q=retreat_goal_arm_q,
        minimum_release_key_clearance_m=minimum_release_key_clearance_m,
        release_height_m=0.0,
    )
    if spec.name == "inspire" and source != target:
        directed = preflight_v8_repose_transition(**common)
        if directed.status == "nominal_reset_preflight_pass_drop_unobserved":
            return directed, scene
    rest = build_v8_repose_rest_pose(
        shared_root=shared_root, mode=mode, calibration=calibration,
        target_pose_stem=target, release_xy_robot_m=target_xy_robot_m,
        asset_support_tolerance_m=limits.goal_position_tolerance_m)
    selected = select_pose_candidates(
        catalog, expected_mode=mode, tabletop_pose_stem=source)
    rows = list(selected.get("candidates", ())) if (
        selected["status"] == "candidates_available") else []
    if preferred_candidate_key is not None:
        preferred = tuple(preferred_candidate_key)
        if len(preferred) != 3 or not all(isinstance(part, str) for part in preferred):
            raise ValueError("preferred restoration grasp needs a v8 candidate key")
        rows.sort(key=lambda row: tuple(row["key"]) != preferred)
    attempts = []
    clearance = (UPRIGHT_REPOSE_CLEARANCE_M
                 if mode.family == "cylinder" and source != target else 0.10)

    def result(status, candidate=None, pickup=None, held=None, release=None):
        return ReposeTransitionPreflight(
            status, source, target, release_height_cm, rest, candidate,
            tuple(attempts), pickup, held, release,
            key_observation.capture_id, capture_time, timestamp,
            max_state_skew_s), scene

    install_curobo_planner_compat()
    for row in rows:
        key = tuple(row["key"])
        analytic = None
        if spec.is_parallel_gripper:
            assert support_normal_robot is not None
            analytic = analytic_franka_gripper_grasp_for_candidate(
                observed, table_normal_robot=support_normal_robot,
                identifier=key, end_effector=spec.name)
            override = analytic.candidate_override(identifier=key)
        else:
            override = planner_candidate_override(
                catalog=catalog, selected=[row], mode=mode,
                session_record=calibration.record, pose_robot_key=observed,
                tabletop_pose_stem=source)
        planner.set_start_state(q)
        pickup = planner.plan(
            scene, mode.key_object, "v8", hand=spec.asset_namespace,
            skip_done=False, success_only=False,
            candidate_override=override)
        seed = {"seed_id": "/".join(key),
                "source": ("analytic_franka_gripper_grasp"
                           if analytic is not None else "stable_v8_grasp"),
                "height_cm": release_height_cm, "cell": f"{source}_{target}"}
        if not pickup.success or pickup.lift_preflight is None:
            attempts.append({**seed, "status": "pickup_unreachable"})
            continue
        if tuple(pickup.scene_info) != key:
            raise ValueError("restoration planner selected a different v8 grasp")
        T_key_hand = (analytic.T_key_hand.copy() if analytic is not None else
                      validate_se3(np.load(
                          Path(row["candidate_dir"]) / "wrist_se3.npy",
                          allow_pickle=False), name="v8 candidate T_key_hand"))
        hold_q = next(iter(nominal_hold_poses(
            pickup.pregrasp_pose, pickup.grasp_pose).values()))
        held = plan_repose_held_chain(
            planner=planner, pickup_plan=pickup, trial_scene=scene,
            shared_root=shared_root, calibration=calibration, mode=mode,
            T_key_hand=T_key_hand, T_robot_key_rest=rest,
            release_height_m=0.0,
            lift_height_m=clearance,
            preplace_vertical_travel_m=clearance,
            minimum_rest_socket_clearance_m=minimum_rest_socket_clearance_m,
            minimum_board_edge_clearance_m=minimum_board_edge_clearance_m,
            held_hand_q=hold_q, held_hand_source="commanded_nominal",
            limits=limits)
        attempts.append({**seed, "status": held.status})
        if held.status != "sampled_held_path_pass_release_unplanned":
            continue
        release = plan_repose_release_exit(
            planner=planner, trial_scene=scene, shared_root=shared_root,
            calibration=calibration, mode=mode, held_plan=held,
            release_hand_q=(np.asarray(INSPIRE_INIT, dtype=float)
                            if spec.name == "inspire" else spec.open_q.copy()),
            retreat_goal_arm_q=np.asarray(retreat_goal_arm_q, dtype=float),
            minimum_release_key_clearance_m=minimum_release_key_clearance_m,
            limits=limits)
        attempts[-1]["release_status"] = release.status
        if release.status == "nominal_release_exit_path_pass_drop_unobserved":
            return result("nominal_reset_preflight_pass_drop_unobserved",
                          seed, pickup, held, release)
    return result("original_pose_restoration_path_unavailable")
