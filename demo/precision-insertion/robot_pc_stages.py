"""Cylinder wiring for the current precision-insertion FR3 pipeline.

This factory only supplies the live wiring around a versioned stage-policy
document. The existing
AutoDex v8 planner, camera session, completion-checked stock executor, local
Qwen backend, force-stop insertion and stock reset callbacks are owned by
``precision_insertion.franka_integration``. Importing this file is non-moving.

The selected policy records the operator-accepted settings and provisional
perception/landing values. They are not calibration or measured performance.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM
from precision_insertion.pickup_execution import PickupExecutionLimits
from precision_insertion.stage_policy import load_zerodex_stage_policy


def _recording_event_marker(*, environment: str, schema: str):
    """Return an optional local event callback for the ZeroDex recorder.

    The marker path is supplied only by the ZeroDex recording wrapper. It is
    intentionally a local control file: capture4 recording must not put NAS
    I/O or remote process management on the robot action thread.
    """
    raw = os.environ.get(environment, "").strip()
    if not raw:
        return None
    target = Path(raw).expanduser().resolve()

    def notify(attempt_id: str, candidate_id: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                json.dump({
                    "schema": schema,
                    "attempt_id": attempt_id,
                    "candidate_id": candidate_id,
                    "confirmed_at_unix_ns": time.time_ns(),
                }, stream, sort_keys=True)
                stream.write("\n")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    return notify


def _preinsert_recording_marker():
    return _recording_event_marker(
        environment="PRECISION_PREINSERT_CONFIRMED_MARKER",
        schema="precision_insertion_preinsert_confirmed_v1")


def _release_clear_recording_marker():
    return _recording_event_marker(
        environment="PRECISION_RELEASE_CLEAR_COMPLETED_MARKER",
        schema="precision_insertion_release_clear_completed_v1")


def _tabletop_pose_error_deg(policy=None) -> float:
    """Keep AutoDex's 10-degree setting unless a profile selects another one."""
    selected = policy or load_zerodex_stage_policy()
    return float(os.environ.get(
        "PRECISION_TABLETOP_POSE_ERROR_DEG",
        str(selected.tabletop_pose_error_deg)))


def _path_limits(policy=None):
    """Return the versioned path-audit values without adding another gate."""
    return (policy or load_zerodex_stage_policy()).path_limits


def _motion_limits(policy=None):
    # Stock executor runs at its default speed. These bounds check feedback;
    # they do not change the executor's arm_speed_scale or traj_speed.
    return (policy or load_zerodex_stage_policy()).motion_limits


def _key_capture(policy=None) -> dict:
    # Only admitted views are used; the current pipeline needs at least two.
    # The pose/mask thresholds are provisional for the first physical trial.
    return dict((policy or load_zerodex_stage_policy()).key_capture)


def _unknown_transfer_recovery(runner, _decision, *, clear_view):
    """Continue the session after the operator resolves an unknown key state.

    A stopped transfer cannot establish whether the key is still held. Let
    the operator remove it, then use the usual empty-hand clear-view motion
    and fresh key observation rather than terminating the action loop.
    """
    answer = input(
        "[recovery] Remove the key from the hand/socket, place it upright "
        "on the board, and leave the socket fixed. Type yes when done: ")
    if answer.strip().lower() not in {"yes", "y"}:
        raise RuntimeError("operator has not completed the key return")
    clear_view.prepare_initial()
    path = Path(runner._attempt_dir) / "supervised_transfer_reset.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({
            "schema": "precision_insertion_supervised_reset_v1",
            "attempt_id": runner.active_attempt.attempt_id,
            "method": "supervised_manual_return",
            "reviewed_by": "operator",
            "completed_at_s": time.time(),
            "socket_clear": True,
            "hand_open": True,
        }, stream, indent=2)
        stream.write("\n")
    return path


def configure(core, live, options) -> dict:
    """Return all ten handlers, reusing the session's stock v8 resources."""
    launch = live.cameras.launch
    scene_mode = getattr(launch, "scene_mode", "charuco_board")
    if (launch.mode.family != "cylinder" or
            launch.mode.gap_mm not in CYLINDER_RADIAL_GAPS_MM or
            not (scene_mode == "charuco_board" and launch.board_id == "9" or
                 scene_mode == "board_free" and launch.board_id is None)):
        raise ValueError(
            "this FR3 binding needs board 9 or board-free cylinder scene")
    if core.sampler is None:
        raise ValueError("live camera stages require exposure-aligned FR3 feedback")
    if options.enable_robot_motion is not True:
        raise PermissionError("FR3 stages require explicit robot motion opt-in")

    def motion_interlock() -> bool:
        # Each motion boundary already checks its own fresh measured start.
        # Do not add a second FR3 read that can fail after that check passes.
        return options.enable_robot_motion is True

    policy = load_zerodex_stage_policy()
    path = _path_limits(policy)
    motion = _motion_limits(policy)
    capture = _key_capture(policy)
    key_prompt = policy.key_prompt
    tabletop_pose_error_deg = _tabletop_pose_error_deg(policy)
    recording_marker = _preinsert_recording_marker()
    recording_stop_marker = _release_clear_recording_marker()
    stationary = dict(policy.stationary)
    # The ChArUco interior is a calibration footprint, not the physical
    # tabletop boundary; the versioned policy therefore keeps its edge
    # clearance at zero.
    reset_geometry = dict(policy.reset_geometry)
    landing = {
        "max_pose_error_deg": tabletop_pose_error_deg,
        **policy.landing,
        **reset_geometry,
    }
    return_parameters = {
        **policy.return_parameters,
        **reset_geometry, **stationary,
    }
    retry = policy.retry_limits
    alignment = policy.alignment_limits

    return {
        "shared": {"motion_interlock": motion_interlock},
        "held_relation_route": "nominal",
        "stages": {
            "key_planning": {
                "key_prompt": key_prompt,
                "limits": path,
                "key_parameters": {
                    **capture, **stationary,
                    "max_pose_error_deg": tabletop_pose_error_deg,
                    "axial_waypoint_step_m": policy.axial_waypoint_step_m,
                },
            },
            "pickup": {
                "pickup_limits": PickupExecutionLimits(
                    **policy.pickup_limits,
                    **stationary),
                "lift_bounds": None,
                "path_limits": path,
                "measured_lift_parameters": {
                    "axial_waypoint_step_m": policy.axial_waypoint_step_m,
                    **stationary,
                },
            },
            "lift": {
                "limits": motion,
                "commissioning_record_path": None,
                "post_lift_settle_s": float(os.environ.get(
                    "PRECISION_POST_LIFT_SETTLE_S",
                    str(policy.post_lift_settle_s))),
                "assessment_parameters": {
                    **policy.lift_assessment,
                    **stationary,
                },
            },
            "held_relation": {
                "limits": path,
                "parameters": {
                    "axial_waypoint_step_m": policy.axial_waypoint_step_m,
                    **stationary,
                },
            },
            "task": {
                "transfer_limits": motion,
                "insertion_limits": motion,
                "path_limits": path,
                "alignment_limits": alignment,
                "same_trial_tip_height_error_m": (
                    retry.max_grasp_translation_drift_m),
                "same_trial_xy_tolerance_m": getattr(
                    options, "same_trial_xy_tolerance_m", None),
                "same_trial_alignment_source": getattr(
                    options, "same_trial_alignment_source", "endpoint"),
                "same_trial_max_step_m": getattr(
                    options, "same_trial_max_step_m", .003),
                "same_trial_axis_tolerance_rad": getattr(
                    options, "same_trial_axis_tolerance_rad", None),
                "same_trial_max_tilt_step_rad": getattr(
                    options, "same_trial_max_tilt_step_rad", math.radians(3.)),
                "same_trial_observation_raise_m": getattr(
                    options, "same_trial_observation_raise_m", 0.),
                "same_trial_observation_lateral_m": getattr(
                    options, "same_trial_observation_lateral_m", 0.),
                "same_trial_observation_lateral_attempts": getattr(
                    options, "same_trial_observation_lateral_attempts", 1),
                "same_trial_observation_yaw_rad": getattr(
                    options, "same_trial_observation_yaw_rad", 0.),
                **({"on_preinsert_confirmed": recording_marker}
                   if recording_marker is not None else {}),
                **({"on_release_and_clear_completed": recording_stop_marker}
                   if recording_stop_marker is not None else {}),
                "transfer_commissioning_path": None,
                "preinsert_parameters": {
                    **policy.preinsert,
                    "render_audit_overlay": os.environ.get(
                        "PRECISION_PREINSERT_AUDIT_OVERLAY",
                        "1" if policy.preinsert["render_audit_overlay"] else
                        "0") != "0",
                    **stationary,
                },
                "handoff_parameters": {},
                "insertion_parameters": {},
                "final_parameters": dict(policy.final),
            },
            "retry": {
                "retry_limits": retry,
                "alignment_limits": alignment,
                "diagnostic_parameters": {},
            },
            "held_return": {
                "key_prompt": key_prompt,
                "limits": path,
                "capture_parameters": capture,
                "return_parameters": return_parameters,
                "landing_parameters": landing,
            },
            "transfer_recovery": {
                "key_prompt": key_prompt,
                "recover_to_tabletop": lambda runner, decision:
                    _unknown_transfer_recovery(
                        runner, decision, clear_view=live.clear_view),
                "capture_parameters": capture,
                "landing_parameters": landing,
            },
            "repose_planning": {
                "path_limits": path,
                "parameters": {
                    **policy.repose,
                    "max_pose_error_deg": tabletop_pose_error_deg,
                    **({"reset_candidate_root": Path(
                        os.environ["PRECISION_RESET_CANDIDATE_ROOT"])}
                       if os.environ.get("PRECISION_RESET_CANDIDATE_ROOT")
                       else {}),
                    **reset_geometry,
                },
            },
            "repose_execution": {
                "key_prompt": key_prompt,
                "capture_parameters": capture,
                "landing_parameters": {
                    "max_pose_error_deg": tabletop_pose_error_deg,
                    "support_tolerance_m": policy.landing[
                        "support_tolerance_m"],
                    **reset_geometry,
                },
            },
        },
    }
