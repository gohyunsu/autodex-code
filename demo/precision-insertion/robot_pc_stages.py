"""Board-9 cylinder wiring for the current AutoDex FR3 pipeline.

This factory only supplies limits and the live feedback gate. The existing
AutoDex v8 planner, camera session, completion-checked stock executor, local
Qwen backend, force-stop insertion and stock reset callbacks are owned by
``precision_insertion.franka_integration``. Importing this file is non-moving.

The values below are the operator-accepted 2026-10-13 proposal where it
covered a field. Remaining perception and landing thresholds are provisional
trial settings, not calibration or physical performance measurements.
"""

from __future__ import annotations

from precision_insertion.grounded_alignment import AlignmentLimits
from precision_insertion.config import CYLINDER_RADIAL_GAPS_MM
from precision_insertion.lift_execution import LiftExecutionLimits
from precision_insertion.path_audit import PathAuditLimits
from precision_insertion.pickup_execution import PickupExecutionLimits
from precision_insertion.retry_session import RetrySessionLimits


KEY_PROMPT = "white plastic cylindrical peg"


def _path_limits() -> PathAuditLimits:
    # Accepted proposal; collision and exact-endpoint checks stay in v8.
    result = PathAuditLimits(
        max_joint_step_rad=.12,
        max_wrist_step_m=.015,
        max_wrist_rotation_deg=8.,
        goal_position_tolerance_m=.005,
        goal_rotation_tolerance_deg=5.,
        axial_lateral_tolerance_m=.005,
        axial_rotation_tolerance_deg=5.,
        minimum_hand_clearance_m=0.,
    )
    result.validate()
    return result


def _motion_limits() -> LiftExecutionLimits:
    # Stock executor runs at its default speed. These bounds check feedback;
    # they do not change the executor's arm_speed_scale or traj_speed.
    result = LiftExecutionLimits(
        max_start_age_s=1., max_end_state_age_s=1.,
        max_start_joint_error_rad=.02, max_end_joint_error_rad=.02,
        max_hand_drift_raw=50., max_arm_hand_skew_s=.10,
        max_hand_command_error_raw=30., max_arm_velocity_rad_s=.05,
        max_execution_duration_s=90.,
        max_watchdog_command_age_s=.2,
        max_commissioning_record_age_s=86400.,
    )
    result.validate()
    return result


def _key_capture() -> dict:
    # Only admitted views are used; the current pipeline needs at least two.
    # The pose/mask thresholds are provisional for the first physical trial.
    return {
        "maximum_multiview_center_error_mm": 10.,
        "maximum_multiview_angle_error_deg": 15.,
        "maximum_socket_mask_overlap_fraction": .2,
        "socket_projection_dilation_px": 2,
        "minimum_refinement_iou": .5,
        "max_key_state_skew_s": 5.,
        "capture_timeout_s": 120.,
        "image_write_timeout_s": 90.,
    }


def _unknown_transfer_recovery(_runner, _decision):
    # No stock AutoDex method can infer where a key went after an unobserved
    # controller abort or slip. Preserve the attempt for supervised recovery.
    raise RuntimeError(
        "key state is unknown after transfer; supervised recovery is required")


def configure(core, live, options) -> dict:
    """Return all ten handlers, reusing the session's stock v8 resources."""
    launch = live.cameras.launch
    if (launch.mode.family != "cylinder" or
            launch.mode.gap_mm not in CYLINDER_RADIAL_GAPS_MM or
            launch.board_id != "9"):
        raise ValueError("this FR3 binding needs board 9 and a configured cylinder gap")
    if core.sampler is None:
        raise ValueError("live camera stages require exposure-aligned FR3 feedback")
    if options.enable_robot_motion is not True:
        raise PermissionError("FR3 stages require explicit robot motion opt-in")

    def motion_interlock() -> bool:
        # Each motion boundary already checks its own fresh measured start.
        # Do not add a second FR3 read that can fail after that check passes.
        return options.enable_robot_motion is True

    path = _path_limits()
    motion = _motion_limits()
    capture = _key_capture()
    stationary = {
        "max_arm_hand_skew_s": .10,
        "max_hand_command_error_raw": 30.,
        "max_arm_velocity_rad_s": .05,
    }
    reset_geometry = {
        "minimum_rest_socket_clearance_m": .005,
        "minimum_board_edge_clearance_m": .005,
    }
    landing = {
        "max_pose_error_deg": 10.,
        "max_return_center_shift_m": .03,
        "support_tolerance_m": .005,
        **reset_geometry,
    }
    return_parameters = {
        "release_height_m": 0.,
        "minimum_release_key_clearance_m": .005,
        "max_withdrawal_state_gap_s": 1.,
        **reset_geometry, **stationary,
    }
    retry = RetrySessionLimits(
        max_state_skew_s=5., **stationary,
        max_grasp_translation_drift_m=.01,
        max_grasp_rotation_drift_deg=10.,
        minimum_anchor_separation_px=3., crop_width_px=320,
        max_frame_age_s=180., max_capture_skew_s=10.,
        axial_waypoint_step_m=.005, path_limits=path,
        max_total_offset_m=None,
    )
    retry.validate()
    alignment = AlignmentLimits(
        pixel_sigma_px=1., max_reprojection_px=3.,
        min_parallax_deg=5., max_axis_tilt_deg=10.,
        max_20mm_axis_sweep_m=.005,
        max_lateral_uncertainty_95_m=.01,
        systematic_lateral_sigma_m=.001,
        cad_spacing_sigma_m=.001,
        minimum_views=2,
    )
    alignment.validate()

    return {
        "shared": {"motion_interlock": motion_interlock},
        "held_relation_route": "nominal",
        "stages": {
            "key_planning": {
                "key_prompt": KEY_PROMPT,
                "limits": path,
                "key_parameters": {
                    **capture, **stationary,
                    "max_pose_error_deg": 10.,
                    "axial_waypoint_step_m": .005,
                    "max_candidate_attempts": 1,
                },
            },
            "pickup": {
                "pickup_limits": PickupExecutionLimits(
                    max_start_age_s=1.,
                    max_start_joint_error_rad=.02,
                    max_arrival_arm_error_rad=.02,
                    **stationary),
                "lift_bounds": None,
                "path_limits": path,
                "measured_lift_parameters": {
                    "axial_waypoint_step_m": .005,
                    **stationary,
                },
            },
            "lift": {
                "limits": motion,
                "commissioning_record_path": None,
                "assessment_parameters": {
                    "max_phase_skew_s": 10.,
                    "minimum_visual_views": 2,
                    **stationary,
                },
            },
            "held_relation": {
                "limits": path,
                "parameters": {
                    "axial_waypoint_step_m": .005,
                    **stationary,
                },
            },
            "task": {
                "transfer_limits": motion,
                "insertion_limits": motion,
                "transfer_commissioning_path": None,
                "preinsert_parameters": {
                    "max_capture_skew_s": 10.,
                    "max_joint_frame_skew_s": 5.,
                    **stationary,
                },
                "handoff_parameters": {
                    "max_start_joint_error_rad": .02,
                    **stationary,
                },
                "insertion_parameters": {},
                "final_parameters": {
                    "max_phase_skew_s": 10.,
                    "minimum_visual_views": 2,
                },
            },
            "retry": {
                "retry_limits": retry,
                "alignment_limits": alignment,
                "diagnostic_parameters": {},
            },
            "held_return": {
                "key_prompt": KEY_PROMPT,
                "limits": path,
                "capture_parameters": capture,
                "return_parameters": return_parameters,
                "landing_parameters": landing,
            },
            "transfer_recovery": {
                "key_prompt": KEY_PROMPT,
                "recover_to_tabletop": _unknown_transfer_recovery,
                "capture_parameters": capture,
                "landing_parameters": landing,
            },
            "repose_planning": {
                "path_limits": path,
                "parameters": {
                    "max_state_skew_s": 5.,
                    "max_pose_error_deg": 10.,
                    **reset_geometry,
                },
            },
            "repose_execution": {
                "key_prompt": KEY_PROMPT,
                "capture_parameters": capture,
                "landing_parameters": {
                    "max_pose_error_deg": 10.,
                    "support_tolerance_m": .005,
                    **reset_geometry,
                },
            },
        },
    }
