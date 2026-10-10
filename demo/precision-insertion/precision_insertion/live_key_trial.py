"""Join one fresh AutoDex key capture to the demo's existing v8 preflight.

This is an observation/planning boundary, not a motion executor. A robot-state
provider must return measured feedback buffered at the camera exposure time;
sampling only after slow SAM/FoundPose inference is not equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Callable, Mapping

from .assets import AssetPaths
from .key_perception import (
    KeyPoseObservation, admit_key_capture, verify_key_capture_artifacts,
    write_key_capture_artifacts,
)
from .live_capture import collect_key_capture
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .perception_evidence import SocketViewLimits
from .session_runner import SessionRunner
from .trial_preflight import TrialPreflight


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PreparedLiveKeyTrial:
    observation: KeyPoseObservation
    state: LiveRobotState
    preflight: TrialPreflight
    key_evidence_dir: Path


def prepare_next_live_key(
    *, runner: SessionRunner, init_orchestrator,
    acquisition_metadata_for_request: Callable[[int], Mapping],
    state_for_observation: Callable[[KeyPoseObservation], LiveRobotState],
    capture_root: Path, key_evidence_dir: Path, capture_id: str,
    calibrated_camera_ids: set[str], intrinsics_full: Mapping,
    extrinsics_full: Mapping, image_hw: tuple[int, int],
    key_prompt: str, view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    maximum_socket_mask_overlap_fraction: float,
    socket_projection_dilation_px: int,
    max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    max_key_state_skew_s: float,
    planner, limits: PathAuditLimits, max_pose_error_deg: float,
    axial_waypoint_step_m: float,
    capture_timeout_s: float, image_write_timeout_s: float,
    silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
    max_candidate_attempts: int | None = None,
    planning_options: Mapping | None = None,
    request_id_factory: Callable[[], int] | None = None,
) -> PreparedLiveKeyTrial:
    """Capture, admit, save, and preflight one fresh tabletop key.

    The socket calibration/catalogue come from ``runner`` and are never
    replaced here. The key pose and 13-DOF feedback must overlap in time.
    The callback must read a continuously sampled robot-state buffer, but
    must not manufacture a camera-time state from a later read. If any gate
    fails, no selected candidate or motor command is produced. Capture/evidence
    directories already created on failure remain available for diagnosis.
    """
    if not isinstance(runner, SessionRunner):
        raise TypeError("fresh key preflight needs a frozen SessionRunner")
    action = runner.current_decision().action
    if action not in {"capture_fresh_key", "reobserve_key_and_preflight"}:
        raise ValueError(f"session is not ready for a fresh key capture: {action}")
    if (not isinstance(capture_id, str) or
            not _SAFE_ID.fullmatch(capture_id)):
        raise ValueError("key capture ID must be path-safe")
    if (not isinstance(key_prompt, str) or not key_prompt.strip() or
            key_prompt == "object"):
        raise ValueError("fresh key needs a specific SAM prompt")
    if not callable(acquisition_metadata_for_request) or not callable(
            state_for_observation):
        raise TypeError("camera evidence and exposure-time state providers are required")
    capture_source = Path(capture_root).expanduser()
    if not capture_source.is_absolute() or not capture_source.is_dir():
        raise ValueError("key capture root must be an existing absolute shared directory")
    capture_root = capture_source.resolve()
    output = Path(key_evidence_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"key evidence already exists: {output}")
    if (not isinstance(calibrated_camera_ids, set) or
            len(calibrated_camera_ids) < view_limits.minimum_accepted_views or
            set(intrinsics_full) != calibrated_camera_ids or
            set(extrinsics_full) != calibrated_camera_ids):
        raise ValueError("key cameras must match frozen session calibration")
    if (not isinstance(image_hw, tuple) or len(image_hw) != 2 or
            any(type(value) is not int or value <= 0 for value in image_hw)):
        raise ValueError("key image shape must be positive H,W")
    for name, value in (
            ("capture_timeout_s", capture_timeout_s),
            ("image_write_timeout_s", image_write_timeout_s),
            ("max_key_state_skew_s", max_key_state_skew_s),
            ("maximum_multiview_center_error_mm",
             maximum_multiview_center_error_mm),
            ("maximum_multiview_angle_error_deg",
             maximum_multiview_angle_error_deg),
            ("silhouette_loss_threshold", silhouette_loss_threshold),
            ("max_arm_hand_skew_s", max_arm_hand_skew_s),
            ("max_hand_command_error_raw", max_hand_command_error_raw),
            ("max_arm_velocity_rad_s", max_arm_velocity_rad_s),
            ("max_pose_error_deg", max_pose_error_deg),
            ("axial_waypoint_step_m", axial_waypoint_step_m)):
        if (type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if (not 0 <= maximum_socket_mask_overlap_fraction < 1 or
            type(socket_projection_dilation_px) is not int or
            socket_projection_dilation_px < 0 or
            type(silhouette_iterations) is not int or
            silhouette_iterations < 0 or
            (max_candidate_attempts is not None and
             (type(max_candidate_attempts) is not int or
              max_candidate_attempts < 1))):
        raise ValueError("key mask, silhouette or candidate limits are invalid")
    if axial_waypoint_step_m > .005:
        raise ValueError("axial waypoint step may not exceed 5 mm")
    view_limits.validate()
    paths = AssetPaths(runner.shared_root, runner.mode)
    mesh = paths.raw_mesh(runner.mode.key_object)
    repre = paths.foundpose_repre(runner.mode.key_object)
    for name, path in (("v8 key mesh", mesh),
                       ("key FoundPose representation", repre)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    if planning_options is None:
        planning_options = {}
    if not isinstance(planning_options, Mapping):
        raise TypeError("planning options must be a mapping")
    allowed_planning_options = {
        "max_reset_center_drift_m", "max_reset_axis_tilt_deg",
        "reset_candidate_root", "attempted_reset",
    }
    unknown = set(planning_options) - allowed_planning_options
    if unknown:
        raise ValueError("unsupported or runner-owned planning options: " +
                         ", ".join(sorted(str(key) for key in unknown)))

    init_orchestrator.init_object(
        obj_name=runner.mode.key_object, mesh_path=str(mesh),
        assets_root=str(paths.foundpose_assets_root(runner.mode.key_object)),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=image_hw, mode="live", load_silhouette=True)
    capture = collect_key_capture(
        init_orchestrator=init_orchestrator,
        key_object=runner.mode.key_object, capture_id=capture_id,
        key_prompt=key_prompt, capture_root=capture_root,
        calibrated_camera_ids=calibrated_camera_ids,
        acquisition_metadata_for_request=acquisition_metadata_for_request,
        timeout_s=float(capture_timeout_s),
        image_write_timeout_s=float(image_write_timeout_s),
        request_id_factory=request_id_factory)
    observation = admit_key_capture(
        capture=capture, init_orchestrator=init_orchestrator,
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration,
        calibrated_camera_ids=calibrated_camera_ids,
        view_limits=view_limits,
        maximum_multiview_center_error_mm=maximum_multiview_center_error_mm,
        maximum_multiview_angle_error_deg=maximum_multiview_angle_error_deg,
        maximum_socket_mask_overlap_fraction=(
            maximum_socket_mask_overlap_fraction),
        socket_projection_dilation_px=socket_projection_dilation_px,
        silhouette_iterations=silhouette_iterations,
        silhouette_loss_threshold=silhouette_loss_threshold)
    if observation.phase != "tabletop":
        raise ValueError("fresh key trial requires a tabletop observation")
    write_key_capture_artifacts(capture, observation, output)
    verify_key_capture_artifacts(output)
    state = state_for_observation(observation)
    if not isinstance(state, LiveRobotState):
        raise TypeError("key trial needs measured Franka/Inspire state")
    state.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    observation.require_state_alignment(
        state_timestamp_s=state.sample_timestamp_s,
        maximum_skew_s=max_key_state_skew_s)
    preflight = runner.preflight_next_key(
        planner=planner, key_observation=observation,
        key_evidence_dir=output, live_start_q=state.full_q.copy(),
        start_q_acquisition_timestamp_s=state.sample_timestamp_s,
        max_key_state_skew_s=max_key_state_skew_s, limits=limits,
        max_pose_error_deg=max_pose_error_deg,
        axial_waypoint_step_m=axial_waypoint_step_m,
        max_candidate_attempts=max_candidate_attempts,
        **dict(planning_options))
    return PreparedLiveKeyTrial(observation, state, preflight, output)
