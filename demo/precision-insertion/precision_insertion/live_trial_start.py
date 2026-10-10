"""Capture a fresh tabletop key and run one non-actuating v8 trial preflight.

This joins existing AutoDex capture/FoundPose and the demo's strict evidence
and SessionRunner gates. It never commands a robot or remeasures the frozen
socket. A caller must already have opened a verified session and commissioned
camera acquisition times, FoundPose assets, and measured joint feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Mapping

from .assets import AssetPaths
from .calibration import validate_session_camera_calibration
from .key_perception import (
    KeyPoseObservation, admit_key_capture, verify_key_capture_artifacts,
    write_key_capture_artifacts,
)
from .live_capture import collect_key_capture
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .perception_evidence import SocketViewLimits
from .session_runner import SessionRunner


@dataclass(frozen=True)
class KeyTrialCaptureLimits:
    view_limits: SocketViewLimits
    maximum_multiview_center_error_mm: float
    maximum_multiview_angle_error_deg: float
    maximum_socket_mask_overlap_fraction: float
    socket_projection_dilation_px: int
    minimum_refinement_iou: float
    maximum_arm_hand_skew_s: float
    maximum_hand_command_error_raw: float
    maximum_arm_velocity_rad_s: float
    maximum_key_state_skew_s: float

    def validate(self) -> None:
        self.view_limits.validate()
        for name in (
                "maximum_multiview_center_error_mm",
                "maximum_multiview_angle_error_deg",
                "maximum_arm_hand_skew_s",
                "maximum_hand_command_error_raw",
                "maximum_arm_velocity_rad_s",
                "maximum_key_state_skew_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (type(self.maximum_socket_mask_overlap_fraction) not in
                (int, float) or
                not math.isfinite(self.maximum_socket_mask_overlap_fraction) or
                not 0 <= self.maximum_socket_mask_overlap_fraction < 1 or
                type(self.socket_projection_dilation_px) is not int or
                self.socket_projection_dilation_px < 0 or
                type(self.minimum_refinement_iou) not in (int, float) or
                not math.isfinite(self.minimum_refinement_iou) or
                not 0 < self.minimum_refinement_iou <= 1):
            raise ValueError("key mask and refinement limits are invalid")

    def measured_state_limits(self) -> dict[str, float]:
        return {
            "max_arm_hand_skew_s": self.maximum_arm_hand_skew_s,
            "max_hand_command_error_raw": self.maximum_hand_command_error_raw,
            "max_arm_velocity_rad_s": self.maximum_arm_velocity_rad_s,
        }


@dataclass(frozen=True)
class PreparedKeyTrial:
    observation: KeyPoseObservation
    evidence_dir: Path
    preflight: object
    next_decision: object
    robot_ready: bool = False


def capture_and_preflight_next_key(
    *, runner: SessionRunner, planner, init_orchestrator,
    acquisition_metadata_for_request: Callable[[int], Mapping],
    read_measured_state: Callable[[], LiveRobotState],
    capture_root: Path, key_evidence_dir: Path, capture_id: str,
    key_prompt: str, calibrated_camera_ids: set[str],
    intrinsics_full: Mapping, extrinsics_full: Mapping,
    image_hw: tuple[int, int], capture_limits: KeyTrialCaptureLimits,
    path_limits: PathAuditLimits, max_pose_error_deg: float,
    axial_waypoint_step_m: float, timeout_s: float,
    image_write_timeout_s: float = 5.0,
    max_candidate_attempts: int | None = None,
    request_id_factory: Callable[[], int] | None = None,
) -> PreparedKeyTrial:
    """Run fresh-key image → pose → measured-state → offline plan in order.

    The key image is taken only after the frozen socket session has reached a
    decision that allows another trial. A state sample is read *after* image
    admission, so a stale pre-capture joint value cannot be silently reused.
    Failed capture/pose/plan leaves any source images or saved evidence for
    review and never implies a grasp or insertion success.
    """
    if not isinstance(runner, SessionRunner):
        raise TypeError("a verified precision SessionRunner is required")
    action = runner.current_decision().action
    if action not in {"capture_fresh_key", "reobserve_key_and_preflight"}:
        raise ValueError(f"session cannot capture a new key: {action}")
    capture_limits.validate()
    path_limits.validate()
    evidence = Path(key_evidence_dir).expanduser()
    source = Path(capture_root).expanduser()
    if (not evidence.is_absolute() or evidence.exists() or
            not source.is_absolute() or not source.is_dir()):
        raise ValueError("new key evidence and existing capture root need absolute paths")
    if (not isinstance(calibrated_camera_ids, set) or
            len(calibrated_camera_ids) <
            capture_limits.view_limits.minimum_accepted_views or
            set(intrinsics_full) != calibrated_camera_ids or
            set(extrinsics_full) != calibrated_camera_ids):
        raise ValueError("key cameras differ from session calibration")
    if (not isinstance(image_hw, tuple) or len(image_hw) != 2 or
            any(type(size) is not int or size <= 0 for size in image_hw)):
        raise ValueError("key camera image H,W is invalid")
    if (not callable(acquisition_metadata_for_request) or
            not callable(read_measured_state)):
        raise TypeError("key capture needs camera-time and measured-state providers")
    if (type(max_pose_error_deg) not in (int, float) or
            not math.isfinite(max_pose_error_deg) or max_pose_error_deg <= 0 or
            type(axial_waypoint_step_m) not in (int, float) or
            not math.isfinite(axial_waypoint_step_m) or
            axial_waypoint_step_m <= 0):
        raise ValueError("key tabletop and axial preflight limits must be positive")
    paths = AssetPaths(runner.shared_root, runner.mode)
    key_mesh = paths.raw_mesh(runner.mode.key_object)
    key_repre = paths.foundpose_repre(runner.mode.key_object)
    for name, path in (("key raw mesh", key_mesh),
                       ("key FoundPose representation", key_repre)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    if (not isinstance(key_prompt, str) or not key_prompt.strip() or
            key_prompt == "object"):
        raise ValueError("key needs a specific SAM prompt")

    init_orchestrator.init_object(
        obj_name=runner.mode.key_object, mesh_path=str(key_mesh),
        assets_root=str(paths.foundpose_assets_root(runner.mode.key_object)),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=image_hw, mode="live", load_silhouette=False)
    validate_session_camera_calibration(
        runner.calibration,
        intrinsics_undist=init_orchestrator.intrinsics_undist,
        extrinsics_full=init_orchestrator.extrinsics,
        calibrated_camera_ids=calibrated_camera_ids)
    capture = collect_key_capture(
        init_orchestrator=init_orchestrator,
        key_object=runner.mode.key_object, capture_id=capture_id,
        key_prompt=key_prompt, capture_root=source,
        calibrated_camera_ids=calibrated_camera_ids,
        acquisition_metadata_for_request=acquisition_metadata_for_request,
        timeout_s=timeout_s, image_write_timeout_s=image_write_timeout_s,
        request_id_factory=request_id_factory)
    observation = admit_key_capture(
        capture=capture, init_orchestrator=init_orchestrator,
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration,
        calibrated_camera_ids=calibrated_camera_ids,
        view_limits=capture_limits.view_limits,
        maximum_multiview_center_error_mm=(
            capture_limits.maximum_multiview_center_error_mm),
        maximum_multiview_angle_error_deg=(
            capture_limits.maximum_multiview_angle_error_deg),
        maximum_socket_mask_overlap_fraction=(
            capture_limits.maximum_socket_mask_overlap_fraction),
        socket_projection_dilation_px=(
            capture_limits.socket_projection_dilation_px),
        minimum_refinement_iou=capture_limits.minimum_refinement_iou)
    saved = write_key_capture_artifacts(capture, observation, evidence)
    manifest = verify_key_capture_artifacts(saved)
    if (manifest["capture_id"] != observation.capture_id or
            manifest["request_id"] != observation.request_id):
        raise ValueError("saved key capture differs from admitted image")
    state = read_measured_state()
    if not isinstance(state, LiveRobotState):
        raise TypeError("trial start requires measured FR3/Inspire feedback")
    state.validate(**capture_limits.measured_state_limits())
    observation.require_state_alignment(
        state_timestamp_s=state.sample_timestamp_s,
        maximum_skew_s=capture_limits.maximum_key_state_skew_s)
    preflight = runner.preflight_next_key(
        planner=planner, key_observation=observation,
        key_evidence_dir=saved, live_start_q=state.full_q,
        start_q_acquisition_timestamp_s=state.sample_timestamp_s,
        max_key_state_skew_s=capture_limits.maximum_key_state_skew_s,
        limits=path_limits, max_pose_error_deg=max_pose_error_deg,
        axial_waypoint_step_m=axial_waypoint_step_m,
        max_candidate_attempts=max_candidate_attempts,
        measured_start_state=state,
        measured_state_limits=capture_limits.measured_state_limits())
    return PreparedKeyTrial(
        observation, saved, preflight, runner.current_decision())
