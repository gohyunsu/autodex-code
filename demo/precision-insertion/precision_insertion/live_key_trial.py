"""Join one fresh AutoDex key capture to the demo's existing v8 preflight.

This is an observation/planning boundary, not a motion executor. A robot-state
provider must return measured feedback buffered at the camera exposure time;
sampling only after slow SAM/FoundPose inference is not equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
import json
import math
from pathlib import Path
import re
from typing import Callable, Mapping

from .assets import AssetPaths
from .calibration import validate_session_camera_calibration
from .clear_view import ClearViewForPerception
from .key_perception import (
    KeyPoseObservation, admit_key_capture,
    write_key_capture_artifacts, write_key_pose_artifacts,
)
from .initial_visual import (
    register_pending_initial_visual, write_initial_visual_capture,
)
from .live_capture import collect_key_capture
from .live_robot_state import LiveRobotState
from .path_audit import PathAuditLimits
from .prompt_contract import reject_incorrect_object_color
from .perception_evidence import SocketViewLimits
from .registered_key_capture import (
    RegisteredKeyAssistance, derive_registered_key_capture,
    verify_registered_key_diagnostic, write_registered_key_diagnostic,
)
from .session_runner import SessionRunner
from .silhouette_compat import prepare_key_silhouette
from .trial_preflight import TrialPreflight
from .timing_profile import timed


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class PreparedLiveKeyTrial:
    observation: KeyPoseObservation
    state: LiveRobotState
    preflight: TrialPreflight
    key_evidence_dir: Path


@dataclass(frozen=True)
class ReturnedKeyCapture:
    observation: KeyPoseObservation
    state: LiveRobotState
    key_evidence_dir: Path


def _capture_admitted_key(
    *, runner: SessionRunner, init_orchestrator,
    acquisition_metadata_for_request: Callable[[int], Mapping],
    capture_root: Path, output: Path, capture_id: str,
    calibrated_camera_ids: set[str], intrinsics_full: Mapping,
    extrinsics_full: Mapping, image_hw: tuple[int, int],
    key_prompt: str, view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    maximum_socket_mask_overlap_fraction: float,
    socket_projection_dilation_px: int, minimum_refinement_iou: float,
    silhouette_iterations: int, silhouette_loss_threshold: float,
    capture_timeout_s: float, image_write_timeout_s: float,
    request_id_factory: Callable[[], int] | None,
    registered_key_assistance: RegisteredKeyAssistance | None,
    pose_only: bool = False,
) -> KeyPoseObservation:
    """Shared AutoDex SAM/FoundPose capture; no candidate planning here."""
    if (not isinstance(capture_id, str) or not _SAFE_ID.fullmatch(capture_id) or
            not isinstance(key_prompt, str) or not key_prompt.strip() or
            key_prompt == "object"):
        raise ValueError("key capture needs safe ID and a specific SAM prompt")
    reject_incorrect_object_color(key_prompt)
    if (not isinstance(image_hw, tuple) or len(image_hw) != 2 or
            any(type(value) is not int or value <= 0 for value in image_hw) or
            type(socket_projection_dilation_px) is not int or
            socket_projection_dilation_px < 0 or
            type(silhouette_iterations) is not int or
            silhouette_iterations < 0 or
            not 0 < minimum_refinement_iou <= 1 or
            not 0 <= maximum_socket_mask_overlap_fraction < 1 or
            any(not math.isfinite(float(value)) or float(value) <= 0
                for value in (
                    maximum_multiview_center_error_mm,
                    maximum_multiview_angle_error_deg,
                    silhouette_loss_threshold, capture_timeout_s,
                    image_write_timeout_s))):
        raise ValueError("key capture needs commissioned finite view limits")
    view_limits.validate()
    root = Path(capture_root).expanduser()
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("key capture root must be an existing absolute directory")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"key evidence already exists: {output}")
    paths = AssetPaths(runner.shared_root, runner.mode)
    mesh = paths.raw_mesh(runner.mode.key_object)
    repre = paths.foundpose_repre(runner.mode.key_object)
    for name, path in (("v8 key mesh", mesh),
                       ("key FoundPose representation", repre)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    if (not isinstance(calibrated_camera_ids, set) or
            len(calibrated_camera_ids) < view_limits.minimum_accepted_views or
            set(intrinsics_full) != calibrated_camera_ids or
            set(extrinsics_full) != calibrated_camera_ids):
        raise ValueError("key cameras differ from frozen session calibration")
    if registered_key_assistance is not None:
        if pose_only:
            raise ValueError("registered ROI assistance requires its source images")
        if not isinstance(registered_key_assistance, RegisteredKeyAssistance):
            raise TypeError("registered key assistance needs a prepared context")
        binding = json.loads((runner.output_dir / "source_evidence_binding.json")
                             .read_text(encoding="utf-8"))
        baseline = registered_key_assistance.baseline
        if (binding.get("schema") !=
                "precision_insertion_session_source_binding_v1" or
                binding.get("session_evidence_dir") !=
                str(baseline.session_evidence_dir) or
                binding.get("session_manifest_sha256") !=
                baseline.session_manifest_sha256 or
                getattr(registered_key_assistance.foundpose_model,
                        "obj_name", None) != runner.mode.key_object):
            raise ValueError("ROI assistance differs from opened key session")
    with timed("key.model_init", phase="key"):
        prepare_key_silhouette(
            init_orchestrator=init_orchestrator,
            object_name=runner.mode.key_object, raw_mesh=mesh)
        if getattr(init_orchestrator, "obj_name", None) != runner.mode.key_object:
            init_orchestrator.init_object(
                obj_name=runner.mode.key_object, mesh_path=str(mesh),
                assets_root=str(paths.foundpose_assets_root(runner.mode.key_object)),
                intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
                image_hw=image_hw, mode="live", load_silhouette=False)
    validate_session_camera_calibration(
        runner.calibration,
        intrinsics_undist=init_orchestrator.intrinsics_undist,
        extrinsics_full=init_orchestrator.extrinsics,
        calibrated_camera_ids=calibrated_camera_ids)
    with timed("key.capture", phase="key", kind="camera",
               capture_id=capture_id, camera_count=len(calibrated_camera_ids)):
        capture = collect_key_capture(
            init_orchestrator=init_orchestrator,
            key_object=runner.mode.key_object, capture_id=capture_id,
            key_prompt=key_prompt, capture_root=root.resolve(),
            calibrated_camera_ids=calibrated_camera_ids,
            acquisition_metadata_for_request=acquisition_metadata_for_request,
            timeout_s=float(capture_timeout_s),
            image_write_timeout_s=float(image_write_timeout_s),
            request_id_factory=request_id_factory,
            pose_only=pose_only)
    if pose_only:
        capture = replace(capture, capture_dir=output)
    derivation = None
    if registered_key_assistance is not None:
        capture, derivation = derive_registered_key_capture(
            capture=capture, assistance=registered_key_assistance,
            intrinsics_undist=init_orchestrator.intrinsics_undist,
            extrinsics_full=init_orchestrator.extrinsics,
            key_object=runner.mode.key_object)
        diagnostic_dir = write_registered_key_diagnostic(
            capture=capture, derivation_report=derivation)
        verify_registered_key_diagnostic(diagnostic_dir)
    with timed("key.pose_admission", phase="key"):
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
            minimum_refinement_iou=minimum_refinement_iou,
            silhouette_iterations=silhouette_iterations,
            silhouette_loss_threshold=silhouette_loss_threshold)
    if observation.phase != "tabletop":
        raise ValueError("key capture requires a tabletop observation")
    with timed("key.evidence_write", phase="key", kind="io"):
        if pose_only:
            write_key_pose_artifacts(capture, observation, output)
        elif derivation is None:
            write_key_capture_artifacts(capture, observation, output)
        else:
            write_key_capture_artifacts(
                capture, observation, output, derivation_report=derivation)
    # preflight_next_key reads and verifies this bundle immediately below;
    # a producer-side full reread here would duplicate the same NAS work.
    return observation


def capture_returned_key_for_reset(
    *, runner: SessionRunner, clear_view: ClearViewForPerception,
    init_orchestrator,
    acquisition_metadata_for_request: Callable[[int], Mapping],
    state_for_observation: Callable[[KeyPoseObservation], LiveRobotState],
    capture_root: Path, key_evidence_dir: Path, capture_id: str,
    calibrated_camera_ids: set[str], intrinsics_full: Mapping,
    extrinsics_full: Mapping, image_hw: tuple[int, int],
    key_prompt: str, view_limits: SocketViewLimits,
    maximum_multiview_center_error_mm: float,
    maximum_multiview_angle_error_deg: float,
    maximum_socket_mask_overlap_fraction: float,
    socket_projection_dilation_px: int, minimum_refinement_iou: float,
    max_key_state_skew_s: float,
    capture_timeout_s: float, image_write_timeout_s: float,
    silhouette_iterations: int = 100,
    silhouette_loss_threshold: float = 0.003,
    request_id_factory: Callable[[], int] | None = None,
    registered_key_assistance: RegisteredKeyAssistance | None = None,
    expected_action: str = "recover_held_key_then_reobserve",
) -> ReturnedKeyCapture:
    """Use the stock key perception for reset/repose landing, without planning.

    This runs while the *old* attempt remains active. Only after the admitted
    fresh tabletop pose is recorded by ``observe_reset_landing`` may the
    runner advance to next-trial v8 candidate filtering.
    """
    if (expected_action not in {
            "recover_held_key_then_reobserve", "recover_key_then_reobserve",
            "recover_released_key_then_reobserve",
            "await_repose_observation"} or
            not isinstance(runner, SessionRunner) or
            runner.current_decision().action != expected_action or
            not isinstance(clear_view, ClearViewForPerception) or
            not callable(state_for_observation) or
            not math.isfinite(max_key_state_skew_s) or
            max_key_state_skew_s <= 0):
        raise ValueError("reset key capture needs active held recovery and feedback")
    clear_view.before_key_capture()
    output = Path(key_evidence_dir).expanduser().resolve()
    observation = _capture_admitted_key(
        runner=runner, init_orchestrator=init_orchestrator,
        acquisition_metadata_for_request=acquisition_metadata_for_request,
        capture_root=capture_root, output=output, capture_id=capture_id,
        calibrated_camera_ids=calibrated_camera_ids,
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=image_hw, key_prompt=key_prompt, view_limits=view_limits,
        maximum_multiview_center_error_mm=maximum_multiview_center_error_mm,
        maximum_multiview_angle_error_deg=maximum_multiview_angle_error_deg,
        maximum_socket_mask_overlap_fraction=(
            maximum_socket_mask_overlap_fraction),
        socket_projection_dilation_px=socket_projection_dilation_px,
        minimum_refinement_iou=minimum_refinement_iou,
        silhouette_iterations=silhouette_iterations,
        silhouette_loss_threshold=silhouette_loss_threshold,
        capture_timeout_s=capture_timeout_s,
        image_write_timeout_s=image_write_timeout_s,
        request_id_factory=request_id_factory,
        registered_key_assistance=registered_key_assistance,
        pose_only=registered_key_assistance is None)
    state = clear_view.verify_state(
        state_for_observation(observation), phase="reset_key_exposure",
        require_fresh=False)
    observation.require_state_alignment(
        state_timestamp_s=state.sample_timestamp_s,
        maximum_skew_s=max_key_state_skew_s)
    return ReturnedKeyCapture(observation, state, output)


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
    minimum_refinement_iou: float,
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
    registered_key_assistance: RegisteredKeyAssistance | None = None,
    capture_initial_visual: Callable[[], object] | None = None,
) -> PreparedLiveKeyTrial:
    """Capture, admit, save, and preflight one fresh tabletop key.

    The socket calibration/catalogue come from ``runner`` and are never
    replaced here. The key pose and 13-DOF feedback must overlap in time;
    the runner hashes the measured feedback alongside this preflight.
    The callback must read a continuously sampled robot-state buffer, but
    must not manufacture a camera-time state from a later read. If any gate
    fails, no selected candidate or motor command is produced. Capture/evidence
    directories already created on failure remain available for diagnosis.
    ``registered_key_assistance`` is an explicit alternative perception
    producer on the *same* acquired frames. It never falls back to stock
    full-frame masks after a failed ROI and does not change the planning gate.
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
    reject_incorrect_object_color(key_prompt)
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
            ("minimum_refinement_iou", minimum_refinement_iou),
            ("silhouette_loss_threshold", silhouette_loss_threshold),
            ("max_arm_hand_skew_s", max_arm_hand_skew_s),
            ("max_hand_command_error_raw", max_hand_command_error_raw),
            ("max_arm_velocity_rad_s", max_arm_velocity_rad_s),
            ("max_pose_error_deg", max_pose_error_deg),
            ("axial_waypoint_step_m", axial_waypoint_step_m)):
        if (type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if (not 0 < minimum_refinement_iou <= 1 or
            not 0 <= maximum_socket_mask_overlap_fraction < 1 or
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
    observation = _capture_admitted_key(
        runner=runner, init_orchestrator=init_orchestrator,
        acquisition_metadata_for_request=acquisition_metadata_for_request,
        capture_root=capture_root, output=output, capture_id=capture_id,
        calibrated_camera_ids=calibrated_camera_ids,
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=image_hw, key_prompt=key_prompt, view_limits=view_limits,
        maximum_multiview_center_error_mm=maximum_multiview_center_error_mm,
        maximum_multiview_angle_error_deg=maximum_multiview_angle_error_deg,
        maximum_socket_mask_overlap_fraction=(
            maximum_socket_mask_overlap_fraction),
        socket_projection_dilation_px=socket_projection_dilation_px,
        minimum_refinement_iou=minimum_refinement_iou,
        silhouette_iterations=silhouette_iterations,
        silhouette_loss_threshold=silhouette_loss_threshold,
        capture_timeout_s=capture_timeout_s,
        image_write_timeout_s=image_write_timeout_s,
        request_id_factory=request_id_factory,
        registered_key_assistance=registered_key_assistance,
        pose_only=capture_initial_visual is not None)
    visual_future = None
    visual_ready = None
    if capture_initial_visual is not None:
        if not callable(capture_initial_visual):
            raise TypeError("initial visual capture must be callable")
        visual_pool = ThreadPoolExecutor(max_workers=1,
                                         thread_name_prefix="initial-visual")
        visual_ready = Future()
        context = copy_context()
        def save_visual():
            try:
                with timed("key.initial_visual_acquire", phase="key",
                           kind="camera"):
                    raw = capture_initial_visual()
                visual_ready.set_result(None)
                with timed("key.initial_visual_save", phase="key", kind="io"):
                    return write_initial_visual_capture(output, raw)
            except BaseException as exc:
                if not visual_ready.done():
                    visual_ready.set_exception(exc)
                raise
        visual_future = visual_pool.submit(context.run, save_visual)
        register_pending_initial_visual(output, visual_future)
        visual_future.add_done_callback(
            lambda _future: visual_pool.shutdown(wait=False))
    try:
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
        with timed("key.candidate_preflight", phase="planning", kind="planning"):
            preflight = runner.preflight_next_key(
                planner=planner, key_observation=observation,
                key_evidence_dir=output, live_start_q=state.full_q.copy(),
                start_q_acquisition_timestamp_s=state.sample_timestamp_s,
                measured_start_state=state,
                measured_state_limits={
                    "max_arm_hand_skew_s": max_arm_hand_skew_s,
                    "max_hand_command_error_raw": max_hand_command_error_raw,
                    "max_arm_velocity_rad_s": max_arm_velocity_rad_s,
                },
                max_key_state_skew_s=max_key_state_skew_s, limits=limits,
                max_pose_error_deg=max_pose_error_deg,
                axial_waypoint_step_m=axial_waypoint_step_m,
                max_candidate_attempts=max_candidate_attempts,
                **dict(planning_options))
    finally:
        if visual_ready is not None:
            with timed("key.initial_visual_acquisition_wait", phase="key",
                       kind="wait"):
                visual_ready.result()
            if visual_future.done():
                visual_future.result()
    return PreparedLiveKeyTrial(observation, state, preflight, output)
