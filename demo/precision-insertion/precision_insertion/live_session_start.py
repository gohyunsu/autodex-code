"""Orchestrate one non-motion AutoDex precision-insertion session start.

Only existing AutoDex capture/FoundPose primitives and the demo's stricter
frame-provenance adapters are used. The order is board, socket init, socket
observation, then frozen collision world and immutable evidence.
This does not commission the camera clock, FoundPose model or robot motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import secrets
from typing import Callable, Mapping

import numpy as np

from .assets import AssetPaths
from .calibration import SessionCalibration, load_session_calibration
from .config import TaskMode
from .geometry import validate_se3
from .fixture_contract import validate_cylinder_socket_fixture
from .live_capture import collect_board_snapshot, collect_socket_capture
from .perception_evidence import SocketViewLimits
from .prompt_contract import reject_incorrect_object_color
from .silhouette_compat import prepare_precision_silhouette
from .socket_refinement import refine_socket_capture
from .session_bootstrap import (
    bootstrap_session, write_session_bootstrap_artifacts,
)
from .timing_profile import timed


@dataclass(frozen=True)
class StartedPrecisionSession:
    calibration: SessionCalibration
    evidence_dir: Path
    board_request_id: int
    socket_request_ids: tuple[int, ...]
    producer_manifest: dict | None = None


def start_precision_session(
    *, mode: TaskMode, shared_root: Path,
    snapshot_orchestrator, init_orchestrator,
    acquisition_metadata_for_request: Callable[[int], Mapping],
    capture_root: Path, evidence_dir: Path,
    calibrated_camera_ids: set[str],
    intrinsics_full: Mapping, extrinsics_full: Mapping,
    image_hw: tuple[int, int], c2r: np.ndarray,
    base_scene: dict, view_limits: SocketViewLimits,
    socket_prompt: str, socket_capture_count: int,
    board_timeout_s: float, socket_timeout_s: float,
    max_socket_translation_mm: float,
    max_socket_angle_deg: float,
    minimum_socket_refined_iou: float,
    socket_perception_mode: str = "iou",
    image_write_timeout_s: float = 5.0,
    request_id_factory: Callable[[], int] | None = None,
    board_id: str = "11",
    minimum_board_corners: int = 54,
    board_capture_phase: str = "empty_before_socket",
    confirm_socket_placement: Callable[[], bool] | None = None,
    fixture_contract_validated: bool = False,
) -> StartedPrecisionSession:
    """Acquire and freeze one session, without touching Franka/Inspire.

    Cameras must already be streaming and the key must be absent. In the
    empty-board phase, the operator places the fixed socket only after board
    preflight; the explicit partial-board phase measures around an already
    placed socket. The supplied metadata provider must
    be the independently commissioned *same-frame* acquisition-time source;
    the normal AutoDex publisher timestamps cannot pass the lower-level gate.
    A failed attempt leaves capture-side files for review and never replaces
    an older session evidence directory.
    """
    root = Path(shared_root).expanduser().resolve()
    capture_source = Path(capture_root).expanduser()
    if not capture_source.is_absolute():
        raise ValueError("socket capture root must be an absolute shared directory")
    capture_root = capture_source.resolve()
    output = Path(evidence_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"session evidence already exists: {output}")
    if not capture_root.is_dir():
        raise ValueError("socket capture root must be an existing shared directory")
    if (type(socket_capture_count) is not int or socket_capture_count < 1 or
            not isinstance(socket_prompt, str) or
            not socket_prompt.strip() or socket_prompt == "object"):
        raise ValueError("session needs a socket capture and a specific prompt")
    reject_incorrect_object_color(socket_prompt)
    for name, value in (("board_timeout_s", board_timeout_s),
                        ("socket_timeout_s", socket_timeout_s),
                        ("image_write_timeout_s", image_write_timeout_s),
                        ("max_socket_translation_mm", max_socket_translation_mm),
                        ("max_socket_angle_deg", max_socket_angle_deg)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (type(minimum_socket_refined_iou) not in (int, float) or
            not math.isfinite(minimum_socket_refined_iou) or
            not 0 < minimum_socket_refined_iou <= 1):
        raise ValueError("commissioned minimum_socket_refined_iou must be in (0,1]")
    if socket_perception_mode not in ("iou", "ignore_sil_loss"):
        raise ValueError("unknown stock socket perception mode")
    view_limits.validate()
    if (not isinstance(calibrated_camera_ids, set) or
            len(calibrated_camera_ids) < view_limits.minimum_accepted_views or
            set(intrinsics_full) != calibrated_camera_ids or
            set(extrinsics_full) != calibrated_camera_ids):
        raise ValueError("active camera IDs must match the frozen calibration")
    if (not isinstance(image_hw, tuple) or len(image_hw) != 2 or
            any(type(size) is not int or size <= 0 for size in image_hw)):
        raise ValueError("undistorted camera image shape must be positive H,W")
    validate_se3(c2r, name="Franka hand-eye C2R")
    if board_id not in ("9", "11"):
        raise ValueError("session board_id must be 9 or 11")
    if (type(minimum_board_corners) is not int or
            not 36 <= minimum_board_corners <= 54 or
            (board_id == "11" and minimum_board_corners != 54)):
        raise ValueError("session minimum_board_corners is invalid")
    if board_capture_phase not in ("empty_before_socket", "socket_present_partial"):
        raise ValueError("session board_capture_phase is invalid")
    if (board_capture_phase == "empty_before_socket" and
            (minimum_board_corners != 54 or
             not callable(confirm_socket_placement))):
        raise ValueError("empty-board session needs all corners and an operator placement gate")
    if (board_capture_phase == "socket_present_partial" and
            (board_id != "9" or minimum_board_corners == 54)):
        raise ValueError("socket-present partial board mode needs board 9 and an explicit lower corner limit")
    if (not isinstance(base_scene, dict) or
            not isinstance(base_scene.get("mesh"), dict) or
            not isinstance(base_scene.get("cuboid"), dict)):
        raise ValueError("base cuRobo scene needs mesh and cuboid dictionaries")
    paths = AssetPaths(root, mode)
    if mode.family == "cylinder" and not fixture_contract_validated:
        # Reject source-PC fixture paths or inconsistent CAD before the first
        # board/socket camera request. The validated SessionLaunch passes its
        # one-time result into this capture boundary; direct callers recheck.
        with timed("socket.fixture_asset_check", phase="socket", kind="io"):
            validate_cylinder_socket_fixture(shared_root=root, mode=mode)
    socket_mesh = paths.raw_mesh(mode.socket_object)
    socket_repre = paths.foundpose_repre(mode.socket_object)
    socket_collision = paths.socket_collision_mesh
    for name, path in (("v8 socket mesh", socket_mesh),
                       ("socket FoundPose representation", socket_repre),
                       ("exact socket collision mesh", socket_collision)):
        if not path.is_file():
            raise FileNotFoundError(f"{name} is missing: {path}")
    if not callable(acquisition_metadata_for_request):
        raise TypeError("session needs an acquisition-time metadata provider")

    used_requests: set[int] = set()

    def next_request_id() -> int:
        value = (request_id_factory() if request_id_factory is not None else
                 secrets.randbelow(2**31 - 1) + 1)
        if (type(value) is not int or not 0 < value < 2**31 or
                value in used_requests):
            raise ValueError("session request IDs must be unique positive int31")
        used_requests.add(value)
        return value

    with timed("board.capture", phase="board", kind="camera"):
        board = collect_board_snapshot(
            snapshot_orchestrator=snapshot_orchestrator,
            calibrated_camera_ids=calibrated_camera_ids,
            acquisition_metadata_for_request=acquisition_metadata_for_request,
            timeout_s=float(board_timeout_s), request_id_factory=next_request_id)
    # Only the empty-board workflow needs a fit before the operator places
    # the socket. With the socket already present, calibration performs the
    # same board fit during the final freeze, so an early fit is redundant.
    if board_capture_phase == "empty_before_socket":
        from src.execution.charuco_tabletop import measure_tabletop_from_images
        with timed("board.charuco_fit", phase="board"):
            measure_tabletop_from_images(
                board.images_bgr, intrinsics_full, extrinsics_full, c2r,
                board_id=board_id, minimum_board_corners=minimum_board_corners)
    if (board_capture_phase == "empty_before_socket" and
            not _confirmed_socket_placement(confirm_socket_placement)):
        raise RuntimeError("operator did not confirm socket placement")
    with timed("socket.model_init", phase="socket"):
        prepare_precision_silhouette(
            init_orchestrator=init_orchestrator,
            object_name=mode.socket_object, raw_mesh=socket_mesh)
        init_orchestrator.init_object(
            obj_name=mode.socket_object, mesh_path=str(socket_mesh),
            assets_root=str(paths.foundpose_assets_root(mode.socket_object)),
            intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
            image_hw=image_hw, mode="live", load_silhouette=False)
    captures = []
    for index in range(socket_capture_count):
        with timed("socket.capture", phase="socket", kind="camera",
                   capture_index=index, camera_count=len(calibrated_camera_ids)):
            captures.append(collect_socket_capture(
                init_orchestrator=init_orchestrator,
                socket_object=mode.socket_object,
                capture_id=f"socket_{index:03d}", socket_prompt=socket_prompt,
                capture_root=capture_root,
                calibrated_camera_ids=calibrated_camera_ids,
                acquisition_metadata_for_request=acquisition_metadata_for_request,
                timeout_s=float(socket_timeout_s),
                image_write_timeout_s=float(image_write_timeout_s),
                request_id_factory=next_request_id))
    with timed("socket.bootstrap_refinement", phase="socket"):
        bootstrap = bootstrap_session(
            mode=mode, object_root=paths.object_root,
            board_request_id=board.request_id,
            board_images_bgr=board.images_bgr,
            board_timestamps_s=board.frame_timestamps_s,
            board_timestamp_source=board.frame_timestamp_source,
            board_frame_evidence=board.frame_evidence,
            socket_captures=captures,
            calibrated_camera_ids=calibrated_camera_ids,
            view_limits=view_limits,
            intrinsics_full=intrinsics_full,
            extrinsics_full=extrinsics_full,
            c2r=c2r, base_scene=base_scene,
            socket_collision_mesh=socket_collision,
            board_id=board_id,
            minimum_board_corners=minimum_board_corners,
            board_capture_phase=board_capture_phase,
            max_socket_translation_mm=float(max_socket_translation_mm),
            max_socket_angle_deg=float(max_socket_angle_deg),
            socket_refiner=lambda capture, admission: refine_socket_capture(
                capture=capture, admission=admission,
                init_orchestrator=init_orchestrator,
                socket_object=mode.socket_object, raw_mesh=socket_mesh,
                minimum_refined_iou=float(minimum_socket_refined_iou),
                perception_mode=socket_perception_mode))
    with timed("socket.evidence_write", phase="socket", kind="io"):
        saved = write_session_bootstrap_artifacts(bootstrap, output)
    # Every source frame was bound before writing and each PNG digest was
    # produced from the encoded bytes. Re-reading all newly written PNGs here
    # duplicates the producer's work; keep the full verifier for offline audit.
    with timed("socket.manifest_read", phase="socket", kind="io"):
        manifest = json.loads((saved / "evidence_manifest.json").read_text(
            encoding="utf-8"))
    if (not isinstance(manifest, dict) or
            manifest.get("schema") !=
            "precision_insertion_session_evidence_bundle_v1" or
            manifest.get("board_request_id") != board.request_id or
            manifest.get("socket_request_ids") != [
                capture.request_id for capture in captures] or
            manifest.get("socket_capture_ids") != [
                capture.capture_id for capture in captures] or
            manifest.get("all_frames_bound_to_acquisition_evidence") is not True):
        raise ValueError("session evidence lacks acquisition-bound camera frames")
    with timed("socket.calibration_reload", phase="socket", kind="io"):
        reopened = load_session_calibration(
            saved / "session_calibration.json", mode=mode, shared_root=root)
    return StartedPrecisionSession(
        calibration=reopened, evidence_dir=saved,
        board_request_id=board.request_id,
        socket_request_ids=tuple(capture.request_id for capture in captures),
        producer_manifest=manifest)


def _confirmed_socket_placement(confirm: Callable[[], bool]) -> bool:
    with timed("socket.operator_placement_wait", phase="socket", kind="operator"):
        return confirm() is True
