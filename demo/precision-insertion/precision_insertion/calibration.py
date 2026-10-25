"""Read-only socket calibration for one insertion session.

The caller acquires synchronized AutoDex images and quality-checked FoundPose
observations. Board sessions reuse AutoDex's ChArUco measurement; board-free
sessions use an explicitly configured support height. Both freeze an
*observed* medoid of one refined pose per independent socket capture, and
adds the exact socket mesh to a copy of the collision scene. It neither
captures images nor connects to, commands, or authorizes the robot.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .config import TaskMode
from .geometry import freeze_fixture_pose, validate_se3
from .symmetry import load_axial_symmetry
from .world import add_fixed_mesh_fixtures


@dataclass(frozen=True)
class SocketObservation:
    capture_id: str
    camera_id: str
    timestamp_s: float
    pose_world: np.ndarray | None
    timestamp_source: str


@dataclass(frozen=True)
class SocketCaptureRefinement:
    """One multi-view socket result from one independently timed capture."""

    capture_id: str
    request_id: int
    pose_world: np.ndarray
    accepted_camera_ids: tuple[str, ...]
    selected_camera_id: str
    per_view_iou: dict[str, float]
    minimum_refined_iou: float
    stock_selection: dict
    raw_mesh_sha256: str
    refinement_gate: str = "all_view_iou"

    def to_record(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "request_id": self.request_id,
            "pose_world": validate_se3(self.pose_world,
                                        name="refined socket pose").tolist(),
            "accepted_camera_ids": list(self.accepted_camera_ids),
            "selected_camera_id": self.selected_camera_id,
            "per_view_iou": dict(self.per_view_iou),
            "minimum_refined_iou": self.minimum_refined_iou,
            "stock_selection": dict(self.stock_selection),
            "raw_mesh_sha256": self.raw_mesh_sha256,
            "refinement_gate": self.refinement_gate,
            "scope": "one_refined_capture_not_independent_pose_ground_truth",
        }


@dataclass(frozen=True)
class SessionCalibration:
    board: dict
    socket_pose_robot: np.ndarray
    socket_diagnostics: dict
    collision_scene: dict
    record: dict


def validate_session_camera_calibration(
    calibration: SessionCalibration, *,
    intrinsics_undist: Mapping[str, np.ndarray],
    extrinsics_full: Mapping[str, np.ndarray],
    calibrated_camera_ids: set[str],
    c2r: np.ndarray | None = None,
) -> None:
    """Reject a key/retry camera rig different from the frozen session rig.

    Older offline records without a camera snapshot remain loadable but cannot
    satisfy this live-perception gate. A digest detects changed saved bytes;
    it does not certify physical camera stability or hand-eye accuracy.
    """
    record = calibration.record
    if c2r is not None and not np.allclose(
            validate_se3(record.get("c2r"), name="frozen hand-eye C2R"),
            validate_se3(c2r, name="live hand-eye C2R"),
            rtol=0, atol=1e-10):
        raise ValueError("live hand-eye C2R differs from frozen session")
    snapshot = record.get("camera_calibration")
    if (not isinstance(snapshot, dict) or
            record.get("camera_calibration_sha256") !=
            _canonical_sha256(snapshot)):
        raise ValueError("session lacks a valid frozen camera calibration")
    stored_intrinsics = snapshot.get("intrinsics_full")
    stored_extrinsics = snapshot.get("extrinsics_full")
    if (not isinstance(stored_intrinsics, dict) or
            not isinstance(stored_extrinsics, dict) or
            set(stored_intrinsics) != calibrated_camera_ids or
            set(stored_extrinsics) != calibrated_camera_ids or
            set(intrinsics_undist) != calibrated_camera_ids or
            set(extrinsics_full) != calibrated_camera_ids):
        raise ValueError("live camera IDs differ from frozen session calibration")
    for serial in calibrated_camera_ids:
        if not isinstance(stored_intrinsics[serial], Mapping):
            raise ValueError(f"invalid saved camera intrinsics: {serial}")
        stored_K = np.asarray(
            stored_intrinsics[serial].get("K_undist"), dtype=float)
        live_K = np.asarray(intrinsics_undist[serial], dtype=float)
        stored_E = validate_se3(stored_extrinsics[serial],
                                name=f"saved {serial} extrinsic")
        live_E = validate_se3(extrinsics_full[serial],
                              name=f"live {serial} extrinsic")
        if (stored_K.shape != (3, 3) or live_K.shape != (3, 3) or
                not np.all(np.isfinite(stored_K)) or
                not np.all(np.isfinite(live_K)) or
                not np.allclose(stored_K, live_K, rtol=0, atol=1e-10) or
                not np.allclose(stored_E, live_E, rtol=0, atol=1e-10)):
            raise ValueError(f"live camera calibration changed: {serial}")


def _json_native(value):
    """Snapshot a cuRobo scene without stringifying unknown Python objects."""
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("collision scene keys must be strings")
        return {key: _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("collision scene contains a non-finite number")
        return value
    if type(value) in (str, int, bool) or value is None:
        return value
    raise TypeError(f"collision scene contains unsupported {type(value).__name__}")


def _canonical_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def write_session_calibration(calibration: SessionCalibration, path: Path) -> Path:
    """Save the versioned evidence record without replacing a prior session.

    Captured images and masks are separate evidence artifacts; the caller
    must retain them under the capture IDs in ``socket_observations``.
    """
    target = Path(path).expanduser().resolve()
    payload = json.dumps(calibration.record, indent=2, sort_keys=True) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    return target


def load_session_calibration(
    path: Path, *, mode: TaskMode, shared_root: Path,
) -> SessionCalibration:
    """Reconstruct and revalidate a saved frozen world for offline replay.

    Older records without a collision-world snapshot cannot be replayed:
    rebuilding from a current default scene would silently change obstacles.
    This is read-only and does not establish live physical fixture stability.
    """
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != (
            "precision_insertion_session_calibration_v1"):
        raise ValueError("unknown session calibration record schema")
    scene = record.get("collision_scene")
    if not isinstance(scene, dict) or not isinstance(scene.get("mesh"), dict) or (
            not isinstance(scene.get("cuboid"), dict)):
        raise ValueError("saved session has no frozen collision world snapshot")
    if record.get("collision_scene_sha256") != _canonical_sha256(scene):
        raise ValueError("saved collision world descriptor hash changed")
    camera_calibration = record.get("camera_calibration")
    camera_calibration_hash = record.get("camera_calibration_sha256")
    if camera_calibration is not None or camera_calibration_hash is not None:
        if (not isinstance(camera_calibration, dict) or
                camera_calibration_hash != _canonical_sha256(
                    camera_calibration)):
            raise ValueError("saved camera calibration changed or is incomplete")
    scene_mode = record.get("scene_mode", "charuco_board")
    if scene_mode not in ("charuco_board", "board_free"):
        raise ValueError("unknown saved scene mode")
    board = (record.get("support_surface") if scene_mode == "board_free"
             else record.get("board"))
    if not isinstance(board, dict):
        raise ValueError("saved support surface is missing")
    if scene_mode == "board_free":
        if (board.get("source") != "configured_table_surface" or
                type(board.get("table_surface_z_m")) not in (int, float) or
                not math.isfinite(board["table_surface_z_m"]) or
                record.get("board") is not None or
                record.get("board_timestamp_source") is not None or
                record.get("socket_pose_source") !=
                "per_capture_stock_multiview_refinement"):
            raise ValueError("board-free support surface is invalid")
    elif record.get("board_timestamp_source") != "camera_acquisition":
        raise ValueError("saved ChArUco timestamp source is invalid")
    if (not isinstance(record.get("socket_observations"), list) or
            not record["socket_observations"] or
            any(not isinstance(row, dict) or
                row.get("timestamp_source") != "camera_acquisition"
                for row in record["socket_observations"])):
        raise ValueError("saved session lacks camera acquisition timestamps")
    from autodex.utils.tabletop_geometry import table_cuboid

    if scene["cuboid"].get("table") != table_cuboid(board):
        raise ValueError("saved table cuboid differs from support surface")
    hashes = record.get("fixed_mesh_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(scene["mesh"]):
        raise ValueError("saved fixed mesh hashes are missing")
    for name, entry in scene["mesh"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("file_path"), str):
            raise ValueError(f"saved fixed mesh {name} has no file path")
        mesh_path = Path(entry["file_path"]).expanduser().resolve()
        if not mesh_path.is_file() or _file_sha256(mesh_path) != hashes[name]:
            raise ValueError(f"saved fixed mesh {name} changed or is missing")
    validate_se3(record.get("c2r"), name="saved C2R")
    socket_pose = validate_se3(record.get("socket_pose_robot"),
                               name="saved socket pose")
    diagnostics = record.get("socket_diagnostics")
    if not isinstance(diagnostics, dict) or diagnostics.get("accepted") is not True:
        raise ValueError("saved socket calibration was not accepted")
    if record.get("socket_pose_source") in (
            "per_capture_stock_multiview_refinement",
            "per_capture_board_constrained_body"):
        _verify_saved_socket_refinements(
            record, mode=mode, shared_root=shared_root, socket_pose=socket_pose)
    elif record.get("socket_pose_source") not in (None, "per_view_foundpose_legacy"):
        raise ValueError("unknown saved socket pose source")
    session = SessionCalibration(board, socket_pose, diagnostics, scene, record)
    from .world import validated_frozen_socket_pose

    validated_frozen_socket_pose(
        mode=mode, shared_root=shared_root, calibration=session)
    return session


def _finite_time(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite_time(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _stock_socket_perception_mode(selection: dict) -> str:
    """Validate the recorded stock silhouette policy without serializing inf."""
    mode = selection.get("perception_mode", "iou")
    if (not isinstance(mode, str) or
            mode not in ("iou", "ignore_sil_loss")):
        raise ValueError("unknown saved socket perception mode")
    try:
        loss = float(selection["sil_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("socket refinement lacks stock silhouette loss") from exc
    if not math.isfinite(loss):
        raise ValueError("socket refinement has nonfinite silhouette loss")
    if mode == "ignore_sil_loss":
        if (selection.get("silhouette_loss_gate_applied") is not False or
                selection.get("silhouette_loss_threshold") is not None):
            raise ValueError("stock ignore_sil_loss record has a loss gate")
    else:
        try:
            limit = float(selection["silhouette_loss_threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("socket refinement lacks stock silhouette limit") from exc
        if (not math.isfinite(limit) or limit <= 0 or loss > limit or
                selection.get("silhouette_loss_gate_applied", True) is not True):
            raise ValueError("socket refinement failed stock silhouette loss")
    return mode


def _check_board_supported_socket(pose_robot: np.ndarray, board: dict) -> None:
    """Check the measured fixture relation, not an arbitrary angle limit."""
    fixture = validate_se3(pose_robot, name="board-supported socket")
    board_pose = validate_se3(board.get("T_robot_board"),
                              name="socket support board")
    up = -board_pose[:3, 2]
    if (not np.allclose(fixture[:3, 2], up, atol=1e-6) or
            abs(float(np.dot(fixture[:3, 3] - board_pose[:3, 3], up))) > 1e-6):
        raise ValueError("socket is not supported upright on measured board")


def _check_body_fit_scores(
    per_view: dict, selection: dict, threshold: float,
) -> None:
    """Validate the saved wall-score summary without a per-camera veto."""
    fit = selection.get("body_fit")
    if not isinstance(fit, dict) or fit.get("per_view_iou") != per_view:
        raise ValueError("socket body fit scores differ from admitted views")
    try:
        scores = [float(score) for score in per_view.values()]
        recorded_mean = float(fit["mean_body_iou"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("socket body fit score is invalid") from exc
    mean = float(np.mean(scores))
    if (not scores or not all(math.isfinite(score) and 0 <= score <= 1
                              for score in scores) or
            not math.isfinite(recorded_mean) or
            not math.isclose(recorded_mean, mean, rel_tol=1e-8, abs_tol=1e-8) or
            mean < threshold):
        raise ValueError("socket body projection does not meet its mean IoU limit")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_saved_socket_refinements(
    record: dict, *, mode: TaskMode, shared_root: Path, socket_pose: np.ndarray,
) -> None:
    """Replay one pose per capture; a record hash alone is not a pose check."""
    refinements = record.get("socket_capture_refinements")
    if (not isinstance(refinements, list) or len(refinements) < 1 or
            record.get("socket_capture_refinements_sha256") !=
            _canonical_sha256({"refinements": refinements})):
        raise ValueError("saved socket refinements changed or are missing")
    expected_board_body = (record["socket_pose_source"] ==
                           "per_capture_board_constrained_body")
    raw_mesh = (Path(shared_root).expanduser().resolve() / "object_processing" /
                mode.socket_object / "raw_mesh" / f"{mode.socket_object}.obj")
    if not raw_mesh.is_file():
        raise ValueError("saved refined socket raw mesh is missing")
    raw_mesh_hash = _file_sha256(raw_mesh)
    c2r = validate_se3(record["c2r"], name="saved C2R")
    robot_from_world = np.linalg.inv(c2r)
    raw_by_capture: dict[str, set[str]] = {}
    for observation in record["socket_observations"]:
        capture_id = observation.get("capture_id")
        camera_id = observation.get("camera_id")
        if not isinstance(capture_id, str) or not isinstance(camera_id, str):
            raise ValueError("saved socket observation identity is invalid")
        cameras = raw_by_capture.setdefault(capture_id, set())
        if camera_id in cameras:
            raise ValueError("saved socket observation repeats a camera")
        cameras.add(camera_id)
        if expected_board_body:
            if (observation.get("pose_world") is None) != (
                    observation.get("pose_robot") is None):
                raise ValueError("saved per-view socket pose is incomplete")
        if not expected_board_body or observation.get("pose_world") is not None:
            raw_world = validate_se3(observation.get("pose_world"),
                                     name="saved per-view socket pose")
            raw_robot = validate_se3(observation.get("pose_robot"),
                                     name="saved per-view robot socket pose")
            if not np.allclose(robot_from_world @ raw_world, raw_robot, atol=1e-8):
                raise ValueError("saved per-view socket pose differs from C2R")
    diagnostics = record["socket_diagnostics"]
    if (len(refinements) != len(raw_by_capture) or
            diagnostics.get("sample_count") != len(refinements)):
        raise ValueError("saved socket refinement count differs from captures")
    refined_robot = []
    seen_captures: set[str] = set()
    seen_requests: set[int] = set()
    for row in refinements:
        if not isinstance(row, dict):
            raise ValueError("invalid saved socket refinement")
        capture_id = row.get("capture_id")
        cameras = raw_by_capture.get(capture_id)
        accepted = row.get("accepted_camera_ids")
        per_view = row.get("per_view_iou")
        selected = row.get("selected_camera_id")
        request_id = row.get("request_id")
        selection = row.get("stock_selection")
        gate = row.get("refinement_gate", "all_view_iou")
        board_body = gate == "board_constrained_body"
        if (not isinstance(capture_id, str) or capture_id in seen_captures or
                board_body != expected_board_body or
                cameras is None or not isinstance(accepted, list) or
                not all(isinstance(camera, str) for camera in accepted) or
                len(accepted) != len(cameras) or set(accepted) != cameras or
                not isinstance(per_view, dict) or set(per_view) != cameras or
                selected not in cameras or type(request_id) is not int or
                request_id <= 0 or request_id in seen_requests or
                row.get("raw_mesh_sha256") != raw_mesh_hash or
                not isinstance(selection, dict) or
                selection.get("selection_mode") != (
                    "board_constrained_body" if board_body else "iou") or
                selection.get("best_serial") != selected or
                selection.get("n_candidates") != len(cameras) or
                selection.get("n_masks") != len(cameras) or
                selection.get("sil_reject") or
                bool(selection.get("sil_skipped")) != board_body or
                type(selection.get("silhouette_iterations")) is not int or
                (selection["silhouette_iterations"] != 0 if board_body else
                 selection["silhouette_iterations"] <= 0)):
            raise ValueError("saved socket refinement source binding is invalid")
        try:
            threshold = float(row["minimum_refined_iou"])
            scores = [float(per_view[camera]) for camera in cameras]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("saved socket refinement quality is invalid") from exc
        perception_mode = ("board_constrained_body" if board_body else
                           _stock_socket_perception_mode(selection))
        if (not 0 < threshold <= 1 or
                gate not in ("all_view_iou", "stock_silhouette",
                             "stock_ignore_sil_loss", "board_constrained_body") or
                (gate == "stock_silhouette" and perception_mode != "iou") or
                (gate == "stock_ignore_sil_loss" and
                 perception_mode != "ignore_sil_loss") or
                any(not math.isfinite(score) or not 0 <= score <= 1
                    for score in scores) or
                (gate == "all_view_iou" and
                 any(score < threshold for score in scores))):
            raise ValueError("saved socket refinement quality gate failed")
        if board_body:
            if mode.family != "cylinder":
                raise ValueError("board body fit is only for a cylinder socket")
            _check_body_fit_scores(per_view, selection, threshold)
        seen_captures.add(capture_id)
        seen_requests.add(request_id)
        refined_pose = validate_se3(
            robot_from_world @ validate_se3(row.get("pose_world")),
            name="saved refined robot socket")
        if board_body:
            _check_board_supported_socket(refined_pose, record["board"])
        refined_robot.append(refined_pose)
    if seen_captures != set(raw_by_capture):
        raise ValueError("saved socket refinements do not cover all captures")
    axis = None
    if mode.family == "cylinder":
        axis = load_axial_symmetry(
            Path(shared_root).expanduser().resolve() / "object_processing",
            mode.socket_object).axis_local
    try:
        replayed, replay_diagnostics = freeze_fixture_pose(
            refined_robot,
            translation_limit_mm=float(diagnostics["translation_limit_mm"]),
            angle_limit_deg=float(diagnostics["angle_limit_deg"]),
            continuous_axis_local=axis)
    except (KeyError, TypeError) as exc:
        raise ValueError("saved socket repeatability limits are invalid") from exc
    if (not np.allclose(replayed, socket_pose, atol=1e-8) or
            replay_diagnostics != diagnostics):
        raise ValueError("saved socket pose differs from refined captures")


def calibrate_session(
    *,
    mode: TaskMode,
    object_root: Path,
    board_images_bgr: Mapping[str, np.ndarray],
    board_timestamps_s: Mapping[str, float],
    board_timestamp_source: str,
    socket_observations: Sequence[SocketObservation],
    intrinsics_full: Mapping,
    extrinsics_full: Mapping,
    c2r: np.ndarray,
    base_scene: dict,
    socket_collision_mesh: Path,
    max_capture_skew_s: float,
    max_socket_translation_mm: float,
    max_socket_angle_deg: float,
    min_socket_captures: int = 1,
    min_views_per_capture: int = 2,
    socket_capture_refinements: Sequence[SocketCaptureRefinement] | None = None,
    board_id: str = "11",
    minimum_board_corners: int = 54,
    board_capture_phase: str = "empty_before_socket",
    measured_board: dict | None = None,
    scene_mode: str = "charuco_board",
    table_surface_z_m: float | None = None,
) -> SessionCalibration:
    """Build one frozen calibration from supplied, already-captured evidence.

    A ``capture_id`` is one simultaneous multi-camera socket snapshot.
    Two different captures are required only to test fixture repeatability.
    ``pose_world`` uses the same world frame as ``c2r`` and AutoDex camera
    calibration. Bounds are explicit commissioning inputs, not defaults.

    When ``socket_capture_refinements`` is provided, each capture contributes
    exactly one independently refined pose. With one capture, its observed
    pose is frozen directly; raw per-view poses remain evidence.
    Geometric agreement alone does not establish absolute sub-mm accuracy or
    FoundPose photometric quality. The caller must reject bad masks/poses
    before supplying them and must separately validate hand-eye calibration.
    """
    skew = _positive(max_capture_skew_s, "max_capture_skew_s")
    translation_limit = _positive(
        max_socket_translation_mm, "max_socket_translation_mm")
    angle_limit = _positive(max_socket_angle_deg, "max_socket_angle_deg")
    if min_socket_captures < 1 or min_views_per_capture < 2:
        raise ValueError("require at least one socket capture and two views per capture")
    if scene_mode not in ("charuco_board", "board_free"):
        raise ValueError("unknown scene mode")
    if scene_mode == "board_free" and socket_capture_refinements is None:
        raise ValueError("board-free socket needs stock FoundPose refinement")
    if scene_mode == "charuco_board" and board_timestamp_source != "camera_acquisition":
        raise ValueError("ChArUco images require camera acquisition timestamps")
    if scene_mode == "charuco_board" and board_capture_phase not in ("empty_before_socket", "socket_present_partial"):
        raise ValueError("unknown board capture phase")
    if scene_mode == "charuco_board" and (board_capture_phase == "empty_before_socket" and
            minimum_board_corners != 54 or
            board_capture_phase == "socket_present_partial" and
            (board_id != "9" or minimum_board_corners == 54)):
        raise ValueError("board capture phase conflicts with board/corner contract")

    camera_ids = set(intrinsics_full) & set(extrinsics_full)
    if not camera_ids or set(intrinsics_full) != set(extrinsics_full):
        raise ValueError("intrinsic and extrinsic camera IDs must match and be nonempty")
    board_ids = set(board_images_bgr)
    if (board_ids != set(board_timestamps_s) or
            (scene_mode == "charuco_board" and not board_ids) or
            (scene_mode == "board_free" and bool(board_ids))):
        raise ValueError("board images and timestamps must have identical camera IDs")
    if not board_ids <= camera_ids:
        raise ValueError("board snapshot contains an uncalibrated camera")
    board_times = {
        camera: _finite_time(board_timestamps_s[camera], f"board timestamp {camera}")
        for camera in board_ids
    }
    if board_times and max(board_times.values()) - min(board_times.values()) > skew:
        raise ValueError("board snapshot is not synchronized")

    c2r_matrix = validate_se3(c2r, name="c2r")
    world_to_robot = np.linalg.inv(c2r_matrix)
    groups: dict[str, list[SocketObservation]] = {}
    for index, observation in enumerate(socket_observations):
        if not isinstance(observation, SocketObservation):
            raise TypeError(f"socket observation {index} must be SocketObservation")
        if not observation.capture_id or not observation.camera_id:
            raise ValueError("socket capture ID and camera ID must be nonempty")
        if observation.camera_id not in camera_ids:
            raise ValueError(f"uncalibrated socket camera {observation.camera_id}")
        if observation.timestamp_source != "camera_acquisition":
            raise ValueError("socket observation requires camera acquisition timestamp")
        _finite_time(observation.timestamp_s, "socket timestamp")
        if observation.pose_world is not None:
            validate_se3(observation.pose_world, name="socket pose_world")
        elif socket_capture_refinements is None:
            raise ValueError("unrefined socket observation needs FoundPose")
        groups.setdefault(observation.capture_id, []).append(observation)
    if len(groups) < min_socket_captures:
        raise ValueError("too few separate socket captures")

    refined_by_capture: dict[str, SocketCaptureRefinement] = {}
    body_fit_count = 0
    if socket_capture_refinements is not None:
        if (not isinstance(socket_capture_refinements, Sequence) or
                len(socket_capture_refinements) != len(groups)):
            raise ValueError("one socket refinement per capture is required")
        for refinement in socket_capture_refinements:
            if (not isinstance(refinement, SocketCaptureRefinement) or
                    refinement.capture_id not in groups or
                    refinement.capture_id in refined_by_capture):
                raise ValueError("socket refinements need unique capture IDs")
            cameras = {item.camera_id for item in groups[refinement.capture_id]}
            if (set(refinement.accepted_camera_ids) != cameras or
                    len(refinement.accepted_camera_ids) != len(cameras) or
                    refinement.selected_camera_id not in cameras or
                    set(refinement.per_view_iou) != cameras):
                raise ValueError("socket refinement cameras differ from admitted views")
            threshold = float(refinement.minimum_refined_iou)
            board_body = refinement.refinement_gate == "board_constrained_body"
            if (not math.isfinite(threshold) or not 0 < threshold <= 1 or
                    refinement.refinement_gate not in
                    ("all_view_iou", "stock_silhouette",
                     "stock_ignore_sil_loss", "board_constrained_body") or
                    any(not math.isfinite(float(score)) or
                            not 0 <= float(score) <= 1 or
                            (refinement.refinement_gate == "all_view_iou" and
                             float(score) < threshold)
                            for score in refinement.per_view_iou.values())):
                raise ValueError("socket refinement failed per-view IoU gate")
            if (type(refinement.request_id) is not int or
                    refinement.request_id <= 0 or
                    not isinstance(refinement.raw_mesh_sha256, str) or
                    len(refinement.raw_mesh_sha256) != 64):
                raise ValueError("socket refinement lacks source binding")
            selection = refinement.stock_selection
            if (not isinstance(selection, dict) or
                    selection.get("selection_mode") != (
                        "board_constrained_body" if board_body else "iou") or
                    selection.get("best_serial") != refinement.selected_camera_id or
                    selection.get("n_candidates") != len(cameras) or
                    selection.get("n_masks") != len(cameras) or
                    selection.get("sil_reject") or
                    bool(selection.get("sil_skipped")) != board_body or
                    type(selection.get("silhouette_iterations")) is not int or
                    (selection["silhouette_iterations"] != 0 if board_body else
                     selection["silhouette_iterations"] <= 0)):
                raise ValueError("socket refinement lacks passing stock selection")
            perception_mode = ("board_constrained_body" if board_body else
                               _stock_socket_perception_mode(selection))
            if (refinement.refinement_gate == "stock_silhouette" and
                    perception_mode != "iou" or
                    refinement.refinement_gate == "stock_ignore_sil_loss" and
                    perception_mode != "ignore_sil_loss"):
                raise ValueError("socket refinement gate differs from stock mode")
            if board_body:
                if mode.family != "cylinder":
                    raise ValueError("board body fit is only for a cylinder socket")
                _check_body_fit_scores(refinement.per_view_iou, selection, threshold)
            validate_se3(refinement.pose_world, name="refined socket pose_world")
            refined_by_capture[refinement.capture_id] = refinement
        request_ids = [row.request_id for row in refined_by_capture.values()]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("socket refinement request IDs must be unique")
        body_fit_count = sum(
            row.refinement_gate == "board_constrained_body"
            for row in refined_by_capture.values())
        if body_fit_count not in (0, len(refined_by_capture)):
            raise ValueError("socket captures use different localization methods")
    if not body_fit_count and any(
            observation.pose_world is None for observations in groups.values()
            for observation in observations):
        raise ValueError("stock socket refinement needs per-view FoundPose")

    ordered_groups = sorted(
        groups.items(), key=lambda item: min(float(o.timestamp_s) for o in item[1]))
    prior_capture_end = max(board_times.values()) if board_times else -math.inf
    for capture_id, observations in ordered_groups:
        ids = [o.camera_id for o in observations]
        if len(observations) < min_views_per_capture or len(ids) != len(set(ids)):
            raise ValueError(f"socket capture {capture_id}: too few or duplicate views")
        times = [float(o.timestamp_s) for o in observations]
        if max(times) - min(times) > skew:
            raise ValueError(f"socket capture {capture_id} is not synchronized")
        if min(times) <= prior_capture_end:
            if board_times:
                raise ValueError("socket captures must follow board capture in time")
            raise ValueError("socket captures must be sequential")
        prior_capture_end = max(times)

    mesh = Path(socket_collision_mesh).expanduser().resolve()
    if not mesh.is_file():
        raise FileNotFoundError(f"exact socket collision mesh not found: {mesh}")
    if refined_by_capture:
        raw_mesh = (Path(object_root).expanduser().resolve() /
                    mode.socket_object / "raw_mesh" /
                    f"{mode.socket_object}.obj")
        if not raw_mesh.is_file():
            raise ValueError("refined socket raw mesh differs from session asset")
        raw_mesh_hash = _file_sha256(raw_mesh)
        if any(row.raw_mesh_sha256 != raw_mesh_hash
               for row in refined_by_capture.values()):
            raise ValueError("refined socket raw mesh differs from session asset")
    if not isinstance(base_scene, dict):
        raise TypeError("base_scene must be a scene dictionary")
    if not isinstance(base_scene.get("mesh", {}), dict):
        raise TypeError("base_scene mesh must be a dictionary")
    if not isinstance(base_scene.get("cuboid", {}), dict):
        raise TypeError("base_scene cuboid must be a dictionary")
    if "target" in base_scene.get("mesh", {}):
        raise ValueError("session fixed collision world must not contain a key target")
    axis = None
    if mode.family == "cylinder":
        symmetry = load_axial_symmetry(Path(object_root), mode.socket_object)
        if symmetry.end_exchange:
            raise ValueError("cylindrical socket may not exchange open and closed ends")
        axis = symmetry.axis_local
    elif mode.family != "square":
        raise ValueError(f"unsupported insertion family: {mode.family}")

    from autodex.utils.tabletop_geometry import table_cuboid

    if scene_mode == "board_free":
        if (type(table_surface_z_m) not in (int, float) or
                not math.isfinite(table_surface_z_m) or measured_board is not None or
                board_timestamp_source != "not_applicable" or body_fit_count):
            raise ValueError("board-free scene needs a configured table surface")
        board = {"source": "configured_table_surface",
                 "table_surface_z_m": float(table_surface_z_m)}
    elif measured_board is None:
        from src.execution.charuco_tabletop import measure_tabletop_from_images
        board = measure_tabletop_from_images(
            board_images_bgr, intrinsics_full, extrinsics_full, c2r_matrix,
            board_id=board_id, minimum_board_corners=minimum_board_corners)
    else:
        board = measured_board
    if scene_mode == "charuco_board":
        if not isinstance(board, dict) or board.get("board") != board_id:
            raise ValueError("measured board does not match session board")
        validate_se3(board.get("T_robot_board"), name="measured T_robot_board")
    for refinement in refined_by_capture.values():
        if refinement.refinement_gate == "board_constrained_body":
            pose_robot = world_to_robot @ validate_se3(refinement.pose_world)
            _check_board_supported_socket(pose_robot, board)
    measured_base_scene = copy.deepcopy(base_scene)
    measured_base_scene.setdefault("cuboid", {})["table"] = table_cuboid(board)
    observations_robot: list[np.ndarray] = []
    source_rows: list[dict] = []
    for capture_id, observations in ordered_groups:
        for observation in sorted(observations, key=lambda o: o.camera_id):
            pose_world = (validate_se3(observation.pose_world,
                                       name="socket pose_world")
                          if observation.pose_world is not None else None)
            pose_robot = (validate_se3(world_to_robot @ pose_world,
                                       name="socket pose_robot")
                          if pose_world is not None else None)
            if socket_capture_refinements is None:
                observations_robot.append(pose_robot)
            source_rows.append({
                "capture_id": capture_id,
                "camera_id": observation.camera_id,
                "timestamp_s": float(observation.timestamp_s),
                "timestamp_source": observation.timestamp_source,
                "pose_world": pose_world.tolist() if pose_world is not None else None,
                "pose_robot": pose_robot.tolist() if pose_robot is not None else None,
            })
        if socket_capture_refinements is not None:
            # One independently captured/refined pose per capture. Repeating
            # it for each admitted camera would manufacture agreement.
            refined_world = validate_se3(
                refined_by_capture[capture_id].pose_world,
                name=f"refined socket {capture_id}")
            observations_robot.append(validate_se3(
                world_to_robot @ refined_world,
                name=f"refined robot socket {capture_id}"))
    selected, diagnostics = freeze_fixture_pose(
        observations_robot,
        translation_limit_mm=translation_limit,
        angle_limit_deg=angle_limit,
        continuous_axis_local=axis,
    )
    scene = add_fixed_mesh_fixtures(measured_base_scene, {
        "fixture_socket": {
            "pose_robot": selected,
            "collision_mesh": mesh,
        },
    })
    frozen_scene = _json_native(scene)
    camera_calibration = {
        "intrinsics_full": _json_native(intrinsics_full),
        "extrinsics_full": _json_native(extrinsics_full),
    }
    fixed_mesh_hashes = {}
    for name, entry in frozen_scene["mesh"].items():
        path = Path(entry["file_path"]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"fixed collision mesh missing: {path}")
        fixed_mesh_hashes[name] = _file_sha256(path)
    if not refined_by_capture:
        pose_source = "per_view_foundpose_legacy"
    elif body_fit_count:
        pose_source = "per_capture_board_constrained_body"
    else:
        pose_source = "per_capture_stock_multiview_refinement"
    record = {
        "schema": "precision_insertion_session_calibration_v1",
        "mode": {
            "family": mode.family,
            "gap_mm": mode.gap_mm,
            "key_object": mode.key_object,
            "socket_object": mode.socket_object,
        },
        "scene_mode": scene_mode,
        "socket_observations": source_rows,
        "socket_pose_source": pose_source,
        "socket_capture_refinements": [
            refined_by_capture[capture_id].to_record()
            for capture_id, _ in ordered_groups] if refined_by_capture else [],
        "socket_pose_robot": selected.tolist(),
        "socket_diagnostics": diagnostics,
        "socket_collision_mesh": str(mesh),
        "socket_collision_mesh_sha256": _file_sha256(mesh),
        "collision_scene": frozen_scene,
        "collision_scene_sha256": _canonical_sha256(frozen_scene),
        "camera_calibration": camera_calibration,
        "camera_calibration_sha256": _canonical_sha256(camera_calibration),
        "fixed_mesh_sha256": fixed_mesh_hashes,
        "c2r": c2r_matrix.tolist(),
        "scope": "read_only_session_calibration_not_robot_authorization",
        "robot_ready": False,
    }
    if scene_mode == "board_free":
        record["support_surface"] = board
    else:
        record.update(board=board, board_capture_phase=board_capture_phase,
                      board_timestamps_s=board_times,
                      board_timestamp_source=board_timestamp_source)
    record["socket_capture_refinements_sha256"] = _canonical_sha256({
        "refinements": record["socket_capture_refinements"]})
    return SessionCalibration(board, selected, diagnostics, scene, record)
