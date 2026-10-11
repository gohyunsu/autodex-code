"""Explicit, non-motion AutoDex camera-session startup from commissioned files.

Preflight loads every timing and CAD input before opening any network socket.
The capture path then reuses stock orchestrators plus demo-only provenance
adapters; it never connects to or commands Franka/Inspire.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import ipaddress
import json
import math
from pathlib import Path
import re
import time
from typing import Iterator

import numpy as np

from .assets import AssetPaths
from .camera_time import AcquisitionTimeProvider, CameraTimeCalibration
from .camera_transport import (
    FrameIdentityRegistry, PrecisionInitOrchestrator,
    ProvenanceSnapshotAdapter, SnapshotMetadataTap,
)
from .config import TaskMode, select_mode
from .fixture_contract import validate_cylinder_socket_fixture
from .geometry import validate_se3
from .live_session_start import StartedPrecisionSession, start_precision_session
from .perception_evidence import SocketViewLimits


_SCHEMA = "precision_insertion_live_session_launch_v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SessionLaunch:
    mode: TaskMode
    shared_root: Path
    capture_root: Path
    evidence_dir: Path
    pc_list: tuple[str, ...]
    capture_ips: tuple[str, ...]
    camera_ids: set[str]
    intrinsics_full: dict
    extrinsics_full: dict
    image_hw: tuple[int, int]
    c2r: np.ndarray
    base_scene: dict
    camera_times: dict[str, CameraTimeCalibration]
    view_limits: SocketViewLimits
    socket_prompt: str
    socket_capture_count: int
    board_timeout_s: float
    socket_timeout_s: float
    image_write_timeout_s: float
    max_socket_translation_mm: float
    max_socket_angle_deg: float
    minimum_socket_refined_iou: float

    def readiness(self) -> dict:
        return {
            "schema": "precision_insertion_live_session_readiness_v1",
            "mode": {"family": self.mode.family, "gap_mm": self.mode.gap_mm},
            "camera_ids": sorted(self.camera_ids),
            "camera_count": len(self.camera_ids),
            "camera_image_hw": list(self.image_hw),
            "capture_pcs": list(self.pc_list),
            "camera_time_calibration_ids": {
                serial: row.calibration_id
                for serial, row in sorted(self.camera_times.items())},
            "minimum_socket_refined_iou": self.minimum_socket_refined_iou,
            "socket_capture_count": self.socket_capture_count,
            "scope": "file_preflight_not_live_camera_or_pose_accuracy",
            "robot_ready": False,
        }


@dataclass(frozen=True)
class OpenSessionCameras:
    """Keep one provenance-bound camera transport alive across trial captures.

    The owner must hold ``open_session_cameras``' context open while using the
    init orchestrator or acquisition provider. Neither object commands a
    robot, and a returned ``StartedPrecisionSession`` is not motion approval.
    """

    launch: SessionLaunch
    snapshot_orchestrator: object
    init_orchestrator: object
    acquisition_metadata_for_request: AcquisitionTimeProvider

    def start(self) -> StartedPrecisionSession:
        """Capture ChArUco then repeated socket observations once."""
        launch = self.launch
        return start_precision_session(
            mode=launch.mode, shared_root=launch.shared_root,
            snapshot_orchestrator=self.snapshot_orchestrator,
            init_orchestrator=self.init_orchestrator,
            acquisition_metadata_for_request=self.acquisition_metadata_for_request,
            capture_root=launch.capture_root, evidence_dir=launch.evidence_dir,
            calibrated_camera_ids=launch.camera_ids,
            intrinsics_full=launch.intrinsics_full,
            extrinsics_full=launch.extrinsics_full,
            image_hw=launch.image_hw, c2r=launch.c2r,
            base_scene=launch.base_scene, view_limits=launch.view_limits,
            socket_prompt=launch.socket_prompt,
            socket_capture_count=launch.socket_capture_count,
            board_timeout_s=launch.board_timeout_s,
            socket_timeout_s=launch.socket_timeout_s,
            image_write_timeout_s=launch.image_write_timeout_s,
            max_socket_translation_mm=launch.max_socket_translation_mm,
            max_socket_angle_deg=launch.max_socket_angle_deg,
            minimum_socket_refined_iou=launch.minimum_socket_refined_iou)


def _path(value, *, name: str, kind: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"{name} needs an absolute {kind} path")
    source = Path(value).expanduser()
    if not source.is_absolute():
        raise ValueError(f"{name} needs an absolute {kind} path")
    path = source.resolve()
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(f"{name} is missing: {path}")
    if kind == "directory" and not path.is_dir():
        raise FileNotFoundError(f"{name} is missing: {path}")
    return path


def _positive(value, *, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} needs a commissioned positive value")
    return float(value)


def _load_stock_camera_calibration(directory: Path, camera_ids: set[str]):
    """Load the same JSON fields and array shapes as stock run_auto._load_calib."""
    intr_raw = json.loads((directory / "intrinsics.json").read_text(
        encoding="utf-8"))
    extr_raw = json.loads((directory / "extrinsics.json").read_text(
        encoding="utf-8"))
    if (not isinstance(intr_raw, dict) or not isinstance(extr_raw, dict) or
            not camera_ids <= set(intr_raw) or not camera_ids <= set(extr_raw)):
        raise ValueError("active cameras are absent from stock camera calibration")
    intrinsics, extrinsics = {}, {}
    image_hw = None
    for serial in sorted(camera_ids):
        row = intr_raw[serial]
        shape = (row["height"], row["width"])
        if (any(type(value) is not int or value <= 0 for value in shape) or
                (image_hw is not None and image_hw != shape)):
            raise ValueError("active AutoDex cameras need one positive image shape")
        image_hw = shape
        original = np.asarray(row["original_intrinsics"], dtype=float).reshape(3, 3)
        undistorted = np.asarray(row["intrinsics_undistort"], dtype=float).reshape(3, 3)
        distortion = np.asarray(row["dist_params"], dtype=float).reshape(-1)
        if (not np.all(np.isfinite(original)) or
                not np.all(np.isfinite(undistorted)) or
                not np.all(np.isfinite(distortion))):
            raise ValueError("camera intrinsics contain nonfinite values")
        intrinsics[serial] = {
            "K_orig": original, "K_undist": undistorted,
            "dist_params": distortion, "height": shape[0], "width": shape[1],
        }
        flat = np.asarray(extr_raw[serial], dtype=float).reshape(-1)
        if flat.size == 12:
            extrinsic = np.vstack((flat.reshape(3, 4), [0, 0, 0, 1]))
        elif flat.size == 16:
            extrinsic = flat.reshape(4, 4)
        else:
            raise ValueError("camera extrinsic needs a 3x4 or 4x4 matrix")
        extrinsics[serial] = validate_se3(
            extrinsic, name=f"{serial} world-to-camera extrinsic")
    return intrinsics, extrinsics, image_hw


def load_session_launch(path: Path) -> SessionLaunch:
    """Validate all static inputs without creating outputs or contacting cameras."""
    config_path = _path(str(path), name="launch config", kind="file")
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("schema") != _SCHEMA:
        raise ValueError("unknown live session launch schema")
    mode_record = record.get("mode")
    if not isinstance(mode_record, dict):
        raise ValueError("launch config needs an explicit socket/key mode")
    mode = select_mode(mode_record["family"], mode_record["gap_mm"])
    root = _path(record.get("shared_root"), name="shared_root", kind="directory")
    capture_root = _path(record.get("capture_root"), name="capture_root",
                         kind="directory")
    evidence = _path(record.get("evidence_dir"), name="evidence_dir",
                     kind="new directory")
    if evidence.exists():
        raise FileExistsError(f"session evidence already exists: {evidence}")
    if not evidence.parent.is_dir():
        raise FileNotFoundError("session evidence parent directory is missing")
    pcs = record.get("pc_list")
    ips = record.get("capture_ips")
    cameras = record.get("camera_ids")
    if (not isinstance(pcs, list) or not isinstance(ips, list) or
            not isinstance(cameras, list) or not pcs or len(pcs) != len(ips) or
            len(set(pcs)) != len(pcs) or len(set(ips)) != len(ips) or
            len(set(cameras)) != len(cameras) or len(cameras) < 2 or
            not all(isinstance(v, str) and v.strip()
                    for v in pcs + ips + cameras)):
        raise ValueError("launch needs unique capture PCs/IPs and camera IDs")
    if any(_SAFE_ID.fullmatch(value) is None for value in pcs + cameras):
        raise ValueError("capture PC and camera IDs must be path-safe")
    try:
        for address in ips:
            ipaddress.ip_address(address)
    except ValueError as exc:
        raise ValueError("capture PCs need literal IP addresses") from exc
    camera_ids = set(cameras)
    calib_dir = _path(record.get("calib_dir"), name="calib_dir",
                      kind="directory")
    intrinsics, extrinsics, image_hw = _load_stock_camera_calibration(
        calib_dir, camera_ids)
    timing_paths = record.get("camera_time_calibrations")
    if not isinstance(timing_paths, dict) or set(timing_paths) != camera_ids:
        raise ValueError("every active camera needs its own time calibration")
    camera_times = {serial: CameraTimeCalibration.load(
        _path(timing_paths[serial], name=f"{serial} time calibration", kind="file"))
        for serial in camera_ids}
    if (any(row.camera_serial != serial or row.valid_until_utc_s <= time.time()
            for serial, row in camera_times.items()) or
            len({row.calibration_id for row in camera_times.values()}) !=
            len(camera_times)):
        raise ValueError("camera-time calibration IDs, serials or expiry are invalid")
    c2r = validate_se3(np.load(_path(record.get("c2r_npy"), name="C2R",
                                     kind="file"), allow_pickle=False),
                       name="saved Franka C2R")
    base_scene = json.loads(_path(record.get("base_scene_json"),
                                  name="base scene", kind="file").read_text(
                                      encoding="utf-8"))
    if (not isinstance(base_scene, dict) or
            not isinstance(base_scene.get("mesh"), dict) or
            not isinstance(base_scene.get("cuboid"), dict) or
            "target" in base_scene["mesh"]):
        raise ValueError("base scene must be a fixed cuRobo world without a key target")
    view_data = record.get("view_limits")
    if not isinstance(view_data, dict):
        raise ValueError("socket view limits are required")
    limits = SocketViewLimits(**view_data)
    limits.validate()
    if len(camera_ids) < limits.minimum_accepted_views:
        raise ValueError("too few active cameras for the view gate")
    paths = AssetPaths(root, mode)
    for name, asset in (
            ("socket raw mesh", paths.raw_mesh(mode.socket_object)),
            ("socket FoundPose representation",
             paths.foundpose_repre(mode.socket_object)),
            ("socket collision mesh", paths.socket_collision_mesh)):
        if not asset.is_file():
            raise FileNotFoundError(f"{name} is missing: {asset}")
    if mode.family == "cylinder":
        validate_cylinder_socket_fixture(shared_root=root, mode=mode)
    count = record.get("socket_capture_count")
    prompt = record.get("socket_prompt")
    if (type(count) is not int or count < 2 or
            not isinstance(prompt, str) or not prompt.strip() or
            prompt.strip() == "object"):
        raise ValueError("socket needs repeated captures and a specific SAM prompt")
    values = {name: _positive(record.get(name), name=name) for name in (
        "board_timeout_s", "socket_timeout_s", "image_write_timeout_s",
        "max_socket_translation_mm", "max_socket_angle_deg",
        "minimum_socket_refined_iou")}
    if values["minimum_socket_refined_iou"] > 1:
        raise ValueError("socket per-view IoU threshold cannot exceed one")
    return SessionLaunch(
        mode, root, capture_root, evidence, tuple(pcs), tuple(ips), camera_ids,
        intrinsics, extrinsics, image_hw, c2r, base_scene, camera_times,
        limits, prompt, count, **values)


@contextmanager
def open_session_cameras(
    launch: SessionLaunch,
) -> Iterator[OpenSessionCameras]:
    """Own AutoDex camera connections until the caller leaves the context.

    A long-running trial process can call ``cameras.start()`` and then reuse
    ``cameras.init_orchestrator`` and
    ``cameras.acquisition_metadata_for_request`` for each fresh key capture.
    All connections close on normal exit or any exception.
    """
    if not isinstance(launch, SessionLaunch):
        raise TypeError("camera session needs a validated launch")
    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

    registry = FrameIdentityRegistry()
    with ExitStack() as stack:
        stock_snapshot = SnapshotOrchestrator(
            list(launch.pc_list), list(launch.capture_ips))
        stack.callback(stock_snapshot.close)
        tap = SnapshotMetadataTap(list(launch.capture_ips))
        stack.callback(tap.close)
        board = ProvenanceSnapshotAdapter(stock_snapshot, tap.buffer, registry)
        stock_init = InitOrchestrator(
            list(launch.pc_list), list(launch.capture_ips))
        stack.callback(stock_init.close)
        init = PrecisionInitOrchestrator(stock_init, registry)
        provider = AcquisitionTimeProvider(registry, launch.camera_times)
        yield OpenSessionCameras(launch, board, init, provider)


def capture_session(launch: SessionLaunch) -> StartedPrecisionSession:
    """One-shot board/socket CLI capture; explicitly no robot connection."""
    with open_session_cameras(launch) as cameras:
        return cameras.start()
