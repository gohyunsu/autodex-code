"""MV-GoTrack bridge for the cylindrical precision-insertion task.

MV-GoTrack publishes a full CAD-frame pose in the frozen camera world frame.
For a D-infinity cylinder, axial yaw and exchange of the two geometrically
identical ends are not observable task variables.  This module therefore
turns the raw pose into the two quantities insertion actually uses:

* the free key end selected by the planned pre-insertion key axis; and
* the unoriented cylinder axis, signed toward that free end.

No score, workspace, jump, or task-success threshold is introduced here.
The publisher's native ``valid`` bit and a finite SE(3) matrix are the only
measurement contract.  Motion bounds remain owned by the existing
same-trial XY/axis planner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Mapping

import numpy as np

from .assets import AssetPaths
from .geometry import validate_se3
from .symmetry import load_axial_symmetry
from .world import validated_frozen_socket_pose


TRACKED_POSE_SCHEMA = "forgehand.tracked_pose.v1"


@dataclass(frozen=True)
class MVGoTrackPose:
    seq: int
    object_name: str
    pose_world: np.ndarray
    valid: bool
    status: str
    score: float | None
    frame_ids: dict[str, int]
    stamp_capture: float
    stamp_publish: float
    calibration_session: str | None

    @classmethod
    def from_message(cls, value: Mapping) -> "MVGoTrackPose":
        if not isinstance(value, Mapping) or value.get("schema") != TRACKED_POSE_SCHEMA:
            raise ValueError("MV-GoTrack message has the wrong schema")
        seq = value.get("seq")
        frame_ids = value.get("frame_ids")
        if (type(seq) is not int or seq < 0 or
                value.get("frame") != "world" or
                not isinstance(value.get("object"), str) or
                not value["object"] or type(value.get("valid")) is not bool or
                not isinstance(value.get("status"), str) or
                not isinstance(frame_ids, Mapping) or len(frame_ids) < 2 or
                any(not isinstance(camera, str) or type(frame) is not int
                    for camera, frame in frame_ids.items())):
            raise ValueError("MV-GoTrack message is incomplete")
        capture = float(value.get("stamp_capture", float("nan")))
        publish = float(value.get("stamp_publish", float("nan")))
        if not math.isfinite(capture) or not math.isfinite(publish):
            raise ValueError("MV-GoTrack timestamps must be finite")
        score_value = value.get("score")
        score = None if score_value is None else float(score_value)
        if score is not None and not math.isfinite(score):
            raise ValueError("MV-GoTrack score must be finite when present")
        pose = validate_se3(value.get("T"), name="MV-GoTrack T_world_key")
        return cls(
            seq=seq, object_name=value["object"], pose_world=pose,
            valid=value["valid"], status=value["status"], score=score,
            frame_ids={str(camera): int(frame)
                       for camera, frame in frame_ids.items()},
            stamp_capture=capture, stamp_publish=publish,
            calibration_session=(
                None if value.get("calib_session") is None else
                str(value["calib_session"])))

    def to_record(self) -> dict:
        return {
            "schema": TRACKED_POSE_SCHEMA,
            "seq": self.seq,
            "object": self.object_name,
            "frame": "world",
            "T": self.pose_world.tolist(),
            "valid": self.valid,
            "status": self.status,
            "score": self.score,
            "frame_ids": dict(self.frame_ids),
            "stamp_capture": self.stamp_capture,
            "stamp_publish": self.stamp_publish,
            "calib_session": self.calibration_session,
        }


class MVGoTrackSubscriber:
    """Keep the newest tracker message while robot stages continue to run."""

    def __init__(self, endpoint: str, *, expected_object: str):
        if (not isinstance(endpoint, str) or not endpoint.startswith("tcp://") or
                not isinstance(expected_object, str) or not expected_object):
            raise ValueError("MV-GoTrack needs a TCP endpoint and object name")
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("pyzmq is required by MV-GoTrack mode") from exc
        self.endpoint = endpoint
        self.expected_object = expected_object
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.connect(endpoint)
        self._condition = threading.Condition()
        self._latest: MVGoTrackPose | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._receive, name="precision-mvgotrack-sub", daemon=True)
        self._thread.start()

    def _receive(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._closed:
                        return
                if not self._socket.poll(100):
                    continue
                message = self._socket.recv_json()
                pose = MVGoTrackPose.from_message(message)
                if pose.object_name != self.expected_object:
                    continue
                with self._condition:
                    self._latest = pose
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                if not self._closed:
                    self._error = exc
                    self._condition.notify_all()

    def newest_seq(self) -> int:
        with self._condition:
            return -1 if self._latest is None else self._latest.seq

    def wait_for_valid(self, *, after_seq: int, timeout_s: float) -> MVGoTrackPose:
        """Return one native-valid pose newer than ``after_seq``.

        Waiting for a new processed frame is a temporal association contract,
        not a pose-quality gate.  In particular, no elapsed-age, score, jump,
        or workspace limit is applied by this subscriber.
        """
        if type(after_seq) is not int or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("tracker wait needs an integer sequence and timeout")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("MV-GoTrack subscriber stopped") from self._error
                pose = self._latest
                if pose is not None and pose.seq > after_seq and pose.valid:
                    return pose
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    detail = "no message" if pose is None else (
                        f"latest seq={pose.seq} valid={pose.valid} status={pose.status}")
                    raise TimeoutError(
                        f"no new native-valid MV-GoTrack pose within {timeout_s:g}s ({detail})")
                self._condition.wait(min(remaining, .1))

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._thread.join(timeout=2.)
        self._socket.close(linger=0)
        self._context.term()


def symmetry_reduced_alignment(
    *, tracked_pose_world: np.ndarray, T_world_robot: np.ndarray,
    T_robot_socket: np.ndarray, T_robot_key_nominal: np.ndarray,
    object_root: Path, key_object: str, task_geometry: Mapping,
) -> dict:
    """Map one raw key pose to task-relevant socket-frame tip and axis.

    ``Dinf`` permits arbitrary axial yaw and end exchange.  The planned
    pre-insertion pose supplies only the sign of the free insertion end; it
    does not replace the measured center or tilt.  Consequently raw tracker
    yaw cannot leak into the robot correction.
    """
    tracked_world = validate_se3(tracked_pose_world, name="tracked key pose")
    world_robot = validate_se3(T_world_robot, name="T_world_robot")
    robot_socket = validate_se3(T_robot_socket, name="T_robot_socket")
    nominal = validate_se3(T_robot_key_nominal, name="nominal key pose")
    symmetry = load_axial_symmetry(Path(object_root), key_object)
    if not symmetry.end_exchange:
        raise ValueError("cylindrical key tracking expects Dinf end exchange")
    key_frame = task_geometry.get("key_frame")
    if not isinstance(key_frame, Mapping):
        raise ValueError("task geometry lacks key_frame")
    tip_z = float(key_frame.get("tip_z_m", float("nan")))
    key_axis = np.asarray(key_frame.get("insertion_axis"), dtype=float)
    if (key_axis.shape != (3,) or not np.all(np.isfinite(key_axis)) or
            np.linalg.norm(key_axis) <= 1e-12 or not math.isfinite(tip_z)):
        raise ValueError("task key tip/axis is invalid")
    key_axis /= np.linalg.norm(key_axis)
    if abs(float(np.dot(key_axis, symmetry.axis_local))) < 1. - 1e-6:
        raise ValueError("task insertion axis differs from symmetry axis")

    tracked_robot = validate_se3(
        np.linalg.inv(world_robot) @ tracked_world,
        name="tracked T_robot_key")
    center_robot = (tracked_robot[:3, :3] @ symmetry.center_local +
                    tracked_robot[:3, 3])
    measured_axis = tracked_robot[:3, :3] @ symmetry.axis_local
    measured_axis /= np.linalg.norm(measured_axis)
    nominal_axis = nominal[:3, :3] @ key_axis
    nominal_axis /= np.linalg.norm(nominal_axis)
    end_exchanged = float(np.dot(measured_axis, nominal_axis)) < 0.
    if end_exchanged:
        measured_axis = -measured_axis

    tip_local = np.array([0., 0., tip_z])
    half_extent = abs(float(np.dot(
        tip_local - symmetry.center_local, symmetry.axis_local)))
    if half_extent <= 0:
        raise ValueError("key tip must differ from its Dinf symmetry center")
    tip_robot = center_robot + half_extent * measured_axis
    socket_robot = np.linalg.inv(robot_socket)
    tip_socket = socket_robot[:3, :3] @ tip_robot + socket_robot[:3, 3]
    axis_socket = socket_robot[:3, :3] @ measured_axis
    axis_socket /= np.linalg.norm(axis_socket)
    correction = -tip_socket[:2]
    return {
        "status": "diagnostic_metric_xy_correction",
        "reason": "mvgotrack_pose_symmetry_reduced",
        "xy_source": "mvgotrack_6d_pose",
        "estimator": "mvgotrack_symmetry_reduced_6d",
        "tip_socket_m": tip_socket.tolist(),
        "insertion_axis_socket": axis_socket.tolist(),
        "xy_correction_socket_m": correction.tolist(),
        "xy_correction_norm_m": float(np.linalg.norm(correction)),
        "symmetry_reduction": {
            "type": "Dinf",
            "axial_yaw_used": False,
            "end_exchange_applied": end_exchanged,
            "center_local_m": symmetry.center_local.tolist(),
            "axis_local": symmetry.axis_local.tolist(),
            "free_tip_half_extent_m": half_extent,
            "sign_reference": "planned_T_robot_key_preinsert_axis",
        },
    }


def alignment_from_tracker(*, runner, pose: MVGoTrackPose) -> dict:
    """Build the existing same-trial alignment record from MV-GoTrack."""
    paths = AssetPaths(Path(runner.shared_root), runner.mode)
    geometry = json.loads(paths.task_geometry.read_text(encoding="utf-8"))
    socket = validated_frozen_socket_pose(
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration)
    alignment = symmetry_reduced_alignment(
        tracked_pose_world=pose.pose_world,
        T_world_robot=np.asarray(runner.calibration.record["c2r"], dtype=float),
        T_robot_socket=socket,
        T_robot_key_nominal=(
            runner._postlift_preflight.targets.T_robot_key_preinsert),
        object_root=paths.object_root, key_object=runner.mode.key_object,
        task_geometry=geometry)
    alignment["inlier_cameras"] = sorted(pose.frame_ids)
    alignment["tracker_pose"] = pose.to_record()
    return alignment
