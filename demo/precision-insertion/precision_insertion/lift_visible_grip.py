"""Judge a lifted key in two measured-hand close-ups of real camera pixels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from .observer import (ImageVLM, LabeledFrame, VLMObservation,
                       resize_classification_images)
from .timing_profile import timed
from .view_parallel import map_view_calls


CAMERAS = ("24122734", "25322638")
LEGACY_QUESTIONS = {
    "24122734": (
        "Is the small white cylindrical key visibly between the black robot "
        "fingers in this image, rather than loose on the checkerboard? "
        "Reply only YES or NO."),
    "25322638": (
        "Is the small white cylindrical key held between the black robot "
        "fingers and lifted clear of the checkerboard? "
        "Answer only YES or NO."),
}
LEGACY_IMAGE_SIZE = {"24122734": 384, "25322638": 512}
LEGACY_SCHEMA = "precision_insertion_visible_grip_v1"
SCHEMA = "precision_insertion_visible_grip_v2"
HAND_PROMPT = (
    "Is the small solid white cylindrical peg visibly held by the robot's "
    "black fingers in this image? The robot also has white finger shells. "
    "Reply only YES or NO.")
HAND_CROP_SIDE = 650
HAND_IMAGE_SIZE = 640
HAND_FOCUS = np.array([0., .07, -.08, 1.])


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def _vote(answer: str) -> bool | None:
    value = answer.strip().upper()
    return True if value == "YES" else False if value == "NO" else None


def _mask_roi(mask_path: Path, image_size: tuple[int, int]) -> tuple[int, ...]:
    with Image.open(mask_path) as source:
        mask = source.convert("L")
        if mask.size != image_size:
            raise ValueError("key mask and after-lift image have different sizes")
        box = mask.getbbox()
    if box is None:
        raise ValueError("initial key mask is empty")
    width, height = image_size
    side = min(900, width, height)
    cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
    left = max(0, min(width - side, cx - side // 2))
    top = max(0, min(height - side, cy - 2 * side // 3))
    return left, top, left + side, top + side


def _legacy_result(rows: list[dict]) -> dict:
    votes = [row["yes"] for row in rows]
    label = ("held" if votes == [True, True] else
             "miss" if votes == [False, False] else "unobservable")
    return {
        "class": label,
        "evidence": "Visible key between fingers in two views" if label == "held"
                    else "No visible held key in two views" if label == "miss"
                    else "Close views disagree or do not resolve the key",
        "evidence_views": list(CAMERAS) if label != "unobservable" else [],
    }


def _hand_transform(joint_q: Sequence[float], robot_urdf: Path) -> np.ndarray:
    from yourdfpy import URDF

    joints = np.asarray(joint_q, dtype=float)
    if joints.shape != (13,) or not np.all(np.isfinite(joints)):
        raise ValueError("hand crop needs thirteen measured robot joints")
    robot = URDF.load(str(robot_urdf), build_scene_graph=True,
                      load_meshes=False)
    names = [joint.name for joint in robot.actuated_joints]
    if len(names) != 13 or not all(name.startswith("fr3_") for name in names[:7]):
        raise ValueError("hand crop needs the FR3/Inspire adapter URDF")
    robot.update_cfg(dict(zip(names, joints)))
    return np.asarray(robot.get_transform("base_link", robot.base_link),
                      dtype=float)


def _hand_roi(*, image_size: tuple[int, int], camera: str,
              calibration: Mapping, T_robot_hand: np.ndarray,
              ) -> tuple[tuple[int, int, int, int], list[float]]:
    camera_cal = calibration["camera_calibration"]
    K = np.asarray(camera_cal["intrinsics_full"][camera]["K_undist"],
                   dtype=float)
    camera_world = np.asarray(camera_cal["extrinsics_full"][camera], dtype=float)
    world_robot = np.asarray(calibration["c2r"], dtype=float)
    if (K.shape != (3, 3) or camera_world.shape != (4, 4) or
            world_robot.shape != (4, 4) or
            not all(np.all(np.isfinite(matrix)) for matrix in
                    (K, camera_world, world_robot, T_robot_hand)) or
            K[0, 0] <= 0 or K[1, 1] <= 0):
        raise ValueError("hand crop needs finite undistorted session calibration")
    point = camera_world @ world_robot @ T_robot_hand @ HAND_FOCUS
    if point[2] <= 0:
        raise ValueError("measured hand is behind the camera")
    pixel = K @ point[:3]
    xy = pixel[:2] / pixel[2]
    if not np.all(np.isfinite(xy)):
        raise ValueError("measured hand projection is not finite")
    width, height = image_size
    side = min(HAND_CROP_SIDE, width, height)
    left = max(0, min(width - side, int(round(xy[0] - side / 2))))
    top = max(0, min(height - side, int(round(xy[1] - side / 2))))
    return (left, top, left + side, top + side), xy.tolist()


def _hand_answer(backend: ImageVLM, camera: str, frame: LabeledFrame,
                 calibration: Mapping, hand_transform: np.ndarray) -> dict:
    roi, pixel = _hand_roi(image_size=frame.image.size, camera=camera,
                           calibration=calibration,
                           T_robot_hand=hand_transform)
    prepared, transform = resize_classification_images(
        [frame.image.crop(roi)], (HAND_IMAGE_SIZE, HAND_IMAGE_SIZE))
    infer_for_camera = getattr(backend, "infer_for_camera", None)
    start = time.perf_counter()
    with timed("vlm.infer", phase="vlm", kind="inference",
               stage="post_lift", camera_id=camera, image_count=1,
               image_sizes=[list(prepared[0].size)],
               model=getattr(backend, "model", None)):
        raw = (infer_for_camera(camera, prepared, HAND_PROMPT)
               if callable(infer_for_camera) else backend.infer(
                   prepared, HAND_PROMPT))
    if not isinstance(raw, str):
        raise TypeError("lift VLM response must be text")
    return {
        "camera_id": camera,
        "image_order": [f"after_lift/{camera}@{frame.timestamp_s:.6f}"],
        "prompt": HAND_PROMPT, "raw_answer": raw,
        "yes": _vote(raw),
        "latency_s": time.perf_counter() - start,
        "input_transform": transform,
        "crop": {"source_roi_xyxy": list(roi),
                 "original_image_size_wh": list(frame.image.size),
                 "projected_hand_px": pixel,
                 "method": "measured_fr3_inspire_fk_and_undistorted_camera"},
    }


def _hand_result(rows: list[dict]) -> dict:
    confirmed = [row["camera_id"] for row in rows
                 if row["yes"] is True]
    if confirmed:
        label = "held"
        evidence = "A close hand view shows the gripped key"
    elif all(row["yes"] is False for row in rows):
        label = "miss"
        evidence = "Neither close hand view shows a gripped key"
    else:
        label = "unobservable"
        evidence = "Close hand views do not resolve the key"
    return {"class": label, "evidence": evidence,
            "evidence_views": list(CAMERAS) if label != "unobservable" else [],
            "confirming_views": confirmed}


def observe_visible_grip(
    backend: ImageVLM, frames: Sequence[LabeledFrame], before_bundle: Path,
    *, calibration: Mapping, joint_q: Sequence[float], robot_urdf: Path,
) -> VLMObservation:
    paired = {(frame.phase, frame.camera_id): frame for frame in frames}
    cameras = {frame.camera_id for frame in frames}
    if (len(paired) != len(frames) or
            len(frames) != 2 * len(cameras) or
            any((phase, camera) not in paired
                for phase in ("before_grasp", "after_lift")
                for camera in cameras) or
            not set(CAMERAS) <= cameras):
        raise ValueError("visible grip needs both saved phases and close cameras")
    source = Path(before_bundle).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("visible grip needs the saved before bundle")
    robot_urdf = Path(robot_urdf).expanduser().resolve()
    hand = _hand_transform(joint_q, robot_urdf)
    rows = map_view_calls(
        backend, list(CAMERAS),
        lambda camera: _hand_answer(
            backend, camera, paired["after_lift", camera], calibration, hand))
    parsed = _hand_result(rows)
    order = tuple(
        f"{frame.phase}/{frame.camera_id}@{frame.timestamp_s:.6f}"
        for frame in frames)
    prompt = ("Images are supplied in this exact order:\n" +
              "\n".join(f"{i + 1}: {label}" for i, label in enumerate(order)) +
              "\n\nPost-lift close-view questions are recorded per camera in aggregation.")
    return VLMObservation(
        "post_lift", parsed, json.dumps(parsed, sort_keys=True), prompt,
        order, None, str(getattr(backend, "model", "local_qwen")),
        sum(row["latency_s"] for row in rows),
        aggregation={
            "schema": SCHEMA, "per_view": rows,
            "calibration_sha256": _digest(calibration),
            "robot_urdf": str(robot_urdf),
            "robot_urdf_sha256": hashlib.sha256(
                robot_urdf.read_bytes()).hexdigest(),
        })


def _verify_legacy_visible_grip(visual: dict, cameras: set[str],
                                before_bundle: Path) -> None:
    aggregation = visual.get("aggregation")
    rows = aggregation.get("per_view") if isinstance(aggregation, dict) else None
    if (not isinstance(aggregation, dict) or
            aggregation.get("schema") != LEGACY_SCHEMA or
            not set(CAMERAS) <= cameras or
            not isinstance(rows, list) or
            [row.get("camera_id") for row in rows] != list(CAMERAS)):
        raise ValueError("visible grip camera evidence changed")
    source = Path(before_bundle).expanduser().resolve()
    for row in rows:
        camera = row["camera_id"]
        mask_path = source / "masks" / f"{camera}.png"
        crop = row.get("crop")
        if not mask_path.is_file():
            raise ValueError("visible grip key mask is missing")
        with Image.open(mask_path) as mask:
            mask_size = mask.size
        order = next((item for item in visual.get("image_order", [])
                      if item.startswith(f"after_lift/{camera}@")), None)
        if (row.get("prompt") != LEGACY_QUESTIONS[camera] or
                not isinstance(row.get("raw_answer"), str) or
                row.get("yes") is not _vote(row["raw_answer"]) or
                row.get("input_transform", {}).get("max_size") !=
                [LEGACY_IMAGE_SIZE[camera], LEGACY_IMAGE_SIZE[camera]] or
                order is None or row.get("image_order") != [order] or
                not isinstance(crop, dict) or
                crop.get("mask_path") != str(mask_path) or
                crop.get("mask_sha256") !=
                hashlib.sha256(mask_path.read_bytes()).hexdigest() or
                crop.get("source_roi_xyxy") != list(_mask_roi(
                    mask_path, mask_size))):
            raise ValueError("visible grip prompt, crop or answer changed")
    expected = _legacy_result(rows)
    if (visual.get("parsed") != expected or
            visual.get("raw_answer") != json.dumps(expected, sort_keys=True) or
            visual.get("parse_error") is not None):
        raise ValueError("visible grip consensus changed")


def verify_visible_grip(visual: dict, cameras: set[str],
                        before_bundle: Path, *, joint_q: Sequence[float] | None = None,
                        calibration: Mapping | None = None) -> None:
    aggregation = visual.get("aggregation")
    if not isinstance(aggregation, dict):
        raise ValueError("visible grip aggregation is missing")
    if aggregation.get("schema") == LEGACY_SCHEMA:
        _verify_legacy_visible_grip(visual, cameras, before_bundle)
        return
    rows = aggregation.get("per_view")
    if (aggregation.get("schema") != SCHEMA or
            not set(CAMERAS) <= cameras or
            not isinstance(rows, list) or
            [row.get("camera_id") for row in rows] != list(CAMERAS) or
            joint_q is None):
        raise ValueError("visible grip camera evidence changed")
    if calibration is None:
        source = Path(before_bundle).expanduser().resolve()
        calibration = json.loads((source.parents[1] /
                                  "frozen_session_calibration.json").read_text())
    robot_urdf = Path(aggregation["robot_urdf"])
    if (aggregation.get("calibration_sha256") != _digest(calibration) or
            aggregation.get("robot_urdf_sha256") !=
            hashlib.sha256(robot_urdf.read_bytes()).hexdigest()):
        raise ValueError("visible grip calibration or robot geometry changed")
    hand = _hand_transform(joint_q, robot_urdf)
    for row in rows:
        camera = row["camera_id"]
        order = next((item for item in visual.get("image_order", [])
                      if item.startswith(f"after_lift/{camera}@")), None)
        if order is None or row.get("image_order") != [order]:
            raise ValueError("visible grip image order changed")
        if (row.get("prompt") != HAND_PROMPT or
                not isinstance(row.get("raw_answer"), str) or
                row.get("yes") is not _vote(row["raw_answer"])):
            raise ValueError("visible grip prompt or answer changed")
        transform = row.get("input_transform", {})
        if transform.get("max_size") != [HAND_IMAGE_SIZE, HAND_IMAGE_SIZE]:
            raise ValueError("visible grip input resolution changed")
        source_size = transform.get("images", [{}])[0].get("source_size")
        if (not isinstance(source_size, list) or len(source_size) != 2 or
                not all(type(value) is int and value > 0
                        for value in source_size)):
            raise ValueError("visible grip crop size changed")
        crop = row.get("crop", {})
        original_size = crop.get("original_image_size_wh")
        if (not isinstance(original_size, list) or len(original_size) != 2 or
                not all(type(value) is int and value > 0
                        for value in original_size)):
            raise ValueError("visible grip source image size changed")
        roi, pixel = _hand_roi(image_size=tuple(original_size), camera=camera,
                               calibration=calibration, T_robot_hand=hand)
        if (crop.get("method") !=
                "measured_fr3_inspire_fk_and_undistorted_camera" or
                crop.get("source_roi_xyxy") != list(roi) or
                source_size != [roi[2] - roi[0], roi[3] - roi[1]] or
                not np.allclose(crop.get("projected_hand_px", []), pixel,
                                atol=1e-6, rtol=0)):
            raise ValueError("visible grip hand crop changed")
    expected = _hand_result(rows)
    if (visual.get("parsed") != expected or
            visual.get("raw_answer") != json.dumps(expected, sort_keys=True) or
            visual.get("parse_error") is not None):
        raise ValueError("visible grip consensus changed")
