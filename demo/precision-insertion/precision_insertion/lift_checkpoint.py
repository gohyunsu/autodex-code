"""Bind a read-only VLM lift assessment to two saved AutoDex key captures.

The first capture is the trial's tabletop key observation. The second is a
fresh post-lift multi-view FoundPose observation admitted using a *candidate*
search prior and measured wrist feedback. Neither the prior nor this report
authorizes motion or establishes stable insertion geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .assets import AssetPaths
from .config import TaskMode
from .endpoint import _load_mesh
from .frame_provenance import bounded_capture_skew_s, image_sha256
from .geometry import validate_se3
from .key_perception import KeyPoseObservation, verify_key_capture_artifacts
from .live_robot_state import LiveRobotState
from .observer import ImageVLM, LabeledFrame, VLMObservation, _parse_object, observe_lift
from .raw_camera_capture import verify_raw_camera_capture
from .session_bootstrap import _safe_id


@dataclass(frozen=True)
class LiftCheckpoint:
    attempt_id: str
    candidate_id: str
    grasp_success: bool | None
    reason: str
    lift_completed_at_s: float
    decision_timestamp_s: float
    before_capture_id: str
    after_capture_id: str
    before_bundle: Path
    after_bundle: Path
    before_manifest_sha256: str
    after_manifest_sha256: str
    frames: tuple[dict, ...]
    center_rise_m: float | None
    minimum_center_rise_m: float | None
    minimum_visual_views: int
    joint_sample: LiveRobotState
    visual: VLMObservation
    evidence_kind: str = "foundpose_key_rise"
    raw_timing: dict | None = None

    def to_record(self) -> dict:
        return {
            "schema": (
                "precision_insertion_raw_lift_checkpoint_v1"
                if self.evidence_kind == "raw_visual" else
                "precision_insertion_lift_checkpoint_v1"),
            "evidence_kind": self.evidence_kind,
            "raw_timing": self.raw_timing,
            "attempt_id": self.attempt_id,
            "candidate_id": self.candidate_id,
            "grasp_success": self.grasp_success,
            "reason": self.reason,
            "lift_completed_at_s": self.lift_completed_at_s,
            "decision_timestamp_s": self.decision_timestamp_s,
            "before_capture_id": self.before_capture_id,
            "after_capture_id": self.after_capture_id,
            "before_bundle": str(self.before_bundle),
            "after_bundle": str(self.after_bundle),
            "before_manifest_sha256": self.before_manifest_sha256,
            "after_manifest_sha256": self.after_manifest_sha256,
            "frames": list(self.frames),
            "center_rise_m": self.center_rise_m,
            "minimum_center_rise_m": self.minimum_center_rise_m,
            "minimum_visual_views": self.minimum_visual_views,
            "joint_feedback": self.joint_sample.to_record(),
            "visual": self.visual.to_record(),
            "scope": (
                "saved_raw_multiview_visible_key_not_6d_pose_or_contact_proof"
                if self.evidence_kind == "raw_visual" else
                "saved_images_vlm_and_observed_key_rise_not_grasp_contact_proof"),
            "robot_ready": False,
        }


def _saved_capture(root: Path) -> tuple[dict, str]:
    directory = Path(root).expanduser().resolve()
    verify_key_capture_artifacts(directory)
    record = json.loads((directory / "key_observation.json").read_text(
        encoding="utf-8"))
    digest = hashlib.sha256(
        (directory / "evidence_manifest.json").read_bytes()).hexdigest()
    return record, digest


def _lift_verdict(
    *, visual_class: str, evidence_views: set[str],
    parse_error: str | None, cameras: set[str], rise_m: float,
    minimum_rise_m: float, minimum_views: int,
) -> tuple[bool | None, str]:
    decisive = (parse_error is None and
                len(evidence_views & cameras) >= minimum_views)
    if decisive and visual_class == "held" and rise_m >= minimum_rise_m:
        return True, "multiview_held_and_key_center_rose"
    if (decisive and visual_class in {"miss", "slip"} and
            rise_m < minimum_rise_m):
        return False, "multiview_miss_or_slip_without_key_lift"
    return None, "visual_and_key_motion_incomplete_or_conflicting"


def assess_lift_checkpoint(
    *, mode: TaskMode, shared_root: Path, calibration,
    attempt_id: str, candidate_id: str, attempt_started_at_s: float,
    lift_completed_at_s: float, decision_timestamp_s: float,
    before_capture_id: str, before_pose_world: np.ndarray,
    before_bundle: Path, after_observation: KeyPoseObservation,
    after_bundle: Path, joint_sample: LiveRobotState,
    expected_candidate_prior_world: np.ndarray, backend: ImageVLM,
    max_state_skew_s: float, max_phase_skew_s: float,
    max_lift_observation_gap_s: float, min_center_rise_m: float,
    max_arm_hand_skew_s: float, max_hand_command_error_raw: float,
    max_arm_velocity_rad_s: float,
    minimum_visual_views: int = 2,
) -> LiftCheckpoint:
    """Judge held/miss/unknown from paired saved pixels and key motion.

    Positive requires VLM held evidence from at least two cameras plus a
    measured rise of the physical key center. A conflicting visual/geometry
    result abstains. The caller must have validated measured Franka/Inspire
    feedback and derived the candidate prior from the selected v8 grasp.
    """
    limits = (max_state_skew_s, max_phase_skew_s,
              max_lift_observation_gap_s, min_center_rise_m)
    if (not all(math.isfinite(float(value)) and float(value) > 0
                for value in limits) or
            type(minimum_visual_views) is not int or
            minimum_visual_views < 2 or
            not all(math.isfinite(float(value)) for value in (
                attempt_started_at_s, lift_completed_at_s,
                decision_timestamp_s)) or
            lift_completed_at_s <= attempt_started_at_s):
        raise ValueError("lift checkpoint needs commissioned timing/rise limits")
    if (not isinstance(after_observation, KeyPoseObservation) or
            after_observation.phase != "held_postlift" or
            after_observation.key_object != mode.key_object or
            after_observation.family != mode.family or
            not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("lift checkpoint needs admitted held key and measured joints")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    before_root = Path(before_bundle).expanduser().resolve()
    after_root = Path(after_bundle).expanduser().resolve()
    before, before_hash = _saved_capture(before_root)
    after, after_hash = _saved_capture(after_root)
    if (before.get("phase") != "tabletop" or
            before.get("capture_id") != before_capture_id or
            before.get("key_object") != mode.key_object or
            before.get("family") != mode.family or
            not np.allclose(validate_se3(before.get("pose_world")),
                            validate_se3(before_pose_world), atol=1e-8, rtol=0) or
            after != after_observation.to_record() or
            before_capture_id == after_observation.capture_id):
        raise ValueError("lift captures differ from selected trial and held key")
    prior = after_observation.consistency.get("held_pose_prior")
    if (not isinstance(prior, dict) or
            prior.get("source") != "measured_wrist_plus_candidate_grasp" or
            prior.get("timestamp_s") != joint_sample.sample_timestamp_s or
            not np.allclose(validate_se3(prior.get("pose_world")),
                            validate_se3(expected_candidate_prior_world),
                            atol=1e-8, rtol=0)):
        raise ValueError("held key admission used another candidate/wrist prior")
    after_observation.require_state_alignment(
        state_timestamp_s=joint_sample.sample_timestamp_s,
        maximum_skew_s=max_state_skew_s)
    before_ids = set(before["consistency"]["accepted_views"])
    after_ids = set(after["consistency"]["accepted_views"])
    cameras = sorted(before_ids & after_ids)
    if len(cameras) < minimum_visual_views:
        raise ValueError("lift checkpoint has too few paired camera views")
    phase_rows = {
        "before_grasp": (before_root, before),
        "after_lift": (after_root, after),
    }
    for _phase, (_root, record) in phase_rows.items():
        subset = {camera: record["frame_evidence"][camera]
                  for camera in cameras}
        if bounded_capture_skew_s(subset) > max_phase_skew_s:
            raise ValueError("lift camera views exceed bounded acquisition skew")
    before_upper = max(
        before["frame_evidence"][camera]["timestamp_s"] +
        before["frame_evidence"][camera]["max_error_s"] for camera in cameras)
    after_lower = min(
        after["frame_evidence"][camera]["timestamp_s"] -
        after["frame_evidence"][camera]["max_error_s"] for camera in cameras)
    if (before_upper >= attempt_started_at_s or
            after_lower <= lift_completed_at_s or
            after_lower <= before_upper or
            after_lower - before_upper > max_lift_observation_gap_s or
            decision_timestamp_s < max(
                after_observation.acquisition_interval_s[1],
                joint_sample.sample_timestamp_s)):
        raise ValueError("lift frames are not fresh around this physical attempt")
    frames: list[LabeledFrame] = []
    inputs: list[dict] = []
    for phase, (root, record) in phase_rows.items():
        for camera in cameras:
            path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            row = record["frame_evidence"][camera]
            if bgr is None or image_sha256(bgr) != row["image_sha256"]:
                raise ValueError("VLM source image differs from saved key frame")
            frames.append(LabeledFrame(
                camera, phase, row["timestamp_s"],
                Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
            inputs.append({
                "phase": phase, "camera_id": camera,
                "path": str(path),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frame_id": row["frame_id"],
                "image_sha256": row["image_sha256"],
                "timestamp_s": row["timestamp_s"],
                "max_error_s": row["max_error_s"],
            })
    visual = observe_lift(backend, frames)
    c2r = validate_se3(calibration.record.get("c2r"), name="session C2R")
    before_robot = validate_se3(
        np.linalg.inv(c2r) @ validate_se3(before_pose_world),
        name="trial tabletop T_robot_key")
    after_robot = validate_se3(
        np.linalg.inv(c2r) @ after_observation.pose_world,
        name="post-lift T_robot_key")
    mesh = _load_mesh(AssetPaths(Path(shared_root), mode).raw_mesh(
        mode.key_object))
    center = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError("key CAD has no finite reference center")
    rise = float((after_robot[:3, :3] @ center + after_robot[:3, 3])[2] -
                 (before_robot[:3, :3] @ center + before_robot[:3, 3])[2])
    label, reason = _lift_verdict(
        visual_class=visual.parsed["class"],
        evidence_views=set(visual.parsed["evidence_views"]),
        parse_error=visual.parse_error, cameras=set(cameras), rise_m=rise,
        minimum_rise_m=min_center_rise_m,
        minimum_views=minimum_visual_views)
    return LiftCheckpoint(
        attempt_id, candidate_id, label, reason,
        lift_completed_at_s, decision_timestamp_s,
        before_capture_id, after_observation.capture_id,
        before_root, after_root, before_hash, after_hash, tuple(inputs),
        rise, min_center_rise_m, minimum_visual_views, joint_sample, visual)


def _raw_lift_verdict(
    *, visual_class: str, evidence_views: set[str],
    parse_error: str | None, cameras: set[str], minimum_views: int,
) -> tuple[bool | None, str]:
    if (parse_error is not None or
            len(evidence_views & cameras) < minimum_views):
        return None, "raw_lift_visual_evidence_incomplete"
    if visual_class == "held":
        return True, "visible_key_held_in_paired_views"
    if visual_class in {"miss", "slip"}:
        return False, "visible_key_miss_or_slip_in_paired_views"
    return None, "raw_lift_visual_evidence_incomplete"


def assess_raw_lift_checkpoint(
    *, mode: TaskMode, attempt_id: str, candidate_id: str,
    attempt_started_at_s: float, lift_completed_at_s: float,
    decision_timestamp_s: float, before_capture_id: str,
    before_bundle: Path, after_bundle: Path, joint_sample: LiveRobotState,
    backend: ImageVLM, max_phase_skew_s: float,
    max_lift_observation_gap_s: float, max_arm_hand_skew_s: float,
    max_hand_command_error_raw: float, max_arm_velocity_rad_s: float,
    minimum_visual_views: int = 2,
) -> LiftCheckpoint:
    """Assess visible held/miss/slip without pretending to recover key 6D pose.

    A positive label is a saved multi-camera *visual* grasp observation, not
    a measured key–hand transform. It cannot by itself unlock transfer.
    """
    if (not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback" or
            type(minimum_visual_views) is not int or
            minimum_visual_views < 2 or
            not all(math.isfinite(float(x)) and float(x) > 0 for x in (
                max_phase_skew_s, max_lift_observation_gap_s)) or
            not all(math.isfinite(float(x)) for x in (
                attempt_started_at_s, lift_completed_at_s,
                decision_timestamp_s)) or
            lift_completed_at_s <= attempt_started_at_s):
        raise ValueError("raw lift needs measured joints and commissioned timing")
    joint_sample.validate(
        max_arm_hand_skew_s=max_arm_hand_skew_s,
        max_hand_command_error_raw=max_hand_command_error_raw,
        max_arm_velocity_rad_s=max_arm_velocity_rad_s)
    before_root = Path(before_bundle).expanduser().resolve()
    after_root = Path(after_bundle).expanduser().resolve()
    before, before_hash = _saved_capture(before_root)
    after = verify_raw_camera_capture(after_root, phase="after_lift")
    after_hash = hashlib.sha256((after_root / "manifest.json").read_bytes()).hexdigest()
    if (before.get("phase") != "tabletop" or
            before.get("capture_id") != before_capture_id or
            before.get("key_object") != mode.key_object or
            before.get("family") != mode.family or
            after["capture_id"] == before_capture_id):
        raise ValueError("raw lift before/after images are not this key trial")
    cameras = sorted(set(before["consistency"]["accepted_views"]) &
                     set(after["frame_evidence"]))
    if len(cameras) < minimum_visual_views:
        raise ValueError("raw lift needs paired calibrated camera views")
    subsets = {
        "before_grasp": {c: before["frame_evidence"][c] for c in cameras},
        "after_lift": {c: after["frame_evidence"][c] for c in cameras},
    }
    if any(bounded_capture_skew_s(rows) > max_phase_skew_s
           for rows in subsets.values()):
        raise ValueError("raw lift camera exposure skew exceeds limit")
    before_upper = max(row["timestamp_s"] + row["max_error_s"]
                       for row in subsets["before_grasp"].values())
    after_lower = min(row["timestamp_s"] - row["max_error_s"]
                      for row in subsets["after_lift"].values())
    after_upper = max(row["timestamp_s"] + row["max_error_s"]
                      for row in subsets["after_lift"].values())
    if (before_upper >= attempt_started_at_s or
            after_lower <= lift_completed_at_s or
            after_lower - before_upper > max_lift_observation_gap_s or
            decision_timestamp_s < max(after_upper,
                                       joint_sample.sample_timestamp_s)):
        raise ValueError("raw lift images do not bracket the physical lift")
    frames, inputs = [], []
    for phase, (root, source) in {
            "before_grasp": (before_root, before),
            "after_lift": (after_root, after)}.items():
        for camera in cameras:
            path = root / "images" / f"{_safe_id(camera, 'camera ID')}.png"
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            row = source["frame_evidence"][camera]
            if image is None or image_sha256(image) != row["image_sha256"]:
                raise ValueError("raw lift image differs from saved pixels")
            frames.append(LabeledFrame(
                camera, phase, row["timestamp_s"],
                Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))))
            inputs.append({
                "phase": phase, "camera_id": camera, "path": str(path),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frame_id": row["frame_id"],
                "image_sha256": row["image_sha256"],
                "timestamp_s": row["timestamp_s"],
                "max_error_s": row["max_error_s"],
            })
    visual = observe_lift(backend, frames)
    label, reason = _raw_lift_verdict(
        visual_class=visual.parsed["class"],
        evidence_views=set(visual.parsed["evidence_views"]),
        parse_error=visual.parse_error, cameras=set(cameras),
        minimum_views=minimum_visual_views)
    return LiftCheckpoint(
        attempt_id, candidate_id, label, reason,
        lift_completed_at_s, decision_timestamp_s,
        before_capture_id, after["capture_id"], before_root, after_root,
        before_hash, after_hash, tuple(inputs), None, None,
        minimum_visual_views, joint_sample, visual, "raw_visual", {
            "attempt_started_at_s": attempt_started_at_s,
            "max_phase_skew_s": max_phase_skew_s,
            "max_lift_observation_gap_s": max_lift_observation_gap_s,
        })


def write_lift_checkpoint(result: LiftCheckpoint, output_dir: Path) -> Path:
    """Persist a VLM answer and immutable references to its exact PNG inputs."""
    target = Path(output_dir).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=False)
    with (target / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(result.to_record(), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return target


def verify_lift_checkpoint(report_path: Path) -> dict:
    """Recheck report-to-capture/image bytes before using a lift verdict."""
    path = Path(report_path).expanduser().resolve()
    report = json.loads(path.read_text(encoding="utf-8"))
    raw_visual = report.get("schema") == "precision_insertion_raw_lift_checkpoint_v1"
    if (not isinstance(report, dict) or report.get("schema") not in {
            "precision_insertion_lift_checkpoint_v1",
            "precision_insertion_raw_lift_checkpoint_v1"} or
            report.get("evidence_kind", "foundpose_key_rise") !=
            ("raw_visual" if raw_visual else "foundpose_key_rise") or
            not isinstance(report.get("frames"), list) or
            not report["frames"] or
            report.get("visual", {}).get("stage") != "post_lift"):
        raise ValueError("invalid saved lift checkpoint")
    bundles = {
        "before_grasp": Path(report["before_bundle"]).expanduser().resolve(),
        "after_lift": Path(report["after_bundle"]).expanduser().resolve(),
    }
    expected = {
        "before_grasp": report["before_manifest_sha256"],
        "after_lift": report["after_manifest_sha256"],
    }
    saved = {}
    for phase, root in bundles.items():
        if raw_visual and phase == "after_lift":
            record = verify_raw_camera_capture(root, phase="after_lift")
            digest = hashlib.sha256(
                (root / "manifest.json").read_bytes()).hexdigest()
        else:
            record, digest = _saved_capture(root)
        if digest != expected[phase]:
            raise ValueError("lift source capture manifest changed")
        saved[phase] = record
    if (saved["before_grasp"].get("phase") != "tabletop" or
            saved["after_lift"].get("phase") !=
            ("after_lift" if raw_visual else "held_postlift") or
            saved["before_grasp"].get("capture_id") !=
            report.get("before_capture_id") or
            saved["after_lift"].get("capture_id") !=
            report.get("after_capture_id")):
        raise ValueError("lift source capture identities changed")
    order = []
    phase_cameras = {"before_grasp": set(), "after_lift": set()}
    for row in report["frames"]:
        phase = row.get("phase")
        camera = row.get("camera_id")
        if phase not in bundles or not isinstance(camera, str):
            raise ValueError("unknown lift input phase or camera")
        if camera in phase_cameras[phase]:
            raise ValueError("duplicate lift VLM camera input")
        phase_cameras[phase].add(camera)
        target = bundles[phase] / "images" / f"{_safe_id(camera, 'camera ID')}.png"
        if Path(row.get("path", "")).expanduser().resolve() != target:
            raise ValueError("lift image path differs from source capture")
        if hashlib.sha256(target.read_bytes()).hexdigest() != row.get(
                "file_sha256"):
            raise ValueError("lift VLM input PNG changed")
        image = cv2.imread(str(target), cv2.IMREAD_COLOR)
        source = saved[phase]["frame_evidence"].get(camera)
        if (image is None or not isinstance(source, dict) or
                image_sha256(image) != row.get("image_sha256") or
                source["image_sha256"] != row.get("image_sha256") or
                source["frame_id"] != row.get("frame_id") or
                source["timestamp_s"] != row.get("timestamp_s") or
                source["max_error_s"] != row.get("max_error_s")):
            raise ValueError("lift VLM input differs from bound frame evidence")
        order.append(f"{phase}/{camera}@{row['timestamp_s']:.6f}")
    if report["visual"].get("image_order") != order:
        raise ValueError("lift VLM prompt order differs from saved inputs")
    prompt_prefix = (
        "Images are supplied in this exact order:\n" +
        "\n".join(f"{index + 1}: {label}"
                  for index, label in enumerate(order)) + "\n\n")
    if not str(report["visual"].get("prompt", "")).startswith(prompt_prefix):
        raise ValueError("lift VLM prompt does not describe its input frames")
    minimum_views = report.get("minimum_visual_views")
    if (type(minimum_views) is not int or minimum_views < 2 or
            phase_cameras["before_grasp"] != phase_cameras["after_lift"] or
            len(phase_cameras["before_grasp"]) < minimum_views):
        raise ValueError("lift VLM report lacks paired camera views")
    if not phase_cameras["before_grasp"] <= set(
            saved["before_grasp"]["consistency"]["accepted_views"]):
        raise ValueError("lift VLM used an unadmitted tabletop view")
    if raw_visual:
        timing = report.get("raw_timing")
        if not isinstance(timing, dict) or set(timing) != {
                "attempt_started_at_s", "max_phase_skew_s",
                "max_lift_observation_gap_s"}:
            raise ValueError("raw lift lacks its acquisition timing limits")
        started = float(timing["attempt_started_at_s"])
        phase_skew = float(timing["max_phase_skew_s"])
        max_gap = float(timing["max_lift_observation_gap_s"])
        if (not all(math.isfinite(x) for x in (started, phase_skew,
                                               max_gap)) or
                min(phase_skew, max_gap) <= 0):
            raise ValueError("invalid raw lift timing limits")
        subsets = {
            phase: {camera: saved[phase]["frame_evidence"][camera]
                    for camera in phase_cameras[phase]}
            for phase in bundles}
        if any(bounded_capture_skew_s(rows) > phase_skew
               for rows in subsets.values()):
            raise ValueError("raw lift camera skew changed")
        before_upper = max(row["timestamp_s"] + row["max_error_s"]
                           for row in subsets["before_grasp"].values())
        after_lower = min(row["timestamp_s"] - row["max_error_s"]
                          for row in subsets["after_lift"].values())
        after_upper = max(row["timestamp_s"] + row["max_error_s"]
                          for row in subsets["after_lift"].values())
        completed = float(report["lift_completed_at_s"])
        if (not started < completed < after_lower or
                before_upper >= started or
                after_lower - before_upper > max_gap or
                report["decision_timestamp_s"] < after_upper):
            raise ValueError("raw lift frame times no longer bracket execution")
    visual = report["visual"]
    parsed = visual.get("parsed")
    if (not isinstance(parsed, dict) or
            parsed.get("class") not in {
                "held", "miss", "slip", "unobservable"} or
            not isinstance(parsed.get("evidence_views"), list) or
            not all(isinstance(view, str)
                    for view in parsed["evidence_views"]) or
            len(set(parsed["evidence_views"])) !=
            len(parsed["evidence_views"]) or
            not set(parsed["evidence_views"]) <=
            phase_cameras["before_grasp"]):
        raise ValueError("saved lift VLM response is invalid")
    if visual.get("parse_error") is None:
        try:
            raw_parsed = _parse_object(visual.get("raw_answer", ""))
        except (TypeError, ValueError) as exc:
            raise ValueError("saved lift raw VLM answer is not JSON") from exc
        if raw_parsed != parsed:
            raise ValueError("lift raw and parsed VLM answers differ")
    if raw_visual:
        if (report.get("center_rise_m") is not None or
                report.get("minimum_center_rise_m") is not None):
            raise ValueError("raw lift must not invent key-center rise")
        label, reason = _raw_lift_verdict(
            visual_class=parsed["class"],
            evidence_views=set(parsed["evidence_views"]),
            parse_error=visual.get("parse_error"),
            cameras=phase_cameras["before_grasp"],
            minimum_views=minimum_views)
    else:
        rise = float(report.get("center_rise_m", float("nan")))
        minimum_rise = float(report.get("minimum_center_rise_m", float("nan")))
        if (not math.isfinite(rise) or not math.isfinite(minimum_rise) or
                minimum_rise <= 0):
            raise ValueError("saved lift key-rise evidence is invalid")
        label, reason = _lift_verdict(
            visual_class=parsed["class"],
            evidence_views=set(parsed["evidence_views"]),
            parse_error=visual.get("parse_error"),
            cameras=phase_cameras["before_grasp"], rise_m=rise,
            minimum_rise_m=minimum_rise, minimum_views=minimum_views)
    if report.get("grasp_success") is not label or report.get("reason") != reason:
        raise ValueError("saved lift label conflicts with its VLM/key evidence")
    return report
