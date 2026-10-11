"""Export a real post-lift key/wrist observation for grasp commissioning.

This is an optional by-product of the visible-key route, not a replacement
for that route's held-transfer preflight. Camera and wrist measurement error
bounds are supplied by the rig operator; the code never invents them.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .config import TaskMode
from .geometry import validate_se3
from .key_perception import KeyPoseObservation, verify_key_capture_artifacts
from .live_robot_state import LiveRobotState


_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
BOUND_FIELDS = frozenset({
    "key_translation_error_bound_m", "key_rotation_error_bound_deg",
    "wrist_translation_error_bound_m", "wrist_rotation_error_bound_deg",
})


def write_observed_physical_pickup_sample(
    *, mode: TaskMode, calibration, planner,
    observation: KeyPoseObservation, evidence_dir: Path,
    joint_sample: LiveRobotState, candidate_id: str, trial_id: str,
    error_bounds: Mapping[str, float], max_state_skew_s: float,
    output_path: Path,
) -> Path:
    """Save one independently image-observed key and measured wrist pair.

    Only a real held-postlift FoundPose observation and synchronized robot
    feedback are accepted. The declared bounds are provenance, not validated
    accuracy; the subsequent calibration remains a descriptive sample summary.
    """
    if (not isinstance(mode, TaskMode) or
            not isinstance(observation, KeyPoseObservation) or
            observation.phase != "held_postlift" or
            observation.family != mode.family or
            observation.key_object != mode.key_object or
            not isinstance(joint_sample, LiveRobotState) or
            joint_sample.source != "robot_joint_feedback"):
        raise ValueError("physical sample needs a real held key and robot feedback")
    parts = candidate_id.split("/") if isinstance(candidate_id, str) else []
    if (len(parts) != 3 or
            any(_PART.fullmatch(part) is None for part in parts) or
            not isinstance(trial_id, str) or not trial_id.strip()):
        raise ValueError("physical sample needs an exact v8 grasp and trial ID")
    if (not isinstance(error_bounds, Mapping) or
            set(error_bounds) != BOUND_FIELDS or
            any(type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0
                for value in error_bounds.values())):
        raise ValueError("physical sample needs four commissioned positive bounds")
    observation.require_state_alignment(
        state_timestamp_s=joint_sample.sample_timestamp_s,
        maximum_skew_s=max_state_skew_s)
    if (joint_sample.full_q.shape != (13,) or
            not np.all(np.isfinite(joint_sample.full_q))):
        raise ValueError("physical sample needs measured FR3/Inspire joints")
    source = Path(evidence_dir).expanduser().resolve()
    manifest = verify_key_capture_artifacts(source)
    saved = json.loads((source / "key_observation.json").read_text(
        encoding="utf-8"))
    if (saved != observation.to_record() or
            manifest["capture_id"] != observation.capture_id or
            manifest["request_id"] != observation.request_id):
        raise ValueError("physical sample key differs from saved camera evidence")
    c2r = validate_se3(calibration.record["c2r"], name="session C2R")
    key_robot = validate_se3(
        np.linalg.inv(c2r) @ observation.pose_world,
        name="physically observed T_robot_key")
    wrist_robot = validate_se3(
        planner.fk_wrist(joint_sample.full_q),
        name="measured T_robot_hand FK")
    output = Path(output_path).expanduser().resolve()
    record = {
        "trial_id": trial_id,
        "source": "physical_independent_key_and_wrist",
        "candidate_key": parts,
        "T_robot_key_observed": key_robot.tolist(),
        "T_robot_hand_measured": wrist_robot.tolist(),
        "hand_q_measured": joint_sample.full_q[7:].tolist(),
        **{name: float(value) for name, value in error_bounds.items()},
        "key_measurement_method": "fresh_multiview_SAM3_FoundPose",
        "wrist_measurement_method": "measured_FR3_joints_plus_planner_FK",
        "key_evidence_manifest_path": str(source / "evidence_manifest.json"),
        "key_evidence_manifest_sha256": hashlib.sha256(
            (source / "evidence_manifest.json").read_bytes()).hexdigest(),
        "capture_id": observation.capture_id,
        "camera_request_id": observation.request_id,
        "joint_sample_timestamp_s": joint_sample.sample_timestamp_s,
        "scope": "physical_observation_not_future_trial_error_bound_or_motion_permit",
        "robot_ready": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output
