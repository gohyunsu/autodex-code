"""Exported commissioning samples come from camera key pose and robot FK."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.key_perception import KeyPoseObservation  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from precision_insertion.physical_pickup_sample import (  # noqa: E402
    write_observed_physical_pickup_sample,
)


def test_visible_postlift_exports_independent_key_and_wrist(tmp_path, monkeypatch):
    import precision_insertion.physical_pickup_sample as module

    mode = select_mode("square", 1.5)
    pose = np.eye(4)
    pose[0, 3] = .12
    observation = KeyPoseObservation(
        capture_id="held_001", request_id=12,
        key_object=mode.key_object, family=mode.family,
        pose_world=pose, selected_camera_id="cam_a",
        selected_acquisition_timestamp_s=100.,
        acquisition_interval_s=(99.99, 100.01),
        frame_evidence={}, per_view={}, consistency={}, selection={},
        source_capture_dir=tmp_path, phase="held_postlift")
    source = tmp_path / "held_key_captures/held_001"
    source.mkdir(parents=True)
    (source / "key_observation.json").write_text(
        json.dumps(observation.to_record()), encoding="utf-8")
    (source / "evidence_manifest.json").write_text(
        '{"capture_id":"held_001"}', encoding="utf-8")
    monkeypatch.setattr(module, "verify_key_capture_artifacts",
                        lambda _: {"capture_id": "held_001", "request_id": 12})
    q = np.zeros(13)
    q[7:] = .2
    state = LiveRobotState(
        q, np.zeros(7), 100., 100., 1., 100.,
        np.zeros(6), np.zeros(6), 0., np.zeros(6))
    wrist = np.eye(4)
    wrist[1, 3] = .3
    output = tmp_path / "physical_pickup_samples/trial_001.json"
    args = dict(
        mode=mode,
        calibration=SimpleNamespace(record={"c2r": np.eye(4).tolist()}),
        planner=SimpleNamespace(fk_wrist=lambda measured: wrist),
        observation=observation, evidence_dir=source, joint_sample=state,
        candidate_id="table/0/184", trial_id="trial_001",
        error_bounds={
            "key_translation_error_bound_m": .001,
            "key_rotation_error_bound_deg": 1.,
            "wrist_translation_error_bound_m": .001,
            "wrist_rotation_error_bound_deg": 1.,
        }, max_state_skew_s=.05, output_path=output)
    assert write_observed_physical_pickup_sample(**args) == output
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["source"] == "physical_independent_key_and_wrist"
    assert saved["candidate_key"] == ["table", "0", "184"]
    assert saved["T_robot_key_observed"][0][3] == pytest.approx(.12)
    assert saved["T_robot_hand_measured"][1][3] == pytest.approx(.3)
    assert saved["hand_q_measured"] == pytest.approx([.2] * 6)
    with pytest.raises(FileExistsError):
        write_observed_physical_pickup_sample(**args)
    args["candidate_id"] = "table/../184"
    args["output_path"] = tmp_path / "other.json"
    with pytest.raises(ValueError, match="exact v8 grasp"):
        write_observed_physical_pickup_sample(**args)
