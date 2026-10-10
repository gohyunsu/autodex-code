"""Separate, append-only grasp, transfer and task-success labels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.outcome import InsertionEvidence  # noqa: E402
from precision_insertion.records import begin_attempt  # noqa: E402


MODE = select_mode("cylinder", 20)
SESSION = {
    "schema": "precision_insertion_session_calibration_v1",
    "mode": {
        "family": MODE.family, "gap_mm": MODE.gap_mm,
        "key_object": MODE.key_object, "socket_object": MODE.socket_object,
    },
    "socket_pose_robot": [[1, 0, 0, 0]] * 4,
}


def _record():
    return begin_attempt(
        attempt_id="trial_001", mode=MODE, session_record=SESSION,
        candidate_id="table/0/3", tabletop_pose_stem="000",
        xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0,
    )


def _grasp_refs():
    return {"vlm_observation": "vlm/lift.json",
            "key_wrist_check": "sensors/key_wrist.json"}


def _transfer_refs():
    return {"trajectory": "planner/transfer.json",
            "key_socket_pose": "sensors/hold_pose.json",
            "grasp_state": "sensors/grasp_state.json"}


def _insertion_refs():
    return {"vlm_observation": "vlm/final.json",
            "key_depth": "sensors/depth.json",
            "alignment": "sensors/alignment.json",
            "force_trace": "sensors/wrench.json"}


def test_separate_stage_labels_and_fused_insertion_success(tmp_path):
    record = _record()
    record.record_stage("grasp_success", True, timestamp_s=2.0,
                        evidence_refs=_grasp_refs())
    assert record.to_record()["preinsert_reached"] is None
    assert record.to_record()["insertion_success"] is None
    record.record_stage("preinsert_reached", True, timestamp_s=3.0,
                        evidence_refs=_transfer_refs())
    assert record.to_record()["insertion_success"] is None
    outcome = record.record_insertion_evidence(
        InsertionEvidence(
            vlm_class="normal_appearance",
            key_depth_interval_m=(0.0201, 0.0204),
            key_depth_source="key_pose_multiview",
            alignment_within_limits=True,
            safety_abort=False, grasp_held=True,
        ), timestamp_s=4.0, evidence_refs=_insertion_refs(),
    )
    assert outcome["insertion_success"] is True
    payload = record.to_record()
    assert [payload[name] for name in (
        "grasp_success", "preinsert_reached", "insertion_success",
        "release_success", "reset_success")] == [True, True, True, None, None]
    assert [event["event_index"] for event in payload["events"]] == [0, 1, 2]
    path = tmp_path / "attempts" / "trial_001.json"
    record.write_new(path)
    assert json.loads(path.read_text())["session_calibration_sha256"] == (
        payload["session_calibration_sha256"])
    with pytest.raises(FileExistsError):
        record.write_new(path)


def test_failed_grasp_does_not_falsely_mark_transfer_or_insertion():
    record = _record()
    record.record_stage("grasp_success", False, timestamp_s=2.0,
                        evidence_refs={"vlm_observation": "vlm/miss.json"})
    record.record_failure("grasp_miss", timestamp_s=2.1,
                          evidence_refs={"camera": "frames/after_lift.png"})
    payload = record.to_record()
    assert payload["grasp_success"] is False
    assert payload["preinsert_reached"] is None
    assert payload["insertion_success"] is None
    with pytest.raises(ValueError, match="requires observed grasp_success"):
        record.record_stage("preinsert_reached", False, timestamp_s=3.0,
                            evidence_refs={"planner": "plan.json"})
    record.record_stage("reset_success", True, timestamp_s=4.0,
                        evidence_refs={"key_pose": "sensors/reset_pose.json"})
    assert record.to_record()["reset_success"] is True


def test_insertion_requires_fusion_and_earlier_milestones():
    record = _record()
    with pytest.raises(ValueError, match="use record_insertion_evidence"):
        record.record_stage("insertion_success", True, timestamp_s=2.0,
                            evidence_refs=_insertion_refs())
    with pytest.raises(ValueError, match="requires observed preinsert_reached"):
        record.record_insertion_evidence(
            InsertionEvidence("normal_appearance", (0.021, 0.022),
                              "key_pose_multiview", True, False, True),
            timestamp_s=2.0, evidence_refs=_insertion_refs())
    record.record_stage("grasp_success", True, timestamp_s=2.0,
                        evidence_refs=_grasp_refs())
    record.record_stage("preinsert_reached", True, timestamp_s=3.0,
                        evidence_refs=_transfer_refs())
    outcome = record.record_insertion_evidence(
        InsertionEvidence("unobservable", (0.021, 0.022),
                          "key_pose_multiview", True, False, True),
        timestamp_s=4.0, evidence_refs=_insertion_refs())
    assert outcome["insertion_success"] is None
    assert record.to_record()["insertion_success"] is None
    with pytest.raises(ValueError, match="stage already recorded"):
        record.record_insertion_evidence(
            InsertionEvidence("normal_appearance", (0.021, 0.022),
                              "key_pose_multiview", True, False, True),
            timestamp_s=5.0, evidence_refs=_insertion_refs())


def test_session_identity_evidence_and_event_order_are_checked():
    wrong = dict(SESSION, mode=dict(SESSION["mode"], gap_mm=1))
    with pytest.raises(ValueError, match="does not match frozen socket"):
        begin_attempt(
            attempt_id="trial_001", mode=MODE, session_record=wrong,
            candidate_id="table/0/3", tabletop_pose_stem="000",
            xy_offset_socket_m=(0.0, 0.0), started_at_s=1.0)
    record = _record()
    with pytest.raises(ValueError, match="missing evidence references"):
        record.record_stage("grasp_success", True, timestamp_s=2.0,
                            evidence_refs={"vlm_observation": "vlm/lift.json"})
    record.record_stage("grasp_success", True, timestamp_s=2.0,
                        evidence_refs=_grasp_refs())
    with pytest.raises(ValueError, match="nondecreasing time"):
        record.record_stage("preinsert_reached", False, timestamp_s=1.9,
                            evidence_refs={"planner": "failed.json"})
