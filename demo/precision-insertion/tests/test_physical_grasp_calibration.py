"""Occluded runtime keys must not turn MuJoCo or nominal poses into evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.physical_grasp_calibration import (  # noqa: E402
    calibrate_physical_held_relation, verify_physical_held_relation,
)


def _sample(tmp_path: Path, index: int, candidate_key, *, source=None):
    evidence = tmp_path / f"physical_pickup_{index}.json"
    key = np.eye(4)
    wrist = np.eye(4)
    wrist[0, 3] = index * 0.0001
    sample = {
        "trial_id": f"trial_{index}",
        "source": source or "physical_independent_key_and_wrist",
        "candidate_key": list(candidate_key),
        "T_robot_key_observed": key.tolist(),
        "T_robot_hand_measured": wrist.tolist(),
        "hand_q_measured": [0.2] * 6,
        "key_translation_error_bound_m": 0.0001,
        "key_rotation_error_bound_deg": 0.2,
        "wrist_translation_error_bound_m": 0.0001,
        "wrist_rotation_error_bound_deg": 0.2,
    }
    evidence.write_text(json.dumps(sample), encoding="utf-8")
    sample["evidence_path"] = str(evidence)
    sample["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    return sample


def _build(tmp_path, samples):
    candidate = ("table", "000", "084")
    return calibrate_physical_held_relation(
        mode=select_mode("square", 1.5), shared_root=tmp_path,
        candidate_key=candidate, candidate_T_key_hand=np.eye(4),
        samples=samples, minimum_independent_trials=5,
        max_nominal_translation_drift_m=0.003,
        max_nominal_rotation_drift_deg=5.0)


def test_distinct_physical_pickups_give_descriptive_not_robot_ready_envelope(tmp_path):
    candidate = ("table", "000", "084")
    result = _build(tmp_path, [_sample(tmp_path, i, candidate) for i in range(5)])
    assert result["sample_count"] == 5
    assert result["medoid_trial_id"] == "trial_2"
    assert result["empirical_translation_radius_m"] == pytest.approx(0.0002)
    assert result["descriptive_translation_envelope_m"] == pytest.approx(0.0004)
    assert result["robot_ready"] is False
    assert "future pickup repeatability or slip after the measured lift" in result["not_validated"]


def test_simulated_nominal_or_duplicate_samples_are_rejected(tmp_path):
    candidate = ("table", "000", "084")
    samples = [_sample(tmp_path, i, candidate) for i in range(5)]
    samples[0]["source"] = "mujoco_end_squeeze"
    with pytest.raises(ValueError, match="MuJoCo"):
        _build(tmp_path, samples)
    samples[0]["source"] = "physical_independent_key_and_wrist"
    samples[1]["trial_id"] = samples[0]["trial_id"]
    with pytest.raises(ValueError, match="distinct physical trial"):
        _build(tmp_path, samples)


def test_changed_source_or_different_grasp_is_rejected(tmp_path):
    candidate = ("table", "000", "084")
    samples = [_sample(tmp_path, i, candidate) for i in range(5)]
    samples[0]["candidate_key"] = ["table", "001", "084"]
    with pytest.raises(ValueError, match="different v8 grasp"):
        _build(tmp_path, samples)
    samples[0]["candidate_key"] = list(candidate)
    Path(samples[0]["evidence_path"]).write_text("modified", encoding="utf-8")
    with pytest.raises(ValueError, match="source evidence changed"):
        _build(tmp_path, samples)


def test_unstated_measurement_error_is_rejected(tmp_path):
    candidate = ("table", "000", "084")
    samples = [_sample(tmp_path, i, candidate) for i in range(5)]
    del samples[0]["key_translation_error_bound_m"]
    with pytest.raises((TypeError, ValueError)):
        _build(tmp_path, samples)


def test_unbound_fields_cannot_be_substituted_after_hashing(tmp_path):
    candidate = ("table", "000", "084")
    samples = [_sample(tmp_path, i, candidate) for i in range(5)]
    samples[0]["T_robot_hand_measured"][0][3] = 0.1
    with pytest.raises(ValueError, match="differ from hashed evidence"):
        _build(tmp_path, samples)


def test_summary_rebuilds_from_sources_and_current_v8_candidate(tmp_path):
    candidate = ("table", "000", "084")
    samples = [_sample(tmp_path, i, candidate) for i in range(5)]
    record = _build(tmp_path, samples)
    candidate_dir = tmp_path / "candidate"
    candidate_dir.mkdir()
    np.save(candidate_dir / "wrist_se3.npy", np.eye(4))
    kwargs = dict(record=record, mode=select_mode("square", 1.5),
                  shared_root=tmp_path, candidate_key=candidate,
                  candidate_dir=candidate_dir)
    assert verify_physical_held_relation(**kwargs) == record
    record["T_key_hand_medoid"][0][3] = 0.1
    with pytest.raises(ValueError, match="summary differs"):
        verify_physical_held_relation(**kwargs)
    record = _build(tmp_path, samples)
    kwargs["record"] = record
    moved = np.eye(4)
    moved[0, 3] = 0.01
    np.save(candidate_dir / "wrist_se3.npy", moved)
    with pytest.raises(ValueError, match="summary differs"):
        verify_physical_held_relation(**kwargs)
