"""Observed retry input must not be confused with a physical grasp medoid."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.config import select_mode  # noqa: E402
from precision_insertion.observed_postlift_source import (  # noqa: E402
    verify_observed_postlift_source,
)


def test_observed_source_binds_candidate_wrist_and_endpoint(tmp_path):
    mode = select_mode("cylinder", 15)
    candidate = tmp_path / "table" / "0" / "194"
    key = np.eye(4)
    key[0, 3] = .1
    relation = np.eye(4)
    relation[2, 3] = .2
    wrist = key @ relation
    record = {
        "schema": "precision_insertion_postlift_preflight_v1",
        "status": "sampled_postlift_preflight_pass",
        "candidate_key": ["table", "0", "194"],
        "robot_ready": False,
        "key_observation_id": "held_view_001",
        "planning": {"sampled_planning_pass": True},
        "observed_held_relation": {
            "schema": "precision_insertion_observed_held_relation_v1",
            "T_robot_key_canonical": key.tolist(),
            "T_key_hand": relation.tolist()},
        "T_robot_wrist_measured": wrist.tolist(),
        "endpoint_screen": {
            "endpoint_pass": True, "T_key_hand": relation.tolist(),
            "mode": {"family": mode.family, "gap_mm": mode.gap_mm,
                     "key_object": mode.key_object,
                     "socket_object": mode.socket_object}},
    }
    file = tmp_path / "postlift.json"
    file.write_text(json.dumps(record))
    assert verify_observed_postlift_source(
        file, mode=mode, candidate_dir=candidate) == record
    record["T_robot_wrist_measured"] = np.eye(4).tolist()
    file.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="relation differs"):
        verify_observed_postlift_source(file, mode=mode,
                                        candidate_dir=candidate)
