"""Check the saved post-lift key/wrist relation used by retry replanning.

This verifies the relation and selected candidate in an observed-route report.
It does not claim that a VLM pose has the accuracy of physical calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import TaskMode
from .geometry import validate_se3


def verify_observed_postlift_source(
    report_path: Path, *, mode: TaskMode, candidate_dir: Path,
) -> dict:
    path = Path(report_path).expanduser().resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    candidate = Path(candidate_dir).expanduser().resolve()
    parts = candidate.parts[-3:]
    if (not isinstance(saved, dict) or
            saved.get("schema") != "precision_insertion_postlift_preflight_v1" or
            saved.get("status") != "sampled_postlift_preflight_pass" or
            saved.get("candidate_key") != list(parts) or
            saved.get("robot_ready") is not False or
            not isinstance(saved.get("planning"), dict) or
            saved["planning"].get("sampled_planning_pass") is not True or
            not isinstance(saved.get("endpoint_screen"), dict) or
            saved["endpoint_screen"].get("endpoint_pass") is not True or
            not isinstance(saved.get("key_observation_id"), str) or
            not saved["key_observation_id"]):
        raise ValueError("observed post-lift source is not a passing selected grasp")
    relation = saved.get("observed_held_relation", {})
    if (not isinstance(relation, dict) or
            relation.get("schema") !=
            "precision_insertion_observed_held_relation_v1"):
        raise ValueError("post-lift source has no observed held relation")
    key = validate_se3(relation.get("T_robot_key_canonical"),
                       name="observed post-lift key")
    wrist = validate_se3(saved.get("T_robot_wrist_measured"),
                         name="observed post-lift wrist")
    transform = validate_se3(relation.get("T_key_hand"),
                             name="observed post-lift T_key_hand")
    endpoint = validate_se3(saved["endpoint_screen"].get("T_key_hand"),
                            name="observed post-lift endpoint relation")
    if (not np.allclose(key @ transform, wrist, atol=1e-6, rtol=0) or
            not np.allclose(transform, endpoint, atol=1e-8, rtol=0) or
            saved["endpoint_screen"].get("mode") != {
                "family": mode.family, "gap_mm": mode.gap_mm,
                "key_object": mode.key_object,
                "socket_object": mode.socket_object}):
        raise ValueError("observed post-lift relation differs from its endpoint")
    return saved
