"""Tri-state VLM/sensor insertion outcome contracts."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.outcome import (  # noqa: E402
    InsertionEvidence, judge_insertion,
)


def _evidence(**changes):
    fields = {
        "vlm_class": "normal_20mm",
        "key_depth_interval_m": (0.0202, 0.0206),
        "key_depth_source": "crosschecked_wrist_key",
        "alignment_within_limits": True,
        "safety_abort": False,
        "grasp_held": True,
    }
    fields.update(changes)
    return InsertionEvidence(**fields)


def test_success_requires_vlm_and_conservative_key_depth():
    result = judge_insertion(_evidence())
    assert result.insertion_success is True
    assert result.to_record()["vlm_insertion_assessment"] == "normal_20mm"
    assert judge_insertion(_evidence(
        key_depth_interval_m=(0.0198, 0.0205)
    )).insertion_success is None


def test_wrist_stroke_or_occlusion_cannot_claim_true():
    with pytest.raises(ValueError, match="key-depth source"):
        judge_insertion(_evidence(key_depth_source="commanded_wrist_stroke"))
    assert judge_insertion(_evidence(
        vlm_class="unobservable"
    )).insertion_success is None
    assert judge_insertion(_evidence(
        key_depth_interval_m=None, key_depth_source=None
    )).insertion_success is None


def test_vlm_failure_and_sensor_contradiction_remain_distinct():
    assert judge_insertion(_evidence(
        vlm_class="rim_jam", key_depth_interval_m=(0.006, 0.010)
    )).insertion_success is False
    conflict = judge_insertion(_evidence(vlm_class="rim_jam"))
    assert conflict.insertion_success is None
    assert conflict.reason == "vlm_and_key_depth_conflict"


def test_hard_safety_and_definite_shortfall_veto_vlm_success():
    assert judge_insertion(_evidence(safety_abort=True)).reason == "safety_abort"
    short = judge_insertion(_evidence(key_depth_interval_m=(0.018, 0.019)))
    assert short.insertion_success is False
    assert short.reason == "key_depth_definitely_short"
    assert judge_insertion(_evidence(alignment_within_limits=False)).insertion_success is False


def test_unknown_sensor_state_is_not_success():
    assert judge_insertion(_evidence(safety_abort=None)).insertion_success is None
    assert judge_insertion(_evidence(grasp_held=None)).insertion_success is None
    assert judge_insertion(_evidence(alignment_within_limits=None)).insertion_success is None
