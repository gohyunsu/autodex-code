"""The ZeroDex rig policy is explicit, versioned, and behavior preserving."""

import json
from pathlib import Path

import pytest

from precision_insertion.stage_policy import (
    DEFAULT_POLICY_PATH,
    ZeroDexStagePolicy,
    load_zerodex_stage_policy,
)


def test_default_stage_policy_records_current_runtime_values(monkeypatch):
    monkeypatch.delenv("PRECISION_STAGE_POLICY", raising=False)
    policy = load_zerodex_stage_policy()
    assert isinstance(policy, ZeroDexStagePolicy)
    assert policy.source == DEFAULT_POLICY_PATH.resolve()
    assert policy.key_prompt == "small white solid peg"
    assert policy.path_limits.minimum_hand_clearance_m == 0
    assert policy.retry_limits.max_total_offset_m is None
    assert policy.alignment_limits.minimum_views == 2
    assert policy.tabletop_pose_error_deg == 20
    assert policy.post_lift_settle_s == 2
    assert policy.preinsert["render_audit_overlay"] is False


def test_environment_selects_an_explicit_stage_policy(tmp_path, monkeypatch):
    record = json.loads(DEFAULT_POLICY_PATH.read_text(
        encoding="utf-8"))
    record["key_prompt"] = "test white cylinder"
    selected = tmp_path / "stage-policy.json"
    selected.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("PRECISION_STAGE_POLICY", str(selected))
    policy = load_zerodex_stage_policy()
    assert policy.source == selected.resolve()
    assert policy.key_prompt == "test white cylinder"


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_stage_policy_rejects_ambiguous_top_level_shape(
        tmp_path, monkeypatch, mutation):
    record = json.loads(DEFAULT_POLICY_PATH.read_text(
        encoding="utf-8"))
    if mutation == "missing":
        record.pop("final")
    else:
        record["unreviewed_gate"] = {"threshold": 1}
    selected = tmp_path / "bad-stage-policy.json"
    selected.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setenv("PRECISION_STAGE_POLICY", str(selected))
    with pytest.raises(ValueError, match="stage-policy document"):
        load_zerodex_stage_policy()
