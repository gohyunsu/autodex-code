"""The experimental and human label modes replay from saved evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.labeling import (  # noqa: E402
    manual_annotation, request_annotation, resolve_label,
)
from precision_insertion.outcome import (  # noqa: E402
    InsertionEvidence, resolve_insertion_outcome,
)
from precision_insertion.lift_checkpoint import (  # noqa: E402
    assess_raw_lift_checkpoint, verify_lift_checkpoint, write_lift_checkpoint,
)
from precision_insertion.preinsert_checkpoint import (  # noqa: E402
    assess_preinsert_checkpoint, verify_preinsert_checkpoint,
    write_preinsert_checkpoint,
)
from precision_insertion.insertion_checkpoint import (  # noqa: E402
    assess_insertion_checkpoint, verify_insertion_checkpoint,
    write_insertion_checkpoint,
)
from test_raw_lift_checkpoint import _raw_case  # noqa: E402
from test_preinsert_checkpoint import _setup as _preinsert_case  # noqa: E402
from test_insertion_checkpoint import _setup as _insertion_case  # noqa: E402


def test_visual_policy_abstains_and_manual_requires_provenance():
    assert resolve_label(
        stage="insertion_success", mode="vlm_only",
        visual_class="unobservable", fused_value=False,
        fused_reason="short_depth") == (None, "vlm_visual_abstained")
    with pytest.raises(ValueError, match="human annotation"):
        resolve_label(stage="grasp_success", mode="manual",
                      visual_class="held", fused_value=True,
                      fused_reason="held")
    with pytest.raises(ValueError, match="human label provider"):
        request_annotation(
            mode="manual", provider=None, stage="grasp_success",
            evidence_dirs=(), attempt_id="a", candidate_id="b")


def test_visual_insertion_label_is_not_a_metric_depth_claim():
    evidence = InsertionEvidence(
        "normal_appearance", None, None, None, False, None)
    result = resolve_insertion_outcome(
        evidence, target_depth_m=.02, label_mode="vlm_only")
    assert result["insertion_success"] is True
    assert result["key_depth_interval_m"] is None
    assert result["scope"] == "visual_task_label_not_verified_20mm_depth"


def test_manual_raw_lift_report_replays_and_tamper_is_rejected(tmp_path):
    args, _ = _raw_case(tmp_path)
    annotation = manual_annotation("grasp_success", False, "researcher", "slip")
    result = assess_raw_lift_checkpoint(
        **args, label_mode="manual", manual_annotation=annotation)
    assert result.grasp_success is False
    path = write_lift_checkpoint(result, tmp_path / "label") / "report.json"
    assert json.loads(path.read_text())["vlm_label"]["class"] == (
        result.visual.parsed["class"])
    assert verify_lift_checkpoint(path)["grasp_success"] is False
    altered = json.loads(path.read_text())
    altered["grasp_success"] = True
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="label conflicts"):
        verify_lift_checkpoint(path)


def test_manual_preinsert_report_replays(tmp_path):
    args = _preinsert_case(tmp_path)
    result = assess_preinsert_checkpoint(
        **args, label_mode="manual",
        manual_annotation=manual_annotation(
            "preinsert_reached", False, "researcher", "tilted"))
    assert result.preinsert_reached is False
    path = write_preinsert_checkpoint(result, tmp_path / "manual_preinsert")
    assert json.loads(path.read_text())["vlm_label"]["class"] == result.visual.status
    assert verify_preinsert_checkpoint(path)["preinsert_reached"] is False


def test_visual_and_manual_insertion_reports_replay(tmp_path):
    args, _ = _insertion_case(tmp_path)
    visual = assess_insertion_checkpoint(**args, label_mode="vlm_only")
    assert visual["outcome"]["insertion_success"] is True
    path = write_insertion_checkpoint(visual, tmp_path / "visual_insertion")
    assert verify_insertion_checkpoint(path)["outcome"] == visual["outcome"]
    annotation = manual_annotation(
        "insertion_success", False, "researcher", "rim contact")
    manual = assess_insertion_checkpoint(
        **args, label_mode="manual", manual_annotation=annotation)
    assert manual["outcome"]["insertion_success"] is False
    assert manual["vlm_label"]["value"] is True
    manual_path = write_insertion_checkpoint(manual, tmp_path / "manual_insertion")
    assert verify_insertion_checkpoint(manual_path)["outcome"] == manual["outcome"]
