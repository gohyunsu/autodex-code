"""VLM observations remain typed, read-only, and fail closed."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from PIL import Image
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.observer import (  # noqa: E402
    LabeledFrame, XYView, ZeroDexLocalBackend, observe_insertion_visual,
    observe_lift, observe_xy_views,
)
from precision_insertion.outcome import InsertionEvidence, judge_insertion  # noqa: E402
from precision_insertion.xy_voting import (  # noqa: E402
    XYChoice, resolve_multiview_choice,
)


class FakeBackend:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def infer(self, images, prompt):
        self.calls.append((images, prompt))
        return self.answers.pop(0)


def _image():
    return Image.new("RGB", (16, 12), (100, 50, 20))


def _frames(before="before_grasp", after="after_lift"):
    return [
        LabeledFrame("front", before, 1.0, _image()),
        LabeledFrame("front", after, 2.0, _image()),
    ]


def test_lift_prompt_preserves_image_order_and_validates_closed_set():
    backend = FakeBackend(json.dumps({
        "class": "held", "evidence_views": ["front"],
        "evidence": "key moves with hand",
    }))
    observed = observe_lift(backend, _frames())
    assert observed.parsed["class"] == "held"
    assert observed.image_order == (
        "before_grasp/front@1.000000", "after_lift/front@2.000000")
    assert "Hidden is not held" in backend.calls[0][1]
    assert len(backend.calls[0][0]) == 2
    assert observed.to_record()["scope"].endswith("not_motion_authorization")
    assert observed.to_record()["latency_s"] >= 0


def test_malformed_lift_answer_abstains_and_missing_phase_fails():
    observed = observe_lift(FakeBackend("the grasp succeeded"), _frames())
    assert observed.parsed["class"] == "unobservable"
    assert observed.parse_error is not None
    with pytest.raises(ValueError, match="before_grasp and after_lift"):
        observe_lift(FakeBackend("{}"), _frames("before_grasp", "after_close"))


def test_insertion_visual_is_not_physical_depth_label():
    observed = observe_insertion_visual(FakeBackend(json.dumps({
        "visual_class": "normal_appearance", "evidence_views": ["front"],
        "evidence": "key passes rim",
    })), _frames("preinsert", "final_or_abort"))
    assert observed.parsed["visual_class"] == "normal_appearance"
    assert "Do not claim numerical depth" in observed.prompt
    outcome = judge_insertion(InsertionEvidence(
        vlm_class=observed.parsed["visual_class"],
        key_depth_interval_m=None, key_depth_source=None,
        alignment_within_limits=True, safety_abort=False, grasp_held=True,
    ))
    assert outcome.insertion_success is None


def test_independent_per_view_xy_calls_produce_proposals_not_motion():
    answer = json.dumps({
        "visible": True, "choice_id": "C1", "failure_class": "misaligned",
        "evidence": "key axis left of rim center",
    })
    backend = FakeBackend(answer, answer)
    views = [
        XYView("front", 10.0, _image(), _image()),
        XYView("side", 10.01, _image(), _image()),
    ]
    choices = [XYChoice("C0", (0.0, 0.0)), XYChoice("C1", (0.001, 0.0))]
    votes, records = observe_xy_views(backend, views, choices)
    assert len(backend.calls) == 2
    assert [vote.choice_id for vote in votes] == ["C1", "C1"]
    assert all("Never invent coordinates" in call[1] for call in backend.calls)
    decision = resolve_multiview_choice(
        choices, votes, current_offset_socket_m=(0.0, 0.0),
        grasp_held=True, hard_abort=False,
        max_step_m=0.001, max_total_m=0.002,
        max_timestamp_skew_s=0.02, decision_timestamp_s=10.02,
        max_frame_age_s=0.2,
    )
    assert decision.status == "propose"
    assert decision.to_record()["scope"].endswith("not_motion_authorization")
    assert all(record.parse_error is None for record in records)


def test_invalid_xy_id_or_occluded_vote_abstains():
    backend = FakeBackend(json.dumps({
        "visible": True, "choice_id": "invented", "failure_class": "misaligned",
        "evidence": "guess",
    }))
    votes, records = observe_xy_views(
        backend, [XYView("front", 1.0, _image(), _image())],
        [XYChoice("C0", (0.0, 0.0))])
    assert votes[0].visible is False
    assert votes[0].choice_id == "abstain"
    assert records[0].parse_error is not None


def test_local_backend_reuses_loaded_vlm():
    class StubModel:
        def infer_images_prompt(self, images, prompt, max_new_tokens):
            assert max_new_tokens == 40
            return type("Result", (), {"answer": "{}"})()

    backend = ZeroDexLocalBackend(StubModel(), max_new_tokens=40)
    assert backend.infer([_image()], "test") == "{}"
