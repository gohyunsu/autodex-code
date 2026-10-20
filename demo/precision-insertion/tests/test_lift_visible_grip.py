"""The lift label must not depend on whether the key began upright or lying."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.lift_visible_grip import (  # noqa: E402
    QUESTIONS, observe_visible_grip, verify_visible_grip,
)
from precision_insertion.observer import LabeledFrame  # noqa: E402


class Answers:
    max_parallel_views = 2
    model = "saved-image-test"

    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def infer_for_camera(self, camera, images, prompt):
        self.requests.append((camera, images[0].size, prompt))
        return self.responses[camera]


def _scene(root, *, lying):
    (root / "masks").mkdir()
    frames = []
    for camera in QUESTIONS:
        mask = Image.new("L", (100, 100), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((30, 45, 65, 55) if lying else
                       (45, 30, 55, 65), fill=255)
        mask.save(root / "masks" / f"{camera}.png")
        frames.extend((
            LabeledFrame(camera, "before_grasp", 1.,
                         Image.new("RGB", (100, 100), "white")),
            LabeledFrame(camera, "after_lift", 2.,
                         Image.new("RGB", (100, 100), "white")),
        ))
    return frames


@pytest.mark.parametrize("lying", [False, True])
@pytest.mark.parametrize("answer,expected", [("YES", "held"), ("NO", "miss")])
def test_grip_label_is_independent_of_initial_key_angle(tmp_path, lying, answer,
                                                        expected):
    backend = Answers({camera: answer for camera in QUESTIONS})
    visual = observe_visible_grip(backend, _scene(tmp_path, lying=lying), tmp_path)
    assert visual.parsed["class"] == expected
    assert len(visual.parsed["evidence_views"]) == 2
    assert {camera for camera, _, _ in backend.requests} == set(QUESTIONS)
    assert all(len(prompt) < 200 for _, _, prompt in backend.requests)
    verify_visible_grip(visual.to_record(), set(QUESTIONS), tmp_path)


def test_disagreeing_cameras_abstain(tmp_path):
    answers = dict(zip(QUESTIONS, ("YES", "NO")))
    visual = observe_visible_grip(
        Answers(answers), _scene(tmp_path, lying=True), tmp_path)
    assert visual.parsed["class"] == "unobservable"
    assert visual.parsed["evidence_views"] == []
    saved = visual.to_record()
    saved["aggregation"]["per_view"][0]["raw_answer"] = "NO"
    with pytest.raises(ValueError, match="answer changed|consensus changed|crop or answer"):
        verify_visible_grip(saved, set(QUESTIONS), tmp_path)
