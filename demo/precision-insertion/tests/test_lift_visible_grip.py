"""The lift label must not depend on whether the key began upright or lying."""

from pathlib import Path
import sys

from PIL import Image, ImageDraw
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.lift_visible_grip import (  # noqa: E402
    HAND_FOCUS, CAMERAS, CAPTURE4_SCHEMA, _hand_roi, _hand_transform,
    observe_visible_grip, verify_visible_grip,
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
        return self.responses.get((camera, prompt), self.responses.get(camera))


def _scene(root, *, lying):
    (root / "masks").mkdir()
    frames = []
    for camera in CAMERAS:
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


def _geometry():
    robot_urdf = (Path(__file__).resolve().parents[3] / "autodex" /
                  "planner/src/curobo/content/assets/robot/" /
                  "fr3_inspire_description/fr3_inspire_oct12_mount.urdf")
    joints = np.zeros(13)
    point = _hand_transform(joints, robot_urdf) @ HAND_FOCUS
    T_camera_robot = np.eye(4)
    T_camera_robot[:3, 3] = [-point[0], -point[1], 1. - point[2]]
    K = [[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]]
    calibration = {
        "c2r": np.eye(4).tolist(),
        "camera_calibration": {
            "intrinsics_full": {camera: {"K_undist": K} for camera in CAMERAS},
            "extrinsics_full": {camera: T_camera_robot.tolist()
                                for camera in CAMERAS},
        },
    }
    return calibration, joints, robot_urdf


@pytest.mark.parametrize("lying", [False, True])
@pytest.mark.parametrize("answer,expected", [("YES", "held"), ("NO", "miss")])
def test_grip_label_is_independent_of_initial_key_angle(tmp_path, lying, answer,
                                                        expected):
    backend = Answers({camera: answer for camera in CAMERAS})
    calibration, joints, urdf = _geometry()
    visual = observe_visible_grip(
        backend, _scene(tmp_path, lying=lying), tmp_path,
        calibration=calibration, joint_q=joints, robot_urdf=urdf)
    assert visual.parsed["class"] == expected
    assert len(visual.parsed["evidence_views"]) == 2
    assert {camera for camera, _, _ in backend.requests} == set(CAMERAS)
    assert all(len(prompt) < 200 for _, _, prompt in backend.requests)
    verify_visible_grip(visual.to_record(), set(CAMERAS), tmp_path,
                        calibration=calibration, joint_q=joints)


def test_one_confirming_close_view_is_enough(tmp_path):
    answers = dict(zip(CAMERAS, ("YES", "NO")))
    calibration, joints, urdf = _geometry()
    visual = observe_visible_grip(
        Answers(answers), _scene(tmp_path, lying=True), tmp_path,
        calibration=calibration, joint_q=joints, robot_urdf=urdf)
    assert visual.parsed["class"] == "held"
    assert visual.parsed["confirming_views"] == [CAMERAS[0]]
    saved = visual.to_record()
    saved["aggregation"]["per_view"][0]["raw_answer"] = "NO"
    with pytest.raises(ValueError, match="prompt or answer changed|consensus changed"):
        verify_visible_grip(saved, set(CAMERAS), tmp_path,
                            calibration=calibration, joint_q=joints)


def test_unreadable_answer_abstains(tmp_path):
    calibration, joints, urdf = _geometry()
    first = CAMERAS[0]
    backend = Answers({
        first: "UNCLEAR",
        **{camera: "NO" for camera in CAMERAS if camera != first},
    })
    visual = observe_visible_grip(
        backend, _scene(tmp_path, lying=False), tmp_path,
        calibration=calibration, joint_q=joints, robot_urdf=urdf)
    assert visual.parsed["class"] == "unobservable"


def test_hand_crop_uses_measured_projection():
    calibration, joints, urdf = _geometry()
    roi, pixel = _hand_roi(
        image_size=(2048, 1536), camera=CAMERAS[0],
        calibration=calibration, T_robot_hand=_hand_transform(joints, urdf))
    assert pixel == pytest.approx([50., 50.])
    assert roi == (0, 0, 650, 650)


def test_capture4_straight_mount_batches_three_hand_crops(tmp_path):
    selected = ("26053248", "25322639", "25305462")
    calibration, joints, _ = _geometry()
    template_camera = CAMERAS[0]
    template_intrinsics = calibration["camera_calibration"]["intrinsics_full"][
        template_camera]
    template_extrinsics = calibration["camera_calibration"]["extrinsics_full"][
        template_camera]
    for camera in selected:
        calibration["camera_calibration"]["intrinsics_full"][camera] = (
            template_intrinsics)
        calibration["camera_calibration"]["extrinsics_full"][camera] = (
            template_extrinsics)
    straight_urdf = (Path(__file__).resolve().parents[3] / "autodex" /
                     "planner/src/curobo/content/assets/robot/" /
                     "fr3_inspire_description/fr3_inspire.urdf")
    class BatchedAnswers:
        visible_grip_camera_ids = selected

        def __init__(self):
            self.calls = []

        def infer_many(self, requests):
            self.calls.append(requests)
            return ["YES", "NO", "NO"]

    backend = BatchedAnswers()
    frames = [
        LabeledFrame(camera, phase, float(stamp),
                     Image.new("RGB", (100, 100), "white"))
        for camera in selected
        for phase, stamp in (("before_grasp", 1), ("after_lift", 2))
    ]
    visual = observe_visible_grip(
        backend, frames, tmp_path, calibration=calibration,
        joint_q=joints, robot_urdf=straight_urdf)
    assert visual.aggregation["schema"] == CAPTURE4_SCHEMA
    assert visual.aggregation["camera_ids"] == list(selected)
    assert visual.aggregation["execution_mode"] == "single_gpu_batch"
    assert len(backend.calls) == 1 and len(backend.calls[0]) == 3
    assert visual.parsed["class"] == "held"
    verify_visible_grip(visual.to_record(), set(selected), tmp_path,
                        calibration=calibration, joint_q=joints)
