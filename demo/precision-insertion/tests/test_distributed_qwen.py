"""Independent capture-PC votes can finish a live visual checkpoint early."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.observer import (  # noqa: E402
    LabeledFrame, XYView, observe_insertion_visual, observe_lift,
    observe_preinsert_temporal_views, observe_released_insertion_visual,
    observe_xy_views,
    verify_per_view_consensus,
)
from precision_insertion.grounded_alignment import observe_grounded_cylinder_axis  # noqa: E402
from precision_insertion.preinsert_checkpoint import (  # noqa: E402
    _verify_visual_record, _verify_visual_replay,
)
from precision_insertion.retry_insertion_checkpoint import _RecordedAnswer  # noqa: E402
from precision_insertion.view_parallel import map_view_calls_until  # noqa: E402
from precision_insertion.distributed_qwen import DistributedQwenBackend  # noqa: E402
from precision_insertion.xy_overlay import CalibratedXYFrame  # noqa: E402
from precision_insertion.xy_voting import XYChoice  # noqa: E402


def test_first_three_consistent_responses_finish_checkpoint_while_others_drain():
    class FiveWorkers:
        model = "saved-test-model"
        preferred_image_batch_size = 2
        max_parallel_views = 5
        early_quorum_views = 3

        def __init__(self):
            self.pools = []

        def register_view_pool(self, pool):
            self.pools.append(pool)

        def infer(self, images, prompt):
            camera = re.search(r"Valid camera IDs: (\d+)\.", prompt).group(1)
            time.sleep(.02 if camera in {"1", "2", "3"} else .3)
            return json.dumps({
                "class": "held" if camera in {"1", "2", "3"} else "miss",
                "evidence_views": [camera], "evidence": "visible key",
            })

    backend = FiveWorkers()
    image = Image.new("RGB", (32, 32), "white")
    frames = [LabeledFrame(camera, phase, stamp, image)
              for camera in ("1", "2", "3", "4", "5")
              for phase, stamp in (("before_grasp", 1.0), ("after_lift", 2.0))]
    started = time.perf_counter()
    result = observe_lift(backend, frames).to_record()
    assert time.perf_counter() - started < .25
    assert result["parsed"]["class"] == "held"
    assert result["aggregation"]["completed_camera_ids"] == ["1", "2", "3"]
    verify_per_view_consensus(
        result, field="class", fallback="unobservable",
        cameras={"1", "2", "3", "4", "5"},
        before="before_grasp", after="after_lift")
    for pool in backend.pools:
        pool.shutdown(wait=True)


def test_failed_worker_does_not_prevent_three_other_responses():
    class Backend:
        max_parallel_views = 5
        early_quorum_views = 3

    def call(camera):
        if camera == "1":
            raise RuntimeError("worker unavailable")
        return camera

    completed = map_view_calls_until(
        Backend(), ["1", "2", "3", "4", "5"], call,
        lambda done: len(done) >= 3)
    assert len(completed) >= 3
    assert "1" not in completed


def test_preinsert_replay_accepts_only_completed_early_views():
    class FiveWorkers:
        model = "saved-test-model"
        max_parallel_views = 5
        early_quorum_views = 3

        def __init__(self):
            self.pools = []

        def register_view_pool(self, pool):
            self.pools.append(pool)

        def infer(self, images, prompt):
            camera = re.search(r"Valid camera IDs: (\d+)\.", prompt).group(1)
            time.sleep(.02 if camera in {"1", "2", "3"} else .2)
            return json.dumps({
                "class": ("coarse_match" if camera in {"1", "2", "3"}
                          else "gross_misalignment"),
                "evidence_views": [camera], "evidence": "visible key",
            })

    backend = FiveWorkers()
    image = Image.new("RGB", (32, 32), "white")
    frames = [LabeledFrame(camera, phase, stamp, image)
              for camera in ("1", "2", "3", "4", "5")
              for phase, stamp in (("after_lift", 1.0), ("preinsert", 2.0))]
    visual = observe_preinsert_temporal_views(
        backend, frames, max_capture_skew_s=.1).to_record()
    assert visual["status"] == "coarse_match"
    assert len(visual["per_view"]) == 3
    views = {camera: {"timestamp_s": 2.0}
             for camera in ("1", "2", "3", "4", "5")}
    after = {camera: {"timestamp_s": 1.0} for camera in views}
    _verify_visual_record(visual, views, after)
    _verify_visual_replay(visual, frames)
    for pool in backend.pools:
        pool.shutdown(wait=True)


def test_retry_replay_uses_actual_first_three_cameras():
    class FiveWorkers:
        model = "saved-test-model"
        preferred_image_batch_size = 2
        max_parallel_views = 5
        early_quorum_views = 3

        def __init__(self):
            self.pools = []

        def register_view_pool(self, pool):
            self.pools.append(pool)

        def infer(self, _images, prompt):
            camera = re.search(r"Valid camera IDs: (\d+)\.", prompt).group(1)
            time.sleep(.01 if camera in {"1", "4", "5"} else .2)
            return json.dumps({
                "class": "held", "evidence_views": [camera],
                "evidence": "visible key",
            })

    backend = FiveWorkers()
    image = Image.new("RGB", (32, 32), "white")
    frames = [LabeledFrame(camera, phase, stamp, image)
              for camera in ("1", "2", "3", "4", "5")
              for phase, stamp in (("before_grasp", 1.0), ("after_lift", 2.0))]
    saved = observe_lift(backend, frames).to_record()
    assert saved["aggregation"]["completed_camera_ids"] == ["1", "4", "5"]
    replay = observe_lift(_RecordedAnswer(saved), frames).to_record()
    replay["latency_s"] = saved["latency_s"]
    for rebuilt, original in zip(replay["aggregation"]["per_view"],
                                 saved["aggregation"]["per_view"]):
        rebuilt["latency_s"] = original["latency_s"]
    assert replay == saved
    for pool in backend.pools:
        pool.shutdown(wait=True)


def test_all_live_vlm_prompt_forms_route_to_one_worker():
    backend = object.__new__(DistributedQwenBackend)
    backend._channels = {camera: None for camera in ("24122734", "25322649")}
    assert backend._camera_from_prompt(
        "Valid camera IDs: 24122734. In evidence_views use camera IDs") == "24122734"
    assert backend._camera_from_prompt(
        "Camera 25322649; ORIGINAL UNDISTORTED image") == "25322649"
    assert backend._camera_from_prompt(
        "Camera: 25322649. Return JSON") == "25322649"


def test_final_and_released_visuals_use_early_per_camera_consensus():
    class FiveWorkers:
        model = "saved-test-model"
        preferred_image_batch_size = 2
        max_parallel_views = 5
        early_quorum_views = 3

        def __init__(self):
            self.pools = []

        def register_view_pool(self, pool):
            self.pools.append(pool)

        def infer(self, _images, prompt):
            camera = re.search(r"Valid camera IDs: (\d+)\.", prompt).group(1)
            time.sleep(.01 if camera in {"1", "2", "3"} else .2)
            label = ("inside_socket" if "initial tabletop" in prompt
                     else "normal_appearance")
            return json.dumps({"visual_class": label,
                               "evidence_views": [camera],
                               "evidence": "visible white key"})

    backend = FiveWorkers()
    image = Image.new("RGB", (32, 32), "white")
    for observer, before, after, label in (
        (observe_insertion_visual, "preinsert", "final_or_abort",
         "normal_appearance"),
        (observe_released_insertion_visual, "initial_tabletop",
         "final_clear_view", "inside_socket"),
    ):
        frames = [LabeledFrame(camera, phase, stamp, image)
                  for camera in ("1", "2", "3", "4", "5")
                  for phase, stamp in ((before, 1.0), (after, 2.0))]
        saved = observer(backend, frames).to_record()
        assert saved["parsed"]["visual_class"] == label
        assert saved["aggregation"]["completed_camera_ids"] == ["1", "2", "3"]
        verify_per_view_consensus(
            saved, field="visual_class", fallback="unobservable",
            cameras={"1", "2", "3", "4", "5"},
            before=before, after=after)
    for pool in backend.pools:
        pool.shutdown(wait=True)


def test_xy_and_metric_grounding_fan_out_to_selected_workers():
    class FiveWorkers:
        model = "saved-test-model"
        max_parallel_views = 5
        native_pixel_coordinates = True

        def infer(self, _images, prompt):
            if "Camera:" in prompt:
                return json.dumps({"visible": True, "choice_id": "hold",
                                   "failure_class": "roughly_aligned",
                                   "evidence": "centres overlap"})
            return json.dumps({"tip_px": [20, 20],
                               "axis_line_px": [[20, 20], [20, 50]],
                               "evidence": "visible shaft"})

    backend = FiveWorkers()
    image = Image.new("RGB", (100, 100), "white")
    cameras = ("1", "2", "3", "4", "5")
    views = [XYView(camera, 1.0, image, image) for camera in cameras]
    votes, records = observe_xy_views(
        backend, views, [XYChoice("hold", (0.0, 0.0))])
    assert [vote.camera_id for vote in votes] == list(cameras)
    assert len(records) == 5
    frames = [CalibratedXYFrame(camera, 1.0, image, np.eye(4),
                                np.eye(3)) for camera in cameras]
    lines, observations = observe_grounded_cylinder_axis(backend, frames)
    assert [line.frame.camera_id for line in lines] == list(cameras)
    assert len(observations) == 5
