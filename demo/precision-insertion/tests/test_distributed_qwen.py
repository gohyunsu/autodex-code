"""Independent capture-PC votes can finish a live visual checkpoint early."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
import re
import shlex
import sys
import threading
import time
from io import StringIO

import numpy as np
import pytest
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
from precision_insertion import distributed_qwen, distributed_qwen_worker  # noqa: E402
from precision_insertion.xy_overlay import CalibratedXYFrame  # noqa: E402
from precision_insertion.xy_voting import XYChoice  # noqa: E402


def test_capture_worker_startup_owns_one_gpu_slot(monkeypatch):
    commands = []
    monkeypatch.setattr(distributed_qwen.subprocess, "Popen",
                        lambda args, **_kwargs: commands.append(args) or object())
    monkeypatch.setattr(DistributedQwenBackend, "_wait_ready",
                        lambda self, **_kwargs: None)
    workers = {str(index): {
        "host": f"capture{index}", "python": f"/home/capture{index}/python",
    } for index in range(1, 6)}
    DistributedQwenBackend(
        model_id="/home/robot/model",
        worker_root=Path("/home/robot/precision_worker_test"),
        workers=workers, max_input_size=(640, 480), max_new_tokens=32,
        classification_image_size=(384, 288))
    assert len(commands) == 5
    for args in commands:
        command = shlex.split(args[-1])
        assert command[:5] == [
            "flock", "-n", "-E", "73",
            "/tmp/precision-insertion-qwen-worker.lock"]


def test_capture_worker_shutdown_releases_model_process(monkeypatch):
    class MemoryStream(StringIO):
        def close(self):
            self.saved = self.getvalue()
            super().close()

    class Process:
        stdin = MemoryStream()

        def poll(self):
            return None

        def wait(self, timeout):
            assert timeout == 15.
            return 0

    process = Process()
    backend = object.__new__(DistributedQwenBackend)
    backend._channels = {"24122734": (process, threading.Lock(), Path("/tmp/log"))}
    backend._background_pools = []
    backend.close()
    assert json.loads(process.stdin.saved.strip()) == {
        "type": "shutdown", "camera_id": "24122734"}


@pytest.mark.parametrize("failure,reply,expected", [
    ("timeout", "", "inference timed out"),
    ("wrong_id", '{"type":"result","request_id":"stale",'
     '"camera_id":"24122734"}\n', "response identity changed"),
    ("bad_json", "model progress\n", "sent invalid JSON"),
])
def test_unbound_qwen_reply_retires_only_its_worker(
        monkeypatch, failure, reply, expected):
    class Process:
        def __init__(self):
            self.stdin = StringIO()
            self.stdout = StringIO(reply)
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            assert timeout == 5.
            return -9

    process = Process()
    other = Process()
    backend = object.__new__(DistributedQwenBackend)
    backend._channels = {
        "24122734": (process, threading.Lock(), Path("/tmp/qwen-test.log")),
        "25322649": (other, threading.Lock(), Path("/tmp/qwen-other.log"))}
    backend.max_new_tokens = 32
    backend.log_path = None
    monkeypatch.setattr(distributed_qwen.select, "select",
                        lambda streams, *_: (
                            [] if failure == "timeout" else streams, [], []))
    image = Image.new("RGB", (32, 32), "white")
    with pytest.raises((RuntimeError, TimeoutError), match=expected):
        backend.infer_for_camera("24122734", [image], "key box")
    assert process.killed
    assert not other.killed
    with pytest.raises(RuntimeError, match="exited"):
        backend.infer_for_camera("24122734", [image], "key box")


def test_remote_worker_exits_cleanly_on_shutdown(monkeypatch):
    class FakeCuda:
        @staticmethod
        def get_device_name(_index):
            return "test GPU"

        @staticmethod
        def memory_allocated():
            return 0

    class FakeModel:
        def __init__(self, **_kwargs):
            pass

        def infer_images_prompt(self, *_args, **_kwargs):
            raise AssertionError("shutdown must not infer")

    monkeypatch.setitem(sys.modules, "torch", type("FakeTorch", (), {
        "cuda": FakeCuda})())
    monkeypatch.setattr(distributed_qwen_worker, "LocalQwenModel", FakeModel)
    monkeypatch.setattr(sys, "stdin", StringIO(
        '{"type":"shutdown","camera_id":"24122734"}\n'))
    output = StringIO()
    distributed_qwen_worker.serve(
        camera_id="24122734", model_id="unused",
        max_input_size=(640, 480), protocol=output)
    assert [json.loads(line)["type"] for line in output.getvalue().splitlines()] == [
        "ready", "stopped"]


def test_capture_worker_reserves_stdout_for_json_before_model_load():
    source_root = Path(__file__).resolve().parents[1]
    code = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from precision_insertion.distributed_qwen_worker import "
        "_isolate_protocol_stdout\n"
        "protocol = _isolate_protocol_stdout()\n"
        "print('model progress', flush=True)\n"
        "os.write(1, b'native progress\\n')\n"
        "print('{\"type\":\"ready\"}', file=protocol, flush=True)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        check=True)
    assert result.stdout.strip() == '{"type":"ready"}'
    assert "model progress" in result.stderr
    assert "native progress" in result.stderr


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
