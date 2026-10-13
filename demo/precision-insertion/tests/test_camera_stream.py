"""Camera acquisition ownership is independent of robot construction."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.camera_stream import open_autodex_camera_stream  # noqa: E402


class Controller:
    name = "precision_insertion_test"

    def __init__(self, owner=None):
        self.owner = owner or self.name
        self.events = []

    def arm(self, **kwargs):
        self.events.append(("arm", kwargs))

    def set_stream(self, value):
        self.events.append(("stream", value))

    def get_status(self):
        return {"pc": {"capture1": {"controller": self.owner}}}

    def stop(self):
        self.events.append(("stop",))

    def end(self):
        self.events.append(("end",))


def test_camera_stream_uses_stock_fps_and_releases_on_exit():
    controller = Controller()
    def build(name, *, pc_list, stall_timeout):
        assert name == controller.name
        assert pc_list == ["capture1"] and stall_timeout == 15.
        return controller
    with open_autodex_camera_stream(
            ("capture1",), name=controller.name, warmup_s=0,
            controller_factory=build):
        assert controller.events == [
            ("arm", {"syncMode": False, "fps": 10}),
            ("stream", True)]
    assert controller.events[-2:] == [("stop",), ("end",)]


def test_camera_stream_rejects_foreign_owner_and_still_closes():
    controller = Controller(owner="another_run")
    with pytest.raises(RuntimeError, match="another controller"):
        with open_autodex_camera_stream(
                ("capture1",), name=controller.name, warmup_s=0,
                controller_factory=lambda *_args, **_kwargs: controller):
            pytest.fail("foreign camera owner must not yield")
    assert controller.events[-2:] == [("stop",), ("end",)]
