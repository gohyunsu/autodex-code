"""Camera acquisition ownership is independent of robot construction."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion.camera_stream import (  # noqa: E402
    attach_running_autodex_camera_stream, open_autodex_camera_stream,
)


class Controller:
    name = "precision_insertion_test"

    def __init__(self, owner=None):
        self.owner = owner or self.name
        self.events = []

    def arm(self, **kwargs):
        self.events.append(("arm", kwargs))

    def start(self, mode, **kwargs):
        self.events.append(("start", mode, kwargs))

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


def test_capture_ip_override_is_local_to_camera_context(monkeypatch):
    import importlib
    module = importlib.import_module(
        "paradex.io.camera_system.remote_camera_controller")
    original = module.get_pc_ip
    controller = Controller()
    with open_autodex_camera_stream(
            ("capture4",), capture_ips=("147.46.219.136",),
            name=controller.name, warmup_s=0,
            controller_factory=lambda *_args, **_kwargs: controller):
        assert module.get_pc_ip("capture4") == "147.46.219.136"
    assert module.get_pc_ip is original


def test_capture4_legacy_stream_mode_uses_native_start(monkeypatch):
    monkeypatch.setenv("PRECISION_CAMERA_ACQUISITION_MODE", "stream")
    controller = Controller()
    with open_autodex_camera_stream(
            ("capture1",), name=controller.name, warmup_s=0,
            controller_factory=lambda *_args, **_kwargs: controller):
        assert controller.events == [
            ("start", "stream", {"syncMode": False, "fps": 10})]
    assert controller.events[-2:] == [("stop",), ("end",)]


def test_capture4_probe_explicit_stream_mode_overrides_default(monkeypatch):
    monkeypatch.delenv("PRECISION_CAMERA_ACQUISITION_MODE", raising=False)
    controller = Controller()
    with open_autodex_camera_stream(
            ("capture1",), name=controller.name, warmup_s=0,
            acquisition_mode="stream",
            controller_factory=lambda *_args, **_kwargs: controller):
        assert controller.events == [
            ("start", "stream", {"syncMode": False, "fps": 10})]


def test_camera_stream_rejects_foreign_owner_and_still_closes():
    controller = Controller(owner="another_run")
    with pytest.raises(RuntimeError, match="another controller"):
        with open_autodex_camera_stream(
                ("capture1",), name=controller.name, warmup_s=0,
                controller_factory=lambda *_args, **_kwargs: controller):
            pytest.fail("foreign camera owner must not yield")
    assert controller.events[-2:] == [("stop",), ("end",)]


def test_attached_camera_stream_reads_health_without_control(monkeypatch):
    events = []
    class Socket:
        def setsockopt(self, *args):
            events.append("setsockopt")
        def connect(self, addr):
            events.append(("connect", addr))
        def recv_json(self):
            return {"running": True, "controller": "rig_manager",
                    "summary": {"expected_camera_count": 1,
                                "detected_camera_count": 1,
                                "errors": {}, "states": {"camera": "CAPTURING"}}}
        def close(self):
            events.append("close")
    class Context:
        def socket(self, kind):
            assert kind == 1
            return Socket()
        def term(self):
            events.append("term")
    class Poller:
        def register(self, sock, kind):
            self.sock = sock
            assert kind == 2
        def poll(self, timeout):
            return [(self.sock, 2)]
    monkeypatch.setitem(sys.modules, "zmq", SimpleNamespace(
        Context=Context, Poller=Poller, SUB=1, POLLIN=2, SUBSCRIBE=3, LINGER=4))
    with attach_running_autodex_camera_stream(
            ("capture1",), ("192.168.0.101",)) as statuses:
        assert statuses["capture1"]["controller"] == "rig_manager"
    assert events == ["setsockopt", "setsockopt",
                      ("connect", "tcp://192.168.0.101:5481"),
                      "close", "term"]


def test_attached_camera_stream_rejects_nonrunning_health(monkeypatch):
    class Socket:
        def setsockopt(self, *args): pass
        def connect(self, addr): pass
        def recv_json(self):
            return {"running": False, "controller": "rig_manager",
                    "summary": {"expected_camera_count": 1,
                                "detected_camera_count": 1,
                                "errors": {}, "states": {"camera": "CAPTURING"}}}
        def close(self): pass
    class Context:
        def socket(self, kind): return Socket()
        def term(self): pass
    class Poller:
        def register(self, sock, kind): self.sock = sock
        def poll(self, timeout): return [(self.sock, 2)]
    monkeypatch.setitem(sys.modules, "zmq", SimpleNamespace(
        Context=Context, Poller=Poller, SUB=1, POLLIN=2, SUBSCRIBE=3, LINGER=4))
    with pytest.raises(RuntimeError, match="not ready"):
        with attach_running_autodex_camera_stream(
                ("capture1",), ("192.168.0.101",)):
            pytest.fail("nonrunning camera stream must not yield")
