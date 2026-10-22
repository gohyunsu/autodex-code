"""Mask-only socket requests skip FoundPose without altering later key runs."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import precision_init_daemon as adapter  # noqa: E402


def test_mask_only_estimator_is_request_scoped(monkeypatch):
    original = object()
    daemon = adapter.PrecisionInitDaemon.__new__(adapter.PrecisionInitDaemon)
    daemon.fp = original
    daemon._precision_active_request_id = None
    daemon.cmd_receiver = SimpleNamespace(event_info={
        "run": {"request_id": 42, "skip_foundpose": True}})
    seen = []

    def stock_run(self):
        seen.append(self.fp.estimate_one_view() if self.fp is not original else
                    "original")

    monkeypatch.setattr(adapter.InitDaemon, "_do_run", stock_run)
    daemon._do_run()
    assert seen == [None]
    assert daemon.fp is original
    assert daemon._precision_active_request_id is None

    daemon.cmd_receiver.event_info["run"] = {"request_id": 43}
    daemon._do_run()
    assert seen == [None, "original"]
    assert daemon.fp is original

    daemon.cmd_receiver.event_info["run"] = {
        "request_id": 44, "skip_foundpose": True}
    def failed_run(self):
        assert self.fp.estimate_one_view() is None
        raise RuntimeError("capture failed")
    monkeypatch.setattr(adapter.InitDaemon, "_do_run", failed_run)
    with pytest.raises(RuntimeError, match="capture failed"):
        daemon._do_run()
    assert daemon.fp is original
    assert daemon._precision_active_request_id is None


def test_mask_only_init_skips_socket_foundpose_and_key_init_can_load(monkeypatch):
    daemon = adapter.PrecisionInitDaemon.__new__(adapter.PrecisionInitDaemon)
    daemon.cmd_receiver = SimpleNamespace(event_info={
        "init": {"obj_name": "socket", "skip_foundpose": True}})
    daemon.fp = None
    daemon.mode = "live"
    daemon.reader = object()
    loaded = []

    def stock_init(self):
        obj_name = self.cmd_receiver.event_info["init"]["obj_name"]
        if self.fp is None or self.fp.obj_name != obj_name:
            loaded.append(obj_name)
            self.fp = SimpleNamespace(obj_name=obj_name)

    monkeypatch.setattr(adapter.InitDaemon, "_do_init", stock_init)
    daemon._do_init()
    assert loaded == []
    assert daemon.fp.obj_name == "socket"
    assert isinstance(daemon.reader, adapter.RecordingReader)

    daemon.cmd_receiver.event_info["init"] = {"obj_name": "key"}
    daemon._do_init()
    assert loaded == ["key"]
    assert daemon.fp.obj_name == "key"
