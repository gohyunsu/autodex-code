"""Mask-only socket requests skip FoundPose without altering later key runs."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import precision_init_daemon as adapter  # noqa: E402


def test_mask_only_estimator_is_request_scoped(monkeypatch):
    original = object()
    daemon = adapter.PrecisionInitDaemon.__new__(adapter.PrecisionInitDaemon)
    daemon.fp = original
    daemon._precision_active_request_id = None
    daemon._precision_undistorted = {}
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


def test_capture4_maps_main_pc_asset_paths_without_changing_inference(monkeypatch):
    daemon = adapter.PrecisionInitDaemon.__new__(adapter.PrecisionInitDaemon)
    info = {
        "obj_name": "key",
        "mesh_path": "~/shared_data/object_processing/key/raw_mesh/key.obj",
        "assets_root": "~/shared_data/AutoDex/foundpose_assets/key",
    }
    daemon.cmd_receiver = SimpleNamespace(event_info={"init": info})
    daemon.mode = "live"
    daemon.reader = object()
    daemon.fp = SimpleNamespace(obj_name="key")
    monkeypatch.setenv("PRECISION_CAPTURE_ASSET_ROOT", "/nas/staged")
    seen = []
    monkeypatch.setattr(adapter.InitDaemon, "_do_init",
                        lambda self: seen.append(dict(info)))
    daemon._do_init()
    assert seen[0]["mesh_path"] == (
        "/nas/staged/object_processing/key/raw_mesh/key.obj")
    assert seen[0]["assets_root"] == (
        "/nas/staged/AutoDex/foundpose_assets/key")


def test_capture4_batch_preserves_serial_order_and_reuses_hash_pixels(monkeypatch):
    monkeypatch.setenv("PRECISION_SAM3_BATCH", "1")
    daemon = adapter.PrecisionInitDaemon.__new__(adapter.PrecisionInitDaemon)
    daemon.K_undist = {"b": np.eye(3), "a": np.eye(3)}
    daemon._precision_undistorted = {
        "b": np.full((4, 5, 3), 23, dtype=np.uint8),
        "a": np.full((4, 5, 3), 42, dtype=np.uint8),
    }
    daemon.sam3 = object()
    seen = []

    def batch(model, images, prompt):
        seen.append((model, [int(image[0, 0, 0]) for image in images], prompt))
        return [np.full((4, 5), index, dtype=np.uint8)
                for index in range(1, len(images) + 1)]

    monkeypatch.setattr(adapter, "segment_batch", batch)
    frames = {serial: (np.zeros((4, 5, 3), dtype=np.uint8), index)
              for index, serial in enumerate(("b", "a"), 1)}
    masks = daemon._segment_snapshot_batch(frames, ["b", "a"], "socket")
    assert seen == [(daemon.sam3, [23, 42], "socket")]
    assert list(masks) == ["b", "a"]
    assert int(masks["b"][0][0, 0]) == 1
    assert int(masks["a"][0][0, 0]) == 2
    assert daemon._undistort_live_image("b", frames["b"][0]) is (
        daemon._precision_undistorted["b"])


def test_capture4_auto_uses_serial_masks_when_source_pngs_are_sent(monkeypatch):
    monkeypatch.delenv("PRECISION_SAM3_BATCH", raising=False)
    daemon = adapter.PrecisionInitDaemon.__new__(adapter.PrecisionInitDaemon)
    daemon._precision_send_source_frames = True
    assert daemon._segment_snapshot_batch({}, [], "socket") is None


def test_stock_auto_dex_daemon_keeps_serial_route():
    daemon = adapter.InitDaemon.__new__(adapter.InitDaemon)
    assert daemon._segment_snapshot_batch({}, [], "key") is None
