"""A long-lived camera session must close transports and never start a robot."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import live_orchestration as orchestration  # noqa: E402
from precision_insertion.clear_view import ClearViewForPerception  # noqa: E402
from precision_insertion.live_robot_state import LiveRobotState  # noqa: E402
from autodex.utils.robot_config import FR3_CLEAR_VIEW  # noqa: E402


def _clear_view(events):
    class Executor:
        arm_dof = 7
        hand_name = "inspire"

        def home(self, *, clear_view):
            assert clear_view
            events.append("home_clear_view")

    def read_state():
        now = time.time()
        raw = np.full(6, 1000., dtype=float)
        return LiveRobotState(
            np.r_[FR3_CLEAR_VIEW, np.zeros(6)], np.zeros(7), now, now,
            10., now, raw, raw.copy(), 0., np.zeros(6))

    return ClearViewForPerception(
        Executor(), read_state, .1, 20., .05, 1.)


def _patch_camera_lifetime(monkeypatch, events):
    launch = SimpleNamespace(
        mode=SimpleNamespace(family="cylinder"), shared_root=Path("/shared"))
    monkeypatch.setattr(orchestration, "load_session_launch",
                        lambda _path: launch)
    monkeypatch.setattr(orchestration, "preflight_endpoint_catalog_file",
                        lambda **_kwargs: events.append("catalog_preflight"))

    @contextmanager
    def cameras(received):
        assert received is launch
        events.append("camera_open")
        try:
            yield SimpleNamespace(
                init_orchestrator="same_init",
                acquisition_metadata_for_request="same_provider",
                start=lambda: (events.append("socket_frozen") or
                               SimpleNamespace(evidence_dir=Path("/evidence"))))
        finally:
            events.append("camera_close")

    monkeypatch.setattr(orchestration, "open_session_cameras", cameras)
    return launch


def test_live_session_keeps_same_camera_transport_for_next_key(
        tmp_path, monkeypatch):
    events = []
    launch = _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    opened = SimpleNamespace(runner="same_runner")

    def open_ledger(**kwargs):
        events.append("ledger_open")
        assert kwargs == {
            "mode": launch.mode, "shared_root": Path("/shared"),
            "evidence_dir": Path("/evidence"), "catalog_path": catalog,
            "output_dir": tmp_path / "new_run", "max_xy_retries": 2,
            "xy_update_weight": 1.0,
        }
        return opened

    monkeypatch.setattr(orchestration, "open_verified_session", open_ledger)
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog, run_output_dir=tmp_path / "new_run",
            max_xy_retries=2, clear_view=_clear_view(events)) as live:
        assert events == ["catalog_preflight", "home_clear_view", "camera_open",
                          "socket_frozen", "ledger_open"]
        assert live.opened is opened
        assert live.cameras.init_orchestrator == "same_init"
        assert live.cameras.acquisition_metadata_for_request == "same_provider"
    assert events[-1] == "camera_close"


@pytest.mark.parametrize("reason", ("missing_catalog", "existing_run",
                                    "negative_retries"))
def test_live_session_static_failures_do_not_connect(
        tmp_path, monkeypatch, reason):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    output = tmp_path / "new_run"
    retries = 1
    if reason != "missing_catalog":
        catalog.write_text("{}", encoding="utf-8")
    if reason == "existing_run":
        output.mkdir()
    if reason == "negative_retries":
        retries = -1
    with pytest.raises((FileNotFoundError, FileExistsError, ValueError)):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog, run_output_dir=output,
                max_xy_retries=retries, clear_view=_clear_view(events)):
            pytest.fail("static failure must not yield a live session")
    assert events == []


def test_failed_session_catalog_binding_closes_camera_but_keeps_capture(
        tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def reject(**_kwargs):
        events.append("catalog_rejected")
        raise ValueError("catalog differs from socket session")

    monkeypatch.setattr(orchestration, "open_verified_session", reject)
    with pytest.raises(ValueError, match="catalog differs"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog, run_output_dir=tmp_path / "run",
                max_xy_retries=1, clear_view=_clear_view(events)):
            pytest.fail("catalog rejection must not yield")
    assert events == ["catalog_preflight", "home_clear_view", "camera_open", "socket_frozen",
                      "catalog_rejected", "camera_close"]


def test_stale_catalog_preflight_does_not_connect_cameras(tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def stale(**kwargs):
        assert kwargs == {
            "mode": SimpleNamespace(family="cylinder"),
            "shared_root": Path("/shared"),
            "catalog_path": catalog,
        }
        raise ValueError("v8 catalog is stale for pose 000")

    monkeypatch.setattr(orchestration, "preflight_endpoint_catalog_file", stale)
    with pytest.raises(ValueError, match="stale for pose 000"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog,
                run_output_dir=tmp_path / "run",
                max_xy_retries=1, clear_view=_clear_view(events)):
            pytest.fail("stale catalogue must not yield a live session")
    assert events == []


def test_initial_clear_view_mismatch_prevents_camera_capture(tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    guard = _clear_view(events)
    guard.read_state = lambda: None
    with pytest.raises(TypeError, match="measured LiveRobotState"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog,
                run_output_dir=tmp_path / "run", max_xy_retries=1,
                clear_view=guard):
            pytest.fail("invalid clear-view feedback must prevent capture")
    assert events == ["catalog_preflight", "home_clear_view"]
