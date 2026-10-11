"""A long-lived camera session must close transports and never start a robot."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import live_orchestration as orchestration  # noqa: E402


def _patch_camera_lifetime(monkeypatch, events):
    launch = SimpleNamespace(mode="square", shared_root=Path("/shared"))
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
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")
    opened = SimpleNamespace(runner="same_runner")

    def open_ledger(**kwargs):
        events.append("ledger_open")
        assert kwargs == {
            "mode": "square", "shared_root": Path("/shared"),
            "evidence_dir": Path("/evidence"), "catalog_path": catalog,
            "output_dir": tmp_path / "new_run", "max_xy_retries": 2,
        }
        return opened

    monkeypatch.setattr(orchestration, "open_verified_session", open_ledger)
    with orchestration.open_live_precision_session(
            launch_config_path=tmp_path / "launch.json",
            endpoint_catalog_path=catalog, run_output_dir=tmp_path / "new_run",
            max_xy_retries=2) as live:
        assert events == ["catalog_preflight", "camera_open",
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
                max_xy_retries=retries):
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
                max_xy_retries=1):
            pytest.fail("catalog rejection must not yield")
    assert events == ["catalog_preflight", "camera_open", "socket_frozen",
                      "catalog_rejected", "camera_close"]


def test_stale_catalog_preflight_does_not_connect_cameras(tmp_path, monkeypatch):
    events = []
    _patch_camera_lifetime(monkeypatch, events)
    catalog = tmp_path / "catalog.json"
    catalog.write_text("{}", encoding="utf-8")

    def stale(**kwargs):
        assert kwargs == {
            "mode": "square", "shared_root": Path("/shared"),
            "catalog_path": catalog,
        }
        raise ValueError("v8 catalog is stale for pose 000")

    monkeypatch.setattr(orchestration, "preflight_endpoint_catalog_file", stale)
    with pytest.raises(ValueError, match="stale for pose 000"):
        with orchestration.open_live_precision_session(
                launch_config_path=tmp_path / "launch.json",
                endpoint_catalog_path=catalog,
                run_output_dir=tmp_path / "run",
                max_xy_retries=1):
            pytest.fail("stale catalogue must not yield a live session")
    assert events == []
