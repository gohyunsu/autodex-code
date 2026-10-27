"""The isolated capture4 stack must be listening before its caller proceeds."""

from types import SimpleNamespace

import pytest

import manage_zerodex_capture4_services as services
from capture4_exposure_daemon import apply_exposure_override


def test_listening_ports_reads_ss_local_address(monkeypatch):
    output = ("LISTEN 0 100 0.0.0.0:5480 0.0.0.0:*\n"
              "LISTEN 0 100 [::]:15007 [::]:*\n"
              "LISTEN 0 100 0.0.0.0:22 0.0.0.0:*\n")
    monkeypatch.setattr(services.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=output))
    assert services._listening_ports() == {22, 5480, 15007}


def test_check_reports_missing_port(monkeypatch):
    monkeypatch.setattr(services, "_listening_ports",
                        lambda: services.READY_PORTS - {15007})
    with pytest.raises(RuntimeError, match="15007"):
        services.check()


def test_foundpose_attention_fallback_is_capture4_only():
    assert "XFORMERS_DISABLED=1" in services._command("precision")
    assert "XFORMERS_DISABLED" not in services._command("snapshot")
    assert "XFORMERS_DISABLED" not in services._command("camera")
    assert "HF_HUB_OFFLINE=1" in services._command("precision")
    assert "HF_HUB_OFFLINE" not in services._command("snapshot")
    assert "HF_HUB_OFFLINE" not in services._command("camera")


def test_start_waits_until_all_ports_listen(monkeypatch):
    commands = []
    observations = iter((set(), services.READY_PORTS, services.READY_PORTS))
    monkeypatch.setattr(services, "deploy", lambda: None)
    monkeypatch.setattr(services, "_ssh", commands.append)
    monkeypatch.setattr(services, "_listening_ports",
                        lambda: next(observations))
    monkeypatch.setattr(services.time, "sleep", lambda *_: None)
    services.start()
    assert len(commands) == 3


def test_start_reuses_ready_services_without_redeploy(monkeypatch):
    probes = []
    monkeypatch.setattr(services, "_listening_ports",
                        lambda: probes.append(True) or services.READY_PORTS)
    monkeypatch.setattr(services, "deploy", lambda: pytest.fail(
        "warm services must not be redeployed"))
    monkeypatch.setattr(services, "_ssh", lambda _cmd: pytest.fail(
        "warm services must not be relaunched"))
    services.start()
    assert len(probes) == 1


def test_capture4_exposure_override_is_process_local():
    original = {"cam1": {"gain": 16, "exposure": 2500},
                "cam2": {"gain": 25, "exposure": 2500}}
    adjusted = apply_exposure_override(original, "20000")
    assert original["cam1"]["exposure"] == 2500
    assert adjusted == {"cam1": {"gain": 16, "exposure": 20000.0},
                        "cam2": {"gain": 25, "exposure": 20000.0}}
    assert apply_exposure_override(original, None) is original
    with pytest.raises(ValueError, match="finite positive"):
        apply_exposure_override(original, "nan")


def test_service_start_passes_exposure_only_to_own_camera_daemon(monkeypatch):
    commands = []
    observations = iter((set(), services.READY_PORTS, services.READY_PORTS))
    monkeypatch.setattr(services, "deploy", lambda: None)
    monkeypatch.setattr(services, "_ssh", commands.append)
    monkeypatch.setattr(services, "_listening_ports", lambda: next(observations))
    monkeypatch.setattr(services.time, "sleep", lambda *_: None)
    services.start(exposure_us=20000)
    assert "PRECISION_CAPTURE_EXPOSURE_US=20000" in commands[0]
    assert "capture4_exposure_daemon.py" in commands[0]
    assert all("PRECISION_CAPTURE_EXPOSURE_US" not in command
               for command in commands[1:])


def test_stop_accepts_earlier_namespaced_camera_command(monkeypatch):
    commands = []
    monkeypatch.setattr(services, "_ssh", commands.append)
    services.stop()
    assert "*capture4_exposure_daemon.py*|*src/camera/server_daemon.py*" in commands[-1]
