"""The isolated capture4 stack must be listening before its caller proceeds."""

from types import SimpleNamespace

import pytest

import manage_zerodex_capture4_services as services


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
