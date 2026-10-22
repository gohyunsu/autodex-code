"""The stock daemon route checks local process identity without robot I/O."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from precision_insertion import lift_execution, stock_daemon_identity as stock  # noqa: E402
from precision_insertion.franka_measured_lift import FrankaMeasuredLiftAdapter  # noqa: E402


def _fixture(tmp_path, monkeypatch, *, timeout_ms=200, duplicate=False):
    binary = tmp_path / "franka_daemon"
    binary.write_bytes(b"audited test daemon")
    monkeypatch.setattr(stock, "is_demo_completion_checked_executor",
                        lambda _executor: True)
    adapter = object.__new__(FrankaMeasuredLiftAdapter)
    adapter.daemon_binary_path = binary
    adapter.executor = SimpleNamespace(arm=SimpleNamespace(
        ip="127.0.0.1", command_port=5555, state_port=5556))
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid in ((123, 124) if duplicate else (123,)):
        process = proc / str(pid)
        process.mkdir()
        (process / "comm").write_text("franka_daemon\n", encoding="utf-8")
        args = ["./build/franka_daemon", "172.17.1.11",
                "--command_port", "5555", "--state_port", "5556"]
        if timeout_ms != 200:
            args.extend(("--stream_timeout_ms", str(timeout_ms)))
        (process / "cmdline").write_bytes(
            b"\0".join(part.encode("utf-8") for part in args) + b"\0")
        (process / "exe").symlink_to(binary)
    return adapter, proc


def test_audited_local_daemon_process_is_recorded_without_physical_test_claim(
        tmp_path, monkeypatch):
    adapter, proc = _fixture(tmp_path, monkeypatch)
    result = stock.verify_stock_daemon_identity(
        adapter=adapter, max_command_age_s=.2, proc_root=proc)
    assert result["daemon_pid"] == 123
    assert result["command_expiry_s"] == .2
    assert result["physical_watchdog_tested"] is False
    assert result["daemon_binary_sha256"] == hashlib.sha256(
        adapter.daemon_binary_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("defect", ["slow", "remote", "missing_binary",
                                     "duplicate", "wrong_port"])
def test_stock_identity_rejects_wrong_runtime(tmp_path, monkeypatch, defect):
    adapter, proc = _fixture(
        tmp_path, monkeypatch, timeout_ms=500 if defect == "slow" else 200,
        duplicate=defect == "duplicate")
    if defect == "remote":
        adapter.executor.arm.ip = "172.16.0.2"
    elif defect == "missing_binary":
        adapter.daemon_binary_path.unlink()
    elif defect == "wrong_port":
        adapter.executor.arm.command_port = 6000
    with pytest.raises(ValueError):
        stock.verify_stock_daemon_identity(
            adapter=adapter, max_command_age_s=.2, proc_root=proc)


def test_rebuilt_binary_records_its_hash_without_rejecting_runtime(tmp_path, monkeypatch):
    adapter, proc = _fixture(tmp_path, monkeypatch)
    adapter.daemon_binary_path.write_bytes(b"same daemon rebuilt")
    result = stock.verify_stock_daemon_identity(
        adapter=adapter, max_command_age_s=.2, proc_root=proc)
    assert result["daemon_binary_sha256"] == hashlib.sha256(
        b"same daemon rebuilt").hexdigest()


def test_only_first_party_adapter_uses_stock_identity(tmp_path, monkeypatch):
    adapter, _proc = _fixture(tmp_path, monkeypatch)
    limits = SimpleNamespace(max_watchdog_command_age_s=.2)
    expected = {"schema": "precision_insertion_stock_daemon_identity_v1"}
    monkeypatch.setattr(lift_execution, "verify_stock_daemon_identity",
                        lambda **_kwargs: expected)
    assert lift_execution._verify_noncontact_controller(
        adapter=adapter, commissioning_record_path=None,
        limits=limits, now_s=100.) is expected
    with pytest.raises(ValueError, match="reviewed watchdog record"):
        lift_execution._verify_noncontact_controller(
            adapter=SimpleNamespace(), commissioning_record_path=None,
            limits=limits, now_s=100.)
