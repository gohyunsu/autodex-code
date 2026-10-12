"""The connection-only probe must use read paths, never controller setup."""

from pathlib import Path
import sys

import msgpack
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_robot_feedback_only as probe  # noqa: E402


def _state(uptime):
    return msgpack.packb({
        "type": "state_update", "timestamp": uptime,
        "qpos": [0.0] * 7, "qvel": [0.0] * 7,
        "wrench": [0.0] * 6,
    }, use_bin_type=True)


def test_franka_probe_subscribes_only_to_state_port(monkeypatch):
    class Socket:
        def __init__(self):
            self.payloads = iter((_state(10.0), _state(10.01)))
            self.endpoints = []

        def setsockopt(self, *_args):
            pass

        def connect(self, endpoint):
            self.endpoints.append(endpoint)

        def poll(self, _timeout):
            return True

        def recv(self):
            return next(self.payloads)

        def close(self):
            pass

    class Context:
        def __init__(self):
            self.sock = Socket()

        def socket(self, kind):
            assert kind == probe.zmq.SUB
            return self.sock

        def term(self):
            pass

    ctx = Context()
    monkeypatch.setattr(probe.zmq, "Context", lambda: ctx)
    report = probe.read_arm_feedback(host="127.0.0.1")
    assert ctx.sock.endpoints == ["tcp://127.0.0.1:5556"]
    assert report["last"]["robot_uptime_s"] == 10.01
    assert report["max_abs_qvel_rad_s"] == 0


def test_inspire_probe_only_reads_actual_and_command_registers():
    events = []

    class Reply:
        registers = [500] * 6

        def isError(self):
            return False

    class Client:
        def __init__(self, **kwargs):
            events.append(("construct", kwargs))

        def connect(self):
            events.append(("connect",))
            return True

        def read_holding_registers(self, **kwargs):
            events.append(("read_holding_registers", kwargs))
            return Reply()

        def close(self):
            events.append(("close",))

        def write_registers(self, *_args, **_kwargs):
            pytest.fail("feedback probe attempted a hand write")

    report = probe.read_hand_feedback(
        host="192.168.11.210", port=6000, client_factory=Client)
    assert [event[1]["address"] for event in events
            if event[0] == "read_holding_registers"] == [1546, 1486]
    assert report["raw_measured"] == [500] * 6
    assert report["max_command_error_raw"] == 0
    assert events[-1] == ("close",)


def test_franka_probe_rejects_malformed_state():
    with pytest.raises(ValueError, match="FR3 qpos"):
        probe._arm_state(msgpack.packb({
            "type": "state_update", "timestamp": 1.0,
            "qpos": [0.0] * 6, "qvel": [0.0] * 7,
            "wrench": [0.0] * 6,
        }), 1.0)


def test_franka_probe_refuses_command_port():
    with pytest.raises(ValueError, match="feedback endpoint"):
        probe.read_arm_feedback(host="127.0.0.1", port=5555)
