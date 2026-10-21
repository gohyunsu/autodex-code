"""Own the stock AutoDex camera stream for one precision session.

Snapshot and InitOrchestrator request images from the camera daemons but do
not arm their acquisition stream. The stock run_auto path arms the capture
PCs at 10 FPS before perception. This context does the same and releases its
controller on every exit; it never opens a robot controller.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Callable, Iterator


@contextmanager
def open_autodex_camera_stream(
    pc_list: tuple[str, ...], *, name: str,
    fps: int = 10, warmup_s: float = 2.0,
    controller_factory: Callable | None = None,
) -> Iterator[object]:
    if (not pc_list or not all(isinstance(pc, str) and pc for pc in pc_list)
            or not isinstance(name, str) or not name
            or type(fps) is not int or fps < 1
            or type(warmup_s) not in (int, float) or warmup_s < 0):
        raise ValueError("camera stream needs PCs, a name and valid stock timing")
    if controller_factory is None:
        from paradex.io.camera_system.remote_camera_controller import (
            remote_camera_controller,
        )
        controller_factory = remote_camera_controller
    controller = controller_factory(name, pc_list=list(pc_list),
                                    stall_timeout=15.0)
    arm_requested = False
    try:
        arm_requested = True
        controller.arm(syncMode=False, fps=fps)
        controller.set_stream(True)
        if warmup_s:
            time.sleep(warmup_s)
        status = controller.get_status()
        rows = status.get("pc", {}) if isinstance(status, dict) else {}
        foreign = {
            pc: row.get("controller") for pc, row in rows.items()
            if isinstance(row, dict) and row.get("controller") not in
            (None, "", "None", controller.name)
        }
        if foreign:
            raise RuntimeError(f"AutoDex cameras belong to another controller: {foreign}")
        yield controller
    finally:
        try:
            if arm_requested:
                controller.stop()
        finally:
            controller.end()


@contextmanager
def attach_running_autodex_camera_stream(
    pc_list: tuple[str, ...], capture_ips: tuple[str, ...], *,
    timeout_s: float = 3.0,
) -> Iterator[dict]:
    """Read an existing rig stream without registering or stopping its owner.

    Snapshot and init daemons consume the capture PCs' shared-memory frames.
    This mode only reads the camera health PUB; their current controller keeps
    acquisition and its heartbeat for the entire precision session.
    """
    if (not pc_list or len(pc_list) != len(capture_ips) or
            len(set(pc_list)) != len(pc_list) or
            not 0 < timeout_s <= 30):
        raise ValueError("attached camera stream needs aligned PCs and IPs")
    import zmq

    context = zmq.Context()
    poller = zmq.Poller()
    sockets = {}
    try:
        for pc, ip in zip(pc_list, capture_ips):
            sock = context.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{ip}:5481")
            poller.register(sock, zmq.POLLIN)
            sockets[sock] = pc
        statuses = {}
        deadline = time.monotonic() + timeout_s
        while len(statuses) < len(pc_list) and time.monotonic() < deadline:
            for sock, _ in poller.poll(100):
                statuses[sockets[sock]] = sock.recv_json()
        missing = set(pc_list) - set(statuses)
        if missing:
            raise RuntimeError(f"camera health unavailable on {sorted(missing)}")
        for pc, status in statuses.items():
            summary = status.get("summary", {})
            states = summary.get("states", {})
            if (status.get("running") is not True or
                    status.get("controller") in (None, "", "None") or
                    summary.get("errors") or
                    summary.get("detected_camera_count") !=
                    summary.get("expected_camera_count") or
                    not states or
                    any(state != "CAPTURING" for state in states.values())):
                raise RuntimeError(f"existing camera stream is not ready on {pc}")
    finally:
        for sock in sockets:
            sock.close()
        context.term()
    yield statuses
