"""Own the stock AutoDex camera stream for one precision session.

Snapshot and InitOrchestrator request images from the camera daemons but do
not arm their acquisition stream. The stock run_auto path arms the capture
PCs at 10 FPS before perception. This context does the same and releases its
controller on every exit; it never opens a robot controller.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
import time
from typing import Callable, Iterator


def _wait_for_stream_frames(capture_ips: tuple[str, ...], timeout_s: float) -> None:
    """Spend the existing warmup budget only until capture4 has real frames.

    The older capture4 PUB exposes per-camera frame IDs under ``cameras``.
    If it is absent, this simply consumes the old warmup duration; it adds no
    new admission criterion or exception to the camera session.
    """
    import zmq

    context = zmq.Context()
    poller = zmq.Poller()
    sockets = {}
    try:
        for ip in capture_ips:
            sock = context.socket(zmq.SUB)
            sock.setsockopt(zmq.SUBSCRIBE, b"")
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(f"tcp://{ip}:5481")
            poller.register(sock, zmq.POLLIN)
            sockets[sock] = ip
        ready = set()
        deadline = time.monotonic() + timeout_s
        while len(ready) < len(capture_ips):
            remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
            if not remaining_ms:
                break
            for sock, _ in poller.poll(min(remaining_ms, 100)):
                cameras = sock.recv_json().get("cameras")
                if (isinstance(cameras, list) and cameras and
                        all(row.get("state") == "CAPTURING" and
                            isinstance(row.get("frame_id"), int) and
                            row["frame_id"] > 0
                            for row in cameras if isinstance(row, dict)) and
                        all(isinstance(row, dict) for row in cameras)):
                    ready.add(sockets[sock])
    finally:
        for sock in sockets:
            sock.close()
        context.term()


@contextmanager
def open_autodex_camera_stream(
    pc_list: tuple[str, ...], *, name: str,
    capture_ips: tuple[str, ...] | None = None,
    fps: int = 10, warmup_s: float = 2.0,
    acquisition_mode: str | None = None,
    controller_factory: Callable | None = None,
) -> Iterator[object]:
    if (not pc_list or not all(isinstance(pc, str) and pc for pc in pc_list)
            or not isinstance(name, str) or not name
            or type(fps) is not int or fps < 1
            or type(warmup_s) not in (int, float) or warmup_s < 0):
        raise ValueError("camera stream needs PCs, a name and valid stock timing")
    controller_module = None
    original_get_pc_ip = None
    if capture_ips is not None:
        if len(capture_ips) != len(pc_list):
            raise ValueError("capture IPs must match the PC list")
        # ParaDex's global system/current can still describe the AutoDex rig.
        # Override only this process's controller lookup for this session.
        controller_module = importlib.import_module(
            "paradex.io.camera_system.remote_camera_controller")
        original_get_pc_ip = controller_module.get_pc_ip
        addresses = dict(zip(pc_list, capture_ips))
        controller_module.get_pc_ip = lambda pc: (
            addresses[pc] if pc in addresses else original_get_pc_ip(pc))
    try:
        if controller_factory is None:
            if controller_module is None:
                controller_module = importlib.import_module(
                    "paradex.io.camera_system.remote_camera_controller")
            controller_factory = controller_module.remote_camera_controller
        controller = controller_factory(name, pc_list=list(pc_list),
                                        stall_timeout=15.0)
    except BaseException:
        if original_get_pc_ip is not None:
            controller_module.get_pc_ip = original_get_pc_ip
        raise
    arm_requested = False
    try:
        arm_requested = True
        acquisition_mode = (acquisition_mode if acquisition_mode is not None
                            else os.environ.get(
                                "PRECISION_CAMERA_ACQUISITION_MODE", "acquire"))
        if acquisition_mode == "stream":
            # capture4's ParaDex camera server predates the decoupled sink
            # API: arm("acquire") enters single-shot mode there. Its native
            # "stream" mode continuously fills the same SHM readers.
            controller.start("stream", syncMode=False, fps=fps)
        elif acquisition_mode == "acquire":
            controller.arm(syncMode=False, fps=fps)
            controller.set_stream(True)
        else:
            raise ValueError("unknown camera acquisition mode")
        if warmup_s:
            if acquisition_mode == "stream" and capture_ips is not None:
                _wait_for_stream_frames(capture_ips, warmup_s)
            else:
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
            try:
                controller.end()
            finally:
                if original_get_pc_ip is not None:
                    controller_module.get_pc_ip = original_get_pc_ip


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
