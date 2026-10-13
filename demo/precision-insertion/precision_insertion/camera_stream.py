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
