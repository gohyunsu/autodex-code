#!/usr/bin/env python3
"""Opt-in ParaDex camera daemon with same-frame chunk-timestamp journals.

Use only on a capture PC after stopping the stock camera daemon. Camera image
transport remains ParaDex's unchanged implementation. The journal records
raw camera ticks, not calibrated UTC exposure times; it cannot yet start a
precision-insertion session or authorize robot motion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import secrets
import signal
import sys
import time


DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from precision_insertion.camera_chunk_tap import (  # noqa: E402
    ChunkTimestampJournal, TimestampedCameraPointer,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal-dir", type=Path, required=True,
                        help="new, empty per-run directory on durable storage")
    args = parser.parse_args(argv)
    root_input = args.journal_dir.expanduser()
    if not root_input.is_absolute() or root_input.exists():
        parser.error("journal directory must be a new absolute per-run path")
    root = root_input.resolve()
    root.mkdir(parents=True, exist_ok=False)

    # Import after argument validation so --help works without camera PySpin.
    from paradex.io.camera_system import camera_loader as loader_module
    from paradex.io.camera_system import camera_server_daemon as server_module
    from paradex.io.camera_system.camera import Camera

    class PrecisionCamera(Camera):
        def connect_camera(self):
            super().connect_camera()
            if not hasattr(self, "camera"):
                return  # stock Camera already recorded the connection error
            try:
                def new_journal():
                    path = (root / f"{self.name}_{int(time.time())}_"
                            f"{secrets.token_hex(4)}.jsonl")
                    return ChunkTimestampJournal(path, camera_serial=self.name)

                self._precision_chunk_pointer = TimestampedCameraPointer(
                    self.camera.cam, journal_factory=new_journal)
                self.camera.cam = self._precision_chunk_pointer
            except BaseException as exc:
                self.last_error = f"chunk timestamp hook failed: {exc}"
                self.event["error"].set()
                self.event["error_reset"].clear()
                raise

        def release(self):
            try:
                super().release()
            finally:
                pointer = getattr(self, "_precision_chunk_pointer", None)
                if pointer is not None:
                    pointer.close_journal()

    # CameraLoader uses its module-level Camera symbol for every initial load
    # and reload. This patch is process-local; no ParaDex source is edited.
    loader_module.Camera = PrecisionCamera
    server = server_module.camera_server_daemon()
    stopped = False

    def stop_once():
        nonlocal stopped
        if not stopped:
            stopped = True
            server.shutdown()

    def shutdown(_signum, _frame):
        stop_once()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        while True:
            time.sleep(1)
    finally:
        stop_once()


if __name__ == "__main__":
    raise SystemExit(main())
