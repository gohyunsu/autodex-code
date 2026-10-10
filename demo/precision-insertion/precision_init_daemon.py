#!/usr/bin/env python3
"""Capture-PC FoundPose daemon retaining exact sensor frame ID and image hash.

Deploy *instead of* the stock init daemon on the selected AutoDex capture PCs,
on the same ports. Do not run both simultaneously. This reuses the unchanged
AutoDex SAM3/FoundPose pipeline and adds only frame provenance metadata.
It does not supply or invent hardware exposure timestamps.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution.daemon.init_daemon import InitDaemon  # noqa: E402
from precision_insertion.camera_transport import (  # noqa: E402
    RecordingReader, install_init_provenance,
)


class PrecisionInitDaemon(InitDaemon):
    """Stock inference with a request-bound SHM frame provenance wrapper."""

    def __init__(self, port_mask: int, port_pose: int, port_cmd: int):
        super().__init__(port_mask, port_pose, port_cmd)
        install_init_provenance(self)

    def _do_init(self) -> None:
        super()._do_init()
        if self.mode != "live" or self.reader is None:
            raise ValueError("precision FoundPose requires live camera mode")
        if not isinstance(self.reader, RecordingReader):
            self.reader = RecordingReader(self.reader, self)

    def _do_run(self) -> None:
        info = self.cmd_receiver.event_info.get("run", {}) or {}
        request_id = info.get("request_id")
        if type(request_id) is not int or request_id <= 0:
            raise ValueError("precision FoundPose run needs request ID")
        if self._precision_active_request_id is not None:
            raise RuntimeError("overlapping precision FoundPose runs")
        self._precision_active_request_id = request_id
        try:
            super()._do_run()
        finally:
            self._precision_active_request_id = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    args = parser.parse_args(argv)
    daemon = PrecisionInitDaemon(args.port_mask, args.port_pose, args.port_cmd)
    try:
        daemon.loop()
    except KeyboardInterrupt:
        pass
    finally:
        daemon.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
