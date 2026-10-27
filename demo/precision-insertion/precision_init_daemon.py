#!/usr/bin/env python3
"""Capture-PC FoundPose daemon retaining exact sensor frame ID and image hash.

Run on dedicated mask, pose and command ports alongside the unchanged stock
init daemon, or on the stock ports while that daemon is stopped. This reuses
the installed AutoDex SAM3/FoundPose pipeline and adds frame provenance
metadata without changing stock code.
It does not supply or invent hardware exposure timestamps.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import cv2

_stock_root_env = os.environ.get("AUTODEX_STOCK_ROOT")
REPO_ROOT = (Path(_stock_root_env).expanduser().resolve()
             if _stock_root_env else Path(__file__).resolve().parents[2])
if not (REPO_ROOT / "src/execution/daemon/init_daemon.py").is_file():
    raise FileNotFoundError(
        f"AutoDex stock init daemon is missing under {REPO_ROOT}; "
        "set AUTODEX_STOCK_ROOT to the capture PC's current AutoDex checkout")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.execution.daemon.init_daemon import InitDaemon  # noqa: E402
from precision_insertion.camera_transport import (  # noqa: E402
    RecordingReader, install_init_provenance,
)
from precision_insertion.sam3_batch import segment_batch  # noqa: E402


class _MaskOnlyEstimator:
    """Let the stock loop publish a frame-bound `ok=False` pose promptly."""

    def __init__(self, obj_name: str):
        self.obj_name = obj_name

    def estimate_one_view(self, **_kwargs):
        return None


class PrecisionInitDaemon(InitDaemon):
    """Stock inference with a request-bound SHM frame provenance wrapper."""

    def __init__(self, port_mask: int, port_pose: int, port_cmd: int):
        super().__init__(port_mask, port_pose, port_cmd)
        install_init_provenance(self)

    def _do_init(self) -> None:
        info = self.cmd_receiver.event_info.get("init", {}) or {}
        capture_asset_root = os.environ.get("PRECISION_CAPTURE_ASSET_ROOT")
        if capture_asset_root:
            for field in ("mesh_path", "assets_root"):
                source = info.get(field)
                if isinstance(source, str) and source.startswith("~/shared_data/"):
                    info[field] = str(Path(capture_asset_root) /
                                      source[len("~/shared_data/"):])
        if info.get("skip_foundpose") is True:
            # Stock initialization already skips a loaded object-specific
            # estimator. A named no-op keeps its camera/undistortion setup
            # unchanged without constructing unused socket FoundPose weights.
            self.fp = _MaskOnlyEstimator(info["obj_name"])
        super()._do_init()
        if self.mode != "live" or self.reader is None:
            raise ValueError("precision FoundPose requires live camera mode")
        if not isinstance(self.reader, RecordingReader):
            self.reader = RecordingReader(self.reader, self)

    def _do_run(self) -> None:
        info = self.cmd_receiver.event_info.get("run", {}) or {}
        source = info.get("save_capture_dir")
        if isinstance(source, str) and source.startswith("/mnt/paradex2/"):
            info["save_capture_dir"] = str(
                Path("/home/capture4/shared_data") /
                source[len("/mnt/paradex2/"):])
        request_id = info.get("request_id")
        if type(request_id) is not int or request_id <= 0:
            raise ValueError("precision FoundPose run needs request ID")
        if self._precision_active_request_id is not None:
            raise RuntimeError("overlapping precision FoundPose runs")
        self._precision_active_request_id = request_id
        self._precision_send_source_frames = info.get("return_source_frames") is True
        original_estimator = self.fp
        try:
            if info.get("skip_foundpose") is True:
                self.fp = _MaskOnlyEstimator(getattr(self.fp, "obj_name", ""))
            super()._do_run()
        finally:
            self.fp = original_estimator
            self._precision_active_request_id = None
            self._precision_send_source_frames = False
            self._precision_undistorted.clear()

    def _undistort_live_image(self, serial, _image_bgr):
        # RecordingReader has already undistorted these exact pixels to bind
        # the published image hash and optional source PNG to this frame ID.
        return self._precision_undistorted[serial]

    def _segment_snapshot_batch(self, frames, serials, prompt):
        batch_mode = os.environ.get("PRECISION_SAM3_BATCH", "auto")
        if batch_mode == "0" or (batch_mode == "auto" and
                                  self._precision_send_source_frames):
            return None
        valid = [serial for serial in serials
                 if serial in self.K_undist and serial in frames and
                 frames[serial][0] is not None]
        if not valid:
            return {}
        images = [cv2.cvtColor(self._precision_undistorted[serial],
                              cv2.COLOR_BGR2RGB) for serial in valid]
        started = time.perf_counter()
        masks = segment_batch(self.sam3, images, prompt)
        elapsed = time.perf_counter() - started
        logger = logging.getLogger("precision_init_daemon")
        logger.info("[run] SAM3 %d-view batch %.3fs", len(valid), elapsed)
        # The stock per-camera timing field is retained as amortized GPU time;
        # the batch wall time above is the latency measurement.
        return {serial: (mask, elapsed / len(valid))
                for serial, mask in zip(valid, masks)}

    def close(self) -> None:
        self._precision_frame_encoder.shutdown(wait=True)
        super().close()


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
