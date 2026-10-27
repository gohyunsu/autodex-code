#!/usr/bin/env python3
"""Run capture4's stock camera daemon with an optional process-local exposure.

No ParaDex source or shared camera.json is edited. The stock daemon retains
its single-instance lock and owns the cameras for this process's lifetime.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import runpy


def stock_camera_daemon_path(paradex_package_file: str) -> Path:
    return Path(paradex_package_file).resolve().parents[1] / "src/camera/server_daemon.py"


def apply_exposure_override(cam_info: dict, value: str | None) -> dict:
    """Return a copy of camera metadata; never mutate the stock config dict."""
    if value is None:
        return cam_info
    exposure_us = float(value)
    if not math.isfinite(exposure_us) or exposure_us <= 0:
        raise ValueError("capture4 exposure must be finite positive microseconds")
    return {
        serial: {**settings, "exposure": exposure_us}
        for serial, settings in cam_info.items()
    }


def main() -> None:
    import paradex
    from paradex.io.camera_system import pyspin

    value = os.environ.get("PRECISION_CAPTURE_EXPOSURE_US")
    pyspin.cam_info = apply_exposure_override(pyspin.cam_info, value)
    if value is not None:
        print(f"[precision] per-session exposure: {float(value):g} us", flush=True)
    runpy.run_path(str(stock_camera_daemon_path(paradex.__file__)),
                   run_name="__main__")


if __name__ == "__main__":
    main()
