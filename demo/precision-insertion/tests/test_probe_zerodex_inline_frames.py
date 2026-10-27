"""Camera-only transport diagnostics report illumination without gating it."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_zerodex_inline_frames import _frame_brightness  # noqa: E402


def test_frame_brightness_reports_dark_and_lit_without_rejecting():
    dark = np.full((8, 8, 3), 5, dtype=np.uint8)
    lit = np.full((8, 8, 3), 200, dtype=np.uint8)
    assert _frame_brightness(dark) == {"gray_mean": 5.0, "gray_p99": 5.0}
    assert _frame_brightness(lit) == {"gray_mean": 200.0, "gray_p99": 200.0}
