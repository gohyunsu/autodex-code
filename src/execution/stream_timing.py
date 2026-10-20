"""Time-based parameters for the FR3 velocity stream."""

from __future__ import annotations

import math


def velocity_retention(per_20ms_retention: float, elapsed_s: float) -> float:
    """Keep the velocity filter's physical response when the tick rate changes."""
    return per_20ms_retention ** (elapsed_s / 0.02)


def stream_command_validity_ms(control_period_s: float) -> int:
    """Keep at least the stock 40 ms daemon margin at faster command rates."""
    return max(40, math.ceil(2000.0 * control_period_s))
