"""Measured progress along a streamed joint path.

The follower may lag its time reference when a joint velocity is capped.  It
must also recover if the arm subsequently travels *ahead* of that reference;
otherwise a symmetric distance-to-reference clamp can freeze it forever.
"""

from __future__ import annotations

import numpy as np


def advance_reference_index(
    path: np.ndarray, measured_q: np.ndarray, index: float, step: float,
    max_lead: float, *, lookahead_points: int = 100,
) -> float:
    """Advance in time if the next target is near, or the arm is on later path.

    The second case advances only at the normal time rate.  It does not skip
    planned waypoints or permit an off-path measured state to move the target.
    """
    proposed = min(index + step, float(len(path) - 1))
    if np.linalg.norm(path[int(proposed)] - measured_q) <= max_lead:
        return proposed
    start = int(index) + 1
    stop = min(len(path), start + lookahead_points)
    if start < stop and np.min(
        np.linalg.norm(path[start:stop] - measured_q, axis=1)
    ) <= max_lead:
        return min(proposed, float(start))
    return index
