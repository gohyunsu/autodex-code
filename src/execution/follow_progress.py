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
    def distance_at(position: float) -> float:
        waypoint = int(position)
        fraction = position - waypoint
        reference = (path[waypoint] if waypoint >= len(path) - 1 else
                     path[waypoint] * (1.0 - fraction) +
                     path[waypoint + 1] * fraction)
        return float(np.linalg.norm(reference - measured_q))

    proposed = min(index + step, float(len(path) - 1))
    if distance_at(proposed) <= max_lead:
        return proposed
    if distance_at(index) <= max_lead:
        # Follow the boundary continuously as the arm catches up.  Holding the
        # reference until the whole next step fits creates a visible stop-go
        # cycle even though a smaller safe advance is available.
        low, high = index, proposed
        for _ in range(12):
            middle = (low + high) * 0.5
            if distance_at(middle) <= max_lead:
                low = middle
            else:
                high = middle
        return low
    start = int(index) + 1
    stop = min(len(path), start + lookahead_points)
    if start < stop and np.min(
        np.linalg.norm(path[start:stop] - measured_q, axis=1)
    ) <= max_lead:
        return min(proposed, float(start))
    return index


def measured_path_index(
    path: np.ndarray, measured_q: np.ndarray, previous: int,
    reference_index: float,
) -> int:
    """Project measured joints onto the path traversed by the reference.

    The fingers follow actual arm progress.  Checking only the next waypoint
    within a fixed tolerance can leave them at an early waypoint forever once
    the arm has passed it with a small tracking error.
    """
    end = min(len(path) - 1, max(previous, int(reference_index)))
    if end <= previous:
        return previous
    distances = np.linalg.norm(
        path[previous:end + 1] - measured_q, axis=1)
    return previous + int(np.argmin(distances))
