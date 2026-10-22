"""AutoDex FR3 force-stop stroke and immediate reverse-path withdrawal.

The 20 mm axial path is planned elsewhere. This adapter uses the stock
Franka follower's 12 N wrist-Z contact stop and configured held speed cap.
It never labels physical key depth or visual task success.
"""

from __future__ import annotations

import math
import time

import numpy as np

from .franka_completion_checked import is_demo_completion_checked_executor


STOCK_PLACE_FORCE_N = 12.0


def reverse_executed_path(axial_arm: np.ndarray, measured_arm: np.ndarray,
                          *, max_path_error_rad: float = .05) -> np.ndarray:
    """Join measured contact position to the saved stroke, then ascend it."""
    path = np.asarray(axial_arm, dtype=np.float64)
    measured = np.asarray(measured_arm, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 or
            measured.shape != (7,) or not np.isfinite(path).all() or
            not np.isfinite(measured).all() or
            not math.isfinite(max_path_error_rad) or max_path_error_rad <= 0):
        raise ValueError("reverse insertion needs finite saved and measured joints")
    segment = path[1:] - path[:-1]
    denom = np.sum(segment * segment, axis=1)
    fraction = np.divide(
        np.sum((measured - path[:-1]) * segment, axis=1), denom,
        out=np.zeros(len(segment)), where=denom > 0)
    fraction = np.clip(fraction, 0.0, 1.0)
    projections = path[:-1] + fraction[:, None] * segment
    errors = np.linalg.norm(projections - measured, axis=1)
    index = int(np.argmin(errors))
    if errors[index] > max_path_error_rad:
        raise RuntimeError("contact stop is off the planned axial path")
    # The first command starts at measured feedback. No endpoint landing or
    # full-stroke reversal can jump to an unvisited joint waypoint.
    return np.vstack((measured, projections[index], path[index::-1]))


class FrankaForceStopInsertionAdapter:
    """Reuse the session's completion-checked stock follower and speed profile."""

    def __init__(self, executor):
        if not is_demo_completion_checked_executor(executor):
            raise TypeError("force-stop insertion needs the session FR3 executor")
        self.executor = executor

    def stop_and_acknowledge(self) -> dict:
        return self.executor.stop_and_acknowledge()

    def follow_stock_force_insertion(self, axial: np.ndarray, *,
                                     force_threshold_n: float = STOCK_PLACE_FORCE_N) -> dict:
        path = np.asarray(axial, dtype=np.float64)
        if (path.ndim != 2 or path.shape[1] != 13 or len(path) < 2 or
                not np.isfinite(path).all() or
                not math.isfinite(force_threshold_n) or force_threshold_n <= 0):
            raise ValueError("stock insertion needs a finite full-joint axial path")
        started = time.time()
        # Deliberately omit speed: stock traj_speed and the executor's
        # configured held-object cap apply. The follower stops on contact.
        self.executor._follow(path[:, :7], stop_wrench_z=force_threshold_n,
                              abort_on_contact=True)
        contact = bool(self.executor._last_stop_on_contact)
        # The required completion-checked _follow already stops the velocity
        # stream and acknowledges stationary feedback, including force stops.
        # A second stop here would delay the reverse ascent after contact.
        retreat = None
        contact_state = None
        if contact:
            contact_state = self.executor.arm.get_data()
            if not isinstance(contact_state, dict):
                raise RuntimeError("contact stop lacks FR3 feedback")
            retreat = reverse_executed_path(
                path[:, :7], np.asarray(contact_state.get("qpos")))
            # Reverse the already checked axis stroke immediately, while the
            # key remains gripped. A reflex/error or failed ascent propagates.
            self.executor._follow(retreat, abort_on_contact=True)
        return {
            "schema": "precision_insertion_stock_force_result_v1",
            "started_at_s": started, "completed_at_s": time.time(),
            "safety_abort": contact,
            "retreat_completed": contact and retreat is not None,
            "force_threshold_n": force_threshold_n,
            "contact_wrench": (None if contact_state is None else
                               np.asarray(contact_state["wrench"],
                                          dtype=np.float64).tolist()),
            "terminal_hold_acknowledged": True,
            "speed_source": "stock_franka_traj_speed",
        }
