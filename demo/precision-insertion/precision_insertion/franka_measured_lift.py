"""Measured, opt-in FR3/Inspire adapter for non-contact held motion.

This reuses the demo's completion-checked stock follower. Every follower
state read also checks fresh FR3 wrench/joints and measured Inspire motors;
the original controller samples, not a planned trajectory, form its trace.
It does not implement a robot-side command-expiry watchdog. The existing
lift boundary still requires a separately reviewed daemon commissioning
record before this adapter can receive a real motion command.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import threading
import time
from uuid import uuid4

import numpy as np

from .franka_completion_checked import is_demo_completion_checked_executor


@dataclass(frozen=True)
class MeasuredLiftMonitorLimits:
    max_force_norm_n: float
    max_torque_norm_nm: float
    max_hand_tracking_error_raw: float
    max_state_gap_s: float
    minimum_samples: int = 2

    def validate(self) -> None:
        values = (self.max_force_norm_n, self.max_torque_norm_nm,
                  self.max_hand_tracking_error_raw, self.max_state_gap_s)
        if (any(type(value) not in (int, float) or
                not math.isfinite(value) or value <= 0 for value in values) or
                type(self.minimum_samples) is not int or
                self.minimum_samples < 2):
            raise ValueError("lift monitor needs finite limits and two samples")


def _vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise RuntimeError(f"{name} lacks finite measured values")
    return array


class _ObservedArm:
    """Intercept only the follower thread; camera feedback reads pass through."""

    def __init__(self, arm, hand, *, expected_hand_raw: np.ndarray,
                 limits: MeasuredLiftMonitorLimits, deadline: float,
                 stage: str):
        self.arm = arm
        self.hand = hand
        self.expected_hand_raw = expected_hand_raw
        self.limits = limits
        self.deadline = deadline
        self.stage = stage
        self.owner_ident = threading.get_ident()
        self.samples: list[dict] = []
        self._last_uptime: float | None = None
        self._last_advance_monotonic = time.monotonic()

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def get_data(self):
        state = self.arm.get_data()
        if threading.get_ident() != self.owner_ident:
            return state
        now_monotonic = time.monotonic()
        if now_monotonic > self.deadline:
            raise TimeoutError(
                f"measured FR3 {self.stage} exceeded its motion deadline")
        if not isinstance(state, dict):
            raise RuntimeError(f"FR3 {self.stage} state stream is absent")
        qpos = _vector(state.get("qpos"), 7, "FR3 joints")
        qvel = _vector(state.get("qvel"), 7, "FR3 velocity")
        wrench = _vector(state.get("wrench"), 6, "FR3 wrench")
        uptime = float(state.get("time", float("nan")))
        if not math.isfinite(uptime) or uptime <= 0:
            raise RuntimeError(f"FR3 {self.stage} has no controller timestamp")
        if self._last_uptime is None or uptime > self._last_uptime:
            self._last_uptime = uptime
            self._last_advance_monotonic = now_monotonic
        elif (uptime < self._last_uptime or
              now_monotonic - self._last_advance_monotonic >
              self.limits.max_state_gap_s):
            raise RuntimeError(
                f"FR3 {self.stage} state stream stopped advancing")
        hand_state = self.hand.get_data()
        if not isinstance(hand_state, dict):
            raise RuntimeError(f"Inspire {self.stage} state stream is absent")
        measured = _vector(hand_state.get("joint_value"), 6,
                           "measured Inspire joints")
        commanded = _vector(hand_state.get("action"), 6,
                            "commanded Inspire joints")
        hand_time = float(hand_state.get("time", float("nan")))
        if (not math.isfinite(hand_time) or hand_time <= 0 or
                np.any(measured < 0) or np.any(measured > 1000) or
                np.any(commanded < 0) or np.any(commanded > 1000)):
            raise RuntimeError(f"Inspire {self.stage} feedback is invalid")
        force = float(np.linalg.norm(wrench[:3]))
        torque = float(np.linalg.norm(wrench[3:]))
        hand_error = float(np.max(np.abs(
            measured - self.expected_hand_raw)))
        self.samples.append({
            "receipt_time_s": time.time(), "robot_uptime_s": uptime,
            "hand_receipt_time_s": hand_time,
            "arm_q": qpos.tolist(), "arm_qvel": qvel.tolist(),
            "wrench": wrench.tolist(),
            "hand_raw_measured": measured.tolist(),
            "hand_raw_commanded": commanded.tolist(),
            "force_norm_n": force, "torque_norm_nm": torque,
            "hand_tracking_error_raw": hand_error,
        })
        if (force > self.limits.max_force_norm_n or
                torque > self.limits.max_torque_norm_nm):
            raise RuntimeError(
                f"measured FR3 {self.stage} force limit exceeded")
        if (hand_error > self.limits.max_hand_tracking_error_raw or
                np.max(np.abs(commanded - self.expected_hand_raw)) >
                self.limits.max_hand_tracking_error_raw):
            raise RuntimeError(f"measured Inspire {self.stage} hold changed")
        if bool(self.arm.is_error()):
            raise RuntimeError(f"FR3 reflex/error during held {self.stage}")
        return state


class FrankaMeasuredLiftAdapter:
    """Supply lift, transfer and lateral boundaries from measured samples.

    The caller must pass the *same* completion-checked executor owned by the
    live session and an exact robot daemon binary path. A commissioning
    record for that binary remains the separate boundary's responsibility.
    """

    def __init__(self, *, executor, daemon_binary_path: Path,
                 trace_root: Path, limits: MeasuredLiftMonitorLimits):
        if not is_demo_completion_checked_executor(executor):
            raise TypeError("lift needs the session's checked FR3 executor")
        if not isinstance(limits, MeasuredLiftMonitorLimits):
            raise TypeError("lift monitor needs explicit measured limits")
        limits.validate()
        self.executor = executor
        self.daemon_binary_path = Path(daemon_binary_path).expanduser().resolve()
        self.trace_root = Path(trace_root).expanduser().resolve()
        self.limits = limits
        self._active = False

    def stop_and_acknowledge(self) -> dict:
        return self.executor.stop_and_acknowledge()

    def follow_lift(
        self, trajectory, *, max_duration_s: float,
        expected_hand_raw, trajectory_sha256: str,
    ) -> dict:
        trace_path, started, completed = self._follow_measured(
            "lift", trajectory, max_duration_s=max_duration_s,
            expected_hand_raw=expected_hand_raw,
            trajectory_sha256=trajectory_sha256)
        return {
            "schema": "precision_insertion_external_lift_result_v1",
            "trajectory_complete": True, "force_abort": False,
            "terminal_hold_acknowledged": True,
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": started, "completed_at_s": completed,
            "controller_trace_path": str(trace_path),
            "controller_trace_sha256": hashlib.sha256(
                trace_path.read_bytes()).hexdigest(),
        }

    def follow_transfer(
        self, trajectory, *, max_duration_s: float,
        expected_hand_raw, trajectory_sha256: str,
    ) -> dict:
        return self._follow_held_motion(
            "transfer", trajectory, max_duration_s=max_duration_s,
            expected_hand_raw=expected_hand_raw,
            trajectory_sha256=trajectory_sha256)

    def follow_lateral(
        self, trajectory, *, max_duration_s: float,
        expected_hand_raw, trajectory_sha256: str,
    ) -> dict:
        return self._follow_held_motion(
            "lateral", trajectory, max_duration_s=max_duration_s,
            expected_hand_raw=expected_hand_raw,
            trajectory_sha256=trajectory_sha256)

    def _follow_held_motion(
        self, stage: str, trajectory, *, max_duration_s: float,
        expected_hand_raw, trajectory_sha256: str,
    ) -> dict:
        trace_path, started, completed = self._follow_measured(
            stage, trajectory, max_duration_s=max_duration_s,
            expected_hand_raw=expected_hand_raw,
            trajectory_sha256=trajectory_sha256)
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        samples = trace["samples"]
        shared = {
            "schema": "precision_insertion_franka_held_motion_source_v1",
            "stage": stage, "trajectory_sha256": trajectory_sha256,
            "started_at_s": started, "completed_at_s": completed,
            "controller_trace_path": str(trace_path),
            "controller_trace_sha256": hashlib.sha256(
                trace_path.read_bytes()).hexdigest(),
        }
        fields = {
            "trajectory_feedback": (
                "robot_uptime_s", "receipt_time_s", "arm_q", "arm_qvel"),
            "safety": (
                "robot_uptime_s", "receipt_time_s", "wrench",
                "force_norm_n", "torque_norm_nm"),
            "grasp_state": (
                "robot_uptime_s", "hand_receipt_time_s",
                "hand_raw_measured", "hand_raw_commanded",
                "hand_tracking_error_raw"),
        }
        sources = {}
        for name, keys in fields.items():
            source_path = trace_path.parent / f"{name}.json"
            record = {
                **shared, "source": name,
                "samples": [{key: sample[key] for key in keys}
                            for sample in samples],
            }
            if name == "grasp_state":
                record["grasp_held_scope"] = (
                    "measured_inspire_motor_hold_only_not_key_presence")
                record["expected_hand_raw"] = _vector(
                    expected_hand_raw, 6, "expected Inspire hold").tolist()
            with source_path.open("x", encoding="utf-8") as stream:
                json.dump(record, stream, indent=2, allow_nan=False)
                stream.write("\n")
            sources[name] = {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            }
        return {
            "schema": f"precision_insertion_external_{stage}_result_v1",
            "trajectory_complete": True, "safety_abort": False,
            "grasp_held": True,
            "grasp_held_scope":
                "measured_inspire_motor_hold_only_not_key_presence",
            "terminal_hold_acknowledged": True,
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": started, "completed_at_s": completed,
            "source_records": sources,
        }

    def _follow_measured(
        self, stage: str, trajectory, *, max_duration_s: float,
        expected_hand_raw, trajectory_sha256: str,
    ) -> tuple[Path, float, float]:
        if stage not in {"lift", "transfer", "lateral"}:
            raise ValueError("unknown held-motion stage")
        path = np.asarray(trajectory, dtype=np.float64)
        hand = _vector(expected_hand_raw, 6, "expected Inspire hold")
        if (path.ndim != 2 or path.shape[1] != 13 or len(path) < 2 or
                not np.all(np.isfinite(path)) or
                not np.allclose(path[:, 7:], path[0, 7:],
                                atol=1e-8, rtol=0) or
                np.any(hand < 0) or np.any(hand > 1000) or
                type(max_duration_s) not in (int, float) or
                not math.isfinite(max_duration_s) or max_duration_s <= 0 or
                not isinstance(trajectory_sha256, str) or
                len(trajectory_sha256) != 64 or
                any(c not in "0123456789abcdef" for c in trajectory_sha256) or
                not bool(getattr(self.executor, "_holding_object", False)) or
                self._active):
            raise ValueError(
                f"{stage} needs one finite held path and active grasp")
        self.trace_root.mkdir(parents=True, exist_ok=True)
        trace_dir = self.trace_root / f"{stage}_{uuid4().hex}"
        trace_dir.mkdir(exist_ok=False)
        started = time.time()
        original_arm = self.executor.arm
        observer = _ObservedArm(
            original_arm, self.executor.hand, expected_hand_raw=hand,
            limits=self.limits, deadline=time.monotonic() + max_duration_s,
            stage=stage)
        self._active = True
        self.executor.arm = observer
        failure: Exception | None = None
        stop_ack: dict | None = None
        try:
            raw_hold = np.tile(hand, (len(path), 1))
            self.executor.follow_joint_trajectory(path, raw_hold)
            stop_ack = self.stop_and_acknowledge()
            if (not isinstance(stop_ack, dict) or
                    stop_ack.get("acknowledged") is not True):
                raise RuntimeError(
                    f"measured FR3 {stage} terminal stop was not acknowledged")
        except Exception as exc:
            failure = exc
        finally:
            self.executor.arm = original_arm
            self._active = False
            if failure is not None:
                try:
                    stop_ack = self.stop_and_acknowledge()
                except Exception as stop_exc:
                    stop_ack = {"acknowledged": False,
                                "error": str(stop_exc)}
        completed = time.time()
        if (failure is None and
                (len(observer.samples) < self.limits.minimum_samples or
                 observer.samples[-1]["robot_uptime_s"] <=
                 observer.samples[0]["robot_uptime_s"] or
                 completed - started > max_duration_s)):
            failure = RuntimeError(
                f"{stage} lacks advancing motion samples or timed out")
            if not isinstance(stop_ack, dict) or stop_ack.get("acknowledged") is not True:
                try:
                    stop_ack = self.stop_and_acknowledge()
                except Exception as stop_exc:
                    stop_ack = {"acknowledged": False, "error": str(stop_exc)}
        trace = {
            "schema": f"precision_insertion_franka_measured_{stage}_trace_v1",
            "trajectory_sha256": trajectory_sha256,
            "started_at_s": started, "completed_at_s": completed,
            "status": "completed" if failure is None else "aborted",
            "limits": vars(self.limits), "samples": observer.samples,
            "stop_acknowledgement": stop_ack,
            "error": None if failure is None else {
                "type": type(failure).__name__, "message": str(failure)},
        }
        trace_path = trace_dir / "controller_trace.json"
        with trace_path.open("x", encoding="utf-8") as stream:
            json.dump(trace, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if failure is not None:
            raise RuntimeError(
                f"measured FR3 {stage} aborted; trace: {trace_path}") from failure
        return trace_path, started, completed
