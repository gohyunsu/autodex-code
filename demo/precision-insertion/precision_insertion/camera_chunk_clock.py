"""Fit camera chunk ticks to UTC from independent per-frame references.

This is an offline calibration/replay boundary. It cannot establish that
external trigger timestamps are physically correct, and it does not provide
the live capture-PC → robot-PC tick transport required for session startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .camera_chunk_tap import load_chunk_journal


_SCHEMA = "precision_insertion_chunk_utc_calibration_v1"
_METHODS = frozenset({
    "independent_per_camera_trigger_metrology",
    "same_imaging_camera_ptp_utc_metrology",
})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive(value, name: str, *, allow_zero: bool = False) -> float:
    if (type(value) not in (int, float) or not math.isfinite(value) or
            (value < 0 if allow_zero else value <= 0)):
        raise ValueError(f"invalid {name}")
    return float(value)


def _references(value, name: str, ticks_by_id: dict[int, int]
                ) -> list[tuple[int, int, float, float]]:
    if not isinstance(value, list) or len(value) < 3:
        raise ValueError(f"{name} needs three independent frame references")
    rows = []
    for row in value:
        if (not isinstance(row, dict) or set(row) != {
                "frame_id", "exposure_utc_s", "max_error_s"} or
                type(row["frame_id"]) is not int or
                row["frame_id"] not in ticks_by_id):
            raise ValueError(f"{name} frame ID is absent from the chunk journal")
        utc = _positive(row["exposure_utc_s"], "UTC exposure reference")
        error = _positive(row["max_error_s"], "reference error",
                          allow_zero=True)
        rows.append((row["frame_id"], ticks_by_id[row["frame_id"]],
                     utc, error))
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"{name} repeats a frame ID")
    return rows


@dataclass(frozen=True)
class ChunkUTCClock:
    calibration_id: str
    camera_serial: str
    journal_path: Path
    journal_sha256: str
    calibration_path: Path
    calibration_sha256: str
    tick_frequency_hz: float
    reference_ticks: int
    reference_utc_s: float
    fitted_rate: float
    fitted_error_bound_s: float
    clock_offset_error_s: float
    drift_error_s_per_s: float
    max_total_error_s: float
    max_extrapolation_s: float
    valid_until_utc_s: float
    first_valid_tick: int
    last_valid_tick: int
    ticks_by_frame_id: dict[int, int]
    source_files: tuple[tuple[Path, str], ...]

    @classmethod
    def load(cls, path: Path) -> "ChunkUTCClock":
        calibration_path = Path(path).expanduser().resolve()
        raw = calibration_path.read_bytes()
        record = json.loads(raw)
        if (not isinstance(record, dict) or record.get("schema") != _SCHEMA or
                not isinstance(record.get("calibration_id"), str) or
                not record["calibration_id"] or
                not isinstance(record.get("camera_serial"), str) or
                not record["camera_serial"] or
                record.get("source_method") not in _METHODS):
            raise ValueError("invalid independent camera chunk clock calibration")
        journal_ref = record.get("chunk_journal")
        if (not isinstance(journal_ref, dict) or
                set(journal_ref) != {"path", "sha256"} or
                not isinstance(journal_ref["path"], str) or
                not Path(journal_ref["path"]).is_absolute()):
            raise ValueError("chunk calibration needs an absolute journal reference")
        journal = Path(journal_ref["path"]).resolve()
        audit, ticks_by_id = load_chunk_journal(
            journal, camera_serial=record["camera_serial"])
        if not ticks_by_id or audit["sha256"] != journal_ref["sha256"]:
            raise ValueError("chunk journal changed or contains no frames")
        source_files = record.get("source_files")
        if not isinstance(source_files, list) or not source_files:
            raise ValueError("UTC references need independent raw source files")
        source_paths = set()
        source_hashes = []
        for row in source_files:
            if (not isinstance(row, dict) or set(row) != {"path", "sha256"} or
                    not isinstance(row["path"], str) or
                    not Path(row["path"]).is_absolute()):
                raise ValueError("invalid independent UTC source reference")
            source = Path(row["path"]).resolve()
            if (not source.is_file() or _sha(source) != row["sha256"] or
                    source == journal or row["sha256"] == audit["sha256"] or
                    source in source_paths):
                raise ValueError("independent UTC source changed or repeats")
            source_paths.add(source)
            source_hashes.append((source, row["sha256"]))
        freq = _positive(record.get("tick_frequency_hz"), "tick frequency")
        max_rate_ppm = _positive(record.get("max_clock_rate_error_ppm"),
                                 "camera clock rate error")
        max_total = _positive(record.get("max_total_error_s"),
                              "maximum UTC error")
        offset_error = _positive(record.get("clock_offset_error_s"),
                                 "UTC clock offset error", allow_zero=True)
        drift_error = _positive(record.get("drift_error_s_per_s"),
                                "future drift bound", allow_zero=True)
        extrapolation = _positive(record.get("max_extrapolation_s"),
                                  "extrapolation duration", allow_zero=True)
        valid_until = _positive(record.get("valid_until_utc_s"),
                                "UTC calibration expiry")
        fit = _references(record.get("fit_samples"), "fit samples",
                          ticks_by_id)
        validation = _references(record.get("validation_samples"),
                                 "held-out validation samples", ticks_by_id)
        if {row[0] for row in fit} & {row[0] for row in validation}:
            raise ValueError("UTC fit and held-out frame IDs overlap")
        # Subtract integer ticks *before* converting to float. The camera
        # counter may be many orders of magnitude larger than one frame step.
        reference_ticks = ticks_by_id[fit[0][0]]
        def x_of(fid: int) -> float:
            return (ticks_by_id[fid] - reference_ticks) / freq

        x = np.asarray([x_of(row[0]) for row in fit], dtype=float)
        y = np.asarray([row[2] for row in fit], dtype=float)
        centered = x - float(x.mean())
        denom = float(centered @ centered)
        if denom <= 0:
            raise ValueError("camera chunk UTC fit lacks clock span")
        slope = float(centered @ (y - y.mean()) / denom)
        if (not math.isfinite(slope) or slope <= 0 or
                abs(slope - 1.) * 1e6 > max_rate_ppm):
            raise ValueError("chunk tick rate disagrees with measured frequency")
        utc_origin = float(y.mean() - slope * x.mean())
        all_samples = fit + validation
        fit_error = max(abs(utc_origin + slope * x_of(row[0]) - row[2]) + row[3]
                        for row in all_samples)
        if fit_error + offset_error + drift_error * extrapolation > max_total:
            raise ValueError("chunk UTC fit exceeds commissioned error budget")
        first_tick = min(ticks_by_id[row[0]] for row in all_samples)
        last_tick = max(ticks_by_id[row[0]] for row in all_samples)
        if valid_until < utc_origin + slope * x_of(max(
                all_samples, key=lambda row: ticks_by_id[row[0]])[0]):
            raise ValueError("chunk UTC calibration expires before validation")
        return cls(
            record["calibration_id"], record["camera_serial"], journal,
            audit["sha256"], calibration_path, hashlib.sha256(raw).hexdigest(),
            freq, reference_ticks, utc_origin, slope, fit_error, offset_error,
            drift_error, max_total, extrapolation, valid_until,
            first_tick, last_tick, ticks_by_id, tuple(source_hashes))

    def exposure(self, frame_id: int) -> tuple[float, float]:
        """Return UTC interval for a journaled frame, within measured bounds.

        This is offline replay. The independent source's claimed physical
        calibration still needs human/rig commissioning before robot use.
        """
        if (_sha(self.journal_path) != self.journal_sha256 or
                _sha(self.calibration_path) != self.calibration_sha256 or
                any(_sha(path) != digest for path, digest in self.source_files)):
            raise ValueError("camera chunk UTC source files changed")
        if type(frame_id) is not int or frame_id not in self.ticks_by_frame_id:
            raise ValueError("camera frame ID is absent from chunk journal")
        ticks = self.ticks_by_frame_id[frame_id]
        outside_ticks = max(self.first_valid_tick - ticks, ticks -
                            self.last_valid_tick, 0)
        outside_s = outside_ticks / self.tick_frequency_hz
        if outside_s > self.max_extrapolation_s:
            raise ValueError("camera chunk frame exceeds extrapolation window")
        timestamp = self.reference_utc_s + self.fitted_rate * (
            (ticks - self.reference_ticks) / self.tick_frequency_hz)
        error = (self.fitted_error_bound_s + self.clock_offset_error_s +
                 self.drift_error_s_per_s * outside_s)
        if timestamp > self.valid_until_utc_s or error > self.max_total_error_s:
            raise ValueError("camera chunk UTC calibration expired or too uncertain")
        return timestamp, error


def evaluate_chunk_utc_calibration(path: Path) -> dict:
    """Summarize the saved fit without treating it as commissioned hardware."""
    clock = ChunkUTCClock.load(path)
    return {
        "schema": "precision_insertion_chunk_utc_calibration_audit_v1",
        "calibration_id": clock.calibration_id,
        "camera_serial": clock.camera_serial,
        "journal_path": str(clock.journal_path),
        "journal_sha256": clock.journal_sha256,
        "calibration_path": str(clock.calibration_path),
        "calibration_sha256": clock.calibration_sha256,
        "journaled_frames": len(clock.ticks_by_frame_id),
        "fitted_rate_seconds_per_declared_second": clock.fitted_rate,
        "observed_fit_and_holdout_error_bound_s": clock.fitted_error_bound_s,
        "max_total_error_s": clock.max_total_error_s,
        "verified_live_camera": False,
        "robot_ready": False,
        "scope": "offline_tick_utc_fit_not_live_camera_commissioning",
    }
