"""Persist lossless camera evidence without idling the inference thread."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .raw_camera_capture import RawCameraCapture
from .timing_profile import timed


_WRITE_POOL = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="precision-capture-save")


@dataclass
class PendingCaptureWrite:
    """One capture write that must finish before its evidence is consumed."""

    output: Path
    future: Future
    _finished: bool = False

    def finish(self) -> Path:
        if not self._finished:
            with timed("evidence.pending_capture_write_wait", phase="evidence",
                       kind="io"):
                saved = Path(self.future.result()).resolve()
            if saved != self.output:
                raise ValueError(
                    "background capture writer returned another path")
            self._finished = True
        return self.output


def begin_capture_write(
    capture: RawCameraCapture,
    output: Path,
    *,
    phase: str,
    writer: Callable,
) -> PendingCaptureWrite:
    """Start one verified capture write on the process-wide serial writer."""
    resolved = Path(output).expanduser().resolve()
    context = copy_context()

    def persist() -> Path:
        writer(capture, resolved, phase=phase)
        return resolved

    return PendingCaptureWrite(
        resolved, _WRITE_POOL.submit(context.run, persist))
