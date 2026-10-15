"""Bounded, ordered parallel calls for independent camera observations.

Only backends explicitly advertising independent concurrent requests opt in.
The default single-GPU Qwen backend remains serial.  Camera acquisition,
robot motion and the session ledger are never called from worker threads.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Callable, Sequence, TypeVar

from .timing_profile import timed


T = TypeVar("T")
R = TypeVar("R")


def map_view_calls(backend: object, items: Sequence[T], fn: Callable[[T], R]) -> list[R]:
    """Return results in input order, propagating the run timing context."""
    workers = getattr(backend, "max_parallel_views", 1)
    if type(workers) is not int or workers < 1:
        raise ValueError("VLM max_parallel_views must be a positive integer")
    active = min(workers, len(items))
    with timed("vlm.view_batch", phase="vlm", kind="inference",
               view_count=len(items), parallel_workers=active):
        if active < 2:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=active,
                                thread_name_prefix="precision-vlm") as pool:
            futures = [pool.submit(copy_context().run, fn, item)
                       for item in items]
            return [future.result() for future in futures]
