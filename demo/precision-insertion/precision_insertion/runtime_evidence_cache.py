"""Session-local producer evidence cache for immutable live NAS sessions.

The producer validates camera/frame provenance before writing each artifact.
During the same live session, readers may reuse the just-written manifest or
preinsert checkpoint instead of immediately replaying the same PNGs and
source files. Offline replay and handoff verification remain strict because
this cache is inactive there.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


_CACHE: ContextVar[dict[tuple[str, str, str], dict] | None] = ContextVar(
    "precision_live_evidence_cache", default=None)


@contextmanager
def live_evidence_cache() -> Iterator[None]:
    token = _CACHE.set({})
    try:
        yield
    finally:
        _CACHE.reset(token)


def remember(kind: str, directory: Path, phase: str, manifest: dict) -> None:
    cache = _CACHE.get()
    if cache is not None:
        cache[(kind, str(Path(directory).resolve()), phase)] = manifest


def recalled(kind: str, directory: Path, phase: str) -> dict | None:
    cache = _CACHE.get()
    return (None if cache is None else
            cache.get((kind, str(Path(directory).resolve()), phase)))
