"""Recognize a stopped motor attempt without inferring the key's state.

This only authorizes the *supervised* tabletop-return branch. It never
authorizes another motor command or declares a grasp successful/failed.
"""

from __future__ import annotations

import json
from pathlib import Path


def acknowledged_failure(
    path: Path, *, schema: str, attempt_id: str, candidate_id: str,
) -> bool:
    """True only for this attempt's saved post-command stop acknowledgement."""
    source = Path(path)
    if not source.is_file():
        return False
    record = json.loads(source.read_text(encoding="utf-8"))
    return (
        isinstance(record, dict) and record.get("schema") == schema and
        record.get("attempt_id") == attempt_id and
        record.get("candidate_id") == candidate_id and
        record.get("status") == "execution_or_feedback_failed" and
        isinstance(record.get("stop_acknowledgement"), dict) and
        record["stop_acknowledgement"].get("acknowledged") is True
    )
