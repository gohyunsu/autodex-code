"""Evidence contract for release and clear-view exit after one insertion stroke.

The robot-PC callback owns motion.  This module only verifies that its saved
record is bound to the completed guarded stroke and reports hand opening
before a measured clear-view arrival.  It is intentionally separate from the
task-success VLM: opening the hand or reaching clear view is not insertion
success.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "precision_insertion_poststroke_release_clear_view_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_post_insertion_release(
    path: Path, *, attempt_id: str, candidate_id: str,
    session_calibration_sha256: str, guarded_execution_path: Path,
) -> dict:
    """Verify one externally produced release/open-hand/clear-view record."""
    source = Path(path).expanduser().resolve()
    guarded = Path(guarded_execution_path).expanduser().resolve()
    if not source.is_file() or not guarded.is_file():
        raise FileNotFoundError("release and guarded execution records are required")
    record = json.loads(source.read_text(encoding="utf-8"))
    metric = json.loads(guarded.read_text(encoding="utf-8"))
    try:
        guarded_completed = float(metric["completed_at_s"])
        release_started = float(record["started_at_s"])
        release_completed = float(record["release_completed_at_s"])
        clear_completed = float(record["clear_view_completed_at_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("post-insertion release needs finite phase times") from exc
    reference = record.get("guarded_execution")
    measurement = record.get("measurement")
    if (not isinstance(record, dict) or record.get("schema") != SCHEMA or
            record.get("attempt_id") != attempt_id or
            record.get("candidate_id") != candidate_id or
            record.get("session_calibration_sha256") !=
            session_calibration_sha256 or
            record.get("status") != "released_and_clear_view" or
            reference != {"path": str(guarded), "sha256": _sha(guarded)} or
            measurement != {"hand_open": True, "clear_view_reached": True} or
            not all(math.isfinite(value) for value in (
                guarded_completed, release_started, release_completed,
                clear_completed)) or
            not guarded_completed <= release_started < release_completed <=
            clear_completed):
        raise ValueError(
            "release record does not prove ordered open-hand and clear-view completion")
    return {**record, "path": str(source), "sha256": _sha(source)}

