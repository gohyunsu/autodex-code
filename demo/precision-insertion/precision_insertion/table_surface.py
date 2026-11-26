"""Load a one-time measured support plane for board-free runtime sessions."""

from __future__ import annotations

import json
import math
from pathlib import Path


TABLE_PLANE_SCHEMA = "precision_insertion_table_plane_v1"


def load_table_surface_z(path: str | Path) -> float:
    """Return the robot-frame support height from a table-plane record."""
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text(encoding="utf-8"))
    value = record.get("table_surface_z_m") if isinstance(record, dict) else None
    if (record.get("schema") != TABLE_PLANE_SCHEMA or
            type(value) not in (int, float) or not math.isfinite(value)):
        raise ValueError(f"invalid measured table-plane record: {source}")
    return float(value)
