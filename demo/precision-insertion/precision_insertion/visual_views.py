"""Choose the operator's camera triplet for slow visual assessments."""

from __future__ import annotations

import os
from collections.abc import Iterable


def assessment_cameras(available: Iterable[str]) -> list[str]:
    """Use all views offline, or exactly the configured three for a live run."""
    cameras = sorted(set(available))
    configured = os.environ.get("PRECISION_VLM_CAMERA_IDS", "").strip()
    if not configured:
        return cameras
    selected = [item.strip() for item in configured.split(",")]
    if (len(selected) != 3 or len(set(selected)) != 3 or
            any(not item.isdecimal() for item in selected)):
        raise ValueError("PRECISION_VLM_CAMERA_IDS needs three distinct camera IDs")
    missing = set(selected) - set(cameras)
    if missing:
        raise ValueError(f"selected VLM cameras lack paired evidence: {sorted(missing)}")
    return sorted(selected)
