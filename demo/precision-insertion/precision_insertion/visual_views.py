"""Choose the operator's fixed camera views for visual assessments."""

from __future__ import annotations

import os
from collections.abc import Iterable


def assessment_cameras(available: Iterable[str]) -> list[str]:
    """Use all views offline, or the configured live camera selection."""
    cameras = sorted(set(available))
    configured = os.environ.get("PRECISION_VLM_CAMERA_IDS", "").strip()
    if not configured:
        return cameras
    selected = [item.strip() for item in configured.split(",")]
    if (len(selected) not in {3, 4, 5} or len(set(selected)) != len(selected) or
            any(not item.isdecimal() for item in selected)):
        raise ValueError("PRECISION_VLM_CAMERA_IDS needs three to five distinct camera IDs")
    missing = set(selected) - set(cameras)
    if missing:
        raise ValueError(f"selected VLM cameras lack paired evidence: {sorted(missing)}")
    return sorted(selected)


def final_assessment_cameras(available: Iterable[str]) -> list[str]:
    """Allow the released-socket label to use a reviewed camera subset.

    The XY estimator and earlier VLM stages keep their own view selection.
    Without a dedicated final subset, retain the existing assessment views.
    """
    configured = os.environ.get("PRECISION_FINAL_VLM_CAMERA_IDS", "").strip()
    if not configured:
        return assessment_cameras(available)
    cameras = set(available)
    selected = [item.strip() for item in configured.split(",")]
    if (len(selected) < 2 or len(selected) > 5 or
            len(set(selected)) != len(selected) or
            any(not item.isdecimal() for item in selected)):
        raise ValueError(
            "PRECISION_FINAL_VLM_CAMERA_IDS needs two to five distinct camera IDs")
    missing = set(selected) - cameras
    if missing:
        raise ValueError(
            f"selected final VLM cameras lack paired evidence: {sorted(missing)}")
    return sorted(selected)
