"""Acquire a replayable exposed-key depth report from final AutoDex frames."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .exposed_depth import ExposedDepthLimits
from .saved_exposed_depth import assess_saved_exposed_depth, write_saved_exposed_depth


def prepare_live_depth_evidence(
    *, runner, final_bundle: Path, backend,
    completed_at_s: float, decision_timestamp_s: float,
    limits: ExposedDepthLimits,
    capture_parameters: Mapping[str, float], slot: str,
) -> tuple[Path, bool]:
    """Save measured visual depth; return whether landmarks were observable.

    Physical error-bound admission remains a separate opt-in. The report
    itself never claims task success, even if its nominal depth exceeds 20 mm.
    """
    if not isinstance(limits, ExposedDepthLimits):
        raise TypeError("live visual depth needs calibrated error limits")
    attempt = runner.active_attempt
    if attempt is None or attempt.candidate_id is None:
        raise ValueError("visual depth needs the selected insertion attempt")
    report = assess_saved_exposed_depth(
        attempt_id=attempt.attempt_id,
        candidate_id=attempt.candidate_id,
        mode=runner.mode, shared_root=runner.shared_root,
        calibration=runner.calibration,
        calibration_path=Path(runner.output_dir) /
                         "frozen_session_calibration.json",
        final_bundle=final_bundle,
        completed_at_s=completed_at_s,
        decision_timestamp_s=decision_timestamp_s,
        backend=backend, limits=limits,
        **dict(capture_parameters))
    target = (Path(runner._attempt_dir) / "depth_assessments" / slot)
    path = write_saved_exposed_depth(report, target)
    return path, report["diagnostic"]["status"] == "bounded_visual_depth"
