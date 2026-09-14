"""Experiment-private state for table-only Jacobian-lift training.

Immutable grasp geometry and v8 coverage metadata stay in their production
locations.  This module mirrors only mutable candidate outcomes below
``experiment/<exp_name>`` using the same layout as ``run_auto --isolate_experiment``.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

CandidateKey = tuple[str, str, str]
TERMINAL_STATUSES = {"coverage_complete", "coverage_unresolved", "training_stalled"}


def candidate_key(value: Sequence[Any]) -> CandidateKey:
    if len(value) != 3:
        raise ValueError(f"candidate key must have three fields, got {value!r}")
    return tuple(str(item) for item in value)  # type: ignore[return-value]


def key_string(key: Sequence[Any]) -> str:
    return "/".join(candidate_key(key))


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON without exposing a partially-written campaign state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=_json_default)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True)
class CampaignPaths:
    experiment_root: Path
    candidate_state_root: Path
    progress_path: Path
    session_root: Path
    episode_root: Path
    analysis_root: Path


def campaign_paths(*, project_dir: str | Path, exp_name: str, hand: str,
                   version: str, obj: str) -> CampaignPaths:
    root = Path(project_dir) / "experiment" / exp_name
    return CampaignPaths(
        experiment_root=root,
        # Exact layout returned by experiment_candidate_state_root(); spelling
        # it from the supplied root also keeps this helper unit-testable.
        candidate_state_root=root / "candidate_state" / hand / version / obj,
        progress_path=root / "coverage" / hand / version / f"{obj}.json",
        session_root=root / hand / "_sessions",
        episode_root=root / hand / obj,
        analysis_root=root / "analysis" / "lift_grid" / hand / obj,
    )


def load_coverage_records(*, project_dir: str | Path, obj: str, version: str,
                          pose_stem: str) -> list[dict[str, Any]]:
    """Return immutable v8 candidates for one tabletop pose, with cover sets."""
    path = (Path(project_dir) / "experiment" / version / "coverage" /
            f"cov_{version}_cand_{obj}.json")
    if not path.is_file():
        raise FileNotFoundError(f"v8 coverage asset missing: {path}")
    with path.open() as stream:
        payload = json.load(stream)
    merged: dict[CandidateKey, dict[str, Any]] = {}
    for source_index, item in enumerate(payload.get("grasps") or []):
        if str(item.get("pose_idx", "")) != str(pose_stem):
            continue
        key = (str(item["type"]), str(item["sid"]), str(item["gid"]))
        record = merged.setdefault(key, {
            "key": list(key), "key_string": key_string(key),
            "source_index": int(source_index), "covers": [],
        })
        record["covers"] = sorted(
            {int(scene) for scene in record["covers"]}
            | {int(scene) for scene in item.get("covers", [])})
    records = sorted(merged.values(), key=lambda item: int(item["source_index"]))
    if not records:
        raise RuntimeError(
            f"no v8 coverage candidates for {obj} tabletop pose {pose_stem}")
    return records


def coverage_universe(records: Iterable[Mapping[str, Any]]) -> set[int]:
    return {int(scene) for record in records for scene in record.get("covers", [])}


def _contract(*, exp_name: str, arm: str, hand: str, version: str, obj: str,
              pose_stem: str, board_proxy: Mapping[str, Any], lift_options: Mapping[str, Any],
              execution_profile: Mapping[str, Any], max_consecutive_failures: int) -> dict:
    return {
        "exp_name": exp_name,
        "arm": arm,
        "hand": hand,
        "grasp_version": version,
        "object": obj,
        "tabletop_pose_stem": str(pose_stem),
        "scene": "table",
        "object_xy_source": "charuco_proxy_center",
        "board_center_xy_m": [float(value) for value in board_proxy["center_xy_m"]],
        "table_surface_z_m": float(board_proxy["table_surface_z_m"]),
        "lift_options": dict(lift_options),
        "execution_profile": dict(execution_profile),
        "max_consecutive_failures": int(max_consecutive_failures),
        "verification_level": "planning_feasible",
    }


def create_or_resume_progress(*, path: Path, exp_name: str, arm: str, hand: str,
                              version: str, obj: str, pose_stem: str,
                              board_proxy: Mapping[str, Any],
                              lift_options: Mapping[str, Any],
                              execution_profile: Mapping[str, Any],
                              max_consecutive_failures: int,
                              records: Sequence[Mapping[str, Any]],
                              board_tolerance_m: float = 0.005) -> dict[str, Any]:
    expected = _contract(
        exp_name=exp_name, arm=arm, hand=hand, version=version, obj=obj,
        pose_stem=pose_stem, board_proxy=board_proxy, lift_options=lift_options,
        execution_profile=execution_profile,
        max_consecutive_failures=max_consecutive_failures)
    if path.is_file():
        with path.open() as stream:
            progress = json.load(stream)
        actual = progress.get("contract") or {}
        exact_fields = (
            "exp_name", "arm", "hand", "grasp_version", "object",
            "tabletop_pose_stem", "scene", "object_xy_source", "lift_options",
            "execution_profile", "max_consecutive_failures",
        )
        mismatch = [field for field in exact_fields if actual.get(field) != expected.get(field)]
        old_xy = np.asarray(actual.get("board_center_xy_m", [np.nan, np.nan]), dtype=float)
        new_xy = np.asarray(expected["board_center_xy_m"], dtype=float)
        old_z = float(actual.get("table_surface_z_m", np.nan))
        new_z = float(expected["table_surface_z_m"])
        if (not np.isfinite(old_xy).all() or np.linalg.norm(old_xy - new_xy) > board_tolerance_m):
            mismatch.append("board_center_xy_m")
        if not np.isfinite(old_z) or abs(old_z - new_z) > board_tolerance_m:
            mismatch.append("table_surface_z_m")
        if mismatch:
            raise RuntimeError(
                "training experiment contract does not match the saved campaign: "
                + ", ".join(sorted(set(mismatch))))
        # A completed/stalled campaign is immutable by default.  This avoids
        # silently changing the ranked verified library after inference maps
        # have already been generated from it.
        progress["updated_at"] = _now()
        return progress

    universe = sorted(coverage_universe(records))
    progress = {
        "schema_version": 1,
        "contract": expected,
        "created_at": _now(),
        "updated_at": _now(),
        "status": "active",
        "stop_reason": None,
        "coverage": {
            "scene_count": len(universe),
            "universe_scene_ids": universe,
            "covered_scene_ids": [],
            "remaining_scene_ids": universe,
            "fraction": 0.0 if universe else 1.0,
        },
        "candidate_count": len(records),
        "attempt_count": 0,
        "success_count": 0,
        "terminal_failure_count": 0,
        "consecutive_failures": 0,
        "verified_grasps": [],
        "terminal_failures": {},
        "attempts": [],
        "timing_summary": {},
    }
    write_json_atomic(path, progress)
    return progress


def _success_keys(progress: Mapping[str, Any]) -> set[CandidateKey]:
    return {candidate_key(item["candidate_key"])
            for item in progress.get("verified_grasps", [])}


def covered_scenes(progress: Mapping[str, Any]) -> set[int]:
    return {int(scene) for scene in progress.get("coverage", {}).get(
        "covered_scene_ids", [])}


def choose_next_candidate(records: Sequence[Mapping[str, Any]],
                          progress: Mapping[str, Any]) -> dict[str, Any] | None:
    covered = covered_scenes(progress)
    successful = _success_keys(progress)
    terminal = {candidate_key(key.split("/"))
                for key in progress.get("terminal_failures", {})}
    options: list[tuple[int, int, Mapping[str, Any], list[int]]] = []
    for record in records:
        key = candidate_key(record["key"])
        if key in successful or key in terminal:
            continue
        remaining = sorted({int(scene) for scene in record.get("covers", [])} - covered)
        if remaining:
            options.append((-len(remaining), int(record["source_index"]), record, remaining))
    if not options:
        return None
    _, _, selected, remaining = min(options, key=lambda item: (item[0], item[1]))
    return {**dict(selected), "marginal_scene_ids": remaining,
            "marginal_gain": len(remaining)}


def determine_status(progress: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> tuple[str, str | None]:
    remaining = progress.get("coverage", {}).get("remaining_scene_ids", [])
    if not remaining:
        return "coverage_complete", "all_v8_scenes_covered"
    limit = int(progress.get("contract", {}).get("max_consecutive_failures", 0))
    if limit > 0 and int(progress.get("consecutive_failures", 0)) >= limit:
        return "training_stalled", "max_consecutive_candidate_failures"
    if choose_next_candidate(records, progress) is None:
        return "coverage_unresolved", "no_positive_gain_candidate_remaining"
    return "active", None


def _timing_summary(attempts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = np.asarray([
        float(item.get("planning_wall_s", 0.0)) for item in attempts
        if item.get("planning_wall_s") is not None
    ], dtype=float)
    by_stage: dict[str, float] = {}
    for item in attempts:
        for name, seconds in (item.get("stage_time_s") or {}).items():
            by_stage[str(name)] = by_stage.get(str(name), 0.0) + float(seconds)
    cumulative = 0.0
    time_to_rank: dict[str, float] = {}
    for item in attempts:
        cumulative += float(item.get("planning_wall_s", 0.0) or 0.0)
        if item.get("success_rank") is not None:
            time_to_rank[str(int(item["success_rank"]))] = round(cumulative, 6)
    return {
        "candidate_planning_wall_s": {
            "sum": round(float(values.sum()), 6) if len(values) else 0.0,
            "mean": round(float(values.mean()), 6) if len(values) else None,
            "median": round(float(np.median(values)), 6) if len(values) else None,
            "p95": round(float(np.percentile(values, 95)), 6) if len(values) else None,
            "max": round(float(values.max()), 6) if len(values) else None,
        },
        "stage_sum_s": {key: round(value, 6) for key, value in sorted(by_stage.items())},
        "planning_time_to_success_rank_s": time_to_rank,
        "note": "predicted robot execution duration is excluded from planning wall time",
    }


def apply_candidate_outcome(*, progress: dict[str, Any], records: Sequence[Mapping[str, Any]],
                            selected: Mapping[str, Any], success: bool,
                            trial_relpath: str, failure_code: str | None,
                            variant_ordinal: int | None,
                            verified_artifact: str | None,
                            planning_wall_s: float,
                            stage_time_s: Mapping[str, float],
                            predicted_execution_duration_s: float | None) -> dict[str, Any]:
    key = candidate_key(selected["key"])
    attempt: dict[str, Any] = {
        "attempt_index": int(progress.get("attempt_count", 0)) + 1,
        "candidate_key": list(key),
        "marginal_gain_before": int(selected.get("marginal_gain", 0)),
        "marginal_scene_ids_before": list(selected.get("marginal_scene_ids", [])),
        "success": bool(success),
        "failure_code": failure_code,
        "trial": trial_relpath,
        "planning_wall_s": round(float(planning_wall_s), 6),
        "stage_time_s": {str(name): round(float(value), 6)
                         for name, value in stage_time_s.items()},
        "predicted_execution_duration_s": (
            None if predicted_execution_duration_s is None
            else round(float(predicted_execution_duration_s), 6)),
        "completed_at": _now(),
    }
    progress["attempt_count"] = attempt["attempt_index"]
    if success:
        rank = int(progress.get("success_count", 0)) + 1
        attempt["success_rank"] = rank
        verified = {
            "success_rank": rank,
            "candidate_key": list(key),
            "variant_ordinal": int(variant_ordinal or 0),
            "covers": list(selected.get("covers", [])),
            "coverage_added": list(selected.get("marginal_scene_ids", [])),
            "trial": trial_relpath,
            "artifact": verified_artifact,
            "planning_wall_s": attempt["planning_wall_s"],
            "predicted_execution_duration_s": attempt["predicted_execution_duration_s"],
            "verified_at": attempt["completed_at"],
            "verification_level": "planning_feasible",
        }
        progress.setdefault("verified_grasps", []).append(verified)
        progress["success_count"] = rank
        progress["consecutive_failures"] = 0
        covered = covered_scenes(progress) | {int(scene) for scene in selected.get("covers", [])}
    else:
        progress["consecutive_failures"] = int(progress.get("consecutive_failures", 0)) + 1
        progress["terminal_failure_count"] = int(progress.get("terminal_failure_count", 0)) + 1
        progress.setdefault("terminal_failures", {})[key_string(key)] = {
            "candidate_key": list(key),
            "covers": list(selected.get("covers", [])),
            "failure_code": failure_code,
            "trial": trial_relpath,
            "failed_at": attempt["completed_at"],
            "all_symmetry_variants_failed": True,
        }
        covered = covered_scenes(progress)
    universe = {int(scene) for scene in progress["coverage"]["universe_scene_ids"]}
    remaining = universe - covered
    progress["coverage"].update({
        "covered_scene_ids": sorted(covered),
        "remaining_scene_ids": sorted(remaining),
        "fraction": (len(covered) / len(universe) if universe else 1.0),
    })
    progress.setdefault("attempts", []).append(attempt)
    status, reason = determine_status(progress, records)
    progress["status"] = status
    progress["stop_reason"] = reason
    progress["updated_at"] = _now()
    progress["timing_summary"] = _timing_summary(progress["attempts"])
    return attempt


def write_candidate_outcome(*, state_root: Path, key: Sequence[Any], payload: Mapping[str, Any]) -> Path:
    target = state_root.joinpath(*candidate_key(key), "result.json")
    write_json_atomic(target, dict(payload))
    # Keep the module importable in CPU-only report/test environments.  The
    # coverage package pulls in geometry dependencies that are only present in
    # the planner environment.
    from autodex.utils.coverage import invalidate_success_cache
    invalidate_success_cache()
    return target


def load_progress(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        return json.load(stream)


def select_verified(progress: Mapping[str, Any], count: int | None) -> list[dict[str, Any]]:
    verified = sorted(
        (dict(item) for item in progress.get("verified_grasps", [])),
        key=lambda item: int(item["success_rank"]))
    if count is None:
        return verified
    if count <= 0:
        raise ValueError("verified count must be positive or 'all'")
    return verified[:count]
