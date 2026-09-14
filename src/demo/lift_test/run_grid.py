#!/usr/bin/env python3
"""Map constructed tabletop lift feasibility over a Charuco-proxy XY grid.

This is an executor-free companion to :mod:`run_session`.  It uses the same
v8 candidate ordering, collision/IK funnel, approach plan, and 5 mm Jacobian
lift validation, but varies only an object's XY translation.  It never starts
a Viser server and deliberately writes no animation trajectories by default.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.demo.lift_test.board import load_proxy
from src.demo.lift_test.candidate_policy import make_candidate_policy
from src.demo.lift_test.campaign_state import (campaign_paths, candidate_key,
                                                load_progress, select_verified)
from src.demo.lift_test.grid_domain import (GridSpec, grid_summary, make_grid,
                                             tabletop_footprint_xy)
from src.demo.lift_test.grid_report import (failure_group, render_feasibility_map,
                                             render_verified_prefix_maps,
                                             summarize_cells, write_cells_csv,
                                             write_cells_npz)
from src.demo.lift_test.jacobian_lift import LiftOptions, options_as_dict
from src.demo.lift_test import run_session as session
from autodex.timing import TimingRecorder


def _write_json(path: Path, payload: Any) -> None:
    session._write_json(path, payload)


def _resolve_object(value: str | None, *, hand: str, version: str) -> str:
    if value is None:
        return session._prompt_object(hand, version, None)
    obj = str(value).strip()
    if not obj:
        raise ValueError("--object must not be empty")
    choices = session._candidate_objects(hand, version)
    if obj not in choices:
        preview = ", ".join(choices[:20])
        raise ValueError(f"unknown object {obj!r}; available examples: {preview}")
    return obj


def _resolve_tabletop_pose(value: str | None, *, obj: str, version: str) -> tuple[int, Path]:
    if value is None:
        return session._prompt_tabletop_pose(obj, version, None)
    requested = str(value).strip().removesuffix(".npy")
    files = session._tabletop_files(obj, version)
    for idx, file_path in enumerate(files):
        if file_path.stem == requested:
            return idx, file_path
    available = ", ".join(file_path.stem for file_path in files)
    raise ValueError(
        f"tabletop pose {value!r} is unavailable for {obj}; choose one of: {available}")


def _sum_span_duration(trace: Mapping[str, Any], name: str) -> float:
    return float(sum(
        float(span.get("duration_s", 0.0))
        for span in trace.get("spans", []) if span.get("name") == name
    ))


def _cell_result(*, cell, planning: Mapping[str, Any] | None,
                 trace: Mapping[str, Any] | None,
                 failure_code: str | None = None) -> dict:
    base = cell.as_dict()
    if not cell.domain_valid:
        base.update({"status": "outside_domain", "failure_code": None,
                     "failure_group": "outside domain"})
        return base
    planning = planning or {}
    success = bool(planning.get("success"))
    timing = planning.get("timing", {})
    final_step = (planning.get("jacobian_steps") or [None])[-1]
    candidate = planning.get("scene_info") if success else None
    code = None if success else (failure_code or planning.get("failure_code") or "grid_cell_exception")
    base.update({
        "status": "feasible" if success else "failed",
        "failure_code": code,
        "failure_group": failure_group("feasible" if success else "failed", code),
        "selected_candidate_key": ("/".join(map(str, candidate)) if candidate else None),
        "candidate_attempt_count": timing.get("approach_attempts"),
        "ik_valid_count": timing.get("ik_valid_count"),
        "completed_lift_steps": (final_step or {}).get("step") if success else None,
        "total_s": timing.get("total_s"),
        "approach_s": (_sum_span_duration(trace, "candidate_approach") if trace else None),
        "jacobian_lift_s": (_sum_span_duration(trace, "candidate_jacobian_lift") if trace else None),
        "execution_trajectory_s": (_sum_span_duration(
            trace, "candidate_execution_trajectory") if trace else None),
        "min_singular_value": ((final_step or {}).get("min_singular_value") if success else None),
        "max_condition_number": ((final_step or {}).get("condition_number") if success else None),
    })
    return base


def _parse_verified_count(value: str, available: int) -> int:
    if value == "all":
        return available
    try:
        requested = int(value)
    except ValueError as exc:
        raise ValueError("--verified-count must be a positive integer or 'all'") from exc
    if requested <= 0:
        raise ValueError("--verified-count must be positive or 'all'")
    return min(requested, available)


def _campaign_transfer_contract(
        contract: Mapping[str, Any], *, target_arm: str, hand: str,
        version: str, obj: str, pose_stem: str) -> dict[str, Any]:
    """Validate reusable grasp geometry while allowing a different target arm.

    A verified artifact stores an object-frame wrist transform plus hand-only
    approach/grasp configurations.  It does not store a reusable source-arm
    IK solution.  The target arm therefore reruns collision filtering, IK,
    approach planning, Jacobian continuation, and C2 validation from scratch.
    """
    required = {
        "hand": hand,
        "grasp_version": version,
        "object": obj,
        "tabletop_pose_stem": pose_stem,
        "scene": "table",
    }
    mismatch = {
        field: {"expected": expected, "saved": contract.get(field)}
        for field, expected in required.items()
        if contract.get(field) != expected
    }
    if mismatch:
        detail = ", ".join(
            f"{field}: expected={values['expected']!r}, saved={values['saved']!r}"
            for field, values in mismatch.items())
        raise ValueError(f"campaign contract mismatch: {detail}")
    source_arm = contract.get("arm")
    if source_arm not in session.ARM_TO_PLANNER_ROBOT:
        raise ValueError(f"campaign source arm is unsupported: {source_arm!r}")
    return {
        "source_arm": str(source_arm),
        "source_planner_robot": session._planner_robot_for_arm(str(source_arm)),
        "target_arm": target_arm,
        "target_planner_robot": session._planner_robot_for_arm(target_arm),
        "cross_arm": str(source_arm) != target_arm,
        "reused_fields": [
            "candidate_key", "variant_ordinal", "wrist_object",
            "pregrasp_qpos", "approach_hand_qpos", "grasp_hand_qpos",
        ],
        "recomputed_for_target_arm": [
            "candidate_collision", "endpoint_ik", "approach_trajectory",
            "jacobian_lift", "c2_execution_trajectory", "trajectory_validation",
        ],
    }


def _campaign_symmetry_variants(
        obj: str, wrist_object: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Expand one verified wrist target through the object's symmetry group.

    The saved target is kept first, so the direction that succeeded during
    training retains priority.  Remaining entries are object-frame rotations
    of that target, matching ``_expand_candidates_cyl`` with an identity
    object pose.  Returning metadata alongside the matrices lets inference
    distinguish the training direction from symmetry-derived alternatives.
    """
    wrist = np.asarray(wrist_object, dtype=np.float64).reshape(4, 4)
    # ``run_session`` resolves these functions together with the rest of the
    # planning runtime. Avoid importing ``autodex.utils`` here because its
    # package initialiser pulls trimesh/curobo into otherwise lightweight
    # ``--help`` and unit-test processes.
    axis_loader = getattr(session, "get_cyl_axis_local", None)
    yaw_loader = getattr(session, "get_cyl_yaw_grid", None)
    axis = axis_loader(obj) if axis_loader is not None else None
    yaw_grid = yaw_loader(obj) if yaw_loader is not None else None
    if axis is None or yaw_grid is None or len(yaw_grid) <= 1:
        return [wrist.copy()], [{
            "symmetry_offset_ordinal": 0,
            "symmetry_offset_rad": 0.0,
        }]

    unit_axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(unit_axis))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError(f"invalid symmetry axis for {obj!r}: {axis!r}")
    unit_axis /= norm
    cross = np.array([
        [0.0, -unit_axis[2], unit_axis[1]],
        [unit_axis[2], 0.0, -unit_axis[0]],
        [-unit_axis[1], unit_axis[0], 0.0],
    ], dtype=np.float64)

    variants: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for ordinal, theta_value in enumerate(np.asarray(yaw_grid, dtype=float)):
        theta = float(theta_value)
        rotation = (np.eye(3) + np.sin(theta) * cross
                    + (1.0 - np.cos(theta)) * (cross @ cross))
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        candidate = transform @ wrist
        # Registry grids exclude 2*pi, but retain this guard so malformed or
        # future custom grids cannot make inference plan an identical target
        # twice within one verified rank.
        if any(np.allclose(candidate, prior, atol=1.0e-9, rtol=0.0)
               for prior in variants):
            continue
        variants.append(candidate)
        metadata.append({
            "symmetry_offset_ordinal": int(ordinal),
            "symmetry_offset_rad": theta,
        })
    return variants, metadata


def _load_campaign_catalogue(*, progress: Mapping[str, Any], count: int,
                             project_dir: str | Path,
                             target_arm: str | None = None) -> session.PreparedCandidateCatalogue:
    """Load ranked verified grasps and expand symmetry within every rank.

    ``count`` remains the number of verified *base grasps*.  A base rank owns
    one or more symmetry-related wrist hypotheses, with the exact direction
    saved by training first.  This preserves the meaning of N-prefix maps
    while allowing a verified grasp to be approached from every equivalent
    object direction.
    """
    selected = select_verified(progress, count)
    wrist, pregrasp, grasp, openpose, scene_info = [], [], [], [], []
    contract = progress["contract"]
    obj = str(contract["object"])
    verified_groups: list[dict[str, Any]] = []
    for selected_index, item in enumerate(selected):
        artifact = Path(project_dir) / str(item["artifact"])
        if not artifact.is_file():
            raise FileNotFoundError(
                f"verified candidate artifact for rank {item['success_rank']} is missing: {artifact}")
        with np.load(artifact, allow_pickle=False) as saved:
            stored_key = tuple(str(value) for value in saved["candidate_key"].tolist())
            expected_key = candidate_key(item["candidate_key"])
            if stored_key != expected_key:
                raise RuntimeError(
                    f"verified candidate key mismatch at rank {item['success_rank']}: "
                    f"state={expected_key}, artifact={stored_key}")
            saved_wrist = np.asarray(saved["wrist_object"], dtype=np.float64)
            saved_pregrasp = np.asarray(saved["pregrasp_qpos"], dtype=np.float32)
            saved_grasp = np.asarray(saved["grasp_hand_qpos"], dtype=np.float32)
            saved_openpose = np.asarray(saved["approach_hand_qpos"], dtype=np.float32)
        variants, variant_metadata = _campaign_symmetry_variants(obj, saved_wrist)
        start = len(wrist)
        training_ordinal = int(item.get("variant_ordinal", 0))
        variant_count = len(variants)
        for variant, metadata in zip(variants, variant_metadata):
            offset_ordinal = int(metadata["symmetry_offset_ordinal"])
            wrist.append(variant)
            pregrasp.append(saved_pregrasp.copy())
            grasp.append(saved_grasp.copy())
            openpose.append(saved_openpose.copy())
            scene_info.append(expected_key)
            metadata["training_variant_ordinal"] = training_ordinal
            metadata["effective_symmetry_ordinal"] = (
                (training_ordinal + offset_ordinal) % variant_count)
        stop = len(wrist)
        verified_groups.append({
            "verified_rank": int(item.get("success_rank", selected_index + 1)),
            "candidate_key": list(expected_key),
            "training_variant_ordinal": training_ordinal,
            "catalogue_start": start,
            "catalogue_stop": stop,
            "symmetry_variant_count": stop - start,
            "variants": variant_metadata,
        })
    ordered = [list(candidate_key(item["candidate_key"])) for item in selected]
    source = {
        "policy": {
            "name": "campaign-verified", "success_only": True,
            "mutable_state_source": "experiment_private_training_progress",
        },
        "candidate_record_gate": "training_success_rank_prefix",
        "coverage_ranking_source": "training_success_rank",
        "selection_mode": "campaign_verified_rank_with_symmetry_variants",
        "ordered_keys": ordered,
        "n_coverage_candidates": len(selected),
        "n_coverage_useful": len(selected),
        "n_coverage_zero": 0,
        "remaining_coverage_by_key": {
            "/".join(candidate_key(item["candidate_key"])): 1 for item in selected
        },
        "verified_count": len(selected),
        "verified_success_ranks": [int(item["success_rank"]) for item in selected],
        "verified_groups": verified_groups,
        "n_candidate_records_after_symmetry_expansion": len(wrist),
        "symmetry_policy": {
            "training_direction_first": True,
            "expand_each_verified_rank": True,
            "axis_source": "autodex.utils.symmetry.get_cyl_axis_local",
            "yaw_grid_source": "autodex.utils.symmetry.get_cyl_yaw_grid",
        },
        "training_arm": contract.get("arm"),
        "target_arm": target_arm,
        "cross_arm_transfer": (
            target_arm is not None and contract.get("arm") != target_arm),
        "arm_transfer_contract": (
            "reuse_object_frame_wrist_and_hand_geometry_then_replan_all_target_arm_motion"
            if target_arm is not None and contract.get("arm") != target_arm
            else "same_arm_replan"),
    }
    return session.PreparedCandidateCatalogue(
        obj=str(contract["object"]), version=str(contract["grasp_version"]),
        hand=str(contract["hand"]), pose_stem=str(contract["tabletop_pose_stem"]),
        source_info=source,
        wrist_object=np.asarray(wrist, dtype=np.float64),
        pregrasp=np.asarray(pregrasp, dtype=np.float32),
        grasp=np.asarray(grasp, dtype=np.float32), openpose=openpose,
        scene_info=scene_info)


def _one_verified_rank(catalogue: session.PreparedCandidateCatalogue,
                       group_index: int) -> tuple[session.PreparedCandidateCatalogue, dict[str, Any]]:
    """Return every symmetry hypothesis belonging to one verified rank."""
    groups = list(catalogue.source_info.get("verified_groups", []))
    if not groups:
        # Backward-compatible in-memory fallback for synthetic callers. New
        # campaign catalogues always carry explicit group boundaries.
        groups = [{
            "verified_rank": index + 1,
            "candidate_key": list(candidate_key(info)),
            "catalogue_start": index,
            "catalogue_stop": index + 1,
            "symmetry_variant_count": 1,
            "variants": [{"symmetry_offset_ordinal": 0,
                          "symmetry_offset_rad": 0.0}],
        } for index, info in enumerate(catalogue.scene_info)]
    group = dict(groups[group_index])
    start = int(group["catalogue_start"])
    stop = int(group["catalogue_stop"])
    if start < 0 or stop <= start or stop > len(catalogue.scene_info):
        raise ValueError(f"invalid verified symmetry group bounds: {start}:{stop}")
    key = candidate_key(group["candidate_key"])
    if any(candidate_key(info) != key for info in catalogue.scene_info[start:stop]):
        raise ValueError(f"verified symmetry group mixes candidate keys at rank {group_index + 1}")
    source = {
        **catalogue.source_info,
        "ordered_keys": [list(key)],
        "n_coverage_candidates": 1,
        "n_coverage_useful": 1,
        "remaining_coverage_by_key": {"/".join(key): 1},
        "verified_rank": int(group.get("verified_rank", group_index + 1)),
        "verified_groups": [group],
        "n_candidate_records_after_symmetry_expansion": stop - start,
    }
    return session.PreparedCandidateCatalogue(
        obj=catalogue.obj, version=catalogue.version, hand=catalogue.hand,
        pose_stem=catalogue.pose_stem, source_info=source,
        wrist_object=catalogue.wrist_object[start:stop],
        pregrasp=catalogue.pregrasp[start:stop],
        grasp=catalogue.grasp[start:stop],
        openpose=catalogue.openpose[start:stop],
        scene_info=catalogue.scene_info[start:stop]), group


def _campaign_cell(*, planner, cell, scenario: Mapping[str, Any], obj: str,
                   version: str, hand: str, pose_stem: str,
                   options: LiftOptions, candidate_policy,
                   catalogue: session.PreparedCandidateCatalogue,
                   execution_profile) -> tuple[dict, dict]:
    """Evaluate verified ranks sequentially and stop at the first success.

    Every rank is planned as one symmetry-expanded candidate group.  Thus all
    equivalent directions participate in the collision/IK batch, while timing
    and prefix-map accounting still advance once per verified base grasp.
    """
    attempts: list[dict[str, Any]] = []
    total_s = 0.0
    stage_totals = {"approach_s": 0.0, "jacobian_lift_s": 0.0,
                    "execution_trajectory_s": 0.0}
    selected_planning: Mapping[str, Any] | None = None
    groups = list(catalogue.source_info.get("verified_groups", []))
    if not groups:
        groups = [{"verified_rank": index + 1,
                   "candidate_key": list(candidate_key(info)),
                   "catalogue_start": index, "catalogue_stop": index + 1,
                   "symmetry_variant_count": 1,
                   "variants": [{"symmetry_offset_ordinal": 0,
                                 "symmetry_offset_rad": 0.0}]}
                  for index, info in enumerate(catalogue.scene_info)]
    for group_index in range(len(groups)):
        grouped_catalogue, group = _one_verified_rank(catalogue, group_index)
        verified_rank = int(group.get("verified_rank", group_index + 1))
        key = candidate_key(group["candidate_key"])
        trace = TimingRecorder()
        parent = trace.begin(
            phase="planning", kind="plan", name="verified_candidate_preflight",
            verified_rank=verified_rank, candidate_key=list(key),
            symmetry_variant_count=len(grouped_catalogue.scene_info))
        planning = session._approach_and_lift(
            planner, scene_cfg=dict(scenario["scene_cfg"]), obj=obj,
            version=version, candidate_hand=hand, pose_stem=pose_stem,
            options=options, candidate_policy=candidate_policy,
            candidate_budget=None, timing=trace, timing_parent_id=parent,
            candidate_catalogue=grouped_catalogue,
            execution_profile=execution_profile, verbose=False)
        trace.end(parent, outcome="success" if planning["success"] else "failure",
                  failure_code=planning.get("failure_code"))
        snapshot = trace.as_dict()
        elapsed = float(planning.get("timing", {}).get("total_s", 0.0))
        total_s += elapsed
        stage = {
            "approach_s": _sum_span_duration(snapshot, "candidate_approach"),
            "jacobian_lift_s": _sum_span_duration(snapshot, "candidate_jacobian_lift"),
            "execution_trajectory_s": _sum_span_duration(
                snapshot, "candidate_execution_trajectory"),
        }
        for name, value in stage.items():
            stage_totals[name] += value
        selected_local = (int(planning["candidate_index"])
                          if planning.get("success") else None)
        variants = list(group.get("variants", []))
        selected_variant = (dict(variants[selected_local])
                            if selected_local is not None and selected_local < len(variants)
                            else None)
        attempts.append({
            "verified_rank": verified_rank,
            "candidate_key": list(key),
            "training_variant_ordinal": group.get("training_variant_ordinal"),
            "symmetry_variant_count": len(grouped_catalogue.scene_info),
            "symmetry_approach_attempt_count": int(
                planning.get("timing", {}).get("approach_attempts", 0)),
            "selected_symmetry_variant": selected_variant,
            "success": bool(planning["success"]),
            "failure_code": planning.get("failure_code"),
            "planning_wall_s": elapsed,
            "cumulative_planning_wall_s": total_s,
            "timing": snapshot,
        })
        selected_planning = planning
        if planning["success"]:
            break
    assert selected_planning is not None
    row = _cell_result(cell=cell, planning=selected_planning, trace=None)
    success_attempt = next((item for item in attempts if item["success"]), None)
    row.update({
        "status": "feasible" if success_attempt else "failed",
        "failure_code": None if success_attempt else "campaign_verified_pool_infeasible",
        "failure_group": "feasible" if success_attempt else "candidate filter",
        "selected_candidate_key": (
            "/".join(success_attempt["candidate_key"]) if success_attempt else None),
        "selected_verified_rank": (
            int(success_attempt["verified_rank"]) if success_attempt else None),
        "min_feasible_rank": (
            int(success_attempt["verified_rank"]) if success_attempt else None),
        "candidate_attempt_count": len(attempts),
        "verified_rank_attempt_count": len(attempts),
        "symmetry_approach_attempt_count": sum(
            int(item["symmetry_approach_attempt_count"]) for item in attempts),
        "selected_training_variant_ordinal": (
            success_attempt.get("training_variant_ordinal") if success_attempt else None),
        "selected_symmetry_offset_ordinal": (
            (success_attempt.get("selected_symmetry_variant") or {}).get(
                "symmetry_offset_ordinal") if success_attempt else None),
        "selected_effective_symmetry_ordinal": (
            (success_attempt.get("selected_symmetry_variant") or {}).get(
                "effective_symmetry_ordinal") if success_attempt else None),
        "verified_attempt_planning_s": [
            float(item["planning_wall_s"]) for item in attempts],
        "total_s": total_s,
        **stage_totals,
    })
    return row, {"row": cell.row, "col": cell.col, "x_m": cell.xy_m[0],
                 "y_m": cell.xy_m[1], "attempts": attempts,
                 "total_planning_wall_s": total_s}


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(payload, default=session._json_default, sort_keys=True) + "\n")


def _write_partial_artifacts(output_dir: Path, *, cells: list[dict]) -> dict:
    """Persist recoverable data after every cell; render only at milestones."""
    write_cells_csv(output_dir / "cells.csv", cells)
    write_cells_npz(output_dir / "cells.npz", cells)
    summary = summarize_cells(cells)
    _write_json(output_dir / "progress.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--object", help="v8 object name; omit to choose interactively")
    p.add_argument("--tabletop-pose", metavar="STEM",
                   help="tabletop pose stem such as 004; omit to choose after object")
    p.add_argument("--board-source", choices=["live-charuco", "file"], default="live-charuco")
    p.add_argument("--board-json", type=Path, help="board proxy JSON; required with --board-source file")
    p.add_argument("--arm", default="franka", choices=sorted(session.ARM_TO_PLANNER_ROBOT),
                   help=("arm kinematic/collision model: franka = FR3 + Inspire; "
                         "xarm = XArm6 + the same Inspire hand"))
    p.add_argument("--hand", default="inspire", choices=["inspire"],
                   help="candidate-hand namespace shared by both arm modes")
    p.add_argument("--grasp-version", default="v8", choices=["v8"])
    p.add_argument("--candidate-policy", choices=[
        "clean-state", "current-state", "verified-only", "campaign-verified"],
                   default="clean-state")
    p.add_argument("--exp-name", "--exp_name", default=None,
                   help=("source training experiment for campaign-verified; its arm may "
                         "differ from --arm because target-arm motion is replanned "
                         "(default: lift_training_<arm>)"))
    p.add_argument("--verified-count", default="all",
                   help=("ranked training base grasps to expose; each includes all "
                         "registered symmetry directions: positive integer or all"))
    p.add_argument("--board-tolerance-m", type=float, default=0.005,
                   help="allowed training/inference Charuco centre displacement")
    p.add_argument("--grid-step-m", type=float, default=0.05,
                   help="XY map spacing in metres (default: 0.05)")
    p.add_argument("--domain", choices=["footprint-inside", "center-only"],
                   default="center-only",
                   help=("grid inclusion rule (default: center-only, so every object "
                         "centre inside the full Charuco proxy is planned)"))
    p.add_argument("--edge-clearance-m", type=float, default=0.0,
                   help="additional inset from the selected proxy boundary in metres (default: 0)")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="per-cell candidate cap; 0 means all candidates")
    p.add_argument("--lift-height-m", type=float, default=0.10)
    p.add_argument("--lift-step-m", type=float, default=0.005)
    p.add_argument("--damping", type=float, default=0.02)
    p.add_argument("--max-iterations", type=int, default=24)
    p.add_argument("--max-joint-step-rad", type=float, default=0.10)
    p.add_argument("--max-segment-joint-delta-rad", type=float, default=0.02)
    p.add_argument("--cuda-graph", choices=["on", "off"], default="on")
    # Keep the live board measurement interface exactly aligned with run_session.
    p.add_argument("--pc-list", nargs="+", default=None)
    p.add_argument("--calib-dir")
    p.add_argument("--port-mask", type=int, default=5006)
    p.add_argument("--port-pose", type=int, default=5007)
    p.add_argument("--port-cmd", type=int, default=6893)
    p.add_argument("--port-snap", type=int, default=5009)
    p.add_argument("--port-snap-cmd", type=int, default=6894)
    p.add_argument("--snapshot-timeout-s", type=float, default=5.0)
    p.add_argument("--stream-fps", type=int, default=10)
    p.add_argument("--stream-warmup-s", type=float, default=2.0)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    if args.board_source == "file" and args.board_json is None:
        raise SystemExit("--board-json is required with --board-source file")
    if args.max_candidates < 0:
        raise SystemExit("--max-candidates must be >= 0")
    if args.grid_step_m <= 0.0:
        raise SystemExit("--grid-step-m must be positive")
    if args.edge_clearance_m < 0.0:
        raise SystemExit("--edge-clearance-m must be non-negative")
    if args.board_tolerance_m < 0.0:
        raise SystemExit("--board-tolerance-m must be non-negative")
    planner_robot = session._planner_robot_for_arm(args.arm)
    args.exp_name = args.exp_name or f"lift_training_{args.arm}"

    # All interactive selection happens before board/camera work, as requested:
    # object first, then a pose from that object's available tabletop assets.
    session._load_runtime_dependencies()
    try:
        obj = _resolve_object(args.object, hand=args.hand, version=args.grasp_version)
        pose_idx, pose_file = _resolve_tabletop_pose(
            args.tabletop_pose, obj=obj, version=args.grasp_version)
    except session.UserQuit:
        return 0

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_progress = None
    campaign_transfer = None
    verified_count = None
    if args.candidate_policy == "campaign-verified":
        paths = campaign_paths(
            project_dir=session.project_dir, exp_name=args.exp_name,
            hand=args.hand, version=args.grasp_version, obj=obj)
        if not paths.progress_path.is_file():
            raise SystemExit(
                f"training coverage state not found: {paths.progress_path}\n"
                "Run src/demo/lift_test/run_training.py first.")
        campaign_progress = load_progress(paths.progress_path)
        contract = campaign_progress.get("contract") or {}
        try:
            campaign_transfer = _campaign_transfer_contract(
                contract, target_arm=args.arm, hand=args.hand,
                version=args.grasp_version, obj=obj, pose_stem=pose_file.stem)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if campaign_transfer["cross_arm"]:
            print(
                f"[campaign] cross-arm transfer: verified on "
                f"{campaign_transfer['source_arm']}, replanning every trajectory for "
                f"{campaign_transfer['target_arm']}", flush=True)
        available = len(campaign_progress.get("verified_grasps", []))
        if available == 0:
            raise SystemExit("training campaign has no planning-feasible verified grasps")
        try:
            verified_count = _parse_verified_count(args.verified_count, available)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        output_dir = paths.analysis_root / stamp
    else:
        output_root = Path(session.project_dir) / "experiment" / "lift_grid"
        output_dir = output_root / args.hand / obj / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    options = LiftOptions(
        height_m=args.lift_height_m, step_m=args.lift_step_m, damping=args.damping,
        max_iterations=args.max_iterations, max_joint_step_rad=args.max_joint_step_rad,
        max_segment_joint_delta_rad=args.max_segment_joint_delta_rad)
    execution_profile = session.profile_for_arm(args.arm)
    grid_spec = GridSpec(step_m=args.grid_step_m, domain=args.domain,
                         edge_clearance_m=args.edge_clearance_m)
    candidate_policy = make_candidate_policy(
        "clean-state" if args.candidate_policy == "campaign-verified"
        else args.candidate_policy,
        clean_state_root=output_dir / "candidate_state_clean")
    capture = session.CaptureContext(args) if args.board_source == "live-charuco" else None
    started = perf_counter()
    run_trace = TimingRecorder()
    rows: list[dict] = []
    proxy: dict | None = None
    interrupted = False
    unexpected_error: Exception | None = None

    try:
        board_span = run_trace.begin(
            phase="preparation", kind="check", name="board_measurement",
            source=args.board_source)
        if args.board_source == "live-charuco":
            print("[board] Clear Charuco board and press Enter to measure (q to quit).")
            session._input("  ready")
            measurement, proxy = capture.measure_board(output_dir)
        else:
            proxy = load_proxy(args.board_json)
            measurement = {"source": "loaded_board_proxy",
                           "table_surface_z_m": proxy["table_surface_z_m"]}
            _write_json(output_dir / "board" / "board_proxy.json", proxy)
        run_trace.end(board_span, center_xy_m=proxy["center_xy_m"])
        if campaign_progress is not None:
            saved_center = np.asarray(
                campaign_progress["contract"]["board_center_xy_m"], dtype=float)
            current_center = np.asarray(proxy["center_xy_m"], dtype=float)
            displacement = float(np.linalg.norm(saved_center - current_center))
            if displacement > args.board_tolerance_m:
                raise RuntimeError(
                    "Charuco centre moved outside the training campaign tolerance: "
                    f"{displacement * 1000.0:.1f} mm > "
                    f"{args.board_tolerance_m * 1000.0:.1f} mm")

        zero_T = session._tabletop_transform(pose_file, np.zeros(2), 0.0)
        mesh_path = session.find_planning_mesh(obj, session.get_obj_root(args.grasp_version))
        mesh = session._load_mesh(mesh_path)
        footprint = tabletop_footprint_xy(np.asarray(mesh.vertices), zero_T)
        grid_cells = make_grid(proxy, footprint_xy=footprint, spec=grid_spec)
        grid_info = grid_summary(grid_cells, grid_spec)
        if grid_info["valid_cell_count"] == 0:
            raise RuntimeError("no grid cell lies in the board proxy under the selected domain")
        print(
            f"[grid] arm={args.arm} object={obj} tabletop={pose_file.stem} "
            f"step={args.grid_step_m * 1000:.1f}mm "
            f"cells={grid_info['cell_count']} valid={grid_info['valid_cell_count']} "
            f"domain={args.domain}", flush=True)

        _write_json(output_dir / "request.json", {
            "schema_version": 1, "object": obj,
            "tabletop": {"idx": pose_idx, "filename": pose_file.name},
            "input_contract": "constructed_xy_grid", "board_source": args.board_source,
            "arm": args.arm, "hand": args.hand, "planner_robot": planner_robot,
            "candidate_policy": candidate_policy.as_dict(), "lift_options": options_as_dict(options),
            "requested_candidate_policy": args.candidate_policy,
            "training_experiment": (args.exp_name if campaign_progress is not None else None),
            "campaign_arm_transfer": campaign_transfer,
            "verified_count": verified_count,
            "execution_profile": execution_profile.as_dict(),
            "candidate_budget": None if args.max_candidates == 0 else args.max_candidates,
            "start_q_source": f"GraspPlanner.{planner_robot} default init state",
            "animation": {"saved": False, "reason": "grid_map_default"},
            "cuda_graph": args.cuda_graph,
        })
        _write_json(output_dir / "board_proxy.json", proxy)
        _write_json(output_dir / "grid_spec.json", {
            **grid_info, "footprint_hull_xy_m": footprint.tolist(),
            "table_surface_z_m": float(measurement["table_surface_z_m"]),
        })

        planner_span = run_trace.begin(
            phase="preparation", kind="setup", name="planner_initialization",
            arm=args.arm, cuda_graph=args.cuda_graph)
        print(f"[planner] initializing {planner_robot} for --arm {args.arm} "
              f"(cuda_graph={args.cuda_graph})...", flush=True)
        init_t0 = perf_counter()
        planner = session.GraspPlanner(hand=planner_robot,
                                       use_cuda_graph=(args.cuda_graph == "on"))
        planner_init_s = perf_counter() - init_t0
        run_trace.end(planner_span)
        print(f"[planner] ready in {planner_init_s:.2f}s", flush=True)
        catalogue_span = run_trace.begin(
            phase="preparation", kind="io", name="candidate_catalogue_load",
            source=args.candidate_policy)
        catalogue_t0 = perf_counter()
        if campaign_progress is not None:
            assert verified_count is not None
            catalogue = _load_campaign_catalogue(
                progress=campaign_progress, count=verified_count,
                project_dir=session.project_dir, target_arm=args.arm)
        else:
            catalogue = session.prepare_candidate_catalogue(
                obj=obj, version=args.grasp_version, candidate_hand=args.hand,
                pose_stem=pose_file.stem, candidate_policy=candidate_policy)
        catalogue_s = perf_counter() - catalogue_t0
        run_trace.end(catalogue_span, candidate_count=len(catalogue.wrist_object))
        _write_json(output_dir / "candidate_source.json", catalogue.source_info)
        if campaign_progress is not None:
            prepared_label = (
                f"{len(catalogue.wrist_object)} symmetry hypotheses from "
                f"{catalogue.source_info['verified_count']} verified rank(s)")
        else:
            prepared_label = str(len(catalogue.wrist_object))
        print(
            f"[candidates] prepared={prepared_label} in {catalogue_s:.2f}s; "
            f"cache=on record_gate={catalogue.source_info.get('candidate_record_gate')}",
            flush=True,
        )

        n_valid = int(grid_info["valid_cell_count"])
        completed_valid = 0
        feasible = 0
        grid_span = run_trace.begin(
            phase="planning", kind="plan", name="grid_evaluation",
            valid_cell_count=n_valid, verified_count=verified_count)
        for cell in grid_cells:
            if not cell.domain_valid:
                rows.append(_cell_result(cell=cell, planning=None, trace=None))
                continue
            completed_valid += 1
            cell_trace = TimingRecorder()
            span = cell_trace.begin(phase="planning", kind="plan", name="cell_candidate_preflight",
                                    row=cell.row, col=cell.col, x_m=cell.xy_m[0], y_m=cell.xy_m[1])
            try:
                scenario = session._constructed_scenario(
                    obj, args.grasp_version, np.asarray(cell.xy_m), pose_idx, pose_file, measurement)
                if campaign_progress is not None:
                    # The campaign path uses one exact verified variant at a
                    # time so N-prefix latency is measured, not inferred from
                    # one large batch-IK call.
                    cell_trace.end(span, outcome="success", delegated="ranked_exact_variants")
                    row, cell_detail = _campaign_cell(
                        planner=planner, cell=cell, scenario=scenario, obj=obj,
                        version=args.grasp_version, hand=args.hand,
                        pose_stem=pose_file.stem, options=options,
                        candidate_policy=candidate_policy, catalogue=catalogue,
                        execution_profile=execution_profile)
                    _append_jsonl(output_dir / "cell_timing.jsonl", cell_detail)
                else:
                    planning = session._approach_and_lift(
                        planner, scene_cfg=scenario["scene_cfg"], obj=obj,
                        version=args.grasp_version, candidate_hand=args.hand,
                        pose_stem=pose_file.stem, options=options,
                        candidate_policy=candidate_policy,
                        candidate_budget=(None if args.max_candidates == 0 else args.max_candidates),
                        timing=cell_trace, timing_parent_id=span,
                        candidate_catalogue=catalogue,
                        execution_profile=execution_profile, verbose=False)
                    cell_trace.end(span, outcome="success" if planning["success"] else "failure",
                                   failure_code=planning.get("failure_code"))
                    row = _cell_result(cell=cell, planning=planning, trace=cell_trace.as_dict())
            except Exception as exc:
                cell_trace.end(span, outcome="failure", failure_code="grid_cell_exception",
                               exception=repr(exc))
                row = _cell_result(cell=cell, planning=None, trace=cell_trace.as_dict(),
                                   failure_code="grid_cell_exception")
                row["exception"] = repr(exc)
                unexpected_error = exc
            rows.append(row)
            if row["status"] == "feasible":
                feasible += 1
            elapsed = perf_counter() - started
            mean_s = elapsed / completed_valid
            eta_s = mean_s * (n_valid - completed_valid)
            print(
                f"[grid] {completed_valid}/{n_valid} cell=({cell.row},{cell.col}) "
                f"xy=({cell.xy_m[0]:.4f},{cell.xy_m[1]:.4f}) {row['status']} "
                f"total={float(row.get('total_s') or float('nan')):.2f}s feasible={feasible} "
                f"ETA={eta_s / 60.0:.1f}m", flush=True)
            _write_partial_artifacts(output_dir, cells=rows)
            if unexpected_error is not None:
                raise unexpected_error
        run_trace.end(grid_span, feasible_count=feasible)
    except session.UserQuit:
        interrupted = True
    except KeyboardInterrupt:
        interrupted = True
        print("\n[grid] interrupted; writing partial map...", flush=True)
    finally:
        if rows and proxy is not None:
            summary = _write_partial_artifacts(output_dir, cells=rows)
            # A static report is the grid product.  It needs neither Viser nor
            # an animation artifact and remains usable without CUDA later.
            report_span = run_trace.begin(
                phase="artifacts", kind="io", name="grid_report_render")
            png, pdf = render_feasibility_map(
                output_dir, proxy=proxy, cells=rows,
                title=f"{obj} · {args.arm} · tabletop pose {pose_file.stem}",
                step_m=args.grid_step_m)
            summary["map_png"] = png.name
            summary["map_pdf"] = pdf.name
            summary.update({"schema_version": 1, "object": obj,
                            "arm": args.arm, "hand": args.hand,
                            "planner_robot": planner_robot,
                            "tabletop": {"idx": pose_idx, "filename": pose_file.name},
                            "interrupted": interrupted,
                            "wall_time_s": perf_counter() - started})
            if campaign_progress is not None:
                assert verified_count is not None
                prefix = render_verified_prefix_maps(
                    output_dir, proxy=proxy, cells=rows,
                    verified_count=verified_count,
                    title=f"{obj} · {args.arm} · tabletop pose {pose_file.stem}",
                    step_m=args.grid_step_m)
                summary["campaign_verified"] = {
                    "training_experiment": args.exp_name,
                    "verified_count": verified_count,
                    "arm_transfer": campaign_transfer,
                    **prefix,
                }
            run_trace.end(report_span)
            timing_trace = run_trace.as_dict(
                trial_total_s=perf_counter() - started)
            _write_json(output_dir / "timing.json", timing_trace)
            summary["timing"] = "timing.json"
            _write_json(output_dir / "result.json", summary)
            print(f"[grid] result -> {output_dir}", flush=True)
            if "map_png" in summary:
                print(f"[grid] map -> {output_dir / summary['map_png']}", flush=True)
        if capture is not None:
            capture.close()

    if unexpected_error is not None:
        raise unexpected_error
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
