"""Pure post-processing and static plots for pipeline lift reachability runs."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import read_jsonl, write_json


STAGES = (
    "bottom_ik_success",
    "both_endpoint_same_variant_success",
    "top_ik_local_success",
    "approach_success",
    "jacobian_lift_success",
    "pipeline_success",
)


def greedy_curve(matrix: np.ndarray) -> tuple[list[int], list[int]]:
    """Return greedy set-cover row order and cumulative covered column counts."""
    values = np.asarray(matrix, dtype=bool)
    if values.ndim != 2:
        raise ValueError("coverage matrix must have shape (grasps, cells)")
    covered = np.zeros(values.shape[1], dtype=bool)
    remaining = set(range(values.shape[0]))
    order: list[int] = []
    counts: list[int] = []
    while remaining:
        best = max(
            remaining,
            key=lambda index: (int(np.sum(values[index] & ~covered)), -index),
        )
        gain = int(np.sum(values[best] & ~covered))
        if gain == 0:
            break
        covered |= values[best]
        order.append(best)
        counts.append(int(covered.sum()))
        remaining.remove(best)
    return order, counts


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _aggregate(rows: Iterable[dict[str, Any]], field: str) -> dict[tuple[str, str], bool]:
    values: dict[tuple[str, str], bool] = {}
    for row in rows:
        key = (str(row["candidate_key_str"]), str(row["cell_id"]))
        values[key] = values.get(key, False) or bool(row.get(field, False))
    return values


def _save_matrices(run_dir: Path, rows: list[dict[str, Any]],
                   snapshot: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    recorded_keys = {str(row["candidate_key_str"]) for row in rows}
    grasp_keys = ["/".join(group["candidate_key"]) for group in snapshot["groups"]
                  if not rows or "/".join(group["candidate_key"]) in recorded_keys]
    cell_order: list[str] = []
    cell_seen: set[str] = set()
    for row in rows:
        cell = str(row["cell_id"])
        if cell not in cell_seen:
            cell_seen.add(cell)
            cell_order.append(cell)
    grasp_index = {key: index for index, key in enumerate(grasp_keys)}
    cell_index = {key: index for index, key in enumerate(cell_order)}
    matrices: dict[str, np.ndarray] = {}
    for field in STAGES:
        matrix = np.zeros((len(grasp_keys), len(cell_order)), dtype=bool)
        for (grasp, cell), success in _aggregate(rows, field).items():
            if grasp in grasp_index and cell in cell_index:
                matrix[grasp_index[grasp], cell_index[cell]] = success
        matrices[field] = matrix
    np.savez_compressed(
        run_dir / "coverage_matrix.npz",
        grasp_keys=np.asarray(grasp_keys), cell_ids=np.asarray(cell_order), **matrices)
    return matrices, grasp_keys, cell_order


def _plot_reports(run_dir: Path, rows: list[dict[str, Any]],
                  matrices: dict[str, np.ndarray], grasp_keys: list[str],
                  cells: list[str], summary: dict[str, Any]) -> list[str]:
    if not rows or not grasp_keys or not cells:
        return []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/autodex_matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plots = run_dir / "plots"
    per_grasp_dir = plots / "per_grasp"
    per_grasp_dir.mkdir(parents=True, exist_ok=True)
    cell_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        cell_meta.setdefault(str(row["cell_id"]), row)
    xs = np.asarray([float(cell_meta[cell]["object_x_m"]) for cell in cells])
    ys = np.asarray([float(cell_meta[cell]["object_y_m"]) for cell in cells])
    point_size = 55.0
    written: list[str] = []

    for index, key in enumerate(grasp_keys):
        endpoint = matrices["both_endpoint_same_variant_success"][index]
        lift = matrices["jacobian_lift_success"][index]
        full = matrices["pipeline_success"][index]
        gap = endpoint & ~lift
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
        for axis, values, title in zip(
                axes, (endpoint, lift, full),
                ("Both endpoint IK", "Jacobian lift", "Full pipeline")):
            axis.scatter(xs[~values], ys[~values], c="#d73027", marker="x", s=point_size)
            axis.scatter(xs[values], ys[values], c="#1a9850", marker="o", s=point_size)
            if title == "Jacobian lift" and gap.any():
                axis.scatter(xs[gap], ys[gap], facecolors="none", edgecolors="#54278f",
                             linewidths=1.5, s=point_size * 1.5)
            axis.set_title(f"{title}: {int(values.sum())}/{len(values)}")
            axis.set_xlabel("robot x (m)")
            axis.set_ylabel("robot y (m)")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.2)
        fig.suptitle(key)
        path = per_grasp_dir / f"{index + 1:04d}_{_safe_name(key)}.png"
        fig.savefig(path, dpi=160, facecolor="white")
        plt.close(fig)
        written.append(str(path.relative_to(run_dir)))

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    for field, label, color in (
            ("both_endpoint_same_variant_success", "both endpoint IK", "#756bb1"),
            ("jacobian_lift_success", "Jacobian lift", "#2ca25f"),
            ("pipeline_success", "full pipeline", "#2171b5")):
        curve = summary["greedy"][field]["covered_cells"]
        ax.plot(np.arange(1, len(curve) + 1), curve, marker="o", label=label, color=color)
    ax.set_xlabel("number of greedily selected base grasps")
    ax.set_ylabel("covered grid cells")
    ax.set_title("Greedy workspace coverage")
    ax.grid(alpha=0.25)
    ax.legend()
    path = plots / "greedy_coverage.png"
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    written.append(str(path.relative_to(run_dir)))

    verified = summary.get("verified_prefix", {})
    if verified.get("candidate_keys"):
        fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
        ax.plot(np.arange(1, len(verified["covered_cells"]) + 1),
                verified["covered_cells"], marker="o", color="#d95f0e")
        ax.set_xlabel("number of prior-success grasps available")
        ax.set_ylabel("full-pipeline covered grid cells")
        ax.set_title("Coverage of the prior verified-grasp prefix")
        ax.grid(alpha=0.25)
        path = plots / "verified_prefix_coverage.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        written.append(str(path.relative_to(run_dir)))
    return written


def analyze_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir)
    with (run_path / "candidate_snapshot.json").open() as stream:
        snapshot = json.load(stream)
    rows = read_jsonl(run_path / "per_grasp.jsonl")
    replay = read_jsonl(run_path / "pipeline_replay.jsonl")
    matrices, grasp_keys, cells = _save_matrices(run_path, rows, snapshot)

    greedy: dict[str, Any] = {}
    for field in ("both_endpoint_same_variant_success",
                  "jacobian_lift_success", "pipeline_success"):
        order, counts = greedy_curve(matrices[field])
        greedy[field] = {
            "candidate_indices": order,
            "candidate_keys": [grasp_keys[index] for index in order],
            "covered_cells": counts,
        }

    prior_success = {
        "/".join(group["candidate_key"])
        for group in snapshot["groups"] if group.get("prior_success")
    }
    verified_indices = [index for index, key in enumerate(grasp_keys)
                        if key in prior_success]
    covered = np.zeros(len(cells), dtype=bool)
    verified_counts = []
    for index in verified_indices:
        covered |= matrices["pipeline_success"][index]
        verified_counts.append(int(covered.sum()))

    failure_counts = Counter(
        str(row.get("failure_code")) for row in rows
        if not row.get("pipeline_success") and row.get("failure_code"))
    approach_gap_rows = [row for row in rows
                         if row.get("both_endpoint_same_variant_success")
                         and not row.get("approach_success")]
    lift_gap_rows = [row for row in rows
                     if row.get("both_endpoint_same_variant_success")
                     and row.get("approach_success")
                     and not row.get("jacobian_lift_success")]
    planner_times = [float(row["planner_wall_s"]) for row in rows
                     if row.get("planner_wall_s") is not None]
    endpoint_times = [float((row.get("timing") or {}).get("total_s", 0.0))
                      for row in rows]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "base_grasp_count": len(grasp_keys),
        "cell_count": len(cells),
        "per_grasp_record_count": len(rows),
        "pipeline_replay_record_count": len(replay),
        "stage_covered_cell_pairs": {
            field: int(matrix.sum()) for field, matrix in matrices.items()},
        "both_endpoint_pass_approach_fail_records": len(approach_gap_rows),
        "both_endpoint_pass_approach_pass_lift_fail_records": len(lift_gap_rows),
        "both_endpoint_pass_lift_fail_records": len(lift_gap_rows),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "greedy": greedy,
        "verified_prefix": {
            "candidate_keys": [grasp_keys[index] for index in verified_indices],
            "covered_cells": verified_counts,
        },
        "timing": {
            "planner_wall_s": ({
                "sum": float(np.sum(planner_times)),
                "mean": float(np.mean(planner_times)),
                "median": float(np.median(planner_times)),
                "p95": float(np.percentile(planner_times, 95)),
            } if planner_times else None),
            "diagnostic_endpoint_s": ({
                "sum": float(np.sum(endpoint_times)),
                "mean": float(np.mean(endpoint_times)),
                "median": float(np.median(endpoint_times)),
                "p95": float(np.percentile(endpoint_times, 95)),
            } if endpoint_times else None),
        },
        "pipeline_replay": {
            "success_count": sum(bool(row.get("pipeline_success")) for row in replay),
            "record_count": len(replay),
            "failure_code_counts": dict(sorted(Counter(
                str(row.get("failure_code")) for row in replay
                if not row.get("pipeline_success") and row.get("failure_code")
            ).items())),
        },
    }
    try:
        summary["plots"] = _plot_reports(
            run_path, rows, matrices, grasp_keys, cells, summary)
    except Exception as exc:
        summary["plot_error"] = repr(exc)
    write_json(run_path / "summary.json", summary)
    return summary
