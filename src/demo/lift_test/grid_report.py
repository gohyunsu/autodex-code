"""Compact artifacts and static figures for lift-feasibility grid runs."""
from __future__ import annotations

import csv
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def failure_group(status: str, failure_code: str | None) -> str:
    """Collapse detailed planner taxonomy into readable map categories."""
    if status == "feasible":
        return "feasible"
    if status == "outside_domain":
        return "outside domain"
    code = failure_code or "unknown"
    if code.startswith("jacobian_") or code.startswith("lift_start_"):
        return "Jacobian lift"
    if code.startswith("approach_"):
        return "approach"
    if "ik" in code:
        return "endpoint IK"
    if ("filtered" in code or "catalogue" in code or "coverage" in code
            or "verified_pool" in code):
        return "candidate filter"
    if code == "candidate_search_truncated":
        return "candidate limit"
    return "other failure"


def write_cells_csv(path: Path, cells: Iterable[Mapping[str, Any]]) -> None:
    rows = list(cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "row", "col", "x_m", "y_m", "status", "failure_code", "failure_group",
        "selected_candidate_key", "selected_verified_rank", "min_feasible_rank",
        "candidate_attempt_count", "verified_rank_attempt_count",
        "symmetry_approach_attempt_count", "selected_training_variant_ordinal",
        "selected_symmetry_offset_ordinal", "selected_effective_symmetry_ordinal",
        "ik_valid_count",
        "completed_lift_steps", "total_s", "approach_s", "jacobian_lift_s",
        "execution_trajectory_s",
        "min_singular_value", "max_condition_number",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def write_cells_npz(path: Path, cells: Iterable[Mapping[str, Any]]) -> None:
    rows = list(cells)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _f(key: str) -> np.ndarray:
        return np.asarray([
            np.nan if row.get(key) is None else float(row[key]) for row in rows
        ], dtype=np.float64)

    def _s(key: str) -> np.ndarray:
        return np.asarray(["" if row.get(key) is None else str(row[key]) for row in rows])

    np.savez_compressed(
        path,
        row=np.asarray([int(row["row"]) for row in rows], dtype=np.int32),
        col=np.asarray([int(row["col"]) for row in rows], dtype=np.int32),
        x_m=_f("x_m"), y_m=_f("y_m"), status=_s("status"),
        failure_code=_s("failure_code"), failure_group=_s("failure_group"),
        selected_candidate_key=_s("selected_candidate_key"),
        selected_verified_rank=_f("selected_verified_rank"),
        min_feasible_rank=_f("min_feasible_rank"),
        candidate_attempt_count=_f("candidate_attempt_count"),
        verified_rank_attempt_count=_f("verified_rank_attempt_count"),
        symmetry_approach_attempt_count=_f("symmetry_approach_attempt_count"),
        selected_training_variant_ordinal=_f("selected_training_variant_ordinal"),
        selected_symmetry_offset_ordinal=_f("selected_symmetry_offset_ordinal"),
        selected_effective_symmetry_ordinal=_f("selected_effective_symmetry_ordinal"),
        ik_valid_count=_f("ik_valid_count"),
        completed_lift_steps=_f("completed_lift_steps"), total_s=_f("total_s"),
        approach_s=_f("approach_s"), jacobian_lift_s=_f("jacobian_lift_s"),
        execution_trajectory_s=_f("execution_trajectory_s"),
        min_singular_value=_f("min_singular_value"),
        max_condition_number=_f("max_condition_number"),
    )


def summarize_cells(cells: Iterable[Mapping[str, Any]]) -> dict:
    rows = list(cells)
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    failure_counts = Counter(
        str(row["failure_code"]) for row in rows
        if row.get("status") == "failed" and row.get("failure_code")
    )
    planned = [row for row in rows if row.get("status") in {"feasible", "failed"}]
    feasible = [row for row in rows if row.get("status") == "feasible"]
    times = [float(row["total_s"]) for row in planned if row.get("total_s") is not None]
    return {
        "cell_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_code_counts": dict(sorted(failure_counts.items())),
        "planned_cell_count": len(planned),
        "feasible_cell_count": len(feasible),
        "feasible_fraction_of_planned": (len(feasible) / len(planned) if planned else None),
        "planning_time_s": ({"mean": float(np.mean(times)), "median": float(np.median(times)),
                               "max": float(np.max(times)), "sum": float(np.sum(times))}
                            if times else None),
    }


def render_feasibility_map(output_dir: Path, *, proxy: Mapping[str, Any],
                           cells: Iterable[Mapping[str, Any]], title: str,
                           step_m: float) -> tuple[Path, Path]:
    """Write a two-panel status/time map without starting a Viser server."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Matplotlib otherwise tries to write below an often read-only user home
    # when the command runs under the robot service account.
    mpl_cache = Path("/tmp") / "autodex_matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = list(cells)
    polygon = np.asarray(proxy["vertices_xy_m"], dtype=float).reshape(4, 2)
    closed = np.vstack([polygon, polygon[:1]])
    fig, (ax_status, ax_time) = plt.subplots(1, 2, figsize=(15, 6.8), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    palette = {
        "feasible": "#2ca25f",
        "outside domain": "#d9d9d9",
        "Jacobian lift": "#de2d26",
        "approach": "#fdae6b",
        "endpoint IK": "#756bb1",
        "candidate filter": "#3182bd",
        "candidate limit": "#e6ab02",
        "other failure": "#636363",
    }
    by_group: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        group = str(row.get("failure_group") or failure_group(
            str(row.get("status", "unknown")), row.get("failure_code")))
        by_group.setdefault(group, []).append(row)
    point_size = max(14.0, min(90.0, (step_m * 950.0) ** 2))
    for group, group_rows in by_group.items():
        xs = [float(row["x_m"]) for row in group_rows]
        ys = [float(row["y_m"]) for row in group_rows]
        marker = "o" if group in {"feasible", "outside domain"} else "X"
        ax_status.scatter(xs, ys, s=point_size, c=palette.get(group, "#636363"),
                          marker=marker, linewidths=0.35, edgecolors="white", label=group)
    ax_status.plot(closed[:, 0], closed[:, 1], color="#00a6b2", linewidth=2.2,
                   label="Charuco proxy")
    ax_status.set_title("Lift feasibility and failure stage")
    ax_status.set_xlabel("robot-base x (m)")
    ax_status.set_ylabel("robot-base y (m)")
    ax_status.set_aspect("equal", adjustable="box")
    ax_status.grid(alpha=0.18)
    ax_status.legend(loc="best", fontsize=8, frameon=True)

    timed = [row for row in rows if row.get("total_s") is not None]
    if timed:
        points = ax_time.scatter(
            [float(row["x_m"]) for row in timed], [float(row["y_m"]) for row in timed],
            c=[float(row["total_s"]) for row in timed], s=point_size, cmap="viridis",
            marker="s", linewidths=0.25, edgecolors="white")
        colorbar = fig.colorbar(points, ax=ax_time, shrink=0.84)
        colorbar.set_label("planning time (s)")
        failed = [row for row in timed if row.get("status") == "failed"]
        if failed:
            ax_time.scatter([float(row["x_m"]) for row in failed],
                            [float(row["y_m"]) for row in failed],
                            marker="x", color="#b2182b", s=22, linewidths=0.8,
                            label="planning failed")
            ax_time.legend(loc="best", fontsize=8)
    else:
        ax_time.text(0.5, 0.5, "No planned cells", transform=ax_time.transAxes,
                     ha="center", va="center", color="#666666")
    ax_time.plot(closed[:, 0], closed[:, 1], color="#00a6b2", linewidth=2.2)
    ax_time.set_title("Planning time per valid grid cell")
    ax_time.set_xlabel("robot-base x (m)")
    ax_time.set_ylabel("robot-base y (m)")
    ax_time.set_aspect("equal", adjustable="box")
    ax_time.grid(alpha=0.18)

    png = output_dir / "feasibility_map.png"
    pdf = output_dir / "feasibility_map.pdf"
    fig.savefig(png, dpi=180, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return png, pdf


def render_verified_prefix_maps(output_dir: Path, *, proxy: Mapping[str, Any],
                                cells: Iterable[Mapping[str, Any]],
                                verified_count: int, title: str,
                                step_m: float) -> dict[str, Any]:
    """Render maps for every prefix of a ranked verified-grasp library.

    Each cell is planned only until its first feasible rank.  Prefix maps are
    therefore exact derivations of ``min_feasible_rank`` and do not repeat GPU
    planning N times.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mpl_cache = Path("/tmp") / "autodex_matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = list(cells)
    planned = [row for row in rows if row.get("status") != "outside_domain"]
    polygon = np.asarray(proxy["vertices_xy_m"], dtype=float).reshape(4, 2)
    closed = np.vstack([polygon, polygon[:1]])
    point_size = max(18.0, min(150.0, (step_m * 950.0) ** 2))
    maps_dir = output_dir / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    fractions: list[float] = []
    latency_by_count: dict[str, dict[str, float | None]] = {}
    map_files: list[str] = []
    for count in range(1, verified_count + 1):
        feasible = [row for row in planned
                    if row.get("min_feasible_rank") is not None
                    and int(row["min_feasible_rank"]) <= count]
        fractions.append(len(feasible) / len(planned) if planned else 0.0)
        latencies = []
        for row in planned:
            attempt_times = [float(value) for value in
                             row.get("verified_attempt_planning_s", [])]
            if attempt_times:
                latencies.append(sum(attempt_times[:min(count, len(attempt_times))]))
        latency_by_count[str(count)] = ({
            "mean_s": float(np.mean(latencies)),
            "median_s": float(np.median(latencies)),
            "p95_s": float(np.percentile(latencies, 95)),
            "max_s": float(np.max(latencies)),
        } if latencies else {
            "mean_s": None, "median_s": None, "p95_s": None, "max_s": None,
        })
        fig, ax = plt.subplots(figsize=(7.4, 6.5), constrained_layout=True)
        for is_feasible, color, label in (
                (False, "#d73027", "not feasible"),
                (True, "#1a9850", "feasible")):
            subset = [row for row in planned if (
                row.get("min_feasible_rank") is not None
                and int(row["min_feasible_rank"]) <= count) == is_feasible]
            if subset:
                ax.scatter([float(row["x_m"]) for row in subset],
                           [float(row["y_m"]) for row in subset],
                           s=point_size, marker="s", c=color,
                           edgecolors="white", linewidths=0.4, label=label)
        ax.plot(closed[:, 0], closed[:, 1], color="#00a6b2", linewidth=2.2,
                label="Charuco proxy")
        ax.set_title(f"{title}\nfirst {count} verified grasp(s): "
                     f"{fractions[-1] * 100.0:.1f}% feasible")
        ax.set_xlabel("robot-base x (m)")
        ax.set_ylabel("robot-base y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.legend(loc="best", fontsize=8)
        path = maps_dir / f"N_{count:03d}.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        map_files.append(str(path.relative_to(output_dir)))

    fig, (ax_rank, ax_curve) = plt.subplots(
        1, 2, figsize=(14.5, 6.4), constrained_layout=True)
    ranked = [row for row in planned if row.get("min_feasible_rank") is not None]
    if ranked:
        points = ax_rank.scatter(
            [float(row["x_m"]) for row in ranked],
            [float(row["y_m"]) for row in ranked],
            c=[int(row["min_feasible_rank"]) for row in ranked],
            s=point_size, marker="s", cmap="viridis_r", vmin=1,
            vmax=max(1, verified_count), edgecolors="white", linewidths=0.4)
        fig.colorbar(points, ax=ax_rank).set_label("minimum verified-grasp rank")
    failed = [row for row in planned if row.get("min_feasible_rank") is None]
    if failed:
        ax_rank.scatter([float(row["x_m"]) for row in failed],
                        [float(row["y_m"]) for row in failed],
                        s=point_size, marker="x", c="#b2182b", label="none feasible")
    ax_rank.plot(closed[:, 0], closed[:, 1], color="#00a6b2", linewidth=2.2)
    ax_rank.set_title("Minimum verified-grasp count required")
    ax_rank.set_xlabel("robot-base x (m)")
    ax_rank.set_ylabel("robot-base y (m)")
    ax_rank.set_aspect("equal", adjustable="box")
    ax_rank.grid(alpha=0.18)
    if failed:
        ax_rank.legend(loc="best", fontsize=8)

    counts = np.arange(1, verified_count + 1)
    ax_curve.plot(counts, np.asarray(fractions) * 100.0, marker="o", color="#2166ac")
    ax_curve.set_ylim(0.0, 102.0)
    ax_curve.set_xlabel("number of verified grasps available (N)")
    ax_curve.set_ylabel("feasible grid area (%)")
    ax_curve.set_title("Workspace coverage vs. verified library size")
    ax_curve.grid(alpha=0.25)
    overview = output_dir / "verified_prefix_summary.png"
    fig.suptitle(title, fontweight="bold")
    fig.savefig(overview, dpi=180, facecolor="white")
    plt.close(fig)
    return {
        "prefix_map_files": map_files,
        "summary_map": overview.name,
        "feasible_fraction_by_verified_count": {
            str(index + 1): float(value) for index, value in enumerate(fractions)
        },
        "online_search_time_by_verified_count": latency_by_count,
    }
