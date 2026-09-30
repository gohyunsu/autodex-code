import json
from pathlib import Path

import numpy as np
import pytest

from src.validation.planning.pipeline_lift_reachability.analysis import (
    analyze_run,
    greedy_curve,
)
from src.validation.planning.pipeline_lift_reachability.core import (
    load_tabletop_transform,
    planner_robot_for,
    polar_cells,
    write_json,
)


def test_planner_robot_mapping_matches_supported_pipeline_pairs():
    assert planner_robot_for("xarm", "inspire") == "inspire"
    assert planner_robot_for("xarm", "allegro") == "allegro"
    assert planner_robot_for("franka", "inspire") == "fr3_inspire"
    with pytest.raises(ValueError):
        planner_robot_for("franka", "allegro")


def test_polar_cells_are_deterministic_and_keep_world_xy():
    cells = polar_cells([0.2, 0.3], [0.0, 90.0])
    assert [cell["cell_id"] for cell in cells] == [
        "r000_t000", "r000_t001", "r001_t000", "r001_t001"]
    assert cells[1]["nominal_x_m"] == pytest.approx(0.0, abs=1e-10)
    assert cells[1]["nominal_y_m"] == pytest.approx(0.2)


def test_tabletop_transform_orbits_position_but_not_orientation(tmp_path: Path):
    pose = np.eye(4)
    pose[:3, 3] = [0.01, 0.02, 0.03]
    path = tmp_path / "000.npy"
    np.save(path, pose)
    placed = load_tabletop_transform(path, 0.4, 90.0, 0.04)
    assert placed[:3, :3] == pytest.approx(np.eye(3))
    assert placed[:3, 3] == pytest.approx([-0.02, 0.41, 0.07])


def test_greedy_curve_uses_marginal_coverage():
    matrix = np.asarray([
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 1],
    ], dtype=bool)
    order, covered = greedy_curve(matrix)
    assert order[0] == 0  # stable tie: first candidate wins
    assert covered == [2, 4]


def test_analysis_separates_endpoint_lift_and_pipeline(tmp_path: Path):
    write_json(tmp_path / "candidate_snapshot.json", {
        "groups": [
            {"candidate_key": ["shelf", "1", "10"], "prior_success": True},
            {"candidate_key": ["wall", "2", "20"], "prior_success": False},
        ]
    })
    rows = [
        {
            "cell_id": "c0", "candidate_key_str": "shelf/1/10",
            "object_x_m": 0.4, "object_y_m": 0.0,
            "bottom_ik_success": True,
            "both_endpoint_same_variant_success": True,
            "top_ik_local_success": True, "approach_success": True,
            "jacobian_lift_success": True, "pipeline_success": True,
            "planner_wall_s": 2.0, "timing": {"total_s": 0.2},
        },
        {
            "cell_id": "c1", "candidate_key_str": "shelf/1/10",
            "object_x_m": 0.5, "object_y_m": 0.0,
            "bottom_ik_success": True,
            "both_endpoint_same_variant_success": True,
            "top_ik_local_success": False, "approach_success": True,
            "jacobian_lift_success": False, "pipeline_success": False,
            "failure_code": "jacobian_joint_limit",
            "planner_wall_s": 1.0, "timing": {"total_s": 0.1},
        },
        {
            "cell_id": "c1", "candidate_key_str": "wall/2/20",
            "object_x_m": 0.5, "object_y_m": 0.0,
            "bottom_ik_success": True,
            "both_endpoint_same_variant_success": True,
            "top_ik_local_success": True, "approach_success": True,
            "jacobian_lift_success": True, "pipeline_success": True,
            "planner_wall_s": 3.0, "timing": {"total_s": 0.3},
        },
    ]
    with (tmp_path / "per_grasp.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    summary = analyze_run(tmp_path)
    matrix = np.load(tmp_path / "coverage_matrix.npz")
    assert matrix["pipeline_success"].tolist() == [[True, False], [False, True]]
    assert summary["both_endpoint_pass_lift_fail_records"] == 1
    assert summary["failure_code_counts"] == {"jacobian_joint_limit": 1}
    assert summary["greedy"]["pipeline_success"]["covered_cells"] == [1, 2]
    assert summary["verified_prefix"]["covered_cells"] == [1]
