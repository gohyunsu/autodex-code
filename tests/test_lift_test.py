import numpy as np

from autodex.planner import jacobian_stroke as js
from src.demo.lift_test import jacobian_lift as jl
from src.demo.lift_test import run_session as lift_session
from src.demo.lift_test.batch_validation import object_bottom_z_batch
from src.demo.lift_test.execution_trajectory import (ExecutionProfile,
                                                      build_execution_trajectory)
from src.demo.lift_test.board import point_in_polygon, polygon_centroid, y_interval_at_x
from src.demo.lift_test.candidate_policy import (make_candidate_policy,
                                                  order_coverage_keys)
from src.demo.lift_test.run_session import (_build_parser,
                                            _planner_robot_for_arm)
from src.demo.lift_test.viser_view import _arm_from_episode
from src.demo.lift_test.run_grid import (_build_parser as _grid_parser,
                                         _campaign_cell,
                                         _campaign_transfer_contract,
                                         _load_campaign_catalogue)
from src.demo.lift_test.run_training import _build_parser as _training_parser
from src.demo.lift_test.campaign_state import (
    apply_candidate_outcome, campaign_paths, choose_next_candidate,
    create_or_resume_progress, select_verified)
from src.demo.lift_test.grid_domain import GridSpec, make_grid
from src.demo.lift_test.grid_report import (render_feasibility_map,
                                             render_verified_prefix_maps, summarize_cells,
                                             write_cells_csv, write_cells_npz)


def _patch_synthetic_batch_validation(monkeypatch):
    """Make lightweight continuation tests mirror the real batch interface."""
    def _batch(planner, qpos):
        checked = [js._check_state(planner, q) for q in np.asarray(qpos)]
        valid = np.asarray([item[0] for item in checked], dtype=bool)
        invalid = np.flatnonzero(~valid)
        status = None if len(invalid) == 0 else checked[int(invalid[0])][1]
        return valid, status, {
            "backend": "synthetic_batch", "sample_count": len(valid),
            "first_invalid_index": None if len(invalid) == 0 else int(invalid[0]),
        }

    monkeypatch.setattr(js, "_check_states_batch", _batch)
    monkeypatch.setattr(
        js, "_fk_batch",
        lambda planner, qpos: np.stack([planner.fk_wrist(q) for q in qpos]),
    )
    monkeypatch.setattr(js, "_arm_execution_limits", lambda _n_arm: (1.0, 4.0))


def test_board_proxy_centroid_and_x_slice_are_inside():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.2, 1.0], [-0.2, 1.0]])
    center = polygon_centroid(vertices)
    assert point_in_polygon(center, vertices)
    interval = y_interval_at_x(vertices, float(center[0]))
    assert interval is not None
    assert interval[0] <= center[1] <= interval[1]


def test_continuation_uses_previous_waypoint_as_seed(monkeypatch):
    class FakePlanner:
        _n_arm = 3
        _motion_gen = object()
        _init_state = np.zeros(4, dtype=np.float32)

        def fk_wrist(self, q):
            T = np.eye(4)
            T[:3, 3] = np.asarray(q[:3], dtype=float)
            return T

    monkeypatch.setattr(
        js, "_joint_bounds",
        lambda planner, n_arm: (np.full(n_arm, -1.0), np.full(n_arm, 1.0)),
    )
    monkeypatch.setattr(js, "_check_state", lambda planner, q: (True, None))
    _patch_synthetic_batch_validation(monkeypatch)
    traj, steps, info = jl.continue_vertical_lift(
        FakePlanner(), np.zeros(4), hand_q=np.zeros(1),
        mesh_vertices=np.array([[0.0, 0.0, 0.0]]), object_pose_at_grasp=np.eye(4),
        table_surface_z_m=0.0,
        options=jl.LiftOptions(height_m=0.01, step_m=0.005,
                               max_iterations=8, position_tolerance_m=1e-7),
    )
    assert info["success"]
    assert traj.shape == (3, 4)  # start + two 5 mm waypoints
    assert np.isclose(traj[-1, 2], 0.01, atol=1e-6)
    assert all(step["success"] for step in steps)
    assert all(step["total_s"] >= step["jacobian_s"] for step in steps)
    assert all("endpoint_validation_s" in step and "segment_validation_s" in step
               for step in steps)
    # The returned execution samples are the same chord samples that passed
    # collision checking, not an implicit re-interpolation by a viewer.
    assert info["execution_qpos"].shape == (3, 4)
    assert info["execution_sample_contract"]["source"] == "collision_checked_linear_joint_chords"


def test_production_jacobian_stroke_supports_positive_and_negative_z(monkeypatch):
    class FakePlanner:
        _n_arm = 3
        _motion_gen = object()
        _init_state = np.zeros(4, dtype=np.float32)

        def fk_wrist(self, q):
            pose = np.eye(4)
            pose[:3, 3] = np.asarray(q[:3], dtype=float)
            return pose

    planner = FakePlanner()
    monkeypatch.setattr(
        js, "_joint_bounds",
        lambda _planner, dof: (np.full(dof, -1.0), np.full(dof, 1.0)))
    monkeypatch.setattr(js, "_check_state", lambda _planner, _q: (True, None))
    monkeypatch.setattr(
        js, "_check_states_batch",
        lambda _planner, q: (
            np.ones(len(np.atleast_2d(q)), dtype=bool), None,
            {"backend": "synthetic", "sample_count": len(np.atleast_2d(q))}))
    monkeypatch.setattr(
        js, "_fk_batch",
        lambda p, q: np.stack([p.fk_wrist(row) for row in np.atleast_2d(q)]))
    monkeypatch.setattr(js, "_arm_execution_limits", lambda _n_arm: (1.0, 4.0))
    options = js.JacobianStrokeOptions(
        step_m=0.005, max_iterations=8, position_tolerance_m=1.0e-7,
        request_start_position_tolerance_m=1.0e-7,
        request_start_orientation_tolerance_rad=1.0e-7)

    for initial_z, target_z, expected_direction in (
            (0.0, 0.01, "+Z"), (0.01, 0.0, "-Z")):
        start = np.array([0.0, 0.0, initial_z, 0.4], dtype=np.float32)
        target = planner.fk_wrist(start)
        target[2, 3] = target_z
        result = js.plan_jacobian_vertical_stroke(
            planner, start, target, options=options)
        assert result.success
        assert result.direction == expected_direction
        assert result.trajectory is not None
        assert result.time_s is not None
        assert np.allclose(result.trajectory[:, 3], 0.4)
        assert np.isclose(result.geometric_qpos[-1, 2], target_z, atol=1.0e-6)
        signed_progress = ((1.0 if expected_direction == "+Z" else -1.0)
                           * np.diff(result.trajectory[:, 2]))
        assert np.all(signed_progress >= -options.monotonic_tolerance_m)


def test_production_stroke_fails_closed_on_bad_request_or_payload(monkeypatch):
    class FakePlanner:
        _n_arm = 3
        _motion_gen = object()
        _init_state = np.zeros(4, dtype=np.float32)

        def fk_wrist(self, q):
            pose = np.eye(4)
            pose[:3, 3] = np.asarray(q[:3], dtype=float)
            return pose

    planner = FakePlanner()
    start = np.zeros(4, dtype=np.float32)
    lateral_target = planner.fk_wrist(start)
    lateral_target[0, 3] = 0.01
    lateral_target[2, 3] = 0.01
    mismatch = js.plan_jacobian_vertical_stroke(
        planner, start, lateral_target,
        options=js.JacobianStrokeOptions(
            request_start_position_tolerance_m=0.001))
    assert not mismatch.success
    assert mismatch.failure_code == "jacobian_request_start_lateral_mismatch"

    monkeypatch.setattr(js, "_check_state", lambda _planner, _q: (True, None))
    vertical_target = planner.fk_wrist(start)
    vertical_target[2, 3] = 0.01
    payload = js.plan_jacobian_vertical_stroke(
        planner, start, vertical_target,
        attached_object_vertices=np.array([[0.0, 0.0, -0.002]]),
        attached_object_pose_at_start=np.eye(4),
        support_surface_z_m=0.0,
    )
    assert not payload.success
    assert payload.failure_code == "jacobian_start_object_below_support"


def test_execution_trajectory_is_uniform_timestamped_and_holds_hand():
    profile = ExecutionProfile(
        arm="synthetic", max_joint_velocity_rad_s=1.0,
        max_joint_acceleration_rad_s2=4.0, held_object_speed_scale=0.4,
        sample_dt_s=0.01, squeeze_duration_s=0.025)
    # Two arm joints + one planner-space hand joint.  The lift nodes are the
    # dense collision-checked chord samples, including the squeeze endpoint.
    result = build_execution_trajectory(
        approach_qpos=np.array([[0.0, 0.0, 0.0], [0.1, -0.1, 0.0]]),
        squeeze_qpos=np.array([0.1, -0.1, 0.8]),
        lift_execution_qpos=np.array([
            [0.1, -0.1, 0.8], [0.12, -0.08, 0.8], [0.14, -0.06, 0.8],
        ]),
        arm_dof=2, profile=profile)
    qpos, time_s, phase = result["qpos"], result["time_s"], result["phase"]
    assert np.allclose(np.diff(time_s), 0.01)
    assert np.all(qpos[phase == "lift", 2] == np.float32(0.8))
    assert result["segments"]["squeeze_duration_s"] == 0.03  # rounded conservatively up
    assert result["segments"]["total_sample_count"] == len(qpos)


def test_execution_trajectory_artifact_preserves_the_rendered_timestamp_contract(tmp_path):
    profile = ExecutionProfile("synthetic", 1.0, 4.0, sample_dt_s=0.01,
                               squeeze_duration_s=0.02)
    execution = build_execution_trajectory(
        approach_qpos=np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]]),
        squeeze_qpos=np.array([0.02, 0.0, 0.5]),
        lift_execution_qpos=np.array([[0.02, 0.0, 0.5], [0.03, 0.01, 0.5]]),
        arm_dof=2, profile=profile)
    artifact = tmp_path / "execution_trajectory.npz"
    lift_session._save_execution_trajectory(
        artifact, {"execution": execution, "execution_validation": {"success": True}})
    with np.load(artifact) as saved:
        assert np.allclose(saved["qpos"], execution["qpos"])
        assert np.allclose(saved["time_s"], execution["time_s"])
        assert np.array_equal(saved["phase"], execution["phase"])


def test_continuation_rejects_collision_between_5mm_endpoints(monkeypatch):
    class FakePlanner:
        _n_arm = 3
        _motion_gen = object()
        _init_state = np.zeros(4, dtype=np.float32)

        def fk_wrist(self, q):
            T = np.eye(4)
            T[:3, 3] = np.asarray(q[:3], dtype=float)
            return T

    monkeypatch.setattr(
        js, "_joint_bounds",
        lambda planner, n_arm: (np.full(n_arm, -1.0), np.full(n_arm, 1.0)),
    )

    def _state_check(_planner, q):
        # Start and 5 mm endpoint are valid, but the executed q chord passes
        # through a forbidden intermediate configuration near z=3 mm.
        return (not (0.0025 < float(q[2]) < 0.0035), "synthetic_collision")

    monkeypatch.setattr(js, "_check_state", _state_check)
    _patch_synthetic_batch_validation(monkeypatch)
    traj, steps, info = jl.continue_vertical_lift(
        FakePlanner(), np.zeros(4), hand_q=np.zeros(1),
        mesh_vertices=np.array([[0.0, 0.0, 0.0]]), object_pose_at_grasp=np.eye(4),
        table_surface_z_m=0.0,
        options=jl.LiftOptions(height_m=0.005, step_m=0.005,
                               max_iterations=8, position_tolerance_m=1e-7,
                               max_segment_joint_delta_rad=0.001),
    )
    assert traj is None
    assert info["failure_code"] == "jacobian_segment_robot_collision"
    assert steps[-1]["segment_collision_valid"] is False
    assert steps[-1]["segment_samples"] == 5


def test_batched_object_bottom_matches_scalar_vertex_transform():
    vertices = np.array([
        [-0.2, -0.1, -0.3], [0.4, -0.2, 0.1], [0.1, 0.3, 0.5],
    ])
    transforms = []
    for angle, translation in [(0.0, [0.0, 0.0, 0.7]),
                               (0.4, [0.2, -0.1, 0.9])]:
        T = np.eye(4)
        c, s = np.cos(angle), np.sin(angle)
        T[:3, :3] = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        T[:3, 3] = translation
        transforms.append(T)
    transforms = np.stack(transforms)
    expected = np.asarray([
        ((T @ np.c_[vertices, np.ones(len(vertices))].T).T[:, 2].min())
        for T in transforms
    ])
    assert np.allclose(object_bottom_z_batch(vertices, transforms), expected)


def test_candidate_policies_keep_clean_current_and_verified_semantics_distinct(tmp_path):
    clean = make_candidate_policy("clean-state", clean_state_root=tmp_path / "empty")
    current = make_candidate_policy("current-state", clean_state_root=tmp_path / "ignored")
    verified = make_candidate_policy("verified-only", clean_state_root=tmp_path / "verified")
    assert clean.skip_done is False
    assert clean.success_only is False
    assert clean.success_root == str(tmp_path / "empty")
    assert current.skip_done is True
    assert current.success_only is False
    assert current.success_root is None
    assert verified.skip_done is False
    assert verified.skip_scenes_with_success is False
    assert verified.success_only is True
    assert verified.success_root == str(tmp_path / "verified")

    # Mirrors run_auto: zero-remaining candidates are dropped; score ties
    # retain catalogue insertion order.
    order = order_coverage_keys({
        ("table", "1", "a"): 2,
        ("table", "1", "b"): 5,
        ("table", "1", "c"): 5,
        ("table", "1", "d"): 0,
    })
    assert order == [("table", "1", "b"), ("table", "1", "c"),
                     ("table", "1", "a")]


def test_lift_test_arm_mode_selects_matching_planner_robot():
    parser = _build_parser()
    assert parser.parse_args([]).arm == "franka"
    assert parser.parse_args(["--arm", "xarm"]).arm == "xarm"
    assert parser.parse_args(["--candidate-policy", "verified-only"]).candidate_policy == "verified-only"
    assert _planner_robot_for_arm("franka") == "fr3_inspire"
    assert _planner_robot_for_arm("xarm") == "inspire"


def test_grid_arm_mode_matches_single_trial_arm_contract():
    parser = _grid_parser()
    assert parser.parse_args([]).arm == "franka"
    assert parser.parse_args([]).grid_step_m == 0.05
    assert parser.parse_args(["--arm", "xarm"]).arm == "xarm"
    assert parser.parse_args(["--candidate-policy", "verified-only"]).candidate_policy == "verified-only"


def test_training_defaults_to_twenty_candidate_failure_stall():
    args = _training_parser().parse_args([])
    assert args.max_consecutive_failures == 20
    assert args.lift_step_m == 0.005
    assert args.arm == "franka"


def test_campaign_paths_match_isolated_pipeline_convention(tmp_path):
    paths = campaign_paths(
        project_dir=tmp_path, exp_name="lift_training_franka",
        hand="inspire", version="v8", obj="apple")
    assert paths.candidate_state_root == (
        tmp_path / "experiment/lift_training_franka/candidate_state/inspire/v8/apple")
    assert paths.progress_path == (
        tmp_path / "experiment/lift_training_franka/coverage/inspire/v8/apple.json")
    assert paths.episode_root == (
        tmp_path / "experiment/lift_training_franka/inspire/apple")


def test_training_coverage_greedy_success_reset_and_failure_stall(tmp_path):
    records = [
        {"key": ["table", "0", "g0"], "source_index": 0, "covers": [0, 1, 2]},
        {"key": ["wall", "1", "g1"], "source_index": 1, "covers": [2, 3]},
        {"key": ["shelf", "2", "g2"], "source_index": 2, "covers": [3]},
    ]
    progress = create_or_resume_progress(
        path=tmp_path / "progress.json", exp_name="train", arm="franka",
        hand="inspire", version="v8", obj="apple", pose_stem="004",
        board_proxy={"center_xy_m": [0.5, 0.1], "table_surface_z_m": 0.0},
        lift_options={"height_m": 0.1}, execution_profile={"sample_dt_s": 0.01},
        max_consecutive_failures=2, records=records)
    first = choose_next_candidate(records, progress)
    assert first["key"] == ["table", "0", "g0"]
    attempt = apply_candidate_outcome(
        progress=progress, records=records, selected=first, success=True,
        trial_relpath="experiment/train/inspire/apple/t0", failure_code=None,
        variant_ordinal=1, verified_artifact="experiment/train/a.npz",
        planning_wall_s=1.5, stage_time_s={"candidate_endpoint_ik": 0.2},
        predicted_execution_duration_s=4.0)
    assert attempt["success_rank"] == 1
    assert progress["consecutive_failures"] == 0
    assert progress["coverage"]["covered_scene_ids"] == [0, 1, 2]
    second = choose_next_candidate(records, progress)
    assert second["key"] == ["wall", "1", "g1"]
    apply_candidate_outcome(
        progress=progress, records=records, selected=second, success=False,
        trial_relpath="experiment/train/inspire/apple/t1",
        failure_code="candidate_all_ik_failed", variant_ordinal=None,
        verified_artifact=None, planning_wall_s=0.4,
        stage_time_s={"candidate_endpoint_ik": 0.4},
        predicted_execution_duration_s=None)
    third = choose_next_candidate(records, progress)
    assert third["key"] == ["shelf", "2", "g2"]
    apply_candidate_outcome(
        progress=progress, records=records, selected=third, success=False,
        trial_relpath="experiment/train/inspire/apple/t2",
        failure_code="candidate_all_jacobian_failed", variant_ordinal=None,
        verified_artifact=None, planning_wall_s=0.8,
        stage_time_s={"candidate_jacobian_lift": 0.7},
        predicted_execution_duration_s=None)
    assert progress["status"] == "training_stalled"
    assert progress["consecutive_failures"] == 2
    assert progress["timing_summary"]["candidate_planning_wall_s"]["sum"] == 2.7
    assert progress["timing_summary"]["planning_time_to_success_rank_s"] == {"1": 1.5}
    assert select_verified(progress, 1)[0]["variant_ordinal"] == 1


def test_training_counts_all_symmetry_variants_as_one_candidate_failure(tmp_path):
    records = [{"key": ["table", "0", "g0"], "source_index": 0, "covers": [0]}]
    progress = create_or_resume_progress(
        path=tmp_path / "progress.json", exp_name="train", arm="franka",
        hand="inspire", version="v8", obj="can", pose_stem="000",
        board_proxy={"center_xy_m": [0.0, 0.0], "table_surface_z_m": 0.0},
        lift_options={}, execution_profile={}, max_consecutive_failures=3,
        records=records)
    selected = choose_next_candidate(records, progress)
    apply_candidate_outcome(
        progress=progress, records=records, selected=selected, success=False,
        trial_relpath="experiment/train/inspire/can/t0",
        failure_code="candidate_all_jacobian_failed", variant_ordinal=None,
        verified_artifact=None, planning_wall_s=1.0, stage_time_s={},
        predicted_execution_duration_s=None)
    assert progress["consecutive_failures"] == 1
    assert progress["terminal_failures"]["table/0/g0"][
        "all_symmetry_variants_failed"] is True


def test_campaign_inference_expands_symmetry_inside_rank(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "experiment/train/inspire/apple/t0/plan"
    artifact_dir.mkdir(parents=True)
    wrist = np.eye(4)
    wrist[0, 3] = 0.12
    np.savez_compressed(
        artifact_dir / "verified_candidate.npz",
        candidate_key=np.asarray(["table", "0", "g0"]),
        wrist_object=wrist,
        pregrasp_qpos=np.asarray([0.1, 0.2]),
        approach_hand_qpos=np.asarray([0.3, 0.4]),
        grasp_hand_qpos=np.asarray([0.5, 0.6]),
    )
    progress = {
        "contract": {"object": "apple", "grasp_version": "v8",
                     "arm": "franka", "hand": "inspire", "tabletop_pose_stem": "004"},
        "verified_grasps": [{
            "success_rank": 1, "candidate_key": ["table", "0", "g0"],
            "artifact": "experiment/train/inspire/apple/t0/plan/verified_candidate.npz",
            "variant_ordinal": 1,
        }],
    }
    monkeypatch.setattr(
        lift_session, "get_cyl_axis_local",
        lambda _obj: np.asarray([0.0, 0.0, 1.0]), raising=False)
    monkeypatch.setattr(
        lift_session, "get_cyl_yaw_grid",
        lambda _obj: np.asarray([0.0, np.pi / 2.0]), raising=False)
    catalogue = _load_campaign_catalogue(
        progress=progress, count=1, project_dir=tmp_path, target_arm="xarm")
    assert len(catalogue.wrist_object) == 2
    assert np.allclose(catalogue.wrist_object[0], wrist)
    assert np.allclose(catalogue.wrist_object[1][:2, 3], [0.0, 0.12])
    assert np.allclose(catalogue.openpose, [[0.3, 0.4], [0.3, 0.4]])
    assert catalogue.scene_info == [("table", "0", "g0")] * 2
    group = catalogue.source_info["verified_groups"][0]
    assert group["verified_rank"] == 1
    assert group["training_variant_ordinal"] == 1
    assert group["symmetry_variant_count"] == 2
    assert [item["effective_symmetry_ordinal"] for item in group["variants"]] == [1, 0]
    assert catalogue.source_info["n_candidate_records_after_symmetry_expansion"] == 2
    assert catalogue.source_info["training_arm"] == "franka"
    assert catalogue.source_info["target_arm"] == "xarm"
    assert catalogue.source_info["cross_arm_transfer"] is True


def test_campaign_cell_keeps_verified_rank_while_trying_all_symmetry(monkeypatch):
    keys = [("table", "0", "g0"), ("table", "1", "g1")]
    groups = [
        {"verified_rank": 1, "candidate_key": list(keys[0]),
         "training_variant_ordinal": 0, "catalogue_start": 0, "catalogue_stop": 2,
         "symmetry_variant_count": 2,
         "variants": [{"symmetry_offset_ordinal": 0},
                      {"symmetry_offset_ordinal": 1}]},
        {"verified_rank": 2, "candidate_key": list(keys[1]),
         "training_variant_ordinal": 1, "catalogue_start": 2, "catalogue_stop": 4,
         "symmetry_variant_count": 2,
         "variants": [{"symmetry_offset_ordinal": 0,
                       "effective_symmetry_ordinal": 1},
                      {"symmetry_offset_ordinal": 1,
                       "effective_symmetry_ordinal": 0}]},
    ]
    catalogue = lift_session.PreparedCandidateCatalogue(
        obj="apple", version="v8", hand="inspire", pose_stem="004",
        source_info={"verified_groups": groups, "ordered_keys": [list(key) for key in keys]},
        wrist_object=np.repeat(np.eye(4)[None], 4, axis=0),
        pregrasp=np.zeros((4, 2), dtype=np.float32),
        grasp=np.ones((4, 2), dtype=np.float32),
        openpose=[np.zeros(2, dtype=np.float32)] * 4,
        scene_info=[keys[0], keys[0], keys[1], keys[1]],
    )
    observed_sizes = []

    def _fake_approach_and_lift(_planner, **kwargs):
        group_catalogue = kwargs["candidate_catalogue"]
        observed_sizes.append(len(group_catalogue.scene_info))
        rank = group_catalogue.source_info["verified_rank"]
        success = rank == 2
        return {
            "success": success,
            "failure_code": None if success else "candidate_all_ik_failed",
            "candidate_index": 1 if success else None,
            "scene_info": keys[1] if success else None,
            "jacobian_steps": [],
            "timing": {"total_s": 0.2 * rank, "approach_attempts": rank},
        }

    monkeypatch.setattr(lift_session, "_approach_and_lift", _fake_approach_and_lift)

    class _Cell:
        row, col = 0, 0
        xy_m = (0.5, 0.1)
        domain_valid = True

        def as_dict(self):
            return {"row": self.row, "col": self.col,
                    "x_m": self.xy_m[0], "y_m": self.xy_m[1]}

    row, detail = _campaign_cell(
        planner=object(), cell=_Cell(), scenario={"scene_cfg": {}}, obj="apple",
        version="v8", hand="inspire", pose_stem="004", options=jl.LiftOptions(),
        candidate_policy=object(), catalogue=catalogue, execution_profile=object())
    assert observed_sizes == [2, 2]
    assert row["status"] == "feasible"
    assert row["min_feasible_rank"] == 2
    assert row["verified_rank_attempt_count"] == 2
    assert row["symmetry_approach_attempt_count"] == 3
    assert row["selected_symmetry_offset_ordinal"] == 1
    assert row["selected_effective_symmetry_ordinal"] == 0
    assert len(detail["attempts"]) == 2


def test_campaign_inference_allows_arm_transfer_but_not_geometry_mismatch():
    contract = {
        "arm": "franka", "hand": "inspire", "grasp_version": "v8",
        "object": "pepsi", "tabletop_pose_stem": "008", "scene": "table",
    }
    transfer = _campaign_transfer_contract(
        contract, target_arm="xarm", hand="inspire", version="v8",
        obj="pepsi", pose_stem="008")
    assert transfer["source_arm"] == "franka"
    assert transfer["target_arm"] == "xarm"
    assert transfer["cross_arm"] is True
    assert "endpoint_ik" in transfer["recomputed_for_target_arm"]

    bad = {**contract, "tabletop_pose_stem": "005"}
    try:
        _campaign_transfer_contract(
            bad, target_arm="xarm", hand="inspire", version="v8",
            obj="pepsi", pose_stem="008")
    except ValueError as exc:
        assert "tabletop_pose_stem" in str(exc)
    else:
        raise AssertionError("non-arm campaign geometry mismatch must be rejected")


def test_viewer_uses_episode_arm_and_keeps_old_episodes_franka(tmp_path):
    xarm_episode = tmp_path / "xarm"
    xarm_episode.mkdir()
    (xarm_episode / "request.json").write_text('{"arm": "xarm"}')
    assert _arm_from_episode(xarm_episode) == "xarm"

    old_episode = tmp_path / "old"
    old_episode.mkdir()
    (old_episode / "request.json").write_text('{"object": "example"}')
    assert _arm_from_episode(old_episode) == "franka"


def test_grid_keeps_outside_cells_but_requires_full_footprint_inside():
    proxy = {
        "vertices_xy_m": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]],
        "center_xy_m": [0.1, 0.1],
    }
    cells = make_grid(
        proxy, footprint_xy=np.array([[-0.04, -0.04], [0.04, -0.04],
                                      [0.04, 0.04], [-0.04, 0.04]]),
        spec=GridSpec(step_m=0.1, domain="footprint-inside", edge_clearance_m=0.0),
    )
    assert len(cells) == 9
    assert sum(cell.domain_valid for cell in cells) == 1
    assert cells[4].domain_valid  # board-centre cell


def test_grid_report_writes_machine_data_and_static_map(tmp_path):
    proxy = {
        "vertices_xy_m": [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]],
    }
    cells = [
        {"row": 0, "col": 0, "x_m": 0.05, "y_m": 0.05, "status": "feasible",
         "failure_code": None, "failure_group": "feasible", "total_s": 0.3,
         "approach_s": 0.2, "jacobian_lift_s": 0.1},
        {"row": 0, "col": 1, "x_m": 0.15, "y_m": 0.05, "status": "failed",
         "failure_code": "jacobian_joint_limit", "failure_group": "Jacobian lift",
         "total_s": 0.4, "approach_s": 0.25, "jacobian_lift_s": 0.15},
        {"row": 1, "col": 0, "x_m": 0.05, "y_m": 0.15, "status": "outside_domain",
         "failure_code": None, "failure_group": "outside domain"},
    ]
    write_cells_csv(tmp_path / "cells.csv", cells)
    write_cells_npz(tmp_path / "cells.npz", cells)
    png, pdf = render_feasibility_map(tmp_path, proxy=proxy, cells=cells,
                                      title="synthetic grid", step_m=0.01)
    summary = summarize_cells(cells)
    assert (tmp_path / "cells.csv").is_file()
    assert (tmp_path / "cells.npz").is_file()
    assert png.is_file() and png.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert summary["feasible_cell_count"] == 1
    assert summary["failure_code_counts"] == {"jacobian_joint_limit": 1}


def test_verified_prefix_report_derives_all_n_maps_without_replanning(tmp_path):
    proxy = {"vertices_xy_m": [[0.0, 0.0], [0.2, 0.0],
                                [0.2, 0.2], [0.0, 0.2]]}
    cells = [
        {"row": 0, "col": 0, "x_m": 0.05, "y_m": 0.05,
         "status": "feasible", "min_feasible_rank": 1,
         "verified_attempt_planning_s": [0.2]},
        {"row": 0, "col": 1, "x_m": 0.15, "y_m": 0.05,
         "status": "feasible", "min_feasible_rank": 2,
         "verified_attempt_planning_s": [0.3, 0.4]},
        {"row": 1, "col": 0, "x_m": 0.05, "y_m": 0.15,
         "status": "failed", "min_feasible_rank": None,
         "verified_attempt_planning_s": [0.1, 0.2]},
    ]
    result = render_verified_prefix_maps(
        tmp_path, proxy=proxy, cells=cells, verified_count=2,
        title="synthetic prefix", step_m=0.05)
    assert result["feasible_fraction_by_verified_count"] == {
        "1": 1 / 3, "2": 2 / 3}
    assert len(result["prefix_map_files"]) == 2
    assert all((tmp_path / path).is_file() for path in result["prefix_map_files"])
    assert (tmp_path / result["summary_map"]).is_file()
    # With N=1 each cell spends only its first candidate time.  With N=2,
    # rank-1 success stops early while the other two cells spend both times.
    assert np.isclose(
        result["online_search_time_by_verified_count"]["1"]["mean_s"], 0.2)
    assert np.isclose(
        result["online_search_time_by_verified_count"]["2"]["mean_s"], 0.4)


def test_grid_candidate_catalogue_keeps_object_frame_wrist_targets(monkeypatch, tmp_path):
    key = ("table", "0", "g0")
    source = {
        "ordered_keys": [list(key)],
        "n_coverage_candidates": 1,
        "n_coverage_useful": 1,
        "n_coverage_zero": 0,
        "remaining_coverage_by_key": {"table/0/g0": 1},
    }
    wrist_obj = np.eye(4)[None]
    wrist_obj[0, 0, 3] = 0.1
    monkeypatch.setattr(lift_session, "_coverage_order", lambda **_kwargs: ([key], source))
    monkeypatch.setattr(
        lift_session, "load_candidate",
        lambda *_args, **_kwargs: (wrist_obj.copy(), np.array([[1.0]], dtype=np.float32),
                                   np.array([[2.0]], dtype=np.float32), [key]), raising=False,
    )
    monkeypatch.setattr(lift_session, "get_cyl_axis_local", lambda _obj: None, raising=False)
    monkeypatch.setattr(lift_session, "get_cyl_yaw_grid", lambda _obj: None, raising=False)
    monkeypatch.setattr(
        lift_session, "_expand_candidates_cyl",
        lambda wrist, pre, grasp, openpose, scene_info, *_args: (wrist, pre, grasp, openpose, scene_info),
        raising=False,
    )
    monkeypatch.setattr(lift_session, "load_openpose_for_candidates", lambda *_args: [None],
                        raising=False)
    policy = make_candidate_policy("clean-state", clean_state_root=tmp_path / "clean")
    catalogue = lift_session.prepare_candidate_catalogue(
        obj="synthetic", version="v8", candidate_hand="inspire", pose_stem="004",
        candidate_policy=policy)
    T_object = np.eye(4)
    T_object[1, 3] = 0.3
    world_wrist = np.matmul(T_object, catalogue.wrist_object)
    assert np.allclose(world_wrist[0, :3, 3], [0.1, 0.3, 0.0])
    assert catalogue.scene_info == [key]


def test_verified_catalogue_passes_success_only_to_candidate_loader(monkeypatch, tmp_path):
    key = ("table", "0", "g0")
    source = {
        "ordered_keys": [list(key)],
        "n_coverage_candidates": 1,
        "n_coverage_useful": 1,
        "n_coverage_zero": 0,
        "remaining_coverage_by_key": {"table/0/g0": 1},
    }
    observed = {}
    monkeypatch.setattr(lift_session, "_coverage_order", lambda **_kwargs: ([key], source))

    def _load(*_args, **kwargs):
        observed.update(kwargs)
        return (np.eye(4)[None], np.array([[1.0]], dtype=np.float32),
                np.array([[2.0]], dtype=np.float32), [key])

    monkeypatch.setattr(lift_session, "load_candidate", _load, raising=False)
    monkeypatch.setattr(lift_session, "get_cyl_axis_local", lambda _obj: None, raising=False)
    monkeypatch.setattr(lift_session, "get_cyl_yaw_grid", lambda _obj: None, raising=False)
    monkeypatch.setattr(
        lift_session, "_expand_candidates_cyl",
        lambda wrist, pre, grasp, openpose, scene_info, *_args: (wrist, pre, grasp, openpose, scene_info),
        raising=False,
    )
    monkeypatch.setattr(lift_session, "load_openpose_for_candidates", lambda *_args: [None],
                        raising=False)
    catalogue = lift_session.prepare_candidate_catalogue(
        obj="synthetic", version="v8", candidate_hand="inspire", pose_stem="004",
        candidate_policy=make_candidate_policy(
            "verified-only", clean_state_root=tmp_path / "empty"))
    assert observed["success_only"] is True
    assert catalogue.source_info["candidate_record_gate"] == "result_json_success_true"
    assert catalogue.source_info["n_candidate_records_after_record_gate"] == 1
