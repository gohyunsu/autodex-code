"""Source-level regression tests for the GPU-only native planning policy.

The base test environment intentionally has no CUDA device, while importing
the vendored cuRobo MotionGen initializes CUDA at module-definition time.
These tests therefore check the routing and fault-containment contract without
importing the GPU planner.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_zero_hold_mask_is_not_sent_to_the_constrained_api():
    planner = _source("autodex/planner/planner.py")
    assert "requires at least one held axis" in planner
    assert "def plan_cartesian_pose(" in planner
    assert "start_full_qpos, target_wrist_pose, None," in planner

    # A pre-place transfer has a free intermediate path.  It must not install
    # a PoseCostMetric with six zero weights merely to reach its endpoint.
    for relative in (
        "src/execution/franka_executor.py",
        "src/experiment/reset/reorient.py",
        "src/execution/paper_lift_run.py",
    ):
        source = _source(relative)
        assert "hold_vec_weight=[0, 0, 0, 0, 0, 0]" not in source
        assert "plan_cartesian_pose(" in source


def test_native_pose_and_lock_joint_route_is_explicit_opt_in():
    planner = _source("autodex/planner/planner.py")

    assert 'os.environ.get("AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS") == "1"' in planner
    assert "if not self._native_pose_constraints_enabled:" in planner
    assert "def _plan_endpoint_approximation(" in planner
    assert 'constraint_mode="endpoint_approximation"' in planner


def test_vertical_policy_is_planner_owned_and_uses_jacobian_continuation():
    planner = _source("autodex/planner/planner.py")
    stroke = _source("autodex/planner/jacobian_stroke.py")
    reorient = _source("src/experiment/reset/reorient.py")
    franka = _source("src/execution/franka_executor.py")
    xarm = _source("autodex/executor/real.py")

    assert "def plan_vertical_stroke(" in planner
    vertical_start = planner.index("    def plan_vertical_stroke(")
    vertical_end = planner.index("    def _plan_endpoint_approximation(", vertical_start)
    vertical = planner[vertical_start:vertical_end]
    assert "plan_jacobian_vertical_stroke(" in vertical
    assert "self.plan_pose_constrained(" not in vertical
    assert "def plan_jacobian_vertical_stroke(" in stroke
    assert "direction = \"+Z\" if sign > 0.0 else \"-Z\"" in stroke
    assert "jacobian_execution_nonmonotonic_z" in stroke
    assert "_plan_franka_verified_vertical_stroke" not in reorient
    assert "_franka_fk_xyz" not in reorient
    assert "planner.plan_vertical_stroke(" in reorient
    assert "planner.plan_vertical_stroke(" in franka
    assert "planner.plan_vertical_stroke(" in xarm
    assert 'label="xarm post-release reset clearance"' in xarm


def test_early_place_contact_releases_then_requires_vertical_reset():
    runner = _source("src/execution/run_auto.py")
    branch_start = runner.index("    if _early_contact:")
    branch_end = runner.index("    # ── 5. Label", branch_start)
    branch = runner[branch_start:branch_end]

    assert branch.index("executor.release(result)") < branch.index(
        "executor.reset(result, planner, scene_cfg)")
    assert "executor.reset_hybrid(" not in branch
    assert "executor.reset_fallback(" not in branch
    assert '"manual_recovery_required": recovery_error is not None' in branch


def test_dynamic_hand_lock_does_not_use_cuda_graph_capture():
    planner = _source("autodex/planner/planner.py")
    hand_lock_start = planner.index("def _locked_motion_gen_for_hand(")
    hand_lock_end = planner.index("    def _set_ik_world(", hand_lock_start)
    hand_lock = planner[hand_lock_start:hand_lock_end]
    assert "use_cuda_graph=False" in hand_lock
    assert '"cuda_graph": "disabled"' in planner


def test_cuda_planning_fault_is_not_downgraded_to_a_candidate_miss():
    planner = _source("autodex/planner/planner.py")
    reorient = _source("src/experiment/reset/reorient.py")
    runner = _source("src/execution/run_auto.py")

    assert "class CudaPlanningFault(RuntimeError)" in planner
    assert "if cuda_fault is not None:\n            raise cuda_fault" in planner
    assert "skipped_after_cuda_fault" in planner
    assert "except CudaPlanningFault:" in reorient
    assert "fatal_cuda_planning_fault" in runner
    assert "CUDA planning fault: stopping this process" in runner
    assert "inprocess_reorient_handler" in runner


def test_reorient_direct_batch_ik_uses_the_cuda_fault_boundary():
    reorient = _source("src/experiment/reset/reorient.py")

    assert "def _solve_reset_ik_batch(" in reorient
    assert "torch.cuda.synchronize(device=planner._tensor_args.device)" in reorient
    assert "raise_cuda_planning_fault(" in reorient
    assert "operation=\"reorient_grasp_seed_ik\"" in reorient
    assert "operation=\"reorient_lift_seed_ik\"" in reorient
