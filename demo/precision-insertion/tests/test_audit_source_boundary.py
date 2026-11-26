from pathlib import Path
import importlib.util


SCRIPT = (Path(__file__).resolve().parents[1] /
          "audit_source_boundary.py")
SPEC = importlib.util.spec_from_file_location("audit_source_boundary", SCRIPT)
audit_source_boundary = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_source_boundary)


def _policy():
    return {
        "schema": "precision_insertion_source_boundary_v1",
        "baseline_commit": "0" * 40,
        "stock_entrypoints": ["src/execution/run_pipeline.py"],
        "shared_runtime_allowlist": ["src/execution/franka_executor.py"],
        "known_inactive_branch_paths": ["autodex/old.py"],
        "known_inactive_branch_prefixes": ["autodex/tasks/"],
        "shared_runtime_prefixes": ["autodex/", "src/execution/"],
    }


def test_reviewed_adapter_and_inactive_history_pass():
    report = audit_source_boundary.classify_changes([
        "src/execution/franka_executor.py",
        "autodex/tasks/base.py",
        "autodex/old.py",
    ], _policy())
    assert report["passed"] is True
    assert report["reviewed_adapter_changes"] == [
        "src/execution/franka_executor.py"]
    assert report["known_inactive_branch_changes"] == [
        "autodex/old.py", "autodex/tasks/base.py"]


def test_stock_entrypoint_change_fails():
    report = audit_source_boundary.classify_changes(
        ["src/execution/run_pipeline.py"], _policy())
    assert report["passed"] is False
    assert report["stock_entrypoint_changes"] == [
        "src/execution/run_pipeline.py"]


def test_new_shared_runtime_edit_fails_without_affecting_demo_files():
    report = audit_source_boundary.classify_changes([
        "autodex/new_runtime_policy.py",
        "demo/precision-insertion/new_task_policy.py",
    ], _policy())
    assert report["passed"] is False
    assert report["unexpected_shared_runtime_changes"] == [
        "autodex/new_runtime_policy.py"]


def test_versioned_policy_declares_dual_end_effector_assets():
    policy = audit_source_boundary.load_policy(
        SCRIPT.parent / "config" / "source_boundary.json")
    allowed = set(policy["shared_runtime_allowlist"])
    assert {
        "autodex/planner/src/curobo/content/assets/robot/"
        "fr3_gripper_description/fr3_gripper.urdf",
        "autodex/planner/src/curobo/content/assets/robot/"
        "fr3_gripper_description/franka_gripper_floating.urdf",
        "autodex/planner/src/curobo/content/configs/robot/fr3_gripper.yml",
        "autodex/planner/src/curobo/content/configs/robot/"
        "franka_gripper_floating.yml",
        "autodex/planner/src/curobo/content/configs/robot/spheres/"
        "fr3_gripper.yml",
        "autodex/utils/robot_config.py",
    } <= allowed
