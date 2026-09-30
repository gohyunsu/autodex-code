from types import SimpleNamespace

import numpy as np

try:
    import trimesh  # noqa: F401 - required by autodex.utils package import
except ModuleNotFoundError:
    import pytest
    pytest.skip("xArm executor tests require trimesh", allow_module_level=True)

import autodex.executor.real as real_module
from autodex.executor.real import (
    RealExecutor,
    XARM_PREPLACE_REUSE_POS_TOL_M,
    XARM_PREPLACE_REUSE_ROT_TOL_RAD,
    XARM_VERTICAL_STROKE_Z_TOL_M,
)
from autodex.executor.timing import new_place_timing


class _FakeArm:
    def __init__(self, qpos, position):
        self.qpos = np.asarray(qpos, dtype=np.float64)
        self.position = np.asarray(position, dtype=np.float64)
        self.arm = object()

    def get_data(self):
        return {
            "qpos": self.qpos.copy(),
            "position": self.position.copy(),
        }


class _FakeMonitor:
    def __init__(self, *_args, **_kwargs):
        pass

    def warmup(self, seconds):
        assert seconds == 1.0

    def tick(self):
        return False


def _pose(x=0.4, y=0.0, z=0.3):
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = [x, y, z]
    return result


def _plan_result():
    return SimpleNamespace(
        success=True,
        grasp_pose=np.zeros(6, dtype=np.float32),
        wrist_se3=np.eye(4, dtype=np.float64),
    )


def _scene_cfg():
    return {"mesh": {"target": {
        "pose": [0.5, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0],
    }}}


def _executor(arm):
    executor = RealExecutor.__new__(RealExecutor)
    executor.arm_dof = 6
    executor.arm = arm
    executor._link6_to_wrist = np.eye(4, dtype=np.float64)
    executor._move_joints = lambda *_args, **_kwargs: None
    return executor


def test_xarm_place_thresholds_match_franka_alignment_policy():
    assert XARM_PREPLACE_REUSE_POS_TOL_M == 0.005
    assert np.isclose(XARM_PREPLACE_REUSE_ROT_TOL_RAD, np.deg2rad(3.0))
    assert XARM_VERTICAL_STROKE_Z_TOL_M == 0.005


def test_xarm_reuses_reposition_despite_2mm_controller_fk_residual(monkeypatch):
    monkeypatch.setattr(real_module, "ContactMonitor", _FakeMonitor)

    high = _pose()
    low = high.copy()
    low[2, 3] -= 0.1
    # Reproduce the reported failure: physical Cartesian pose is 2.234 mm
    # away from planner FK, but remains inside the arm-level 5 mm gate.
    physical = high.copy()
    physical[0, 3] += 0.002234
    arm = _FakeArm(np.zeros(6), physical)
    executor = _executor(arm)
    executed_corrections = []
    executor.follow_joint_trajectory = executed_corrections.append

    preplace = np.zeros((2, 12), dtype=np.float32)

    class _Planner:
        def __init__(self):
            self.vertical_call = None

        def fk_wrist(self, _qpos):
            return high.copy()

        def plan_cartesian_pose(self, *_args, **_kwargs):
            raise AssertionError("valid reposition endpoint must be reused")

        def plan_vertical_stroke(self, *args, **kwargs):
            self.vertical_call = (args, kwargs)
            return np.zeros((2, 12), dtype=np.float32)

    planner = _Planner()
    timing = new_place_timing()
    result = executor._place_planned(
        _plan_result(), planner, _scene_cfg(),
        lift_height=0.1, overshoot=0.0, mcc_model_path="unused.pt",
        timing_s=timing, placement_wrist=low,
        preplace_traj=preplace, preplace_wrist_target=high)

    args, kwargs = planner.vertical_call
    assert np.allclose(args[0], preplace[-1])
    assert np.allclose(args[1], high)
    assert np.allclose(args[2], low)
    assert kwargs["travel_tolerance_m"] == 0.005
    assert executed_corrections == []
    assert result["preplace_reused"] is True
    assert result["preplace_plan_source"] == "runner_reposition"
    assert np.isclose(timing["preplace_model_pos_err_m"], 0.002234)


def test_xarm_replans_preplace_when_live_endpoint_exceeds_5mm(monkeypatch):
    monkeypatch.setattr(real_module, "ContactMonitor", _FakeMonitor)

    high = _pose()
    low = high.copy()
    low[2, 3] -= 0.1
    physical = high.copy()
    physical[0, 3] += 0.010
    arm = _FakeArm(np.zeros(6), physical)
    executor = _executor(arm)
    correction = np.zeros((2, 12), dtype=np.float32)
    correction[-1, :6] = 0.5
    executed_corrections = []

    def _execute_correction(traj):
        executed_corrections.append(np.asarray(traj).copy())
        arm.qpos = np.asarray(traj[-1, :6], dtype=np.float64)
        arm.position = high.copy()

    executor.follow_joint_trajectory = _execute_correction

    class _Planner:
        def __init__(self):
            self.correction_call = None
            self.vertical_call = None

        def fk_wrist(self, qpos):
            if np.allclose(np.asarray(qpos)[:6], 0.5):
                return high.copy()
            return physical.copy()

        def plan_cartesian_pose(self, *args, **kwargs):
            self.correction_call = (args, kwargs)
            return correction.copy()

        def plan_vertical_stroke(self, *args, **kwargs):
            self.vertical_call = (args, kwargs)
            return np.zeros((2, 12), dtype=np.float32)

    planner = _Planner()
    timing = new_place_timing()
    result = executor._place_planned(
        _plan_result(), planner, _scene_cfg(),
        lift_height=0.1, overshoot=0.0, mcc_model_path="unused.pt",
        timing_s=timing, placement_wrist=low,
        preplace_traj=np.zeros((2, 12), dtype=np.float32),
        preplace_wrist_target=high)

    assert planner.correction_call is not None
    assert len(executed_corrections) == 1
    assert np.allclose(executed_corrections[0], correction)
    assert np.allclose(planner.vertical_call[0][0][:6], correction[-1, :6])
    assert result["preplace_reused"] is False
    assert result["preplace_plan_source"] == "live_correction"
    assert timing["preplace_live_pos_err_m"] == 0.01
    assert timing["preplace_final_pos_err_m"] == 0.0
