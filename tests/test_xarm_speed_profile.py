import numpy as np

try:
    import trimesh  # noqa: F401 - required by autodex.utils package import
except ModuleNotFoundError:
    import pytest
    pytest.skip("xArm executor tests require trimesh", allow_module_level=True)

from autodex.executor.real import (
    RealExecutor,
    XARM_BASE_SPEED_SCALE,
    XARM_COMMAND_SMOOTHING,
    XARM_CONTINUOUS_PLAYBACK,
    XARM_HELD_SPEED_SCALE,
    XARM_LEGACY_JOINT_STEP_RAD,
    XARM_NEAR_SPEED_SCALE,
    XARM_PLAYBACK_RATE_SMOOTHING,
    XARM_PLACE_CONTACT_STARTUP_BLANK_S,
    XARM_PLACE_CONTACT_SUSTAINED_TICKS,
    XARM_PLACE_CONTACT_THRESHOLD_NM,
    XARM_PROXIMITY_QUERY_PERIOD_S,
    XARM_TRACKING_ERROR_HARD_RAD,
    XARM_TRACKING_ERROR_SOFT_RAD,
)


def test_xarm_defaults_use_double_base_and_legacy_near_held_speed():
    assert XARM_BASE_SPEED_SCALE == 2.0
    assert XARM_NEAR_SPEED_SCALE == 0.5
    assert XARM_HELD_SPEED_SCALE == 0.5
    assert XARM_BASE_SPEED_SCALE * XARM_NEAR_SPEED_SCALE == 1.0
    assert XARM_BASE_SPEED_SCALE * XARM_HELD_SPEED_SCALE == 1.0
    assert XARM_COMMAND_SMOOTHING == 0.80
    assert XARM_CONTINUOUS_PLAYBACK is True
    assert XARM_PLAYBACK_RATE_SMOOTHING == 0.85
    assert XARM_TRACKING_ERROR_SOFT_RAD == 0.08
    assert XARM_TRACKING_ERROR_HARD_RAD == 0.16
    assert XARM_PROXIMITY_QUERY_PERIOD_S == 0.05


def test_xarm_place_contact_defaults_reject_small_motion_residuals():
    assert XARM_PLACE_CONTACT_THRESHOLD_NM == 20.0
    assert XARM_PLACE_CONTACT_SUSTAINED_TICKS == 8
    assert XARM_PLACE_CONTACT_STARTUP_BLANK_S == 0.5


def _profile_executor() -> RealExecutor:
    executor = RealExecutor.__new__(RealExecutor)
    executor.arm_dof = 6
    executor.base_speed_scale = 4.0
    executor.near_speed_scale = 0.25
    executor.held_speed_scale = 0.25
    executor.slowdown_near_m = 0.15
    executor.slowdown_far_m = 0.30
    executor.command_smoothing = 0.80
    executor.continuous_playback = True
    executor._holding_object = False
    executor._speed_profile_object_query = None
    executor._last_speed_profile_band = None
    return executor


def test_xarm_distance_profile_is_linear_between_near_and_far():
    executor = _profile_executor()

    assert np.isclose(executor._approach_speed_scale(0.10), 0.25)
    assert np.isclose(executor._approach_speed_scale(0.15), 0.25)
    assert np.isclose(executor._approach_speed_scale(0.225), 0.625)
    assert np.isclose(executor._approach_speed_scale(0.30), 1.0)
    assert np.isclose(executor._approach_speed_scale(1.00), 1.0)


def test_xarm_held_cap_is_quarter_of_new_base_even_when_far():
    executor = _profile_executor()
    executor._holding_object = True
    executor._hand_mesh_distance = lambda *_args: (1.0, "right_index_2")

    rate, step, band, distance, link = executor._motion_speed(np.zeros(6))

    assert rate == 1.0
    assert step == XARM_LEGACY_JOINT_STEP_RAD
    assert band == "held"
    assert distance is None
    assert link is None


def test_xarm_near_profile_is_quarter_of_new_base():
    executor = _profile_executor()
    executor._hand_mesh_distance = lambda *_args: (0.10, "right_index_2")

    rate, step, band, distance, link = executor._motion_speed(np.zeros(6))

    assert rate == 1.0
    assert step == XARM_LEGACY_JOINT_STEP_RAD
    assert band == "near"
    assert distance == 0.10
    assert link == "right_index_2"


def test_xarm_far_profile_uses_four_times_legacy_rate():
    executor = _profile_executor()
    executor._hand_mesh_distance = lambda *_args: (0.40, "base_link")

    rate, step, band, _, _ = executor._motion_speed(np.zeros(6))

    assert rate == 4.0
    assert step == 4.0 * XARM_LEGACY_JOINT_STEP_RAD
    assert band == "far"


def test_xarm_pending_first_distance_is_conservatively_near():
    executor = _profile_executor()
    executor._speed_profile_object_query = object()
    executor._hand_mesh_distance = lambda *_args: None

    rate, step, band, _, _ = executor._motion_speed(np.zeros(6))

    assert rate == 1.0
    assert step == XARM_LEGACY_JOINT_STEP_RAD
    assert band == "proximity-pending"


def test_xarm_neutral_profile_skips_distance_fk():
    executor = _profile_executor()
    executor.near_speed_scale = 1.0
    executor.held_speed_scale = 1.0
    executor._hand_mesh_distance = lambda *_args: (_ for _ in ()).throw(
        AssertionError("neutral profile must not run distance FK"))

    rate, step, band, distance, link = executor._motion_speed(np.zeros(6))

    assert rate == 4.0
    assert step == 4.0 * XARM_LEGACY_JOINT_STEP_RAD
    assert band == "neutral"
    assert distance is None
    assert link is None


def test_xarm_base_rate_advances_dense_trajectory_four_samples_per_tick():
    executor = _profile_executor()
    executor.dt = 0.0
    executor.command_smoothing = 0.0
    executor._motion_speed = lambda *_args: (4.0, 0.2, "far", 0.4, "base_link")
    executor._log_speed_profile = lambda *_args: None

    class _ImmediateArm:
        def __init__(self):
            self.qpos = np.zeros(6)
            self.moves = []

        def get_data(self):
            return {"qpos": self.qpos.copy()}

        def move(self, qpos, is_servo=True):
            assert is_servo
            self.qpos = np.asarray(qpos, dtype=np.float64).copy()
            self.moves.append(self.qpos.copy())

        def clear_error(self):
            raise AssertionError("instant fake arm must not stall")

    class _Hand:
        def move(self, _target):
            pass

    executor.arm = _ImmediateArm()
    executor.hand = _Hand()
    # Keep the synthetic tracking error below the lag guard so this test
    # isolates trajectory-index rate behavior.
    trajectory = np.repeat((np.arange(9) * 0.005)[:, None], 6, axis=1)

    executor._move_joints(trajectory, threshold=1.0e-8)

    assert len(executor.arm.moves) == 3
    assert np.allclose(executor.arm.moves[-1], trajectory[-1])


def test_xarm_continuous_player_does_not_wait_at_intermediate_waypoints():
    executor = _profile_executor()
    executor.base_speed_scale = 1.0
    executor.dt = 0.0
    executor.command_smoothing = 0.0
    executor._motion_speed = lambda *_args: (1.0, 0.05, "neutral", None, None)
    executor._log_speed_profile = lambda *_args: None

    class _LaggingArm:
        def __init__(self):
            self.qpos = np.zeros(6)
            self.moves = []

        def get_data(self):
            return {"qpos": self.qpos.copy()}

        def move(self, qpos, is_servo=True):
            assert is_servo
            target = np.asarray(qpos, dtype=np.float64).copy()
            self.moves.append(target)
            self.qpos += 0.5 * (target - self.qpos)

        def clear_error(self):
            raise AssertionError("lagging fake arm must not stall")

    class _Hand:
        def move(self, _target):
            pass

    executor.arm = _LaggingArm()
    executor.hand = _Hand()
    trajectory = np.repeat((np.arange(5) * 0.01)[:, None], 6, axis=1)

    executor._move_joints(trajectory, threshold=1.0e-4)

    commanded = np.asarray(executor.arm.moves[:5])[:, 0]
    assert np.allclose(commanded, np.arange(5) * 0.01)
    assert np.linalg.norm(executor.arm.qpos - trajectory[-1]) < 1.0e-4


def test_xarm_continuous_player_smooths_speed_band_changes():
    executor = _profile_executor()
    executor.dt = 0.0
    executor.command_smoothing = 0.0
    requested_rates = iter([2.0, 1.0])
    executor._motion_speed = lambda *_args: (
        (rate := next(requested_rates)),
        XARM_LEGACY_JOINT_STEP_RAD * rate,
        "transition",
        0.2,
        "base_link",
    )
    logged_rates = []
    executor._log_speed_profile = (
        lambda rate, *_args: logged_rates.append(rate))

    class _ImmediateArm:
        def __init__(self):
            self.qpos = np.zeros(6)

        def get_data(self):
            return {"qpos": self.qpos.copy()}

        def move(self, qpos, is_servo=True):
            assert is_servo
            self.qpos = np.asarray(qpos, dtype=np.float64).copy()

        def clear_error(self):
            raise AssertionError("instant fake arm must not stall")

    class _Hand:
        def move(self, _target):
            pass

    executor.arm = _ImmediateArm()
    executor.hand = _Hand()
    trajectory = np.repeat((np.arange(3) * 0.005)[:, None], 6, axis=1)

    executor._move_joints(trajectory, threshold=1.0e-8)

    assert np.allclose(logged_rates, [2.0, 1.85])


def test_xarm_waypoint_wait_fallback_remains_available():
    executor = _profile_executor()
    executor.base_speed_scale = 1.0
    executor.dt = 0.0
    executor.command_smoothing = 0.0
    executor._motion_speed = lambda *_args: (1.0, 0.05, "neutral", None, None)
    executor._log_speed_profile = lambda *_args: None

    class _LaggingArm:
        def __init__(self):
            self.qpos = np.zeros(6)
            self.moves = []

        def get_data(self):
            return {"qpos": self.qpos.copy()}

        def move(self, qpos, is_servo=True):
            assert is_servo
            target = np.asarray(qpos, dtype=np.float64).copy()
            self.moves.append(target)
            self.qpos += 0.5 * (target - self.qpos)

        def clear_error(self):
            raise AssertionError("lagging fake arm must not stall")

    class _Hand:
        def move(self, _target):
            pass

    executor.arm = _LaggingArm()
    executor.hand = _Hand()
    trajectory = np.repeat((np.arange(3) * 0.01)[:, None], 6, axis=1)

    executor._move_joints_waypoint_wait(trajectory, threshold=1.0e-4)

    commanded = np.asarray(executor.arm.moves)[:, 0]
    assert commanded[0] == 0.0
    assert commanded[1] == 0.01
    assert commanded[2] == 0.01
    assert np.linalg.norm(executor.arm.qpos - trajectory[-1]) < 1.0e-4


def test_xarm_continuous_playback_switch_routes_to_fallback():
    executor = RealExecutor.__new__(RealExecutor)
    executor.continuous_playback = False
    calls = []
    executor._move_joints_waypoint_wait = (
        lambda *args, **kwargs: calls.append((args, kwargs)))
    trajectory = np.zeros((2, 6))

    executor._move_joints(trajectory, threshold=0.03)

    assert len(calls) == 1
    assert calls[0][0][0] is trajectory
    assert calls[0][1]["threshold"] == 0.03


def test_xarm_command_smoothing_filters_waypoints_but_reaches_endpoint():
    executor = _profile_executor()
    executor.base_speed_scale = 1.0
    executor.dt = 0.0
    executor.command_smoothing = 0.80
    executor._motion_speed = lambda *_args: (1.0, 0.05, "neutral", None, None)
    executor._log_speed_profile = lambda *_args: None

    class _ImmediateArm:
        def __init__(self):
            self.qpos = np.zeros(6)
            self.moves = []

        def get_data(self):
            return {"qpos": self.qpos.copy()}

        def move(self, qpos, is_servo=True):
            assert is_servo
            self.qpos = np.asarray(qpos, dtype=np.float64).copy()
            self.moves.append(self.qpos.copy())

        def clear_error(self):
            raise AssertionError("instant fake arm must not stall")

    class _Hand:
        def move(self, _target):
            pass

    executor.arm = _ImmediateArm()
    executor.hand = _Hand()
    trajectory = np.repeat(np.array([[0.0], [0.01], [0.02]]), 6, axis=1)

    executor._move_joints(trajectory, threshold=1.0e-4)

    assert np.allclose(executor.arm.moves[1], 0.002)
    assert np.linalg.norm(executor.arm.moves[-1] - trajectory[-1]) < 1.0e-4
