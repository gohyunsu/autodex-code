"""Run the actual FR3 follower with inert controllers, without CUDA or I/O."""

from pathlib import Path
import subprocess
import sys
import textwrap


def test_incomplete_stream_stops_without_final_hand_or_arm_command():
    repo = Path(__file__).resolve().parents[3]
    script = textwrap.dedent("""
        import sys
        import types
        import numpy as np

        # Import the real follower without constructing cuRobo or an FR3 client.
        planner = types.ModuleType("autodex.planner")
        planner.PlanResult = object
        sys.modules["autodex.planner"] = planner
        from src.execution import franka_executor as module
        from precision_insertion.franka_completion_checked import (
            CompletionCheckedFrankaMixin)

        class Checked(CompletionCheckedFrankaMixin, module.FrankaExecutor):
            pass

        module.time.sleep = lambda _seconds: None

        class Arm:
            def __init__(self, wrench=0., stop_response=None):
                self.wrench = wrench
                self.stop_response = stop_response
                self.stops = 0
                self.velocities = []

            def get_data(self):
                return {"qpos": np.zeros(7),
                        "qvel": np.zeros(7),
                        "wrench": np.array([0., 0., self.wrench])}

            def set_joint_velocity(self, velocity, **_kwargs):
                self.velocities.append(np.asarray(velocity).copy())

            def stop_streaming(self):
                self.stops += 1
                return (self.stop_response if self.stop_response is not None
                        else {"type": "success"})

            def is_error(self):
                return False

            def move(self, *_args, **_kwargs):
                raise AssertionError("incomplete stream made a direct arm move")

        def executor(arm, timeout=10., ctrl_dt=.02,
                     executor_type=module.FrankaExecutor):
            instance = object.__new__(executor_type)
            instance.arm = arm
            instance.hand = types.SimpleNamespace(move=lambda _command: None)
            instance._clip_hand = lambda command: command
            instance._holding_object = False
            instance.object_safety_speed_scale = .4
            instance.traj_speed = 1.
            instance.ctrl_dt = ctrl_dt
            instance.traj_dt = .01
            instance.follow_timeout_s = timeout
            instance.follow_log_every_s = 100.
            instance.pos_kp = 4.
            instance.joint_vmax = 1.2
            instance.vel_smooth = .6
            instance.accel_max = 4.
            instance.max_lead = .12
            instance.land_tol = .02
            instance.follow_tol = .04
            instance._ramp_hand = lambda _target: (_ for _ in ()).throw(
                AssertionError("incomplete stream ramped the hand"))
            return instance

        path = np.stack((np.zeros(7), np.full(7, .2)))
        hand = np.zeros((2, 6))

        # A reached endpoint still completes normally.
        arrived = Arm()
        normal = executor(arrived)
        normal._ramp_hand = lambda _target: None
        normal._follow(np.zeros((2, 7)), hand)
        assert arrived.stops == 1

        # Exercise the real precision MRO: the base stop result is enough;
        # measured rest must not send a duplicate stop command.
        checked_arm = Arm()
        checked = executor(checked_arm, executor_type=Checked)
        checked._ramp_hand = lambda _target: None
        checked._follow(np.zeros((2, 7)), hand)
        assert checked_arm.stops == 1

        # A daemon refusal cannot be followed by a hand ramp or landing move.
        rejected = Arm(stop_response={"type": "error", "message": "offline"})
        try:
            executor(rejected)._follow(np.zeros((2, 7)), hand)
        except RuntimeError as exc:
            assert "stream stop failed" in str(exc)
        else:
            raise AssertionError("failed daemon stop was treated as complete")
        assert rejected.stops == 1

        # A deliberate stock wrench stop returns for reverse-axis recovery.
        contact = Arm(wrench=13.)
        stopped = executor(contact)
        stopped._follow(path, stop_wrench_z=12., abort_on_contact=True)
        assert stopped._last_stop_on_contact and contact.stops == 1

        unacknowledged_contact = Arm(
            wrench=13., stop_response={"type": "error", "message": "offline"})
        try:
            executor(unacknowledged_contact)._follow(
                path, stop_wrench_z=12., abort_on_contact=True)
        except RuntimeError as exc:
            assert "stream stop failed" in str(exc)
        else:
            raise AssertionError("unacknowledged force stop could reverse")

        timed_out = Arm()
        try:
            executor(timed_out, timeout=-1.)._follow(path, hand)
        except RuntimeError as exc:
            assert "timeout" in str(exc)
        else:
            raise AssertionError("timed-out stream was treated as complete")
        assert timed_out.stops == 1
        assert np.allclose(timed_out.velocities[-1], 0.)

        frozen = Arm()
        original_advance = module.advance_reference_index
        module.advance_reference_index = lambda _p, _q, index, _step, _lead: index
        try:
            executor(frozen, ctrl_dt=1.)._follow(path, hand)
        except RuntimeError as exc:
            assert "stalled reference" in str(exc)
        else:
            raise AssertionError("stalled stream was treated as complete")
        finally:
            module.advance_reference_index = original_advance
        assert frozen.stops == 1
        assert np.allclose(frozen.velocities[-1], 0.)

        exhausted = Arm()
        module.advance_reference_index = lambda _p, _q, index, _step, _lead: index
        try:
            executor(exhausted, ctrl_dt=.001)._follow(path, hand)
        except RuntimeError as exc:
            assert "tick budget exhausted" in str(exc)
        else:
            raise AssertionError("exhausted stream was treated as complete")
        finally:
            module.advance_reference_index = original_advance
        assert exhausted.stops == 1
        assert np.allclose(exhausted.velocities[-1], 0.)
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=repo,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
