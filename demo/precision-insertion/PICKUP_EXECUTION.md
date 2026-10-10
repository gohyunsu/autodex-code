# Demo-local AutoDex pickup command boundary

`precision_insertion.pickup_execution.execute_bound_pickup` is the first
physical-motion adapter for this demo. It reuses the original v8
`FrankaExecutor.execute` for **approach → pregrasp → grasp → squeeze**, with
`skip_lift=True` and `start_from_current=True`. It does not modify stock
AutoDex code or implement a separate grasp controller.

The function refuses to call the executor unless:

- A fresh `SessionRunner` attempt has one passing, hash-verified trial
  preflight and the selected approach, pregrasp, grasp and wrist target still
  match its saved NPZ bytes. Older preflight bundles must be regenerated.
- The attempt is still awaiting lift observation, has no physical-stage
  event, and its saved candidate/session/start-state binding matches.
- Immediately sampled **measured** FR3/Inspire feedback is stationary,
  synchronous, recent, and within commissioned joint tolerance of the v8
  approach's first waypoint. `FrankaExecutor.get_hand_qpos()` is not used:
  it reports a commanded nominal hand pose.
- The caller opts into physical motion and a separately commissioned live
  hardware/intervention interlock returns `True`.

After the stock executor returns, fresh measured feedback must match the
approach arm endpoint and final raw Inspire squeeze command. Before motor
invocation it saves `attempts/<id>/pickup_started.json` exclusively; an
orphaned marker blocks an automatic re-command after a process crash. It then
saves `pickup_execution.json` exclusively. An exception after
command onset is saved as `execution_or_feedback_failed`, without asserting
that the arm is safe or changing the attempt's grasp label. A completed
squeeze is only `squeeze_command_and_feedback_complete`; the lift and paired
camera checkpoint must establish `grasp_success` separately.

Integration outline after real camera timestamps, asset QA and robot safety
commissioning (not a runnable command for this workstation):

```python
from precision_insertion.pickup_execution import (
    PickupExecutionLimits, execute_bound_pickup,
)

# runner.begin_selected_attempt(...) has already saved its binding.
# read_measured_state() must return LiveRobotState from the actual controllers.
result = execute_bound_pickup(
    runner=runner, executor=franka_executor, planner=v8_planner,
    pre_state=read_measured_state(),
    read_post_state=read_measured_state,
    limits=commissioned_pickup_limits,  # PickupExecutionLimits
    motion_interlock=commissioned_hardware_interlock,
    enable_robot_motion=True,
)
```

The default `enable_robot_motion=False` cannot send commands. The interlock
callback must be supplied by the robot PC; a test lambda is **not** a safety
interlock. This code has only fake-executor tests, no Franka hardware test.
Nothing here commissions an E-stop, workspace, hand-eye calibration or force
limits. Do not deploy it until those external gates are implemented and
reviewed.

The stock `execute(lift_traj_override=...)` must **not** receive the demo's
separately replanned nominal lift. The stock start check refers to its own
candidate lift model, whereas squeeze changes the hand. The next physical
stage must read the achieved squeeze state, re-screen the held geometry,
plan a lift from that exact state, verify the matching start again, and then
perform an observed lift checkpoint. Transfer and guarded 20 mm insertion
need separate adapters; `follow_joint_trajectory` alone is not a contact
controller.
