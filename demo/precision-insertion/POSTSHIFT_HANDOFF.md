# Post-shift transfer/axial handoff

`SessionRunner.prepare_postshift_path_handoff(...)` saves one evidence packet
after the VLM-guided shift, fresh post-shift multi-view checkpoint, and
passing sampled 20 mm replan. It does not command Franka/Inspire.

```python
path = runner.prepare_postshift_path_handoff(
    preflight=postshift_preflight,
    preflight_report_path=postshift_20mm_report,
    checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan,
    measured_start=fresh_stationary_robot_state,
    decision_timestamp_s=decision_time,
    max_state_age_s=commissioned_state_age,
    max_start_joint_error_rad=commissioned_joint_error,
    max_hand_drift_raw=commissioned_hand_drift,
    max_arm_hand_skew_s=commissioned_arm_hand_skew,
    max_hand_command_error_raw=commissioned_hand_tracking_error,
    max_arm_velocity_rad_s=commissioned_stationary_velocity,
)
```

The report binds the exact `planned_trajectories.npz` bytes, post-shift
checkpoint, original physical-grasp medoid, selected candidate, frozen
session calibration, measured 13-DOF start, unchanged Inspire pose, and
the 20 mm axial target. `transfer_required` is true when the saved transfer
changes joints. The axial path starts at the **end of the transfer**, not
at the shifted hold. A future executor must verify this packet again at
command time, execute and verify the transfer first if required, acquire a
new measured arrival/visual observation, and only then create a separate
contact-controller handoff. Reusing the first centered-path guarded handoff
or playing the axial path directly from the shifted hold is invalid.

The packet's `robot_ready=false` is intentional. It proves source
consistency and sampled planning, not continuous collision safety, robot
watchdog/contact commissioning, actual key penetration, or task success.
Unit tests use synthetic paths and limit values; they are not hardware
commissioning data.

For a packet with `transfer_required=true`, the opt-in
`execute_bound_postshift_transfer(...)` boundary can invoke an externally
commissioned `follow_transfer` adapter. It replays the packet and exact path,
checks fresh measured arm/hand state, reviewed watchdog binary, a live
motion interlock and a command-time expiration limit. It defaults to
`enable_robot_motion=False`. The adapter must enforce its own robot-side
watchdog, grip-loss and contact stops; the Python checks cannot replace them.

```python
log = execute_bound_postshift_transfer(
    runner=runner, expected=postshift_preflight,
    checkpoint=postshift_checkpoint, shift_plan=completed_shift_plan,
    handoff_report_path=path, adapter=commissioned_transfer_adapter,
    pre_state=fresh_state, read_post_state=read_fresh_state,
    limits=commissioned_execution_limits,
    max_handoff_age_s=commissioned_handoff_age,
    commissioning_record_path=reviewed_watchdog_record,
    motion_interlock=operator_and_robot_interlock,
    enable_robot_motion=True,
)
```

The success log is saved at
`postshift_transfer_executions/NNN/execution.json` and can be replayed with
`verify_postshift_transfer_execution(...)`. It records controller completion
and terminal feedback only. A controller error latches `failure.json` and
requires supervised recovery. Neither log sets `preinsert_reached` or
`insertion_success`: a new camera/robot observation at the transfer endpoint
is still required before any guarded 20 mm retry.
