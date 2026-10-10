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
