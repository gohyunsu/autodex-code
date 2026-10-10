# Measured post-squeeze lift and full-chain replan

The initial v8 trial estimates the key–hand relation and the held Inspire
pose before the robot moves. The actual squeeze can change the finger joints
and shift the key. Consequently the nominal `lift_trajectory` saved with
that trial must **not** be replayed after pickup.

After `execute_bound_pickup` saves a successful *command-and-feedback* record,
sample fresh stationary FR3/Inspire feedback and call
`SessionRunner.prepare_measured_lift_chain(...)`. This read-only stage:

1. Rechecks the pickup command-start marker, execution log and original
   hash-bound v8 trial; rejects changed or unrelated inputs.
2. Requires a fresh measured 13-DOF state whose arm and raw Inspire joints
   remain within commissioned drift limits of the post-squeeze measurement.
3. Re-screens the exact 20 mm key/socket endpoint with the **measured** finger
   joints, the full key/hand mesh and the nominal BODex key–hand transform.
4. Calls the unchanged `GraspPlanner.plan_lift_preflight` from those measured
   joints, then the demo's existing `plan_held_transfer_and_axial` for lift →
   pre-insertion hold → centered 20 mm insertion. Its sampled full-key/hand
   audit includes the frozen board, table and socket world.
5. Requires explicit `SurfaceDeviationBounds` covering future key/hand
   surface error relative to the fixture. If nominal clearance cannot absorb
   these commissioned bounds, the result is rejected.

Passing status is `sampled_measured_chain_pass`; it is **not** grasp success,
a contact-control result or a motor command. The bound is not learned from
one simulated grasp or one camera image. It must cover the full future-trial
error budget, including squeeze-induced relation shift and any slip. Without
such physical commissioning, a positive geometric result cannot be promoted
to a real-robot launch.

From the demo Python environment, after one selected attempt and the pickup
execution record:

```python
from precision_insertion.uncertainty_margin import SurfaceDeviationBounds

result = runner.prepare_measured_lift_chain(
    planner=v8_planner,
    pickup_execution_log=attempt_dir / "pickup_execution.json",
    joint_sample=read_measured_state(),
    bounds=SurfaceDeviationBounds(
        key_surface_m=commissioned_key_error_m,
        hand_surface_m=commissioned_hand_error_m,
        source="commissioned_future_trial_surface_bound",
    ),
    limits=commissioned_path_audit_limits,
    max_state_age_s=commissioned_state_age_s,
    max_post_squeeze_arm_drift_rad=commissioned_arm_drift_rad,
    max_post_squeeze_hand_drift_raw=commissioned_hand_drift_raw,
    max_arm_hand_skew_s=commissioned_feedback_skew_s,
    max_hand_command_error_raw=commissioned_hand_tracking_error_raw,
    max_arm_velocity_rad_s=commissioned_hold_velocity_rad_s,
    axial_waypoint_step_m=0.002,
)
```

Rejected and passing plans are saved exclusively under
`attempts/<id>/measured_lift_preflights/<index>/`. The report and dense NPZ
can be rechecked with `verify_measured_lift_chain(report_path)`. An altered
pickup log or trajectory fails verification.

There is deliberately **no** lift motor call yet. Stock
`FrankaExecutor.execute_lift()` calls `follow_joint_trajectory()` without
`abort_on_contact`; its `_follow()` may recover a collision reflex. Even
calling `_follow(..., abort_on_contact=True)` is insufficient as-is: its
timeout/stall branches fall through to a blocking final landing move. A
demo-local monitored lift controller must stop and remain stopped on reflex,
force threshold, timeout or stall, verify actual endpoint/hand state, and
write a fresh `precision_insertion_lift_execution_v1` log. Only then can the
existing paired-camera VLM lift checkpoint label `grasp_success`. The same
execution discipline is needed for transfer; the 20 mm contact stroke needs
a dedicated guarded controller, not free-space trajectory playback.
