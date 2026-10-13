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
   audit includes the frozen board, table and socket world. The actual
   Cartesian planner mode must match the bound initial preflight and is
   recorded again; an implicit switch between default and experimental
   locked-hand modes is rejected before the replan.
5. With `SurfaceDeviationBounds`, checks that the sampled clearance absorbs
   a commissioned future-trial key/hand surface error. Cylinder mode may
   instead explicitly use `bounds=None` to keep the stock nominal CAD audit
   without claiming a measured physical error margin. Square retains the
   bound requirement.

The bounded passing status is `sampled_measured_chain_pass`; the cylinder
nominal status is `sampled_nominal_measured_chain_pass`. Both are geometric
path results, not grasp success, contact results or motor commands. The latter
saves `sampled_margin_pass: null` and the explicit provenance
`nominal_BODex_relation_without_future_surface_bound`. It cannot be reported
as a verified physical clearance margin. A bound, when supplied, must cover
future squeeze-induced relation shift, slip, calibration, robot tracking and
mesh error; a single reset landing or one camera image does not establish it.
In particular, the reset landing error is the difference between the key's
requested and actual tabletop pose after a failed attempt. The next attempt
captures that key afresh. It does not measure how far any point on the key or
hand may deviate from its planned pose relative to the fixed socket while the
robot carries it.

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

For the current cylinder stock-style nominal trial, pass `bounds=None` in
that call. This changes only the uncertainty-margin claim; the measured
post-squeeze start, exact endpoint, sampled full-mesh collision and complete
axial path checks still run. A failed nominal audit still stops the attempt.

Rejected and passing plans are saved exclusively under
`attempts/<id>/measured_lift_preflights/<index>/`. The report and dense NPZ
can be rechecked with `verify_measured_lift_chain(report_path)`. An altered
pickup log or trajectory fails verification.

The current `FrankaMeasuredLiftAdapter` follows the saved measured-start
lift with the session's completion-checked stock FR3 executor, records the
FR3/Inspire/force trace and verifies a stopped endpoint. The execution
boundary checks the exact local stock daemon binary and running process with
a stream expiry of at most 200 ms for this first-party adapter; external
adapters still need a reviewed watchdog record. The installed ParaDex source
and binary contain a default 200 ms stream-expiry watchdog, but inspection is
not a physical timeout or E-stop test. No daemon process was visible in this
session's offline check. The 20 mm cylinder contact stroke uses the
separate stock 12 N force-stop adapter and reverse-axis ascent. A positive
paired-camera VLM lift label is evaluated only after a completed lift record.
