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
new measured arrival/visual observation after that motion, and only then create a separate
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
`insertion_success`.

After the transfer, save a new same-request, undistorted AutoDex capture via
`write_raw_camera_capture(..., phase="preinsert")`. Its request and per-view
frame IDs must advance beyond the earlier post-lateral-hold capture, and its
exposure bounds must follow the completed transfer. Then call
`SessionRunner.assess_postshift_transfer_arrival(...)` with the same passing
replan/checkpoint/shift, the handoff and transfer log, fresh stationary
Franka/Inspire feedback, the frozen camera rig, a selected VLM backend, and
commissioned timing/visual-error limits. It triangulates the cylinder tip
and projected axis again and checks observed tip/axis against the measured
hand and frozen socket. The report is saved in
`postshift_arrival_checkpoints/NNN/report.json`; possible statuses include
`visual_alignment_within_budget`, `residual_requires_new_shift`,
`held_relation_inconsistent`, and `visual_abstain`. Its
`axial_retry_allowed=false` even on visual alignment: fresh visual evidence
is necessary, but does not substitute for a guarded-contact handoff,
independent force/depth evidence, or physical success validation.
An estimator response of `continuous_xy_correction_not_confident` may also
be accepted as aligned **only** when it contains independently triangulated
tip/axis inliers and their observed rim/depth residuals plus uncertainty fit
the commissioned budget: an already centered key need not have a beneficial
next XY step.

## Fresh-arrival axial replan (read-only)

A positive arrival checkpoint is **not** permission to use the axial path
planned before transfer. The transfer may have changed the key/hand relation.
Use the same session runner to reconstruct that relation from the newly
observed tip/axis and measured wrist, then rerun the exact 20 mm endpoint,
cuRobo axial waypoints and sampled held-key/hand margin checks:

```python
replan = runner.prepare_postshift_arrival_axial_replan(
    planner=fr3_inspire_planner,
    previous=postshift_preflight,
    previous_report_path=postshift_20mm_report,
    arrival=arrival_checkpoint,
    arrival_report_path=arrival_report,
    checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan,
    bounds=commissioned_future_surface_bounds,
    max_visual_tip_error_m=commissioned_tip_worst_case_m,
    max_visual_axis_error_deg=commissioned_axis_worst_case_deg,
    max_axis_prior_residual_deg=commissioned_prior_axis_limit_deg,
)
```

`sampled_arrival_20mm_axial_preflight_pass` means the measured arrival was
already at the newly reconstructed pre-insertion hand target and its **new**
axial path passed sampled geometry. A mismatch returns
`arrival_hold_goal_residual`; it does not quietly plan or execute
another transfer. The report and `planned_axial.npz` are saved under
`postshift_arrival_axial_replans/NNN/`. The archive contains no transfer
command; the two identical transfer samples in the internal audit mean a
stationary hold. The saved report binds both earlier preflight and new
arrival source hashes, checks CAD/candidate inputs and stores the new
tip/axis-derived yaw gauge. It explicitly marks the old axial path
non-reusable and `axial_contact_authorized=false`.

All error bounds above must come from physical commissioning. This remains
a read-only preflight with synthetic-unit-test coverage, not a contact
command or a physical 20 mm success observation. Do not
feed either old or newly planned axial path to the stock trajectory follower
as a contact insertion command.

After a passing replan, explicitly record the **completed and reobserved**
continuous XY shift as one pending retry:

```python
pending = runner.record_grounded_xy_retry(
    replan=replan,
    replan_report_path=arrival_axial_report,
    previous=postshift_preflight,
    arrival=arrival_checkpoint,
    checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan,
    shift_plan_report_path=lateral_preflight_report,
    timestamp_s=decision_time_after_arrival,
)
```

This verifies the saved lateral diagnostic, withdrawal/shift sources, transfer
arrival and fresh axial replan before writing an append-only `xy_retry` event.
Unlike the older four-cardinal-choice policy, the socket-frame increment may
point in any XY direction but must be nonzero and at most 1 mm. The attempt
then awaits a **new** guarded execution and a fresh post-choice camera pair;
the previous insertion verdict remains false until new evidence is recorded.
Recording a pending retry is not a motion command or insertion-success label.
Now bind that pending event to its **new** axial-only archive and a second
measured, stationary arm/hand sample:

```python
retry_packet = runner.prepare_retry_axial_handoff(
    replan=replan,
    replan_report_path=arrival_axial_report,
    previous=postshift_preflight,
    arrival=arrival_checkpoint,
    checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan,
    measured_start=fresh_stationary_robot_state,
    decision_timestamp_s=decision_time_after_state,
    max_state_age_s=commissioned_state_age,
    max_arrival_age_s=commissioned_arrival_age,
    max_start_joint_error_rad=commissioned_joint_error,
    max_hand_drift_raw=commissioned_hand_drift,
    max_arm_hand_skew_s=commissioned_arm_hand_skew,
    max_hand_command_error_raw=commissioned_hand_tracking_error,
    max_arm_velocity_rad_s=commissioned_stationary_velocity,
)
```

This writes an exclusive
`retry_guarded_axial_handoffs/NNN/report.json`. It re-verifies the fresh
arrival/CAD/path source chain, checks that the archive contains **only** axial
samples starting at the measured arrival, and binds the latest append-only
`grounded_continuous_xy` event and current measured 13-DOF state. A stale
arrival, changed archive, different candidate or earlier/duplicate event
cannot be reused. `verify_retry_axial_handoff(...)` replays these checks
without querying the VLM or moving the robot.

The packet deliberately says `robot_ready=false` and
`read_only_grounded_retry_axial_packet_not_contact_permission`. Its saved
joint sample becomes stale; the contact boundary must recheck fresh
feedback, a command-time deadline, force/depth limits, watchdog and interlock
before *any* stroke. The first-attempt guarded executor and v2 insertion
checkpoint still reject this retry source. Fresh post-contact observations
and physical key-depth admission remain to be commissioned and integrated.
Neither a passing packet nor a VLM label is evidence that the key entered
20 mm.

## Retry contact metric contract (read-only)

An external, separately commissioned guarded controller must emit a
`precision_insertion_guarded_retry_execution_v1` metric record. It references
the **retry** handoff and its exact `planned_axial.npz` digest, stroke start/end
times, five measurement fields (`key_depth_interval_m`,
`key_depth_source`, `alignment_within_limits`, `safety_abort`,
`grasp_held`), the exact contact limits and four hashed producer records:
`key_depth`, `alignment`, `force_trace`, `grasp_state`. Its force trace must
be replayable v2 and bind the same handoff/archive digests. Use the existing
`precision_insertion_external_metric_claim_v1` format for the other three
source claims; each claim references its own unchanged raw evidence bytes.

```python
metric, started, completed = verify_retry_guarded_metric(
    retry_metric_path,
    handoff_report_path=retry_packet,
    expected=replan,
    previous=postshift_preflight,
    arrival=arrival_checkpoint,
    checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan,
    mode=runner.mode,
    shared_root=runner.shared_root,
    calibration=runner.calibration,
)
```

Import this from `precision_insertion.retry_guarded_metric`. The verifier
checks file hashes, capture/source identity, event timing, trace replay and
the new path binding. It does **not** authenticate the external sensor or
admit its claimed key-depth interval as a task-success measurement.

## Opt-in retry contact boundary

`execute_bound_retry_guarded_insertion(...)` accepts the same independently
commissioned `follow_guarded_insertion` adapter contract as the first stroke,
but requires the **new** retry packet, pending continuous XY event and fresh
stationary robot feedback. It reuses watchdog and contact-controller
commissioning checks; it defaults to `enable_robot_motion=False` and will
not request motion without an explicit live interlock. The adapter must
enforce its own robot-side watchdog, force/contact and grasp-loss stops.
No production adapter or reviewed commissioning record is bundled here.

```python
from precision_insertion.retry_guarded_execution import (
    execute_bound_retry_guarded_insertion,
)

execution = execute_bound_retry_guarded_insertion(
    runner=runner, expected=replan, previous=postshift_preflight,
    arrival=arrival_checkpoint, checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan, handoff_report_path=retry_packet,
    adapter=commissioned_contact_adapter,
    pre_state=fresh_robot_state, read_post_state=read_fresh_robot_state,
    limits=commissioned_execution_limits,
    contact_limits=commissioned_contact_limits,
    max_handoff_age_s=commissioned_handoff_age,
    watchdog_commissioning_path=reviewed_watchdog_record,
    contact_commissioning_path=reviewed_contact_record,
    motion_interlock=operator_and_robot_interlock,
    enable_robot_motion=True,
)
```

This writes `retry_guarded_executions/NNN/started.json` before calling the
adapter and either an `execution.json` or a latched `failure.json`. A
controller/metric/feedback error requests `stop_and_acknowledge()` and
requires supervised recovery. `verify_retry_guarded_execution(...)` replays
the saved path, state, timeline, controller output, metric sources and
commissioning references without a robot call. Even a complete external
stroke does **not** change `insertion_success`. Synthetic fake-adapter tests
are not a contact commissioning test.

## Fresh retry insertion observation

After the controller holds or aborts, obtain two **new** same-request raw
AutoDex captures: one shortly before the stroke, after the retry handoff,
and one after contact. Reusing the transfer-arrival frames is rejected. The
request/frame IDs must advance, camera exposure intervals must bracket this
stroke, and paired views must be among the arrival's grounded inlier cameras.

```python
report = runner.prepare_observed_retry_insertion_label(
    replan=replan, previous=postshift_preflight,
    arrival=arrival_checkpoint, checkpoint=postshift_checkpoint,
    shift_plan=completed_shift_plan,
    handoff_report_path=retry_packet,
    execution_log_path=retry_execution_log,
    preinsert_bundle=new_raw_precontact_capture,
    final_bundle=new_raw_final_or_abort_capture,
    backend=selected_vlm_backend,
    decision_timestamp_s=decision_time_after_final_images,
    max_phase_skew_s=commissioned_camera_skew,
    max_preinsert_age_s=commissioned_precontact_age,
    max_final_observation_gap_s=commissioned_final_gap,
)
```

`verify_retry_insertion_checkpoint(...)` replays the saved VLM answer against
the same PNG bytes and the exact execution/metric sources without another
model call. The first-attempt `prepare_observed_insertion_label(...)` now
rejects **all** retry records, including the formerly unbound v1 metric.
The new attempt event can be `false` on a supported visible jam/slip or a
sensor abort, or `unknown` on an ambiguous/normal-looking result. A claimed
external key-depth interval is intentionally **not** admitted; therefore
this path cannot yet record `insertion_success=true`. Independently
commissioned physical key-depth admission is still needed for that label.
