# Observed-hold → guarded 20 mm axial handoff

`guarded_axial_handoff.py` binds a **positive, verified** pre-insertion hold
checkpoint to the exact axial trajectory already saved by the demo's observed
post-lift AutoDex/cuRobo preflight. It is an evidence packet, not a controller.
It supports the first centered insertion attempt; a VLM-adjusted retry needs
its own new post-shift 20 mm path and a separate handoff.

The packet checks the same attempt/candidate/frozen session in the post-lift
and pre-insertion reports, a sampled-clear 20 mm path audit and every successful
axial waypoint query. It checks the saved transfer-to-axial joint continuity,
constant measured Inspire grasp, a new stationary FR3/Inspire sample at the
observed hold, and the CAD target's `(preinsert clearance + 20 mm)` displacement
along the socket axis. Source reports and trajectory archive are hashed. A
subsequent verifier reopens them and repeats the checks; changed bytes fail.

Call this only after `preinsert_reached=True` has been recorded from
`prepare_observed_preinsert_label` and `observe_stage`. The session method
requires that recorded positive event to cite the exact checkpoint, then
passes its frozen source paths and a **new** robot feedback sample to the
lower-level packet builder:

```python
from precision_insertion.guarded_axial_handoff import verify_guarded_axial_handoff

report_path = runner.prepare_guarded_axial_handoff(
    measured_start=fresh_fr3_inspire_feedback,
    decision_timestamp_s=decision_time_on_robot_clock,
    max_state_age_s=commissioned_feedback_age_s,
    max_start_joint_error_rad=commissioned_start_joint_error_rad,
    max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
    max_hand_command_error_raw=commissioned_hand_tracking_error_raw,
    max_arm_velocity_rad_s=commissioned_stationary_arm_velocity_rad_s,
)
packet = verify_guarded_axial_handoff(report_path)
```

## Opt-in external contact boundary

`guarded_insertion_execution.py` can pass the exact saved axial joint array to
an **independently commissioned** adapter. It defaults to no motion. Before
calling the adapter it replays the handoff and archive hashes, checks a fresh
stationary FR3/Inspire start against the trajectory, verifies separate
watchdog and contact-controller review records for the adapter's exact binary,
and requires an explicit live interlock. It rejects the unchanged stock
`FrankaExecutor` as a guarded-contact follower.

The adapter must expose `follow_guarded_insertion(axial,
handoff_sha256=..., trajectory_archive_sha256=..., contact_limits=...,
expected_hand_raw=..., max_duration_s=...)` and
`stop_and_acknowledge()`. It must enforce its own robot-side dead-man,
force/torque and hand-hold stops even if this Python process dies. Its return
must cite a hashed `precision_insertion_guarded_execution_v2` metric record;
the boundary replays that record and its bound contact trace before accepting
the controller's terminal hold. A safety abort is logged, not silently
converted into insertion success. An exception after the command request
creates a failure marker and calls the adapter's stop method.

```python
from precision_insertion.guarded_insertion_execution import (
    execute_bound_guarded_insertion,
)

# `safe_controller` and the two commissioning records must be supplied and
# tested on the AutoDex robot PC. This demo ships no commissioned controller.
execution = execute_bound_guarded_insertion(
    runner=runner, adapter=safe_controller,
    handoff_report_path=report_path,
    pre_state=fresh_fr3_inspire_feedback,
    read_post_state=read_stationary_fr3_inspire_feedback,
    limits=commissioned_execution_limits,
    contact_limits=commissioned_contact_limits,
    max_handoff_age_s=commissioned_hold_age_s,
    watchdog_commissioning_path=watchdog_review_json,
    contact_commissioning_path=contact_review_json,
    motion_interlock=live_operator_and_hardware_interlock,
    enable_robot_motion=False,  # defaults to denied
)
```

The contact review has schema
`precision_insertion_contact_controller_commissioning_v1` and must name a
reviewer, test time, absolute daemon binary path and SHA-256, the **exact**
`GuardedContactLimits` values, and passing `force_limit`, `contact_abort`,
`sample_gap_abort`, and `terminal_hold` test flags (each field ends in
`_test_passed`). The separate watchdog review is the same record required
for held lift/transfer. These are operator-reviewed assertions, not proof
that a real controller has been commissioned on this workstation. For an
actual authorized trial, a separately reviewed caller must deliberately set
`enable_robot_motion=True`; this alone cannot bypass missing reviews, stale
state, wrong paths or a false interlock. The boundary returns an execution
record, never a task-success label.

For a centered first attempt, a separately produced guarded-stroke record can
use `precision_insertion_guarded_execution_v2`. Its `axial_handoff` field is
`{"path": absolute_report_path, "sha256": report_file_sha256}`. The referenced
`force_trace` must use `precision_insertion_guarded_contact_trace_v2` and carry
`path_binding` with that same handoff file SHA-256 and the packet's
`trajectory_archive_sha256`. The insertion checkpoint replays both records,
checks attempt/candidate/session and stroke timing, and requires the exact raw
pre-insertion camera bundle cited by the observed-hold checkpoint. Its
`execution_plan_binding` is then `observed_hold_axial_path`. Version 1 records
remain readable as `unbound_legacy_diagnostic` and cannot claim a handoff.
This v2 binding is for the **first centered attempt only**; a shifted XY retry
needs a separate post-shift axial path and handoff.

The saved packet can also be independently checked on the AutoDex PC:

```bash
python demo/precision-insertion/run_pipeline.py \
  verify-guarded-axial-handoff --report /path/to/report.json
```

The output has `robot_ready: false` and contains no motion call. A real
controller still needs a robot-side dead-man/watchdog, force/torque and grip
monitoring in the socket frame, a guarded trajectory follower, stop
acknowledgement, per-sample trace production, continuous/swept safety checks,
and commissioning of all numeric limits. Even successfully following these
joints establishes only a **nominal wrist stroke**. The task label still
requires post-stroke raw camera/VLM evidence and independently verified
physical **key** depth; a wrist endpoint is not 20 mm key penetration.
Matching hashes also do not authenticate the external controller or prove
that its samples came from the commanded path; producer commissioning and
actual robot-state logging remain necessary.

The canonical v8 candidate, object-processing and fixed socket assets remain
under the selected `shared_root`. This module and its tests live only under
`demo/precision-insertion`; stock AutoDex execution files are untouched.
