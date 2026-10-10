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
