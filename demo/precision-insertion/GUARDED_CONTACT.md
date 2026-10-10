# Guarded insertion sample policy (not an actuator)

`precision_insertion.guarded_contact.GuardedContactMonitor` is the demo-local
decision policy for a **single** preflighted 20 mm insertion stroke. It
receives consecutive measured samples and returns one of:

| Decision | Meaning |
| --- | --- |
| `continue_preplanned_stroke` | The latest sample is within configured limits; the external, independently safeguarded actuator may consider the **next waypoint of the same validated plan**. |
| `hold_for_key_depth_and_vlm` | The wrist/FK **nominal** 20 mm endpoint was reached without this policy seeing an abort. Stop and acquire independent key-depth and VLM evidence; this is not task success. |
| `abort_hold_for_supervised_recovery` | A sample/timing/force/pose limit failed. The decision is latched; no automatic retry or withdrawal is commanded. |

The policy contains no ParaDex/Franka connection. It is not called by the
current `run_pipeline.py` as a motor controller and does not write the
`precision_insertion_guarded_execution_v1` record. That record still needs a
commissioned external execution producer.

## Input contract

Create `GuardedContactLimits` **only from measured and approved rig-specific
limits**. Its target is fixed to 0.020 m. The `max_depth_step_m`, force,
moment, pose, age and timeout values in the unit tests are synthetic examples
and must not be copied to the robot. For each `GuardedContactSample`:

- `timestamp_s` is the actual sample-acquisition time on the same clock as
  `started_at_s` and `decision_time_s`. A publication timestamp substituted
  for acquisition time invalidates the age and gap checks.
- `nominal_depth_m` is the current wrist/FK progress projected onto the frozen
  socket axis using a separately bounded key–hand relation. It is **not** a
  measured depth of the physical key. A squeeze shift or slip can break it.
  The first monitored sample must still be at or above the socket entry
  plane (nonpositive nominal depth); an already advanced stroke is rejected.
- `force_socket_n` and `moment_socket_nm` must be baseline/gravity corrected
  and transformed to a documented socket-frame origin. The FR3 controller's
  raw `wrench` must not be passed through by renaming it: a full wrench
  transform includes the moment induced by the origin translation.
- `lateral_error_m`, `axis_tilt_deg`, and (for the square key)
  `yaw_error_deg` are measured/uncertainty-bounded pose residuals. Cylinder
  axial yaw may be `None` because it is symmetric. An occluded key cannot be
  silently replaced by the nominal CAD prediction.
- `hand_command_tracked` means only that Inspire controller feedback follows
  the commanded fingers. It is **not** evidence that the key remains held.

Offline policy replay has the following shape; `samples` and `limits` must
come from independent validated sources, not the code example:

```python
from precision_insertion.guarded_contact import GuardedContactMonitor

monitor = GuardedContactMonitor(
    family=mode.family,
    limits=commissioned_limits,
    started_at_s=stroke_start_acquisition_time_s,
)
for sample, decision_time_s in saved_samples:
    decision = monitor.observe(sample, decision_time_s=decision_time_s)
    if decision.action != "continue_preplanned_stroke":
        break
```

`check_sample_deadline(now_s=...)` detects a missing sample only while this
Python process is running. The checked-out ParaDex velocity-stream daemon
parses but does not enforce `duration_ms`; Python process failure could leave
the last velocity target active. **A verified robot-side dead-man/watchdog,
E-stop, stop acknowledgement, force sensing and contact-control validation
are prerequisites for any physical integration.** `continue` is not a robot
command or safety certification.

After a nominal stroke, `insertion_checkpoint.py` combines a fresh paired
multiview VLM comparison with an independent conservative **physical key**
depth interval, alignment assertion, force trace and grasp-state evidence.
Its success label remains unknown if those sources are absent. An abort from
this monitor must be reflected in the external guarded-execution record; the
current module does not fabricate that record from wrist progress.
