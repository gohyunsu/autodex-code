# Demo-local measured Franka held-motion adapter

`FrankaMeasuredLiftAdapter` fills the existing `LiveLiftHandlers.adapter`,
`LiveTaskHandlers.transfer_adapter`, and retry transfer/lateral adapter
protocols for **non-contact** held motion. The historical class name is kept
for the lift-only config. It shares the exact
completion-checked Franka/Inspire executor used for pickup. During the stock
joint-path follower, every state read on that motion thread also reads actual
FR3 joints, velocity and wrist wrench plus Inspire measured/commanded motor
positions. A stale state stream, exceeded force/torque/hand limit, FR3 reflex,
motion timeout, insufficient samples or unacknowledged terminal stop aborts
without producing a positive lift controller result. It writes an exclusive
controller trace even on an observed abort. Successful transfer/lateral
calls also write three distinct SHA256-bound source records: sampled FR3 joint
feedback, sampled wrist-wrench limits, and sampled Inspire motor hold. The
`grasp_held` field in those controller results means **motor hold only**; it
does not prove that the key is physically present. The subsequent raw
multi-view checkpoint decides that separately.

To let the first-party FR3 integration build it when the stage factory omits
`lift.adapter`, add this optional block to the robot-PC JSON:

```json
{
  "lift_monitor": {
    "daemon_binary_path": "/absolute/path/to/franka_daemon",
    "max_force_norm_n": 20.0,
    "max_torque_norm_nm": 3.0,
    "max_hand_tracking_error_raw": 25.0,
    "max_state_gap_s": 0.05,
    "minimum_samples": 3
  }
}
```

These numbers illustrate the config shape, **not commissioned limits**. The
adapter writes real measurements to
`<run_output_dir>/controller_traces/lift_<id>/controller_trace.json` and
returns its SHA256 to the already-existing lift execution verifier. A robot-PC
stage factory still supplies the lift `motion_interlock`, `limits`,
`commissioning_record_path`, and checkpoint parameters. A custom adapter
supplied as `lift.adapter` takes precedence over this default.

For one shared first-party adapter across lift, first transfer, observed
square retry transfer, and cylinder retry transfer/lateral shift, use
`held_motion_monitor` with the same six JSON fields instead of
`lift_monitor`. Do not specify both. The stage factory can still override
individual stage adapters. Transfer and lateral results are recorded under
`controller_traces/transfer_<id>/` and `controller_traces/lateral_<id>/`.
This setting does **not** supply the robot-PC stage factory's other limits,
watchdog record, motion interlock, guarded contact, withdrawal or reset.

Crucially, polling from the Python follower is **not** a robot-side watchdog.
The current checked-out ParaDex daemon accepts `duration_ms` in velocity
commands but does not expire a stale command on that basis. The unchanged
lift execution boundary requires a reviewed, hash-bound daemon record for
command expiry, disconnect stop, stop acknowledgement and E-stop tests **before
motion**. Such a record has not been produced for this workstation or the
AutoDex robot PC. Do not invent it or interpret CPU-only adapter tests as
physical commissioning. These held-motion methods have not been run on the
real FR3.
