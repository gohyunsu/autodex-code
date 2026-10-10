# Demo-local measured held-lift execution boundary

`precision_insertion.lift_execution.execute_bound_measured_lift` is an
**opt-in** command boundary for the first 10 cm lift after a measured squeeze.
It consumes `SessionRunner.prepare_measured_lift_chain(...)` output, not the
original BODex candidate's nominal lift. Before motion, it rechecks the
saved complete-chain report and trajectory hash, fixed Inspire joints, exact
selected attempt/session, fresh stationary FR3/Inspire feedback, a reviewed
robot-side watchdog record tied to the executing daemon binary, and a live
interlock. `enable_robot_motion` defaults to `False`.

The injected adapter must implement:

```python
daemon_binary_path: str | Path
follow_lift(full_q_path, *, max_duration_s, expected_hand_raw,
            trajectory_sha256) -> dict
stop_and_acknowledge() -> dict
```

The result is a `precision_insertion_external_lift_result_v1` dictionary
containing `trajectory_complete`, `force_abort`,
`terminal_hold_acknowledged`, `trajectory_sha256`, `started_at_s`,
`completed_at_s`, and an absolute `controller_trace_path` plus SHA-256.
The adapter must implement online force, state, stall and timeout handling;
this Python function validates its returned record and a fresh measured
stationary endpoint, but cannot replace a controller-side watchdog. In
particular, the checked-out ParaDex daemon accepts `duration_ms` without
expiring a stale streaming velocity target. It **does not currently satisfy**
this prerequisite. The commissioning JSON is a reviewed external assertion,
not proof that a hardware test was honestly performed.

The commissioning record has schema
`precision_insertion_robot_watchdog_commissioning_v1`, the exact binary path
and SHA-256, `reviewed_by`, `tested_at_s`, `command_expiry_s`, and four true
booleans: `stale_command_timeout_test_passed`,
`disconnect_stop_test_passed`, `stop_ack_test_passed`, and
`e_stop_test_passed`. Each must be verified on the AutoDex robot PC with its
real controller and a recovery operator present before a record is created.
The demo's synthetic test record is not a deployable example.

A one-use `lift_started.json` marker is written before the adapter call.
Failure or missing final feedback creates `lift_failure.json`, asks the
adapter to stop/acknowledge, and leaves the robot state unknown for
supervised recovery. A completed lift creates `lift_execution.json` in the
existing schema consumed by the session runner's paired-camera lift
checkpoint. It does **not** set `grasp_success`; fresh multi-view visual
evidence must do that. It commands no transfer, XY shift or socket contact.

Only fake-adapter tests have run on this workstation. No current adapter or
watchdog record is commissioned, and no real Franka motion has been tested.
