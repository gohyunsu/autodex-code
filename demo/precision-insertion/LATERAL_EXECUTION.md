# VLM-grounded held XY shift: execution boundary

`precision_insertion.lateral_execution.execute_bound_lateral_shift` is the
opt-in motion boundary between a source-verified **at-most-1 mm** socket-XY
hold-shift preflight and the fresh `postshift_checkpoint` camera assessment.
The actual robot-side controller is **not** included or commissioned here.
This module never interprets a VLM direction as permission to move, never
labels a retry or insertion, and defaults to no motor command.

Before calling the injected adapter, it verifies the saved
`GroundedLateralPreflight`, its exact joint-array hash, the failed held
attempt, a new stationary FR3/Inspire start, the age of the plan, the
adapter's reviewed watchdog binary and a live interlock. The existing stock
`FrankaExecutor` is refused. The adapter must provide:

```python
follow_lateral(
    path, max_duration_s=..., expected_hand_raw=...,
    trajectory_sha256=...,
)
stop_and_acknowledge()
daemon_binary_path
```

It must return schema `precision_insertion_external_lateral_result_v1` with
`trajectory_complete=true`, `safety_abort=false`, `grasp_held=true`,
`terminal_hold_acknowledged=true`, matching trajectory SHA-256, start/end
times, and three independently hashed source files named
`trajectory_feedback`, `safety`, and `grasp_state`. The boundary checks a new
stationary terminal robot sample and writes
`lateral_executions/NNN/execution.json`. Exceptions after command request
create `failure.json` and call the adapter's stop method; supervised recovery
is required. The accepted execution record uses the exact schema expected
by `postshift_checkpoint._execution_log`.

Illustrative programmatic use after `prepare_grounded_lateral_hold_preflight`
returns a passing plan and writes its `report.json`:

```python
from precision_insertion.lateral_execution import execute_bound_lateral_shift

record = execute_bound_lateral_shift(
    runner=runner, plan=passing_lateral_plan,
    plan_report_path=saved_lateral_report,
    adapter=independently_commissioned_controller,
    pre_state=fresh_withdrawn_hold_robot_state,
    read_post_state=read_new_stationary_robot_state,
    limits=commissioned_lateral_execution_limits,
    max_plan_age_s=commissioned_plan_age_s,
    commissioning_record_path=reviewed_watchdog_json,
    motion_interlock=live_operator_and_hardware_interlock,
    enable_robot_motion=False,  # default: denied
)
```

Only a separately reviewed robot-PC caller may explicitly opt in to motion.
The external controller must independently enforce a stale-command
dead-man, force/contact stop and grip-loss hold even if Python dies. File
hashes and a claimed commissioning record do not prove those properties.

After a completed shift, save **new** synchronized AutoDex-camera images and
measured joints, then call `runner.assess_postshift_lateral_alignment` with
`execution_log_path` pointing to this `execution.json`. If alignment is
credible, `runner.prepare_postshift_insertion_preflight` can check a new
20 mm endpoint and path. That latter result is still planning-only: a
post-shift axial handoff, contact execution and post-contact observation
remain to be integrated before a physical retry is possible.
