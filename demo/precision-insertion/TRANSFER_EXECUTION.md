# Held transfer execution boundary

`precision_insertion.transfer_execution.execute_bound_held_transfer()` is the
demo-local bridge between a passing **post-lift** transfer preflight and the
existing pre-insertion camera checkpoint. It does not touch the original
AutoDex execution modules. It does not command the 20 mm axial stroke.

The caller must supply the active `SessionRunner`, a newly measured stationary
`LiveRobotState`, a fresh endpoint-state reader, explicit
`TransferExecutionLimits`, a motion interlock, and an adapter.
`enable_robot_motion` defaults to `False`. The first-party
`FrankaMeasuredLiftAdapter` must be bound to the exact audited local ParaDex
daemon process with a stream expiry within the limit. An external adapter
still needs a reviewed watchdog record. The unchanged `FrankaExecutor` is
rejected; the first-party adapter uses the completion-checked subclass.

The boundary verifies the session/candidate/post-lift report hash, checks the
exact saved `transfer` array in `planned_trajectories.npz`, requires an
unchanged Inspire grasp along it, and compares its start with current measured
arm and hand feedback. It then writes `transfer_started.json` before calling
the adapter. A second call to the same attempt is refused even if the first
one failed.

The external adapter contract is:

```python
result = adapter.follow_transfer(
    trajectory_13dof, max_duration_s=limits.max_execution_duration_s,
    expected_hand_raw=pre_state.hand_raw_measured,
    trajectory_sha256=saved_trajectory_sha256,
)
adapter.stop_and_acknowledge()  # required on post-start exceptions
adapter.daemon_binary_path    # exact local stock binary or reviewed external binary
```

`result` must have schema `precision_insertion_external_transfer_result_v1`,
matching trajectory hash, bounded start/completion times, and explicit
`trajectory_complete=True`, `safety_abort=False`, `grasp_held=True`, and
`terminal_hold_acknowledged=True`. It must reference three distinct external
files with absolute paths and SHA-256 hashes under `source_records`:
`trajectory_feedback`, `safety`, and `grasp_state`. The Python boundary
checks their bytes and reads a new measured stationary endpoint state; it
cannot prove that an external controller honestly produced them or that it
stopped promptly when the Python process died. Those properties require
physical testing of the daemon, watchdog, E-stop and force/grip stops. The
first-party identity check records that physical watchdog testing is absent.

After all checks, `transfer_execution.json` uses the schema already read by
`preinsert_checkpoint._transfer_log`. It is **not** `preinsert_reached=True`.
The session still needs synchronized raw AutoDex camera frames, measured
FR3/Inspire state, and `SessionRunner.prepare_observed_preinsert_label()`.
An exception after command onset writes `transfer_failure.json`, requests
an adapter stop acknowledgement, and leaves the robot state unknown for
supervised recovery. The first-party measured adapter exists, but no real
transfer has been tested; the execution tests use a fake controller.
