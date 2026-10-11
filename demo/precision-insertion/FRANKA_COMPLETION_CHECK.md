# Demo-local Franka completion check

The first-party `open_franka_core(...)` now constructs a
`CompletionCheckedFrankaExecutor`, a demo-only subclass of the **unchanged**
AutoDex `FrankaExecutor`. It reuses the stock velocity-streaming path follower,
planner, squeeze sequence and clear-view behavior. The subclass intercepts
only the follower's terminal behavior:

1. Disable the stock blocking endpoint `arm.move` for that trajectory.
2. Defer the stock final hand ramp until after motion.
3. Require an acknowledged `stop_streaming` response and measured stationary
   joints within 0.02 rad of the planned endpoint.
4. Only then apply the deferred final hand command. A stall, timeout, reflex,
   absent feedback or unacknowledged stop raises without that command.

The original stock executor and other arbitrary subclasses remain rejected
by `execute_bound_pickup`; this exact demo mixin is admitted. The check also
affects other stock `_follow` callers on the shared executor. It is a
**Python-side terminal check**, not a robot-side command-expiry watchdog, a
guarded contact controller or a guarantee that the path was physically safe.
The ParaDex FR3 daemon currently does not expire a stale velocity command
just because its `duration_ms` elapsed. Its independent watchdog/E-stop and
the motion-interlock/commissioning records required by the stage boundaries
remain AutoDex-PC work. The wrapper has CPU-only fake-arm tests; it has **not**
been exercised on the real FR3.

`run_live.py` adds the repository, `src` and demo directories to its own
module path, so an invocation from another working directory can still load
both stock AutoDex modules and the demo package. The robot-PC integration
module itself must remain importable, for example via `PYTHONPATH`.
