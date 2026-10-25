# Demo-local Franka completion check

The first-party `open_franka_core(...)` now constructs a
`CompletionCheckedFrankaExecutor`, a demo-only subclass of the
AutoDex-derived `FrankaExecutor`. The precision checkout's FR3 executor also
has explicit mount geometry and stream-timing changes relative to the stock
checkout; see [PIPELINE_CODE_AUDIT_20261026.md](PIPELINE_CODE_AUDIT_20261026.md).
The subclass intercepts the follower's terminal behavior:

1. Disable the blocking endpoint `arm.move` while carrying a key or near
   contact. Keep it for an empty-hand clear-view return. Use the FR3 follower's
   0.04 rad L2 streaming completion tolerance.
2. Defer the stock final hand ramp until after motion.
3. Require an acknowledged `stop_streaming` response, maximum measured joint
   speed at most 0.05 rad/s, and measured endpoint error at most 0.04 rad L2.
4. While an object is held, request reflex-abort rather than the stock
   recover-and-continue behavior.
5. Only then apply the deferred final hand command. A stall, timeout, reflex,
   absent feedback or unacknowledged stop raises without that command.

The original stock executor and other arbitrary subclasses remain rejected
by `execute_bound_pickup`; this exact demo mixin is admitted. The check also
affects other `_follow` callers on the shared executor. It is a
**Python-side terminal check**, not a robot-side command-expiry watchdog, a
guarded contact controller or a guarantee that the path was physically safe.
The installed ParaDex FR3 daemon source expires a streaming setpoint after
`duration_ms` or its configured maximum, then ramps velocity to zero. The
Python follower currently sends a 40 ms validity period at its default 50 Hz
rate. The watchdog and this completion check address different failures.
Saved controller traces show completed lift and transfer motions on the real
FR3; they do not establish the accuracy of a physical key pose or contact.

`run_live.py` adds the repository, `src` and demo directories to its own
module path, so an invocation from another working directory can still load
both stock AutoDex modules and the demo package. The robot-PC integration
module itself must remain importable, for example via `PYTHONPATH`.
