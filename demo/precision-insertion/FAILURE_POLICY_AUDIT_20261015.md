# Live failure checks against stock AutoDex (2026-10-15)

Scope: the board-9, cylindrical 20 mm live path in this checkout. This is a
code-path review; no new robot motion or insertion trial was run.

Stock `FrankaExecutor.home(clear_view=True)` issues a blocking arm move and
polls the endpoint. Its `_follow` checks arm errors and contact, but stock
`run_auto.py` has no session-long exposure feedback sampler and no rule that
every read must observe a new state within 0.2 seconds. The local ParaDex
daemon publishes state stamped with its host `steady_clock`; on this PC, a
read-only 30-packet probe found daemon-to-Python monotonic age 0.16–0.26 ms.

| Precision-only check or failure path | Decision | Reason |
| --- | --- | --- |
| Require a second FR3 state packet within 0.2 s for every feedback read; permanently fail the session on one timeout | Removed | The stock blocking move can pause the daemon publisher. The current state carries a host-monotonic timestamp, so its actual age can be checked with one read. The old `max_arm_update_wait_s` JSON field is accepted but ignored. |
| Permanently stop the background sampler when one FR3 state is absent or old | Removed | Omit that sample and retry. A key or checkpoint exposure still needs actual feedback samples on both sides; a motion boundary still needs a recent measured start. Unrelated controller/read errors still stop the sampler. |
| Read FR3 state again inside `motion_interlock` after each stage checked its measured start | Removed | Every active motor boundary checks its own recent measured start and plan start; the second read added a separate failure opportunity without changing the commanded path. The callback now checks the explicit CLI motion opt-in. |
| Duplicate early board fit and immediate readback/hash of all just-written session PNGs | Removed in this checkout's latency changes | The final calibration still fits the board, and the producer's small manifest carries the frame binding. Full saved-data verification remains available offline. |
| Verify the hand is open and arm is clear at key exposure | Retained | Stock `home` verifies its arm endpoint, but does not confirm measured Inspire opening. Key capture needs an unobstructed object and a measured planning start. |
| Require key/checkpoint images to have matching frame IDs, camera calibration and exposure-time feedback | Retained | Stock's ordinary grasp task has no fixed socket or temporal VLM checkpoint. These bind insertion decisions to the actual scene; missing data cannot be replaced by a later joint read. |
| Check the sampled held key/hand path and stop on contact or a failed motor endpoint | Retained | The fixed socket adds a new obstacle and contact task. The completion-checked follower also prevents the stock follower's blocking final landing after an incomplete stream. |
| Compare two socket observations once at session start | Retained for current calibration contract | Stock has no fixed socket. The two observations establish its frozen pose; changing to one requires updating calibration and session evidence contracts together. |
| Exact running daemon-binary identity before first-party held motion | Retained pending a separate controller change | This is additional to stock, but currently supplies evidence that the streaming controller's command-expiry watchdog is the audited local binary. It is a motor-control boundary, rather than a camera or feedback sampling retry. |

The old terminal log reached socket and key pose acquisition, then surfaced a
background feedback failure before candidate planning. Its trace does not
timestamp the sampler exception itself, so the exact paused packet cannot be
identified from that run. The code order and daemon mutex explain a plausible
pause, and the state port was subsequently observed publishing normally.
