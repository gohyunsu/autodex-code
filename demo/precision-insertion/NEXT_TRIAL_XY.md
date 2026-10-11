# Next-physical-trial XY correction

This document describes the live `run_live.py` route for `cylinder` only.
The `square` mode remains available for offline assets and planning but has
effective XY update weight and retry budget zero. The live entry point also
rejects it before creating robot integration objects. This
is a correction of the earlier *same-held-attempt* retry
design. The first failed insertion is **not** followed by a lateral command
or second insertion while the same key remains in the hand.

1. After a negative insertion checkpoint, the commissioned controller
   withdraws axially to the pre-insertion hold. The open AutoDex camera
   session captures new, undistorted raw views. Qwen marks the insertion tip
   and key-axis evidence. Camera intrinsics/extrinsics and the frozen socket
   pose convert these image observations to a socket-frame metric XY error.
   The live cylinder route only proposes translation. If the image evidence
   is insufficient, it abstains.
2. The diagnostic records a continuous suggested displacement and caps the
   **per-failure increment** at 1 mm Euclidean norm. Let `p_0=(0,0)` in the
   socket XY frame. After failed attempt `n`, with measured residual `d_n`,
   the **next** attempt receives `p_(n+1)=p_n + w*d_n`.
   `--xy-update-weight` sets `w` (default `1.0`, allowed `(0,1]`). The
   configured `RetrySessionLimits.max_total_offset_m` separately bounds
   `||p_(n+1)||`; reaching that bound records an unavailable correction and
   returns the held key rather than applying an out-of-budget update.
   `SessionRunner` binds each update to
   the failed attempt's candidate ID, tabletop pose, frozen calibration,
   catalogue and saved multi-view report. This changes no pose or actuator
   command in the current attempt.
3. The existing held-return route takes the withdrawn key back to a supported
   tabletop pose, releases it, moves Franka/Inspire to clear view and captures
   a fresh key pose. The reset must be observed before a new attempt starts.
4. The previous candidate is allowed once despite the normal attempted-grasp
   exclusion and is tried first **only for the same tabletop pose**. If it
   cannot be planned or the pose changed, the advice is not applied to a
   different grasp. `--max-xy-retries 0` disables this path. By default there
   is **no retry-count cap**, so successive failed-trial updates accumulate
   until success, an unavailable visual estimate, a changed pose/grasp, a
   failed preflight, or the separately configured cumulative-offset budget
   stops the sequence. A nonnegative `--max-xy-retries` may set an explicit
   count cap for a particular experiment.
5. A new attempt receives `wrist_correction_socket_m`; its historical
   `xy_offset_socket_m` (key CAD target offset) remains zero. After a
   successful fresh pickup, the default physical-calibration branch uses
   that correction to shift the *commanded wrist relation* and re-runs the
   full-key/hand 20 mm endpoint screen and sampled held transfer/axial path.
   The CAD key target remains centered on the socket axis. To make this
   learned update effective on every cylinder live attempt, the live
   assembly requires the physical held-relation route; the separate observed
   6D relation route remains available to offline/square code.
6. The new attempt has its own lift, pre-insertion and final labels. A
   positive task label ends the session; another failure can contribute
   another residual if the retry count and cumulative-offset budgets remain.

The transform used in step 5 is
`T_robot_hand = T_robot_socket · T_socket_key_centered · T_key_hand_command`.
For cumulative socket-frame correction `p=(px,py,0)`, the command relation's translation
is changed by `R_socket_keyᵀ p`. Consequently the wrist target moves by `p`
while the *modelled key target* stays on-axis. This is a learned compensation
for a repeatable grasp bias, not an observation of the newly held key. It
cannot compensate an unknown yaw, tilt or nonrepeatable slip; it does not
remove contact-force limits or establish physical insertion success.

Important implementation references: `session_policy.py`, `live_retry_handlers.py`,
`session_runner.py`, `next_trial_xy.py`, `bounded_postlift.py`. The old
`LiveObservedXYRetryHandlers` and same-attempt post-shift helpers remain for
reading historical experimental reports but are not assembled into the live
action loop. A legacy pending-retry record routes to `stop_for_review`.

The repository tests cover the event boundary, weighted cumulative updates,
fresh attempt ID, 1 mm per-step bound, same-grasp preference and wrist-transform
equation. They do **not** establish
VLM pixel accuracy, physical grasp repeatability, controller behavior or
real-robot success; these require AutoDex-PC commissioning and experiments.
