# 2026-10-21 transfer stop and endpoint timing

The 11:06 board-9, 20 mm run stopped after lifting and transferring the key,
before the preinsert camera check or insertion. The controller trace reports a
completed, acknowledged transfer with 154 advancing samples. Its last measured
arm joint error from the planned endpoint was 0.01235 rad, below the configured
0.02 rad endpoint limit. The operator confirmed the key remained held above the
board. The episode records `measured transfer hold differs from plan` and has no
`transfer_execution.json`.

The old post-motion check required the joint feedback timestamp to be strictly
later than the controller's completion timestamp. The controller records
completion immediately after its last stationary sample; the combined arm/hand
sample timestamp can legitimately be slightly earlier. The failed run did not
persist that post-state sample, so its exact failed subcondition cannot be
proven retrospectively. Transfer failures now save the post-state when present.

Lift, transfer, lateral and retry transfer, and insertion motion checks now
accept a stationary measured endpoint sampled within the existing feedback gap
on either side of controller completion. They still require a sample after
motion began, the existing age bound, the planned joint endpoint and the
original controller/force status. The same timeline rule is applied when
verifying saved retry records.

The stock Inspire release ramp returns before the motors finish opening. The
release/exit paths now wait for measured open-hand feedback before following
the already-planned arm trajectory. This addresses an independently observed
empty-hand return where immediate feedback still showed a partly closed hand.
The empty-hand final retract uses the existing 0.05 rad clear-view gate as
its follower endpoint tolerance. Two audited returns ended stationary at
0.030 and 0.033 rad maximum arm error, yet the generic 0.02 rad completion
check rejected them. Other motion endpoints retain the 0.02 rad check.

The saved timing trace reports 160.06 s for the failed session. It records
58.39 s in sampled geometry audit, 29.33 s key pose admission, 16.23 s v8
pickup planning, 6.85 s lift motion, 4.85 s transfer motion, and 1.37 s
cumulative VLM inference. These operation durations can overlap; they are not
summed into the session duration. No insertion or release time was recorded
because those stages were not reached.
