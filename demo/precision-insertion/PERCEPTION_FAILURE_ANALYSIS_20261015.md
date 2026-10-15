# 2026-10-15 live run: feedback failure and perception latency

Source: terminal attachment `ec22ad8c-79bf-4990-a7fe-1ede5dc6677c` and
`/tmp/precision-insertion-timing/20261015_140518_038989.timing`.
The run started at 14:05:18 KST, ended at 14:11:40 KST, and failed before an
episode, candidate planning, VLM inference or insertion. Only the clear-view
robot action was reached. No camera or robot was contacted during this review.

## Immediate failure

`RobotFeedbackSampler` stopped on one `ValueError: FR3/Inspire feedback is
stale or not time-aligned`. `state_for_observation` then raised `feedback
sampling failed`. Five capture PCs had already returned all 20 key masks and
poses; this is not a missing capture daemon or a FoundPose error. The exact
stale/skew component cannot be recovered from the terminal log because the
failing sample's arm and hand timestamps were not recorded. The active robot
configuration allowed at most 0.1 s arm/hand skew and 1.0 s sample age; the
background sampler requested a new read every 0.01 s.

The traceback displays source line 158 as an Inspire hold-command check, but
that line's source was edited at 14:10:17, while the process ran from 14:05
through 14:11. The exception *text* identifies the actual check. Treating
the displayed line as evidence of a hand position error would be incorrect.

The adapter now raises a dedicated `FeedbackTimeAlignmentUnavailable` for an
unusable arm/hand timestamp pair. The background sampler discards that one
sample and keeps collecting. It does not admit that sample: an image lookup
still requires valid measured states bracketing its acquisition interval.
The existing hard failures for malformed or implausible joint values remain.
The preceding `6da83c7` change also replaced the old new-packet wait with the
daemon's host monotonic timestamp, but this run started before that edit.

## Measured time in this run

Times below are wall durations from completed trace spans. Child spans are
included in parent spans, so rows should not all be added together.

| Work | Duration | Location / meaning |
| --- | ---: | --- |
| Whole failed process | 382.656 s | Robot PC process, no episode |
| Board and two socket setup | 288.316 s | Session parent span |
| Three distributed FoundPose requests | 8.994 s total | Five capture PCs; 3.1, 2.9, 2.9 s |
| Read 20 same-request PNGs per request | 97.187 s total | Robot PC read/decode: 55.974, 26.816, 14.397 s |
| Board/two-socket evidence save | 189.651 s | Robot PC wrote 100 PNGs, 187 MB on NAS |
| Two socket silhouette refinements | 3.947 s total | Robot PC, 100 iterations each |
| Operator key-placement wait | 9.644 s | Human wait |
| Qwen model load | 2.594 s | No VLM inference in this run |
| Key pose admission and silhouette | 2.485 s | Robot PC, 100 iterations |
| Key evidence save | 50.885 s | Robot PC image/mask bundle |

The three saved-frame reads plus the two evidence writes occupy 337.723 s,
88.3% of the failed process. FoundPose compute/transport occupies 9.0 s.
Source socket PNG mtimes show all first-request files written by 14:05:32,
yet its saved-frame read ended around 14:06:26. This points to NAS read/decode
latency rather than waiting for 20 FoundPose results. The mount is NFS4.

## Stock AutoDex comparison

Stock `InitOrchestrator.collect_payloads` dispatches one request to all five
capture PCs. Each `InitDaemon` processes its assigned cameras sequentially,
while the PCs run concurrently. The robot PC receives per-camera masks and
poses and performs cross-view selection and silhouette refinement. The
precision path reuses those functions and the same 100-iteration optimizer.
The capture-PC work has **not** moved wholesale to the robot PC.

Stock `run_auto.py` asks the capture PCs to save init images but continues
from the returned pose. This precision path additionally waits until every
same-request PNG can be read and bound to its frame ID, then writes board,
two socket captures and key evidence to a new NAS bundle. The socket is
captured twice per session to compare/freeze its pose. Before this change,
those board/socket/key evidence PNGs were encoded and written serially on the
robot PC: `PRECISION_CAPTURE_IO_WORKERS=4` affected later raw checkpoints only.

The code now uses that bounded worker setting for the source-PNG read/decode
and for board/socket/key evidence writes. The writer keeps the same PNG bytes,
per-file SHA-256 manifest, request/frame metadata, and no-overwrite behavior.
All writes must finish before the manifest is published. This change cannot
remove the NAS transfer or guarantee a live speedup. The first cold NFS read
and writes should be measured on the next authorized run.

Read-only warm replay of the same 20 real socket PNGs on the mounted NAS,
alternating worker order, took 0.830/0.790 s with one worker and 0.212/0.203 s
with four. Cache state differs from the 14:05 live run, so those replay times
are **not** predicted live durations.

Validation: relevant planner-environment tests, `run_robot_pc.sh --check`, and
`git diff --check`. No `--run` was performed during this analysis.
