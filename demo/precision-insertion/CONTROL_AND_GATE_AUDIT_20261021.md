# FR3 control and active gate review (2026-10-21)

This is a source and saved-data review of the board-9, white cylinder,
20 mm-gap live route. The separate checkout is at `b81a564` before the
uncommitted changes described here. The stock comparison is local AutoDex
`main` at `448732d`. No camera, daemon, hand, or arm was started for this
review. The original AutoDex checkout was not edited.

## Active route versus stock AutoDex

| Stage | Stock AutoDex | Precision-insertion route | Review |
| --- | --- | --- | --- |
| FR3 approach and lift | `FrankaExecutor._follow`, 50 Hz requested, 1.2 rad/s joint cap, 4 rad/s² command slew, 0.4 near-object/held speed scale, 0.12 rad reference lead | The same limits and proximity policy, with elapsed-tick pacing and partial reference progress after the 20 October stop-go fix | Preserve the stock speed values and the measured-progress fix. A 100 Hz command period remains optional; no current live trace proves that this Python loop sustains it. |
| Stream completion | Stock `_follow` can leave a short trajectory and make a blocking endpoint move | `CompletionCheckedFrankaMixin` stops and checks measured endpoint before the next action or final finger ramp | Retain: the saved 09:16 transfer stopped short. Removing this check could carry an unmeasured key pose into insertion. |
| Lift result | Stock `auto_label_charuco` judges board visibility | Same-camera before/after Qwen views judge visible key hold; measured lift and visual result are separate | Task-specific; the 10:10 saved run was visibly lifted but received an old false-negative visual label. The newer two-view classifier has only saved-image replay. |
| Held transfer and insertion planning | Stock v8 grasp and cuRobo follower | Selected v8 grasp relation, measured arm/hand state, socket-axis path and sampled collision check | Keep relation binding and path checks; stock has no fixed socket or 20 mm insertion to substitute. |
| Preinsert and final observation | No fixed-socket checkpoints | Raw same-camera temporal views and distributed Qwen; diagnostic CAD overlay runs alongside remote preinsert inference | Preserve temporal frame order. The overlay is diagnostic; no saved live preinsert time establishes that it is the current critical path. |
| Contact and failure return | Stock place stops at 12 N wrist-Z contact | Same 12 N follower, followed by ascent along the executed socket-axis path; visual outside-socket failure can feed the next trial's XY diagnosis | Preserve force stop and measured reverse-path corridor. They prevent a contact failure from commanding an unplanned return. |

The previous active-path audits removed absolute photo-age vetoes, a separate
held-lift wrench cutoff, hand-command transition aborts, and redundant hand
drift limits. This review found no further **motion** gate that the saved
runs show to be unnecessary. File hashes remain in provenance records, but
the same live capture no longer repeats an already performed image transform
just to validate those records.

## Changes in this review

1. `checkpoint_camera.py` now uses a fixed-point OpenCV remap for each
   snapshot. This produces the same undistorted pixels as the previous
   `cv2.undistort` on every one of the 20 real saved camera views checked.
2. `raw_camera_capture.py` no longer recalculates the same undistortion in
   the live writer. Offline `verify_raw_camera_capture` still reconstructs
   the derivative from the saved original PNG and rejects mismatched pixels.
   Frame ID, exposure identity, original and undistorted PNG hashes, and
   calibration values remain recorded.
3. FR3 velocity EMA retention is now expressed per elapsed second. At the
   default 20 ms tick its response is unchanged; a 10 ms tick retains the
   same physical smoothing time. The daemon command validity keeps the stock
   40 ms margin when the requested tick is shortened to 10 ms. The default
   controller remains 50 Hz. The source comment incorrectly claiming that
   the daemon ignored `duration_ms` was corrected.

The ParaDex daemon publishes state at 100 Hz and runs its libfranka callback
at 1 kHz. A 100 Hz outer Python command loop still has to fit cached-state
read, optional cuRobo FK, hand command update, and synchronous ZMQ command
acknowledgement into each 10 ms period. The saved 20 October traces measured
about 30 ms per command **before** the 20 October pacing fix; there is no
post-fix physical-rate measurement. Changing the default to 100 Hz on those
old traces would be an unsupported claim of smoother motion.

## Read-only measurements and tests

The same 20 actual 1536×2048 saved original/undistorted image pairs were
used for the pixel comparison. All 20 remap outputs were byte-for-byte
identical to the saved undistorted PNG pixels. With the 20 originals already
loaded in memory, three CPU passes gave:

| Work for 20 views | Pass 1 | Pass 2 | Pass 3 | Median |
| --- | ---: | ---: | ---: | ---: |
| Previous direct undistort | 0.386 s | 0.402 s | 0.395 s | 0.395 s |
| Fixed-point map and remap | 0.096 s | 0.079 s | 0.076 s | 0.079 s |

The old live writer also repeated the direct transform once, about 0.395 s
for these images under the same CPU conditions. The combined transform-only
saving is roughly 0.71 s per such checkpoint. This excludes camera/network
time, PNG encoding, NAS I/O, VLM, and robot movement; it is not a measured
episode speedup.

`bash demo/precision-insertion/run_robot_pc.sh --check` returned
`scope=static_files_only_no_camera_cuda_or_robot_connection` and 112 catalog
candidates. The full demo suite passed with 1059 tests and 4 skips. An
additional stock speed-profile/FR3 subset passed 44 tests. The new tests
verify that 10 and 20 ms velocity filters have the same wall-time response,
that the daemon command validity stays at 40 ms, and that offline provenance
replay still detects a deliberately altered undistorted image. These are
software tests; the actual 50 or 100 Hz controller rate and post-lift
insertion outcome remain unmeasured on hardware.
