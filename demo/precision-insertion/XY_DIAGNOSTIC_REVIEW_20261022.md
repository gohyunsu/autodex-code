# Saved preinsert XY diagnostic review (2026-10-22)

Scope: the separate precision-insertion checkout, compared with stock
`/home/robot/AutoDex` at `448732d`. No camera capture or robot command was
issued for this review. The stock and separate checkouts have identical
`src/execution/run_auto.py` contents. Stock AutoDex supplies v8 object
processing, grasp planning, camera orchestration, FR3 execution and reset, but
has no fixed-socket preinsert snapshot or next-trial XY diagnostic. A task
specific consistency check therefore cannot be removed solely because stock
has no equivalent step.

## Recorded failure and code path

Run `20261021_111900_186891` reached preinsert, guarded insertion, release,
clear view and a final visual label before `action.assess_saved_preinsert_xy`
failed at 128.88 s session wall time. Its five `grounded_axis` calls ran in
parallel and all raised capture-PC Qwen CUDA out-of-memory errors. Their logs
show a 15.48 GiB GPU with only 114–162 MiB free, about 4.75 GiB used by the
current worker, about 4.89 GiB by another process and about 5.68 GiB by the
AutoDex init process. The other 4.89 GiB process is consistent with an orphaned
Qwen worker. The prior session's Qwen close occupied the old 3 s SSH timeout;
the trace does not prove the identity of that other PID. Current capture-PC
inspection found the init process and no active Qwen worker.

`live_retry_handlers.py` already turns a capture-PC Qwen backend exception
into an `xy_retry_unavailable` record and routes to released-key recovery.
That change postdates the failed run, so the old terminal exception is not
the current policy. The current fix also prevents a second worker from loading
on any capture PC: the SSH command holds a nonblocking per-PC `flock` for the
worker lifetime. `close()` asks the worker to shut down, closes stdin, and
allows 15 s for the remote model process to exit before terminating SSH. A
second-start saved-frame replay was rejected promptly while the first worker
remained usable, then the worker exited without a leftover process.
The recipient's shared worker stage at
`/home/robot/shared_data/hyunsu/tmp/precision_distributed_qwen_20261020`
was updated; its worker module and this checkout's module both hash to
`d0a7e5820e901063b7cd8f3cdbecd1f929413c04386b83f82a12b284e5754be7`.
That staged file is an external runtime asset, not part of the Git commit.

## Stock comparison and changes

| Stage | Stock AutoDex | Precision path before review | Current change |
| --- | --- | --- | --- |
| Object pose and plan | v8 FoundPose/candidate planning on capture PCs and robot PC | Reuses those assets with a frozen socket and full-key candidate | No new motion or planning threshold. |
| Saved preinsert image | No fixed-socket XY snapshot | Raw undistorted images, camera/frame IDs and timestamps; live manifest cache avoids rereading all saved PNGs | Same source evidence and coordinate system. |
| Qwen process | No distributed per-camera XY model | Five persistent capture-PC Qwen workers; short close could leave duplicate GPU consumers | One worker per PC, explicit remote shutdown. |
| Qwen XY input | No XY grounding | 640 × 640 lossless crop from projected tip and rim | Rectangular native-pixel crop covers projected tip, rear and rim with 96 px margin; no resampling. |
| XY update | No fixed-socket next-trial offset | First triangulatable pair could become a metric XY correction even when its depth contradicted the commanded pose | Reuse the existing parallax, reprojection and 10 mm held-key translation limits before calling tip-only advice metric. A valid one-view mark remains a labeled planar hypothesis; otherwise the existing labeled 2 mm exploratory policy applies. |
| Physical execution | Stock force stop, measured state and collision/path planning | Adds socket-aware path and contact/return handling | No relaxation of force stop, collision sampling or current-state checks. |

The five selected saved frames in the failed run were replayed read-only using
the actual distributed Qwen workers. With one worker per PC and the new crop,
all five calls returned without OOM. The new crops were 20–52% of the old
640 × 640 area and contained the three projected nominal points in all 15
view checks across three saved runs. The old square crop also fit with only
one worker per PC. Thus duplicate model residency is the best-supported OOM
cause; cropping lowers load but is not by itself a proven OOM cure.

| Saved-frame replay | Model startup | Five-view parallel inference wall | Per-call inference | CUDA peak per worker | Close |
| --- | ---: | ---: | ---: | ---: | ---: |
| New rectangular crop | 7.45 s | 1.43 s | 1.29–1.36 s | 4.02–4.05 GiB | 0.73 s |
| Prior 640 × 640 crop, one worker/PC | 3.87 s | 1.58 s | 1.35–1.43 s | about 4.10 GiB | 0.67 s |

These are two read-only replays with differing model warm state, so their
startup values are not a controlled speed comparison. They measure neither
live camera network time nor motor/contact time. Qwen returned degenerate
shaft lines in several views and out-of-crop points in two views on this
saved attempt; a shorter English prompt also failed to recover a consistent
axis. The old tip-only fallback would have treated a triangulated tip 47.7 mm
above the commanded height as metric and proposed almost 10 mm of XY motion.
The updated resolver rejects that depth as a measurement and labels the
unchanged 2 mm search step exploratory. This prevents a false measured
correction; it does **not** establish a correct physical XY offset. Future
physical XY success remains unverified.

## Validation

`/home/robot/anaconda3/envs/planner/bin/python -m pytest -q
demo/precision-insertion/tests --ignore=demo/precision-insertion/tests/test_franka_completion_checked.py`
passed 1,097 tests with four skips after the axis-fit height regression was
added.
The omitted module requires CUDA hardware unavailable in this test environment.
The saved-frame replay used the October 21 raw preinsert PNGs and frozen
session calibration; it did not move the robot. `git diff --check` passed.
