# Precision-insertion timing profile

This demo now records the **live run**, including startup, operator waits,
camera/FoundPose work, planning, motion callbacks, VLM calls, evidence I/O and
cleanup. It reuses the stock AutoDex `autodex.pipeline_trace.PipelineTrace`
event format and monotonic clock. It does **not** start cameras or move a robot
to obtain a profile; the trace is produced by an otherwise normal live run.

## Files and reading them

For an output directory `.../_pipeline_runs/RUN_ID`, the profiler writes a
sibling `.../_pipeline_runs/RUN_ID.timing/` containing:

- `events.jsonl`: append-only stock trace; individual start/end records,
  monotonic duration and UTC timestamps for synchronization.
- `timing_summary.json`: spans in execution order and per-operation `count`,
  `inclusive_s`, `self_s` and failure count. Each span also records direct
  `child_cumulative_s`, `child_occupied_s` and `child_overlap_s`.
- Stock trace `manifest.json` and `timeline.json`.

The sibling location means startup failures can be timed even if no session
ledger was opened. On the robot PC, after a live attempt:

```bash
cd /home/robot/autodex-precision-insertion
python3 demo/precision-insertion/report_timing.py \
  /home/robot/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/RUN_ID.timing
python3 demo/precision-insertion/report_timing.py \
  /home/robot/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/RUN_ID.timing \
  --tree --max-depth 8 --min-duration-s 0.01
```

For the separate ZeroDex capture4 checkout on `kds-Z790`, use its local
evidence root instead. This command was run on the saved 5 mm insertion trace:

```bash
cd /home/hyunsu/autodex-code
python3 demo/precision-insertion/report_timing.py \
  /home/hyunsu/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/20261029_132835_069139.timing
```

## ZeroDex capture4: two successful 5 mm episodes (2026-10-29)

The archived traces for `20261029_131932_872369` and
`20261029_132835_069139` correspond to human-confirmed inside-socket
episodes using grasp `table/0/20195`. They predate the latest provisional
cuRobo warmup overlap, so that optimization still needs a fresh live trace.
The measured costs below are **within** those runs, not a prediction of the
next run:

| Recorded span | First run | Second run | Interpretation |
| --- | ---: | ---: | --- |
| Lift + preinsert + final manual-label spans (inclusive) | 355.29 s | 256.06 s | Mostly operator response time; each span includes a small VLM inference child. |
| All `vlm.infer` spans, cumulative | 2.76 s | 2.70 s | GPU work across checkpoints, not the source of the long manual waits. |
| Fixed-world warmup joins, cumulative | 20.91 s | 21.04 s | Almost entirely the first wait on provisional cuRobo world creation; latest launcher overlaps its start with initial clear-view motion, unmeasured live. |
| `startup.sam3_roi_model_load` | 9.56 s | 10.22 s | One model startup per process. |
| `cleanup.integration_close` | 0.43 s | 55.55 s | The second span ended in `KeyboardInterrupt()` and is not a normal shutdown-time sample. The insertion itself was human-confirmed. |

`evidence.png_write_hash` sums to about 4.6 s in each run, but the PNG jobs
run in parallel; that sum is not checkpoint wall time. The writer already uses
lossless PNG compression level 1 and local parallel I/O. Changing its format
or skipping evidence is not justified by the cumulative number.

The second saved run also spent 1.926 s in
`planning.nominal_insertion > planning.held_path_audit >
audit.minimum_distance_scan`, and 1.783 s in the nominal post-lift path
audit. Initial grasp selection and the nominal post-lift replan consume the
sampled collision result, but not the exact minimum-distance table. Both now
request a collision-only sampled audit, retaining the same key/hand collision
decision and omitting that unused diagnostic scan. Bounded post-lift paths
still request exact distances for their commissioned surface-deviation margin.
The 1.926 s and 1.783 s are historical scan times, not measured speedups;
compare a fresh trace after these changes.

An offline startup-order probe on this PC used the saved 5 mm fixed collision
world and the same local cuRobo, Qwen and SAM3 assets. With the FR3 planner
core constructed first, starting all three remaining GPU jobs together took
16.79 s wall time; finishing cuRobo world setup before starting Qwen and SAM3
took 21.10 s. The planner's own job was slower under overlap (16.79 versus
12.20 s), but the concurrent schedule finished sooner. This is one offline
measurement without camera or robot traffic, so it does not establish a live
startup speedup. It gives no reason to serialize the current model preloads.
The probe and raw results are grouped under
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_gpu_startup_*_20261029.*`.

Future runs now nest `cleanup.planner_warmup_join`, `cleanup.vlm_load_join`,
`cleanup.vlm_backend_close`, `cleanup.sam_roi_close`,
`cleanup.sam_load_join`, and `cleanup.franka_core_close` under
`cleanup.integration_close`. The core span separates
`cleanup.feedback_sampler_close` from `cleanup.franka_executor_shutdown`.
These spans only observe shutdown and do not alter resource ownership or
motion. The 55.55 s archived span was interrupted by the operator, so do not
attribute it to normal cleanup. Compare their **inclusive** and **self** times
in an uninterrupted fresh trace before changing shutdown behavior.

`run_robot_pc.sh --run` now runs only the Python pipeline. Start/check/stop
long-lived services independently with `manage_robot_pc_services.sh`; its
service-start report is `/tmp/precision-insertion-services-*.tsv`. The
manager starts the five capture hosts concurrently. Its timing is separate
from the Python session; do not add service startup to a warm-run comparison.
Each live invocation creates a new launch JSON with a fresh evidence path;
the static `--check` only validates the original launch template.
When `PRECISION_REUSE_SESSION_EVIDENCE` is set, the launch JSON still gets a
fresh unused evidence path, while the live session verifies and reuses the
specified saved board/socket images and calibration. Set
`PRECISION_KEY_ALREADY_PLACED=1` to skip the initial placement prompt. This
keeps the key on the board for the whole startup and avoids recapturing the
fixed socket. The live key capture and cuRobo planning still run.

For the board-9, 20 mm socket already measured on 2026-10-15:

```bash
cd /home/robot/autodex-precision-insertion
bash demo/precision-insertion/manage_robot_pc_services.sh check
PRECISION_REUSE_SESSION_EVIDENCE=/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical/live_board9_gap20_20261013/session_evidence_20261015_140513_805015685_553881 \
PRECISION_KEY_ALREADY_PLACED=1 \
PRECISION_CAPTURE_IO_WORKERS=4 \
  bash demo/precision-insertion/run_robot_pc.sh --plan-only
```

`--plan-only` allows the initial clear-view move and live key perception,
then stops before pickup motion when a plan is ready. If all candidates fail
planning, it returns `planning_no_candidate` without trying a repose move.
This saved socket calibration is applicable only while the board, socket and
camera calibration remain unchanged. The verifier checks saved file digests,
board ID and current camera calibration; it cannot detect physical fixture
movement. `--run` with the same environment variables proceeds beyond
planning and commands robot actions.

The 2026-10-15 15:09 plan-only attempt reused that socket evidence, captured
the key and reached real v8/curobo pickup planning. Three pickup candidates
failed (`IK: 0/1`), so there was no accepted pickup plan or trial motion. Its
recorded `live.session` duration was 80.872 s: saved-session verification
2.905 s, clear-view 0.424 s, key capture 43.302 s (40.817 s waiting for
saved camera frames and 2.245 s FoundPose request), key pose admission
2.398 s, key evidence write 3.484 s, candidate preflight 14.555 s and v8
pickup planning 13.004 s across three candidates. These spans are nested and
must not be summed into the session duration. The first cuRobo candidate
took 12.464 s including warmup; subsequent candidates took 0.320 and
0.221 s. The old plan-only dispatcher then spent 3.350 s on repose preflight;
the current dispatcher stops before that step. See run
`20261015_150915_090115.timing` for the source trace. No pickup, insertion,
contact or failed-trial recovery times were measured.

The current live tabletop-key path no longer waits for the init daemon's
same-request PNG files before v8 planning. FoundPose mask/pose frame IDs and
camera acquisition times produce a pose-only `key_captures/...` bundle; no
robot-PC image pixel hash or saved-PNG comparison is used to admit that pose.
The pose-only request also omits `save_capture_dir`, so the init daemons do
not start the unused per-camera PNG writes for that request.
After pose admission, a separate AutoDex snapshot starts on a worker while
candidate planning runs. Its calibrated undistorted frames are saved under
`key_captures/.../initial_tabletop/`, with a separate request ID, per-camera
frame ID and timestamp. Lift/final Qwen comparisons read these visual frames;
the candidate planner reads the FoundPose pose frames. Before pickup, the
action handler waits only for the initial snapshot exposure/undistortion;
its PNG save continues in the background. The first VLM consumer waits for
that save if it is still pending, and run cleanup drains remaining saves. The initial
snapshot does not write a second set of distorted source PNGs or recompute
their pixel hashes. Undistorted PNG hashes remain in the visual bundle for
VLM replay; they do not gate the start of candidate planning. The historical
15:09 numbers above are for the previous pipeline; the new overlap and any
speedup need a new live timing trace.

Static validation used `run_robot_pc.sh --check` (no camera or robot
connection) and the planner environment's pytest suite for live capture,
pose admission, concurrent planning, raw lift and released-insertion replay.
This path was not exercised with `--run`; no motor, network-round-trip or new
live latency number is implied by these tests.

On the robot PC, `run_robot_pc.sh` sets `PRECISION_TIMING_LOCAL_ROOT` to
`/tmp/precision-insertion-timing` by default. The stock trace writes its
events there during the run; orderly exit copies the complete sidecar once
to the NAS path above. If killed before cleanup, recover the local trace;
the NAS copy may not exist. This avoids a network write for every event.

`inclusive_s` contains child spans. Use `self_s` to rank mutually exclusive
costs within one serial branch; adding all inclusive values double-counts.
With parallel workers, even the sum of `self_s` across workers is **work
time**, not elapsed critical-path time. The summary unions overlapping direct
child intervals when computing a parent's `self_s`. It also reports
`vlm_inference_cumulative_s` (sum of all camera requests),
`vlm_inference_occupied_s` (union of their wall intervals) and
`vlm_inference_overlap_s`; compare the occupied value and enclosing
`trial.*_label`/`vlm.view_batch` duration when judging latency. `socket.capture` is the
entire camera request plus saved-frame/provenance work, while
`camera.socket_mask_request` (board/body mode), `camera.saved_frame_wait` and
`camera.frame_binding_verify` separate its children. Subsequent
`socket.body_fit` is outside `socket.capture` and contains CAD dimension
extraction, per-camera centroid ray unprojection, the multiview median and
per-camera wall IoU spans. The stock path retains `camera.foundpose_request`
and `socket.refine_capture`. `vlm.infer` carries stage, camera ID, model and image count,
so the report shows both total per-checkpoint cost and slow views. Parent
`trial.*_label` spans additionally include parsing/evidence/manual wait.

The raw event log is the primary evidence if the process is interrupted.
`timing_summary.json` is written on orderly exit or Python exception, not a
power loss or SIGKILL. Timings are elapsed wall time, including operator wait,
network, NAS I/O and robot motion; they do not measure isolated GPU kernel
time. The Python trace itself writes files and has a small overhead, even on
local storage. The service TSV is a different clock/process and is not
nested in the trace.

The current distributed Qwen setting starts five capture-PC workers after
the socket capture, before prompting the operator to place the key. It uses
one fixed camera per PC: capture1/24122734, capture2/25305463,
capture3/25322638, capture5/25322649 and capture6/25305461. The
`startup.vlm_model_load` span includes parallel worker startup; per-call
`vlm.infer` spans and `distributed_qwen.jsonl` separate wall time, remote
inference time, actual image sizes and peak CUDA allocation. Lift,
preinsert, insertion and final classification can finish on the first three
consistent responses. The two remaining requests finish in background and
are joined during backend cleanup. XY voting and native-pixel grounding use
all five selected cameras concurrently. Therefore a five-view inference sum
is GPU work time, not the elapsed checkpoint duration.

## Where the extra time enters versus stock AutoDex v8

| Executed stage | Stock `run_pipeline.py` / `run_auto.py` | Precision-insertion live demo | Timing span |
| --- | --- | --- | --- |
| Process timeline | Already uses `PipelineTrace` and detailed planning/execution spans. | Reuses the same trace format but a separate demo run and sidecar. | `live.session` |
| Service startup | Existing AutoDex camera/robot services may already be running. | Independent manager starts missing daemons; pipeline runs reuse them. | Service TSV, outside run trace |
| ChArUco | Board/camera measurement used for tabletop geometry. | One board capture and fit at session start; the saved frames are also used in the frozen calibration. | `board.*` |
| Socket | No fixture-pose estimation or socket refinement. | Current board-9 / 20 mm launch uses one 20-camera SAM mask acquisition and one board-normal/body-mask CPU fit, then freezes that pose. Socket FoundPose and stock silhouette fitting are skipped; repeatability is unmeasured. | `socket.capture`, `socket.body_fit` |
| Source/evidence checks | Stock live entry points do **not** hash the complete grasp pool and session image bundle on each run. Stock trace hashes an artifact only when explicitly written through its helper. | Live catalog parses once and assumes immutable NAS sources; explicit offline audit still rehashes candidate files. Socket evidence keeps its existing source checks. A live tabletop key pose uses frame IDs and timestamps without waiting for or hashing its FoundPose PNGs; its independent VLM snapshot is saved concurrently with planning and keeps its own raw visual provenance. | `startup.catalog_load`, `socket.evidence_write`, `key.capture`, `key.initial_visual_capture`, `key.candidate_preflight` |
| Per-trial perception | Object SAM/FoundPose for a fresh tabletop pose. | Fresh key SAM/FoundPose with frozen-socket, camera-time and mask admission; repeated after failure/repose. | `key.model_init`, `key.capture`, `key.pose_admission` |
| Grasp planning | v8 grasp candidates, cuRobo plan, pickup and lift. | Reuses v8 candidate files but adds a 20 mm whole-hand/socket endpoint catalog and per-current-pose full-chain preflight; measured post-squeeze lift chain. | `key.candidate_preflight`, `trial.measured_lift_preflight` |
| Motion | Grasp/lift, then stock task/coverage/reorientation flow. | Pickup/lift, held transfer, preinsert hold, guarded axial stroke, release/clear; failed trials may include restoration/repose. | `trial.*_motion`, `trial.release_and_clear`, `action.*` |
| Visual labels | Existing lift/ChArUco heuristic and task-specific labels. | Local Qwen checkpoint labels on same-camera temporal pairs, plus per-view grounded key-axis observations for next-trial XY correction. | `vlm.infer` by stage/camera |
| Outcome | Lift and task outcomes tracked by stock task flow. | Initial/final key images compared after hand release; success ends session, failure may trigger accumulated XY and restore the original tabletop state. | `trial.final_label`, recovery `action.*` |

The last robot-PC session saved a 188 MB socket evidence directory and ended
after board/socket calibration, before key perception. Its two socket-pose
estimates differed by 0.522 mm and 0.711 degrees, within the configured
1 mm/1 degree repeatability limits. It had **no profiler**, so those are not
latency measurements. No measured claim that VLM or hashing dominated that
run is justified yet.

## Reading the finer breakdown

`report_timing.py` ranks mutually exclusive `self_s` by operation and phase,
and prints per-candidate operation costs plus `vlm.infer` by stage/camera.
Each attempted grasp in the trial-preflight report also carries the stock
planner's bounded `pickup_timing` breakdown (world/filter/IK/approach/lift
times and candidate counts). This records why planning a pose took time
without rerunning collision or IK checks; it adds no acceptance criterion.
Important nested stages include:

- `planning.candidate_load`, `planning.v8_pickup`, `planning.held_lift`,
  `planning.held_transfer`, `planning.axial_jacobian`,
  `planning.held_path_audit` and its child `audit.geometry_samples`;
- `audit.fixed_world_build` versus `audit.fixed_world_lookup`,
  `audit.key_model_lookup`, `audit.hand_urdf_fk`, `audit.wrist_fk_rollout`;
- `camera.socket_mask_request` (or stock `camera.foundpose_request`),
  `camera.saved_frame_wait`,
  `camera.frame_binding_verify`, `camera.undistort_batch`,
  `camera.undistort_generation` and
  `socket.body_fit` (or stock `socket.refine_capture`);
- Live tabletop key `key.initial_visual_acquire` and
  `key.initial_visual_save` overlap `key.candidate_preflight`;
  `key.initial_visual_acquisition_wait` is only the remaining capture wait
  after planning. `camera.saved_frame_wait` remains for socket and held
  key captures, not for the new tabletop pose-only path;
- `evidence.undistort_validate`, `evidence.png_write_hash`,
  `evidence.png_read_verify`, `evidence.undistort_replay` and
  `evidence.live_manifest_reuse`.
- Failed-trial `recovery.clear_view`, `recovery.key_capture`,
  `recovery.restore_plan`, `recovery.restore_motion` and
  `recovery.restore_attempt`; pose-exhaustion `repose.preflight`,
  `repose.motion`, `repose.key_capture` and `repose.landing_assessment`.

`audit.fixed_world_build` should normally occur once per immutable socket
session; dynamic key/arm trajectory checks still run for each candidate.
PNG SHA-256 is calculated from encoded bytes in memory when written,
instead of rereading every new PNG from NAS. Offline readback remains.

Independent camera PNG reads and PNG encoding, writes and in-memory SHA-256
for board, socket, key and raw checkpoints now use at most eight workers by
default. Set
`PRECISION_CAPTURE_IO_WORKERS=1` for the previous serial path or `4` to
replay the older parallel setting; accepted range is 1–8. `camera.saved_frame_read` records each
camera's source-file wait and decode, `evidence.png_batch_write` is the
elapsed batch time, and `evidence.png_write_hash` records every written
camera/source pair. The manifest is written only after every file completes.
Capture-PC SAM/FoundPose processing remains distributed; this worker setting
affects robot-PC file I/O only. The saved after-lift 40-PNG NAS replay in
`LAST_RUN_AUDIT_20261020.md` measured median 20.032 s at four workers and
16.893 s at eight workers across three runs each; live storage contention can
change this result.

The local single-GPU Qwen backend still handles same-camera pairs serially.
Lift, pre-insertion transfer and final released-key labels are **gates**: the
next physical action waits for the corresponding label. After insertion
failure, the saved pre-insertion images are used for XY inference before the
next attempt; no new failure-time image capture is needed. The VLM model is
loaded once at startup. Optional Gemini mode can set `vlm.parallel_views`
to 2–8 in the robot configuration (default 1). This changes Gemini from one
all-camera request to independent same-camera pairs and uses separate API
clients in a bounded worker pool. Per-camera results stay in deterministic
camera order and use the existing conservative per-camera consensus rule. It sends images
to an external service, can hit rate limits, and **does not speed up the
default local Qwen mode**. Setting `parallel_views > 1` with `mode: local` is
rejected. The metric XY grounding backend still requires native-pixel local
Qwen, so Gemini parallelism is for visual labels only.

### Reproducing the timing comparison on the robot PC

Keep the same saved data, model, NAS mount and service state when comparing.
`--check` is static; `--run` moves the robot and must only be used at the
commissioned rig. Use separate fresh run directories for each run.

```bash
cd /home/robot/autodex-precision-insertion
git pull --ff-only origin feat/precision-insertion
PRECISION_CAPTURE_IO_WORKERS=1 bash demo/precision-insertion/run_robot_pc.sh --check
PRECISION_CAPTURE_IO_WORKERS=8 bash demo/precision-insertion/run_robot_pc.sh --check
# Only after physical authorization, readiness and operator approval:
PRECISION_CAPTURE_IO_WORKERS=8 bash demo/precision-insertion/run_robot_pc.sh --run
python3 demo/precision-insertion/report_timing.py \
  /home/robot/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/RUN_ID.timing
```

The two static `--check` commands validate configuration only; they do not
exercise PNG writing or VLM. For a camera/robot-free replay, use the saved
frame benchmark or preflight scripts with explicit real evidence paths.
`evaluate_saved_vlm.py` requires a separately annotated, hash-bound case
manifest and reports **visual-label** behavior, not end-to-end robot latency.
For a prepared manifest and the installed local Qwen model:

```bash
PRECISION_TIMING_LOCAL_ROOT=/tmp/precision-insertion-timing \
  /home/robot/anaconda3/envs/planner/bin/python \
  demo/precision-insertion/evaluate_saved_vlm.py \
  --manifest /absolute/path/to/reviewed_manifest.json \
  --backend local --model-id /absolute/path/to/Qwen3-VL-2B-Instruct \
  --output /tmp/precision-vlm-replay-001.json --profile
python3 demo/precision-insertion/report_timing.py \
  /tmp/precision-vlm-replay-001.json.timing
```

The manifest must have independent human labels and real saved image SHA-256
values. The evaluator now also supports the live released-key visual
`inside_socket` label with its production final-socket crop. The replay profile
separates manifest/image load, model load and individual VLM calls.
Do not report animation/video playback duration or an offline VLM replay as
a physical trial time.

The 2026-10-15 board-9/gap-20 saved-camera timing measurements, input
inventory, reproducible commands and unavailable live stages are in
[`SAVED_REPLAY_TIMING_20261015.md`](SAVED_REPLAY_TIMING_20261015.md).
`replay_saved_timing.py` profiles the production saved-bundle verifier,
catalog loader and PNG worker path on real preserved pixels.
`replay_saved_live_vlm.py` is a separate read-only harness for a later complete
released checkpoint: it uses the live `inside_socket` function and, when a
same-failed-attempt preinsert bundle exists, the production grounded XY core.
Neither tool acquires camera frames or commands the robot. The 2026-10-13
session stopped before these VLM checkpoints, so there are no valid VLM timing
numbers for that session.

## Optimization experiments (one change at a time)

1. First collect one profiled run and separate service/initialization,
   operator wait, socket work, NAS verification, key/plan and VLM time. Repeat
   on the same robot PC, rig, model cache and candidate pool before comparing.
2. Keep all calibrated raw camera frames and multi-view geometric pose evidence
   initially. For VLM **only**, evaluate two or three predetermined cameras
   with visible key tip and socket rim and a useful baseline; compare their
   labels and grounded XY against all-view replay. A dynamic selector should
   use visibility, non-occlusion, camera geometry and prior observed reliability,
   and fall back to more views for disagreement. Do not select just the nearest
   or one high-confidence view. Two separated views are the minimum for metric
   triangulation; three provide redundancy. The original distributed-camera
   benchmark used individual Qwen calls. The current capture4 local backend
   batches the lift and final-label views; verify the actual timing trace
   before assuming another view reduction saves model latency.
3. The fixed socket is measured only at **session startup**, not every trial.
   The current board-9 / 20 mm launch uses one capture and one stock refinement.
   Launch, bootstrap, saved calibration replay, evidence and ledger now accept
   one capture. Its calibration record labels `repeatability_checked: false`;
   the old two-capture comparison remains available when explicitly configured.
4. Catalog source hashes and the complete image-bundle reread have moved out
   of the live hot path under an explicit immutable-NAS assumption. Use
   `verify_sources=True` or the standalone evidence verifier for offline
   handoff auditing. Keep frame-ID/image-source binding; a cached producer
   manifest is not a tamper-resistant signature.
5. Measure whether full-resolution 20-camera original-plus-undistorted PNG
   saving, the second undistortion check, or candidate-by-candidate path
   audit dominates. These are the next candidates for an evidence-format or
   planning-budget change. Keep actual elapsed motion and contact-control
   time distinct from computation time.

For a fair comparison, report total time **to first key attempt** and median
time **per trial** separately. Exclude human placement wait only as a second
reported statistic, not by silently subtracting it from end-to-end time.

## ZeroDex capture4 pre-insertion timing audit (2026-10-29)

The archived four-camera runs from 2026-10-28 measured
`trial.preinsert_label` at 2.84–3.52 s, with the nested Qwen inference at
0.90–1.01 s. The checkpoint also builds a measured-joint Franka/Inspire plus
nominal held-key scene and renders four CAD overlays **for audit**, while the
VLM receives only raw camera crops. A read-only replay of the saved 5 mm
scene on this PC measured the repeated scene build at 0.39–0.42 s and the
warm four-view render at 0.21 s. The first render in that isolated process
took 16.3 s including renderer/CUDA initialization, so it must not be used
as a steady-state per-trial estimate. These are component measurements from
saved data, not a new robot-run timing result.

New nested spans `preinsert.prediction_build` and
`preinsert.audit_overlay_render` separate those two costs on the next live
run. They do not change the VLM input, checkpoint evidence or motion policy.
The capture4 profile now disables the diagnostic CAD overlay by default;
its raw held-scene comparison retains the measured transforms and camera
hashes. The remaining preinsert-label time includes image input, crop
construction, VLM inference, report writing and other checkpoint work.
The larger observed startup costs remain
planner world setup and SAM3 model loading; both have asynchronous warmup
paths that need measurement on a new live run before claiming a speedup.

The live preinsert checkpoint now reuses both just-captured camera arrays:
the current preinsert frames and the earlier after-lift frames from the same
attempt. It still writes the source PNGs and binds each reused array to the
saved frame hash. Saved-data replay keeps the PNG decode path. On four
archived 2048×1536 after-lift views, ten local repetitions measured 0.243 s
median for PNG decoding versus 0.020 s for the in-memory pixel hashes. The
new `preinsert.temporal_pair_input` span records which source was used. The
roughly 0.22 s component difference is not a measured live-pipeline saving.

The four-view Qwen key-box batch added after the archived 2026-10-28 runs
was checked on all four saved XY snapshots of attempt `9542c1bd`. With the
production local backend, four independent camera requests took 1.36–1.93 s
per snapshot, while one four-view batch took 0.57–0.58 s. Thirteen of the
sixteen raw boxes matched exactly; the other three differed by at most
4.1 source-crop pixels. Replaying both sets of answers through the existing
SAM3 ROI worker and XY geometry reproduced the saved serial offsets exactly.
The batched answers kept the same inlier cameras and metric status in all
four snapshots; the largest XY-offset difference was 0.130 mm. This is a
saved-scene result, not a new physical alignment or insertion trial.

On the two human-confirmed 5 mm `20195` episodes, the preferred first three
views did not yield metric XY, so the fourth Qwen request was needed. A
read-only replay of their exact saved crops measured a warm median of
0.840 s for a 3+1 Qwen split versus 0.572 s for one four-view batch
(three repetitions on the first episode). On the second episode, the same
comparison was 0.833 s versus 0.567 s. Seven of eight boxes matched exactly;
the remaining box edge moved 5.12 crop pixels. Replaying both answer sets
through the production SAM3/geometry code returned the same status
(`abstain`), no metric XY, and identical per-view reasons in both episodes.
The raw Qwen answers and times are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/qwen_split_quartet_20195_20261029.json`.
The local batch-capable backend now asks all four key-box questions in one
batch even when SAM still tests the preferred three views first. The fourth
SAM view is still used only when the first three do not yield metric XY.
This changes no motion or XY admission rule. The component saving is about
0.27 s after model warmup; it is not a measured live-session speedup.

A separate SAM3 batch probe on the first snapshot returned masks with IoU
at least 0.9996 against serial inference. Its warm four-view runtime was
0.63 s versus 0.68 s serial. That approximately 0.05 s gain does not yet
justify changing the worker protocol and mask provenance. These read-only
probes and their JSON results are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/`
as `probe_qwen_key_box_batch_20261029`,
`probe_grounding_batch_geometry_20261029`, and
`probe_sam_roi_batch_20261029`.
