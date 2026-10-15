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
  `inclusive_s`, `self_s` and failure count.
- Stock trace `manifest.json` and `timeline.json`.

The sibling location means startup failures can be timed even if no session
ledger was opened. On the robot PC, after a live attempt:

```bash
cd /home/robot/autodex-precision-insertion
python3 demo/precision-insertion/report_timing.py \
  /home/robot/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/RUN_ID.timing
```

`run_robot_pc.sh --run` now runs only the Python pipeline. Start/check/stop
long-lived services independently with `manage_robot_pc_services.sh`; its
service-start report is `/tmp/precision-insertion-services-*.tsv`. The
manager starts the five capture hosts concurrently. Its timing is separate
from the Python session; do not add service startup to a warm-run comparison.

On the robot PC, `run_robot_pc.sh` sets `PRECISION_TIMING_LOCAL_ROOT` to
`/tmp/precision-insertion-timing` by default. The stock trace writes its
events there during the run; orderly exit copies the complete sidecar once
to the NAS path above. If killed before cleanup, recover the local trace;
the NAS copy may not exist. This avoids a network write for every event.

`inclusive_s` contains child spans. Use `self_s` to rank mutually exclusive
costs; adding all inclusive values double-counts. `socket.capture` is the
entire camera request plus saved-frame/provenance work, while
`camera.foundpose_request`, `camera.saved_frame_wait` and
`camera.frame_binding_verify` separate its children. `socket.refine_capture`
is the subsequent silhouette refinement; it is **not** included in
`socket.capture`. `vlm.infer` carries stage, camera ID, model and image count,
so the report shows both total per-checkpoint cost and slow views. Parent
`trial.*_label` spans additionally include parsing/evidence/manual wait.

The raw event log is the primary evidence if the process is interrupted.
`timing_summary.json` is written on orderly exit or Python exception, not a
power loss or SIGKILL. Timings are elapsed wall time, including operator wait,
network, NAS I/O and robot motion; they do not measure isolated GPU kernel
time. The Python trace itself writes files and has a small overhead, even on
local storage. The service TSV is a different clock/process and is not
nested in the trace.

## Where the extra time enters versus stock AutoDex v8

| Executed stage | Stock `run_pipeline.py` / `run_auto.py` | Precision-insertion live demo | Timing span |
| --- | --- | --- | --- |
| Process timeline | Already uses `PipelineTrace` and detailed planning/execution spans. | Reuses the same trace format but a separate demo run and sidecar. | `live.session` |
| Service startup | Existing AutoDex camera/robot services may already be running. | Independent manager starts missing daemons; pipeline runs reuse them. | Service TSV, outside run trace |
| ChArUco | Board/camera measurement used for tabletop geometry. | One board capture and fit at session start; the saved frames are also used in the frozen calibration. | `board.*` |
| Socket | No fixture-pose estimation or socket refinement. | Currently two 20-camera socket acquisitions, each with SAM/FoundPose; two stock silhouette refinements, then repeatability/frozen collision world. | `socket.capture`, `socket.refine_capture` |
| Source/evidence checks | Stock live entry points do **not** hash the complete grasp pool and session image bundle on each run. Stock trace hashes an artifact only when explicitly written through its helper. | Live catalog parses once and assumes immutable NAS sources; explicit offline audit still rehashes candidate files. Socket evidence is written/verified once, then passed to the ledger in memory. Producer-validated live key/raw manifests are reused in-session; offline replay still checks bytes. | `startup.catalog_load`, `socket.evidence_write`, `socket.evidence_verify`, `ledger.small_binding_hashes`, `evidence.*` |
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
Important nested stages include:

- `planning.candidate_load`, `planning.v8_pickup`, `planning.held_lift`,
  `planning.held_transfer`, `planning.axial_jacobian`,
  `planning.held_path_audit` and its child `audit.geometry_samples`;
- `audit.fixed_world_build` versus `audit.fixed_world_lookup`,
  `audit.key_model_lookup`, `audit.hand_urdf_fk`, `audit.wrist_fk_rollout`;
- `camera.foundpose_request`, `camera.saved_frame_wait`,
  `camera.frame_binding_verify`, `camera.undistort_generation` and
  `socket.refine_capture`;
- `evidence.undistort_validate`, `evidence.png_write_hash`,
  `evidence.png_read_verify`, `evidence.undistort_replay` and
  `evidence.live_manifest_reuse`.

`audit.fixed_world_build` should normally occur once per immutable socket
session; dynamic key/arm trajectory checks still run for each candidate.
PNG SHA-256 is now calculated from encoded bytes in memory when written,
instead of rereading every new PNG from NAS. Offline readback remains.

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
   triangulation; three provide redundancy. Qwen currently processes each
   camera pair/grounded view serially, so this is a likely scaling lever.
3. The fixed socket is already measured only at **session startup**, not every
   trial. Reducing two captures to one would remove one acquisition/refinement
   but also remove the independent repeatability check. The present launch,
   bootstrap, evidence and ledger contracts all require at least two; changing
   only `socket_capture_count` in JSON will fail. For 20 mm bring-up, a
   *separate, explicitly marked single-capture exploratory mode* is reasonable
   after measured pose-quality review, never represented as repeatability-
   verified or silently reused for tighter gaps.
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
