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

If `run_robot_pc.sh` launched the session, its **separate**, shell-side service
startup profile is printed as `/tmp/precision-insertion-RUN_TAG-service-timing.tsv`.
It contains file preflight, each capture host, Franka daemon, Python session,
and launcher/cleanup duration. The Python trace begins only when `run_live.py`
starts; do not compare it to a stock process including shell startup without
also including this service profile.

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
time. The Python trace itself writes files and has a small overhead. The
shell-side TSV is a different clock/process and is not nested in the trace.

## Where the extra time enters versus stock AutoDex v8

| Executed stage | Stock `run_pipeline.py` / `run_auto.py` | Precision-insertion live demo | Timing span |
| --- | --- | --- | --- |
| Process timeline | Already uses `PipelineTrace` and detailed planning/execution spans. | Reuses the same trace format but a separate demo run and sidecar. | `live.session` |
| Service startup | Existing AutoDex camera/robot services may already be running. | Robot-PC launcher checks/starts services on five capture hosts plus the Franka daemon. | Shell TSV |
| ChArUco | Board/camera measurement used for tabletop geometry. | One board capture and fit at session start; the saved frames are also used in the frozen calibration. | `board.*` |
| Socket | No fixture-pose estimation or socket refinement. | Currently two 20-camera socket acquisitions, each with SAM/FoundPose; two stock silhouette refinements, then repeatability/frozen collision world. | `socket.capture`, `socket.refine_capture` |
| Source/evidence checks | Stock live entry points do **not** hash the complete grasp pool and session image bundle on each run. Stock trace does hash an artifact when explicitly written through its artifact helper. | Full endpoint catalog source checks before camera startup and again at ledger open, image/frame binding, evidence bundle SHA256 write/verify/reverify and source hashes. | `startup.catalog_precheck`, `catalog.pose_source_validation`, `socket.evidence_verify`, `ledger.*` |
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

## Optimization experiments (one change at a time)

1. First collect one unmodified profiled run and separate service/initialization,
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
4. Profile the two catalog/evidence verification passes before changing them.
   A future immutable, content-addressed session snapshot could perform one
   full hash at opening and reuse the attested binding during that session;
   do not replace provenance with an untracked cache or remove the frame-ID
   binding. The 188 MB evidence bundle alone can make repeated NAS reads
   costly. Keep producer and recipient file checks for NAS handoff separate
   from per-trial runtime checks.
5. Only after those measurements consider reducing model reloads, camera file
   copies or the number of planning candidates. Keep actual elapsed motion
   and contact-control time distinct from computation time.

For a fair comparison, report total time **to first key attempt** and median
time **per trial** separately. Exclude human placement wait only as a second
reported statistic, not by silently subtracting it from end-to-end time.
