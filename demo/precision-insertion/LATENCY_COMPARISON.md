# Live latency audit: AutoDex v8 vs. precision insertion

This is a code-path comparison, **not** a measured speedup claim. The
2026-10-15 robot-PC runs have timing traces; the latest ended after key capture
but before planning. Collect paired warm runs on the same robot PC
before ranking later causes by seconds. Use `report_timing.py` on the run's `.timing` directory and
compare startup, first trial, and later trials separately. Include operator
wait and physical motion in the end-to-end total, but also show each on its
own; neither is GPU inference time.

| Executed operation | AutoDex v8 (`src/execution`) | Precision demo (`demo/precision-insertion`) | Extra cost, and whether it is necessary |
| --- | --- | --- | --- |
| Daemons before Python | `run_auto.py` assumes the camera/robot daemons are available; its trial loop does not start them. | The old combined robot-PC launcher checked/started services serially on five hosts and stopped its own services at exit. The current **separate** manager starts hosts concurrently and keeps them warm across runs. | Old startup was a launcher cost, not an insertion algorithm cost. Compare warm runs or report service TSV separately. |
| Python/CUDA/Qwen start | One FoundPose object and one `GraspPlanner` are constructed at run startup. | The FR3 planner plus local Qwen model are loaded before the camera loop. The socket renderer is initialized, then the key object is initialized once on object switch. | Qwen load is new and can dominate first-run startup, but should not recur every trial. `startup.franka_planner_core`, `startup.vlm_model_load`, `socket.model_init`, `key.model_init`. |
| Board | Optional one-time ChArUco preflight, followed by stock key perception. | One board capture; calibration derives the frozen table scene. The early board fit now runs only when an empty board must be accepted before the operator places the socket. | The active board-9/socket-present path no longer repeats the fit. Its removed early fit took 2.227 s in the failed 2026-10-15 run; this is an observed old-path duration, not a measured speedup. |
| Fixed socket | No socket object, pose or collision world. | One full-rig SAM/FoundPose acquisition followed by one 100-iteration stock silhouette refinement and frozen collision pose; once per session. | Insertion-specific startup cost. Repeatability is not measured in the current board-9 / 20 mm launch. `socket.capture`, `socket.refine_capture`. |
| Initial static asset checks | Existence/frame checks for the target and FoundPose representation. | The validated launch's cylinder fixture contract is reused at session-start, avoiding a second full contract check. Calibration still hashes fixed meshes. The catalog is parsed once without candidate-source hashes in the live path. | Remaining cold-start I/O includes a few repeated CAD reads and small session binding hashes. Offline handoff may still use full hashes. `startup.launch_config`, `ledger.small_binding_hashes`. |
| Saved socket evidence | Stock ChArUco/preflight writes images but does not construct the insertion-specific board/socket evidence bundle. | Writes board, two socket image sets, masks, payloads, calibration and manifest. The producer passes the small manifest forward without rereading all newly written PNGs. The bounded PNG worker setting now applies to these writes. | The offline full verifier remains for handoff/replay. The latest live run spent 189.651 s saving socket/board evidence on NAS with the previous serial writer; speedup from bounded workers has not been measured live. `socket.evidence_write`, `evidence.png_batch_write`. |
| Fresh key recognition | One `trigger_init` SAM/FoundPose pass per trial; stock may add an automatic ChArUco reposition image. | Fresh key SAM/FoundPose on the open cameras, then mask/pose/frame admission and evidence write. After a failed restoration, it runs again because the tabletop pose can change. | Camera/FoundPose is mostly shared; additional admission and saved provenance are new. Key model init no longer repeats for the same object. `key.capture`, `key.pose_admission`, `key.evidence_write`. |
| Candidate pool | Stock `planner.plan` loads v8 candidates and searches until a pickup succeeds; coverage can skip tested grasps. Stock `load_candidate` walks the complete object tree even with `candidate_order` set. | Offline-screened v8+20 mm catalog restricts the pool. The live override now reads the exact three NPY files of each catalog-selected grasp, without per-candidate whole-tree NAS walking; stock pickup planning is then called. | A candidate can pass stock pickup but fail insertion planning and force another pickup plan. Report attempts and time per candidate, not just trial total. `planning.candidate_load`, `planning.v8_pickup`. |
| Pre-execution held path | Stock planner includes its normal lift preflight. | Because squeeze changes hand joints, the demo additionally plans a held-hand lift, Cartesian transfer to socket, Jacobian axis stroke, then samples the held full key/hand against table/socket. | This is the largest **algorithmic** difference in the planning loop. `planning.held_lift`, `planning.held_transfer`, `planning.axial_jacobian`, `planning.held_path_audit`. Shorten only if the experiment intentionally accepts less path evidence. |
| Collision-world construction | cuRobo world setup is already timed in the stock planner and may reuse MotionGen according to world structure; moving from free key to held key still requires a world update. | The session socket/table world is frozen, and the demo additionally constructs Coal/solid-occupancy models for sampled whole-hand/key checks. Fixed models now cache for the session; dynamic pose checks still run for each candidate. | Do not count every `update_world` as a full planner initialization. Check stock `PlanResult.timing.world_setup_s` plus `audit.fixed_world_build/lookup`, `audit.geometry_samples`. |
| Fixed-hand cuRobo model | Stock `GraspPlanner._locked_motion_gen_for_hand` caches at most four models keyed by the **exact** six Inspire joint values. A cache miss creates and warms a separate lock-joint MotionGen. | Held transfer requests `lock_hand=True` with the candidate's grasp-specific finger values. Different candidates can therefore miss that cache even though the socket world is unchanged. The returned `hand_lock_prepare_s`, `hand_lock.cache`, and world-update timings are retained in the transfer preflight record. | Likely large on the first distinct grasp, but not measured yet. Reusing the fixed socket world does **not** make different locked-hand robot models interchangeable. Rank `planner_timing.hand_lock_prepare_s` before changing the hand-lock cache policy. |
| Runtime file checks | Stock candidate loader reads candidate arrays/records but does not hash the entire pool at every trial. | The live catalog now assumes immutable NAS and skips per-row source hashes and directory scans; strict source checks remain opt-in for offline audit. Small session/catalog JSON binding hashes remain. | Previously avoidable O(pool-size × files × NAS) startup cost has been removed from the live path. A repeated `task_geometry.json` read/validation per target and repeated fixture CAD checks remain measurable, likely smaller. |
| After-lift decision | Stock optional ChArUco heuristic or manual label after lift. | Fresh full-rig after-lift snapshot, exposure-bound state, original+undistorted PNG save, and same-camera temporal-pair Qwen label. | New camera/I/O/inference costs, but replaces an unreliable grasp label. `trial.after_lift_capture`, `evidence.*`, `trial.lift_label`, `vlm.infer`. |
| Preinsert decision | Stock has no socket transfer checkpoint. | Held transfer motion, fresh preinsert snapshot, temporal-pair Qwen label; preinsert images are retained for a failed-trial XY update. | Additional physical motion plus camera/inference. `trial.transfer_motion`, `trial.preinsert_capture`, `trial.preinsert_label`. |
| Final decision | Stock task-specific placement/coverage logic, not insertion imagery. | Guarded 20 mm stroke, hand release, clear-view motion, full-rig final snapshot and initial/final Qwen label. | New task work. `trial.guarded_insertion_motion`, `trial.release_and_clear`, `trial.final_clear_view`, `trial.final_capture`, `trial.final_label`. |
| Full-rig checkpoint image processing | Stock `trigger_init` saves its FoundPose frames; ordinary lift labeling need not persist two copies of every checkpoint frame. | Each raw checkpoint undistorts every camera image, recomputes the undistortion in the writer to compare pixels, and saves **both** original and undistorted full-resolution PNGs. In-session consumers now reuse producer manifests. | CPU plus NAS I/O scale with camera count, image size and checkpoint count. The encode-once hash removes immediate PNG rereads; the second undistort and two-image format are still present. `camera.undistort_generation`, `evidence.undistort_validate`, `evidence.png_write_hash`. |
| VLM view count | No local Qwen per-camera temporal-pair loop. | `observer._per_view_consensus` invokes the same Qwen model **serially** for each camera pair at a checkpoint; retry grounding has its own per-view calls. | If 20 views are admitted, label inference can involve 20 sequential calls per checkpoint. This is a plausible large cost, not yet a measured bottleneck. First compare fixed 2–3 geometrically useful views against all-view replay; avoid reducing geometric key/socket pose capture by accident. `vlm.infer` by stage/camera. |
| Failure recovery | Stock coverage/reorient may attempt a different grasp or pose. | A failed released insertion requires key re-observation, physical restoration/repose to original tabletop state, a new key capture, and next-trial accumulated XY. | Time belongs to the failed attempt and is not a verification overhead. Distinguish it from the next trial's planning/perception when reporting throughput. |
| Timing itself | Stock `PipelineTrace` emits event files/artifacts. | Same trace format with more nested events. The run script now writes event files to local `/tmp` and copies once to NAS on orderly exit. | Fine-grained tracing adds small local I/O; writing each span over NAS was unnecessary. Service TSV and Python trace use different processes. |

## Priority for the next measured run

The active 20 mm path has already removed these extra checks: an inference-duration
cross-camera skew veto for stationary captures, a duplicate board-9 ChArUco
fit, the immediate full session PNG readback, repeated live catalog-source
hashes, and repeated raw checkpoint PNG readback. Source frame IDs, image
hashes produced at capture/write time, calibrated camera identity, and the
offline full verifiers remain available. The partial-board path still has
two socket FoundPose captures because it presently uses their independent
pose agreement to freeze the fixture; changing to one capture also requires
changing calibration and ledger contracts and loses that observation.

The 2026-10-15 failed run measured 45.960 s for the first FoundPose response,
2.572 s for the second, 29.670 s total waiting for saved camera PNGs, and
2.227 s for the now-removed early board fit. It did not reach evidence writing,
key recognition, planning, VLM, insertion or recovery, so their live times
remain unmeasured.

The later 14:05 run reached fresh key recognition. Its three FoundPose requests
took 8.994 s total, but reading the capture-PC PNGs took 97.187 s and saving
board/socket/key evidence took 240.536 s on NFS. The new four-worker file I/O
path addresses the serial robot-PC read/write component; it has only been
checked with saved images, not a repeat live run. See
`PERCEPTION_FAILURE_ANALYSIS_20261015.md` for the measured stage breakdown and
feedback failure.

1. Separate **cold session startup** (daemon manager, CUDA/Qwen load, board,
   socket, evidence) from **warm trial** (key perception onward). Socket is
   measured once per session, not per trial.
2. For each trial, report candidate count and `planning.v8_pickup`,
   `planning.held_transfer`, `planning.axial_jacobian`,
   `audit.geometry_samples` self-time and the transfer report's
   `planner_timing.hand_lock_prepare_s`. If many
   candidates fail only at insertion planning, an offline pre-ranking or a
   bounded live candidate budget is worth testing; it changes coverage, so
   record the skipped candidates.
3. Compare all-view Qwen against a fixed, calibrated 2–3-view subset on the
   **same saved frames**, including disagreement and XY error. Camera
   acquisition for FoundPose and view selection for VLM are distinct knobs.
4. Quantify checkpoint `camera.undistort_generation`,
   `evidence.undistort_validate`, PNG encode/write and `vlm.infer` separately.
   If evidence dominates, use a deliberate evidence-format change rather than
   dropping the frame-ID/pose binding.
5. Keep fixed-world caching and catalog/manifest reuse. They remove repeated
   computation while preserving offline strict audit; never cache dynamic
   key pose, current joint path, or contact result across trials.
