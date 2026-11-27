# Current ZeroDex implementation status

This page is the concise completion ledger for the active
`feat/zerodex-precision-insertion` deployment. Detailed operation, stage and
timing contracts remain in [`OPERATIONS.md`](OPERATIONS.md),
[`ACTION_LOOP.md`](ACTION_LOOP.md) and [`TIMING_PROFILE.md`](TIMING_PROFILE.md).
Historical dated reports are evidence, not current run instructions.

Last audited: **2026-11-27** on kds-Z790. The current live branch is
`feat/zerodex-precision-insertion-robotiq`; the paired released-outcome VLM
path is present from commit `1f11a75` onward. Capture4
perception services and the Robotiq state service were reached read-only; no
robot motion was issued during this audit.

## Implemented

| Requirement | Current implementation |
| --- | --- |
| Independent runner | `run_zerodex_capture4.sh` and `run_live.py` run the demo-local state machine. Stock `run_auto.py`, `run_pipeline.py` and `scene_cfg.py` are not entry points. |
| ZeroDex machine binding | `zerodex_capture4.env.sh` owns capture4 serials/host, local cache paths, NAS namespace and the straight-mount FR3 profile. `zerodex_franka_integration.py` changes routing only inside the process. |
| Board-free perception | The default launcher records `scene_mode=board_free`, uses the commissioned robot/camera extrinsics and measured table plane, and removes board-ID/corner fields from the effective launch. The fixed cylindrical socket is localized directly from its four SAM masks, known CAD body and saved support plane (`table_constrained_body`); socket FoundPose is not run or required. Key pose remains the same-session SAM3+FoundPose input to MV-GoTrack. Socket and key prompts do not require an object to be “on a board,” and both upright and side-down cylindrical-key conditioning use the same measured support plane. |
| End-effector modes | ZeroDex defaults to the commissioned `robotiq_2f85`. The arm remains on the local FR3 FCI daemon, while the physical Robotiq RS-485 adapter remains owned by capture4 and is commanded through its isolated ports 5565/5566. `franka_gripper` and `inspire` remain separate selectable modes; neither implementation was removed. Every mode has an independent URDF, joint convention and candidate namespace. |
| Tabletop grasp coverage | The Robotiq catalogue contains eight symmetry-equivalent radial approaches for the upright/end-down class (`000`) and two top-down wrist flips for the side-down class (`001`). The measured 6D key pose and frozen support plane reconstruct every grasp at runtime; the files remain stable candidate identifiers. All ten passed the 20 mm endpoint catalogue screen. The cuRobo model uses the official 2F-85 meshes, mimic kinematics and CAD-derived collision spheres. Inspire keeps its independent screened BODex catalogue and the stock Franka Hand keeps its own analytic catalogue. |
| Four-camera evidence | Every checkpoint captures all four commissioned cameras. Socket-pose inliers and measured residuals may select a smaller inference subset; camera evidence is not discarded or permanently blacklisted. |
| Full action loop | The active profile supplies all ten stage handlers: key planning, pickup, lift, held relation, transfer/insertion, retry diagnosis, held return, transfer recovery, repose planning and repose execution. |
| Same-trial correction | Pre-insertion capture, metric endpoint/partial-shape observation, bounded wrist correction, recapture and measured-motion feedback run before the first insertion stroke. The active profile permits a 10 mm per-update step and has no cumulative XY cap. |
| Continuous key tracking | The `mvgotrack_axis_alignment_capture4.env.sh` profile sends the accepted same-session SAM3+FoundPose 6D key pose and mesh to the isolated capture4 tracker immediately after pose/state admission. Capture4 initialization and its first pose overlap local cuRobo candidate preflight; the post-planning join completes before pickup can be selected. At the stopped pre-insertion hold it consumes a newer pose, reduces the cylinder pose under its D-infinity symmetry, and supplies XY plus optional axis correction. |
| Tracker camera-world binding | The capture4 manager publishes a canonical identity of the four selected intrinsic/extrinsic entries with every tracker session. The main runtime derives the same identity from the active `PRECISION_CALIB_DIR` and checks the first pose before pickup. Formatting and unused camera entries are ignored; this prevents combining FoundPose and tracking poses expressed in different calibrated world frames without adding a confidence or task-outcome threshold. |
| Distributed visual inference | `mvgotrack_axis_alignment_capture4_remote_qwen.env.sh` keeps MV-GoTrack and one four-camera persistent Qwen worker on capture4. Camera IDs share one model and same-host batch path; kds retains cuRobo, the state machine and Franka control. The deployment is isolated under `/home/capture4/precision_insertion_qwen` and does not modify shared capture services or Conda environments. |
| Insertion execution | The corrected hold is replanned along the measured socket axis. If the corrected 20 mm endpoint/path is unavailable, the profile tries the configured 10 mm axial fallback. |
| Failed-attempt recovery | A released insertion failure first reuses its saved pre-insertion image for next-attempt XY advice, then perceives and restores the released key to the original tabletop pose. Model, saved-input or advice-validation errors are recorded as unavailable diagnostics and cannot suppress that recovery. |
| Labels | Manual collection stores the independent Qwen result too. For each camera the released-outcome model receives the saved initial raw key crop followed by the final clear-view socket crop; no CAD overlay is shown to the model. Grasp, pre-insertion arrival, descent-time tip entry and released-key `inside_socket` outcome remain separate fields. |
| Storage | Runtime JSON and environment variables resolve NAS paths only below `/mnt/paradex2/hyunsu/precision_insertion`; repeated-use model caches and source checkout remain local. Capture4 acquisition is the documented external ParaDex exception. |
| Timing | Nested spans cover startup, perception, planning, motion, camera, VLM, evidence writes, human waits and cleanup. Overlap-aware occupied time prevents parallel PNG writes from being counted as serial latency. Rejected grasp candidates retain their cuRobo stage, failure stage and timing records instead of reducing the result to only `transfer_unreachable`. |

## Verified evidence

- The static ZeroDex preflight passes for the Robotiq, stock Franka Hand and
  Inspire contracts. The Robotiq mode has 10/10 eligible deterministic candidates: eight radial approaches
  for upright/end-down and two wrist flips for side-down;
  Inspire retains 126 catalogued and 114 eligible candidates. Both report
  four configured cameras, board-free mode, no legacy runtime path and no NAS
  reference outside `/mnt/paradex2/hyunsu`.
- The explicit
  `PRECISION_EXPERIMENT_PROFILE=mvgotrack_axis_alignment_capture4.env.sh`
  preflight also passes with the default `robotiq_2f85` route. It resolves
  `xy_estimator=mvgotrack`, the capture4 endpoint at port 5591, the isolated
  remote manager/session paths and both tabletop grasp candidates. Selecting
  `PRECISION_END_EFFECTOR=inspire` remains an independent supported route; it
  is not removed or silently translated to a parallel-gripper command.
- The corresponding remote-Qwen profile also passes static preflight. Its
  generated runtime maps all four serials to one capture4 host/process. A real
  SSH worker smoke test loaded Qwen3-VL-2B-Instruct on the RTX 5060 Ti and
  returned the expected synthetic-image label. The latest 2026-11-27 repeat
  measured 3.592 s cold ready time and 0.589 s inference round trip; an earlier
  instrumented run recorded 4.30 GB peak CUDA allocation. This verifies
  process, model, CUDA and transport wiring only.
- The current calibration profile resolves the final extrinsic solution at
  `~/calibration/zerodex_capture4/extrinsic/20261118_232125/solution_filtered`
  and hand-eye transform at
  `~/calibration/zerodex_capture4/handeye/20261118_233621/solution/C2R.npy`.
  All four selected world-to-camera transforms and C2R reload as finite
  SE(3) matrices (rotation determinant 1.0 and orthogonality error below
  `5e-16`). Their intrinsics all declare 2048x1536 images. The current
  board-free support surface was measured once from all four views and all 54
  ChArUco corners; its saved plane RMS is 0.424 mm and its runtime height is
  0.063360839 m in the robot frame. The board is measurement equipment for
  this calibration artifact, not a live-perception dependency.
- A read-only 2026-11-26 capture4 probe reported all isolated ports ready:
  `5009, 5480, 5481, 5482, 6894, 15006, 15007, 16893`. The four configured
  serials remain `26053248, 25322639, 25305462, 25322642`.
- A fresh, robot-free four-camera run on 2026-11-27 exercised the default
  board-free socket path with the lit physical 5 mm-gap fixture. All four SAM
  masks contributed to the support-plane/CAD fit. The frozen socket origin was
  `[0.559026, 0.105217, 0.063251] m` in the robot frame, its local +Z axis was
  `[-0.011407, -0.001576, 0.999934]`, and the mean whole-body mask IoU was
  `0.9273` (per-view `0.9437, 0.9146, 0.9391, 0.8966`). The saved source is
  `per_capture_table_constrained_body`; the session reload check passed. The
  key was already present during this one-off diagnostic, so it validates the
  estimator and runtime binding rather than the production key-absent socket
  capture procedure. No robot command was issued and IoU remains a diagnostic,
  not a newly added veto.
- A later read-only capture4 check enumerated those same four cameras and
  reported `stream_pid=null`, `tracker_pid=null`. This is the expected idle
  state before a same-session accepted key pose initializes tracking; the
  audit did not start MV-GoTrack from a stale or fabricated pose.
- Physical attempt `attempt_46ab969b91cd48f0b46132b9df113df8`
  bound a same-session four-view key capture, accepted FoundPose initializer,
  matching capture4 calibration identity, first native-valid tracker pose,
  Robotiq pickup/squeeze feedback and a continuous local tracker journal. The
  journal contains 24,027 messages over 1,678.78 s with native-valid fraction
  1.0 and one unchanged calibration identity. Pickup reached a measured jaw
  command of 0.65255 with `gripper_grasping=true`; the following measured-lift
  preflight, rather than tracking, stopped the trial.
- That measured-lift stop was reproduced from the archived joint state. The
  old streamed pickup stopped with a 0.01305 rad arm endpoint residual, at
  which cuRobo reported `INVALID_START_STATE_WORLD_COLLISION` against the
  table. Replaying the nominal endpoint with the measured jaw width passed;
  the arm-residual sweep passed through interpolation factor 0.7 and collided
  from 0.8 onward. The active ZeroDex adapter now continues the already
  planned stream to a 0.002 rad endpoint tolerance. This is a targeted motion
  completion fix, not an added perception or task-success gate, and still
  requires one physical rerun.
- Robot-free four-camera requests on 2026-11-27 reached the isolated camera,
  SAM3 and FoundPose services. At the commissioned 2,500 us exposure the
  frames were nearly black (per-view gray means 2.3--15.1 on 0--255), all key
  masks were empty, and no physical 6D key pose was claimed. A reversible
  project-only exposure audit found one key pose at 10,000 us; at 30,000 us it
  found key poses in 2/4 views and a socket pose in only 1/4 views. This is not
  sufficient evidence for the intended multi-view socket/key measurement.
  The stack was restored to 2,500 us. Restore workspace illumination before
  the physical run; the raw audit is under
  `development/zerodex_capture4/probes/20261127_exposure_audit`. These counts
  are current acquisition evidence, not new perception thresholds.
- The isolated capture4 MV-GoTrack deployment now carries the same active
  four-camera calibration as kds-Z790. Both machines compute the identity
  `precision_calib_v1:26053248,25322639,25305462,25322642:7d412a3ae834d0e7b53f9cfd4576fe1e4b63f409d95ec4e9ecede07322384f70:a43905fc73e2186548b70a7402ad84eb7bf14eed7a261a37a192605c5bf95a4d`.
  The focused calibration/track tests passed 15/15, and the active board-free
  gripper/MV-GoTrack/Qwen profile passed static preflight.
- The source-boundary audit at the current branch head reports zero
  stock-entrypoint changes and zero unexpected shared-runtime changes. The
  reviewed adapter contract now explicitly includes the official Robotiq
  visual and collision meshes used by its selected URDF; the stock AutoDex
  entry points remain unchanged.
- The current worktree passed the complete demo suite with
  **1655 passed, 4 skipped**. This includes regression coverage for the ten
  deterministic gripper candidates and for retaining the measured board-free
  support-plane tilt in key conditioning, tabletop classification, grasp
  construction and failed-attempt restoration, plus same-camera paired
  initial-key/final-socket Qwen requests and replay verification.
- A real local-CUDA synthetic gripper diagnostic now preserves the FR3
  profile's 32 trajectory-optimization seeds even though the ZeroDex planner
  binds explicit checkout-local robot YAML paths. The former generic
  explicit-YAML fallback selected 1,024 seeds and exhausted GPU memory. With
  the corrected 32-seed setting and stock Franka hand/finger collision
  spheres, the `side_down` analytic grasp passed table/world collision, IK,
  approach and 10 cm held lift in two hypothetical workspaces. The following
  transfer query was unreachable in those arbitrary synthetic socket
  placements, so this is pickup/lift evidence, not a full insertion result.
- A focused 2026-11-26 completion audit additionally passed **167 tests**
  covering board-free key/support-plane conditioning, analytic gripper
  grasps, planner/preflight assembly, MV-GoTrack initialization and pose use,
  symmetry-aware endpoint/axis correction, insertion/release/outcome/retry,
  and both end-effector executors.
- A 2026-11-27 local-CUDA preflight used the freshly measured upright key and
  frozen socket pose with the expanded eight-candidate upright catalogue. It
  inspected candidates `000` through `006` and stopped at the first complete
  pass. Candidates `000`--`003` had no pickup IK, `004` reached pickup IK but
  not the held lift, `005` passed pickup/lift/transfer but reached a joint
  limit at the 10 mm axial waypoint, and `006` passed pickup, held lift,
  transfer, the complete sampled axial path and the nominal 20 mm insertion
  endpoint. Candidate `007` was not inspected after that success. The saved
  result is offline cuRobo evidence, not a physical grasp or insertion label.
- Real 5 mm-gap episodes exercised the full pickup/transfer/insertion/release
  route. In particular, attempts
  `attempt_3c8ff1b830d347a2b8c45aa7ae1a5363` and
  `attempt_43e8596aec534c959c18b3a5e16f5f92` store human-observed insertion
  success for fixed grasp `table/0/20195`.
- The 2026-11-03 read-only outcome audit found 16 pickup attempts: 13/15
  manually labelled lifts succeeded, 12 reached the labelled pre-insertion
  hold, 9 executed an insertion stroke and 2 ended with the released key
  inside the socket. Only 1/9 executed insertion strokes has the newer,
  independent `tip_entry_success` label. These are a development funnel, not
  a success-rate estimate, because conditions and code revisions changed
  across the attempts.
- The corresponding Qwen/manual audit has 15 paired lift labels (13/15
  agreement), 12 paired pre-insertion labels (12/12, all positive) and 6
  paired released-key labels (6/6, including two positives and four
  negatives). Reaggregating the saved **primary per-view votes** with the
  current two-view lift rule, without rerunning Qwen, gives 13/14 agreement,
  one abstention, no false positives and one false negative. This isolates the
  aggregation-policy change but is not a current-model evaluation. Missing
  negative pre-insertion examples and the small released-key set still rule
  out autonomous label claims.
- Historical closed-loop evidence contains 57 saved observations. Only four
  are metric multi-view endpoint corrections; most are single-view planar or
  measured-motion temporal hypotheses. The two observed insertion successes
  both acquired a metric endpoint before converging. Large single-view drift
  in older failed attempts predates commit `3c87fd9`, which now permits one
  active probe and returns to the original hold unless the recapture produces
  a real multi-view 3D endpoint. That revised behavior is covered by a
  regression test but has not yet been exercised on the powered robot.
- This physical evidence proves that the route has run; it does **not** prove
  robustness. Later saved attempts include both grasp and insertion failures,
  which remain part of the evaluation dataset.

## Not currently verified or externally blocked

| Item | Current evidence / next action |
| --- | --- |
| Current hardware execution | The RT kernel is active and `enp4s0` is configured for the FR3. A paired read-only sample combined the live FR3 7-axis state with the capture4 Robotiq position without fault. The local FCI daemon is not currently listening, so it must be restarted before motion. The physical Robotiq is not a Franka Hand; the daemon message that its built-in gripper is unavailable is expected. |
| Local cuRobo GPU preflight | Planner construction succeeds with the new 8-DoF FR3+Robotiq model, official 2F-85 geometry and the restored FR3 32-seed profile. A fresh measured-scene pickup/transfer/insertion preflight still has to be run after the clean socket/key recapture; model construction alone is not a trajectory result. |
| Physical Robotiq/MV-GoTrack run | Same-session FoundPose-to-tracker initialization, continuous four-view tracking, pickup and measured Robotiq squeeze feedback have been physically exercised. The latest attempt stopped at post-squeeze measured-lift preflight because the old trajectory follower left a 0.013 rad arm endpoint residual. The active 0.002 rad completion fix and tracker reinitialization fixes have not yet been exercised through lift, transfer, correction, insertion and release. Rerun from a clean socket capture and current key pose; do not initialize from an archived pose. |
| Reproduction bundle | The writable NAS manifest now archives merged AutoDex commit `299ddc3`, ParaDex `e2cd4fd`, and nvdiffrast `253ac4f`, and retains 6,861 declared asset files. `--refresh-code-only` no longer scans or mutates the established asset/rig inventories. Both NFS and SMB full-tree verification later entered remote-filesystem response waits while traversing the runtime view, so the post-refresh metadata/SHA256 audit remains to be repeated when NAS metadata service is responsive; the manifest and all three declared code archives are present from both mount views. |
| VLM accuracy claim | Saved replay supports development, but class balance and independent held-out trials are insufficient for a general accuracy claim. Manual labels remain authoritative during data collection. |
| Robust insertion claim | Two observed successes are not a controlled success-rate experiment. Repeat fixed-grasp/fixed-socket trials and report stage-wise confidence intervals before making this claim. |
| All recovery orientations | Code paths and assets exist, but every reset/reorient direction has not been physically exercised on this rig. Do not present catalogue/preflight evidence as physical success. |

## Read-only audit commands

These commands do not open cameras, CUDA or robot devices:

```bash
cd /home/hyunsu/autodex-code
python3 demo/precision-insertion/audit_source_boundary.py
source demo/precision-insertion/config/zerodex_capture4.env.sh
./demo/precision-insertion/run_zerodex_capture4.sh --check
```

When the NAS is writable, refresh and verify the reproducible project bundle:

```bash
source demo/precision-insertion/config/zerodex_capture4.env.sh
"$PRECISION_CHECK_PY" demo/precision-insertion/prepare_zerodex_nas_project.py \
  --refresh-code-only --nas-root "$PRECISION_NAS_PROJECT_ROOT"
"$PRECISION_CHECK_PY" demo/precision-insertion/prepare_zerodex_nas_project.py \
  --verify-project-sha256 --nas-root "$PRECISION_NAS_PROJECT_ROOT"
```
