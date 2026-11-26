# Current ZeroDex implementation status

This page is the concise completion ledger for the active
`feat/zerodex-precision-insertion` deployment. Detailed operation, stage and
timing contracts remain in [`OPERATIONS.md`](OPERATIONS.md),
[`ACTION_LOOP.md`](ACTION_LOOP.md) and [`TIMING_PROFILE.md`](TIMING_PROFILE.md).
Historical dated reports are evidence, not current run instructions.

Last audited: **2026-11-26** on kds-Z790. The read-only capture4 service
probe passed; no local Franka daemon was running and no robot command was
issued during this audit.

## Implemented

| Requirement | Current implementation |
| --- | --- |
| Independent runner | `run_zerodex_capture4.sh` and `run_live.py` run the demo-local state machine. Stock `run_auto.py`, `run_pipeline.py` and `scene_cfg.py` are not entry points. |
| ZeroDex machine binding | `zerodex_capture4.env.sh` owns capture4 serials/host, local cache paths, NAS namespace and the straight-mount FR3 profile. `zerodex_franka_integration.py` changes routing only inside the process. |
| Board-free perception | The default launcher records `scene_mode=board_free`, uses the commissioned robot/camera extrinsics and measured table plane, and removes board-ID/corner fields from the effective launch. Socket and key prompts do not require an object to be “on a board.” |
| End-effector modes | ZeroDex defaults to `franka_gripper`; `PRECISION_END_EFFECTOR=inspire` preserves the original Inspire/BODex route. Each mode has an independent URDF, joint width and candidate namespace. Gripper startup does not import Inspire Modbus. |
| Tabletop grasp coverage | The gripper catalogue contains one deterministic analytic cylindrical grasp for each supported tabletop class (`end_down`/`000` and `side_down`/`001`). Both retain the same endpoint collision contract; Inspire keeps its screened BODex catalogue. |
| Four-camera evidence | Every checkpoint captures all four commissioned cameras. Socket-pose inliers and measured residuals may select a smaller inference subset; camera evidence is not discarded or permanently blacklisted. |
| Full action loop | The active profile supplies all ten stage handlers: key planning, pickup, lift, held relation, transfer/insertion, retry diagnosis, held return, transfer recovery, repose planning and repose execution. |
| Same-trial correction | Pre-insertion capture, metric endpoint/partial-shape observation, bounded wrist correction, recapture and measured-motion feedback run before the first insertion stroke. The active profile permits a 10 mm per-update step and has no cumulative XY cap. |
| Continuous key tracking | The `mvgotrack_axis_alignment_capture4.env.sh` profile sends the accepted same-session SAM3+FoundPose 6D key pose and mesh to the isolated capture4 tracker before pickup. It consumes a pose newer than the stopped pre-insertion sequence barrier, reduces the cylinder pose under its D-infinity symmetry, and supplies XY plus optional axis correction. |
| Insertion execution | The corrected hold is replanned along the measured socket axis. If the corrected 20 mm endpoint/path is unavailable, the profile tries the configured 10 mm axial fallback. |
| Failed-attempt recovery | A released insertion failure first reuses its saved pre-insertion image for next-attempt XY advice, then perceives and restores the released key to the original tabletop pose. Model, saved-input or advice-validation errors are recorded as unavailable diagnostics and cannot suppress that recovery. |
| Labels | Manual collection stores the independent Qwen result too. Grasp, pre-insertion arrival, descent-time tip entry and released-key `inside_socket` outcome remain separate fields. |
| Storage | Runtime JSON and environment variables resolve NAS paths only below `/mnt/paradex2/hyunsu/precision_insertion`; repeated-use model caches and source checkout remain local. Capture4 acquisition is the documented external ParaDex exception. |
| Timing | Nested spans cover startup, perception, planning, motion, camera, VLM, evidence writes, human waits and cleanup. Overlap-aware occupied time prevents parallel PNG writes from being counted as serial latency. |

## Verified evidence

- The static ZeroDex preflight passes in both end-effector modes. The gripper
  mode has 2/2 eligible deterministic candidates, one for each tabletop pose;
  Inspire retains 126 catalogued and 114 eligible candidates. Both report
  four configured cameras, board-free mode, no legacy runtime path and no NAS
  reference outside `/mnt/paradex2/hyunsu`.
- The current calibration profile resolves the final extrinsic solution at
  `~/calibration/zerodex_capture4/extrinsic/20261118_232125/solution_filtered`
  and hand-eye transform at
  `~/calibration/zerodex_capture4/handeye/20261118_233621/solution/C2R.npy`.
- A read-only 2026-11-26 capture4 probe reported all isolated ports ready:
  `5009, 5480, 5481, 5482, 6894, 15006, 15007, 16893`. The four configured
  serials remain `26053248, 25322639, 25305462, 25322642`.
- The source-boundary audit reports zero stock-entrypoint changes and zero
  unexpected shared-runtime changes.
- The complete demo test suite last passed with **1604 passed, 5 skipped**.
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
| Current hardware execution | The local Franka daemon is not running. After CUDA is restored, start the RT daemon and select either `franka_gripper` (default) or `inspire`; the current audit intentionally issued no motion command. |
| CUDA runtime | The PREEMPT_RT kernel and NVIDIA 535.309.01 modules are loaded and the RTX 3090 is bound to the driver, but `/dev/nvidia*` is absent, `nvidia-smi` fails and PyTorch reports zero CUDA devices. Run `sudo /sbin/ub-device-create --verbose`, then verify `nvidia-smi` before Qwen/SAM/curobo execution. |
| Physical gripper/MV-GoTrack run | Code, assets, capture4 services and tracker deployment are present, but the new gripper pickup-to-insertion route and same-session tracker initialization have not been physically exercised. First run GPU cuRobo preflight, then initialize tracking from a fresh accepted key pose; do not substitute an arbitrary or stale tracker initializer. |
| Reproduction bundle | The NAS is currently mounted read-only. A 2026-11-03 metadata audit found the live runtime view internally valid, but the archived checkout remains at `f0f9f1d` rather than the current branch, `nvdiffrast` is absent from the archived code inventory, two reset `source_evidence.json` files no longer match their recorded sizes, and twelve managed paths are unmanifested. The tested `--refresh-code-only` path rescans these managed trees as well as archiving all three code repositories, but applying it and then running full SHA256 verification must wait for a writable mount. |
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
