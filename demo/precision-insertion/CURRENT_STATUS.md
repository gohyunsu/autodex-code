# Current ZeroDex implementation status

This page is the concise completion ledger for the active
`feat/zerodex-precision-insertion` deployment. Detailed operation, stage and
timing contracts remain in [`OPERATIONS.md`](OPERATIONS.md),
[`ACTION_LOOP.md`](ACTION_LOOP.md) and [`TIMING_PROFILE.md`](TIMING_PROFILE.md).
Historical dated reports are evidence, not current run instructions.

Last audited: **2026-11-03**, with the robot, hand and cameras off.

## Implemented

| Requirement | Current implementation |
| --- | --- |
| Independent runner | `run_zerodex_capture4.sh` and `run_live.py` run the demo-local state machine. Stock `run_auto.py`, `run_pipeline.py` and `scene_cfg.py` are not entry points. |
| ZeroDex machine binding | `zerodex_capture4.env.sh` owns capture4 serials/host, local cache paths, NAS namespace and the straight-mount FR3/Inspire profile. `zerodex_franka_integration.py` changes routing only inside the process. |
| Four-camera evidence | Every checkpoint captures all four commissioned cameras. Socket-pose inliers and measured residuals may select a smaller inference subset; camera evidence is not discarded or permanently blacklisted. |
| Full action loop | The active profile supplies all ten stage handlers: key planning, pickup, lift, held relation, transfer/insertion, retry diagnosis, held return, transfer recovery, repose planning and repose execution. |
| Same-trial correction | Pre-insertion capture, metric endpoint/partial-shape observation, bounded wrist correction, recapture and measured-motion feedback run before the first insertion stroke. The active profile permits a 10 mm per-update step and has no cumulative XY cap. |
| Insertion execution | The corrected hold is replanned along the measured socket axis. If the corrected 20 mm endpoint/path is unavailable, the profile tries the configured 10 mm axial fallback. |
| Failed-attempt recovery | A released insertion failure first reuses its saved pre-insertion image for next-attempt XY advice, then perceives and restores the released key to the original tabletop pose. Model, saved-input or advice-validation errors are recorded as unavailable diagnostics and cannot suppress that recovery. |
| Labels | Manual collection stores the independent Qwen result too. Grasp, pre-insertion arrival, descent-time tip entry and released-key `inside_socket` outcome remain separate fields. |
| Storage | Runtime JSON and environment variables resolve NAS paths only below `/mnt/paradex2/hyunsu/precision_insertion`; repeated-use model caches and source checkout remain local. Capture4 acquisition is the documented external ParaDex exception. |
| Timing | Nested spans cover startup, perception, planning, motion, camera, VLM, evidence writes, human waits and cleanup. Overlap-aware occupied time prevents parallel PNG writes from being counted as serial latency. |

## Verified evidence

- The static ZeroDex preflight passes with 126 catalogued candidates, 114
  eligible candidates, four configured cameras, no legacy runtime path and no
  NAS reference outside `/mnt/paradex2/hyunsu`.
- The source-boundary audit reports zero stock-entrypoint changes and zero
  unexpected shared-runtime changes.
- The complete demo test suite last passed with **1552 passed, 5 skipped**.
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
| Current hardware execution | Robot and hand are off. No hardware command is permitted until the operator restarts FCI, the Franka daemon and Inspire. |
| CUDA runtime | NVIDIA kernel modules are present, but `/dev/nvidia*` is absent and `nvidia-smi` fails. Recreate the device nodes and verify `nvidia-smi` before Qwen/SAM/curobo execution. |
| Reproduction bundle | The NAS is currently mounted read-only. The existing bundle predates the current branch and is not reproduction-ready; refresh and SHA256 verification must wait for a writable mount. |
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
