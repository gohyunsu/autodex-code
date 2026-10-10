# Real AutoDex-camera handoff: current evidence and limits

The NAS runtime addendum contains real AutoDex-rig images, not just CAD. The
read-only audit snapshot at
`/home/hyunsu/shared_data/AutoDex/precision_insertion/nas_perception_audit_20261010.json`
checked the selected capture index on 2026-10-10. Run it again whenever the
`.partial` perception evaluation changes:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/audit_perception_handoff.py \
  --capture-index /mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909/latest_capture_index.json \
  --candidate-index /mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909/existing_foundpose_candidates.json \
  --evaluation-root /mnt/paradex2/hyunsu/autodex_precision_insertion_perception_eval_20261010_652ac909.partial \
  --output /path/to/new_audit_snapshot.json
```

`--output` is exclusive. The audit reads NAS files but never promotes a
FoundPose candidate or modifies a capture. Use `--verify-candidate-hashes`
when a full ~9 GB NAS read is acceptable; the initial snapshot checked the
12 candidate PTHs' existence/expected size, **not** their SHA-256 bytes.

At the initial snapshot, all 35 selected conditions had complete 20-camera
shots and every indexed JPEG matched its recorded hash. Five planned square
0.5 mm key tabletop poses were not captured. There were 52 evaluation
reports over the selected conditions, including multiple methods on the
*same* shot:

| Evaluator variant | Reports | Numeric silhouette pass |
| --- | ---: | ---: |
| socket-only | 7 | 7 |
| cylinder key, direct SAM3 | 12 | 12 |
| key, registered-difference assisted | 18 | 14 |
| square key, ROI-assisted SAM3 | 15 | 14 |

Thus 47/52 reports passed **only the evaluator's numeric silhouette gate**.
The five failing reports were four cylinder registered-difference cases and
square 0.3 mm pose 05 under ROI-assisted SAM3. This is not a 47/52 physical
pose-success rate: duplicated conditions, method differences and absent
ground-truth key/socket poses prevent that interpretation. In the cylinder
direct-SAM3 20 mm-gap `pose_01` report, one admitted view has IoU 0 even
though the mean is high; per-view screening and consensus still matter.

The key ROI/difference methods crop or subtract a separately recorded
socket-only baseline. The current live `InitOrchestrator` +
`admit_key_capture()` does **not** reproduce those same preprocessing steps.
The direct-cylinder SAM3 variant is closer to the live segmentation path,
but its visual/numeric fit still needs held-out pose and symmetry-aware
orientation checks. Socket overlay examples look plausible in some views;
no exhaustive manual rim, socket-axis or calibration audit is recorded here.

None of the 35 selected `shot.json` files provides a verified **per-camera
frame ID and exposure UTC interval**. Its overall `captured_at_utc` and
request timing cannot replace that evidence for the session bootstrap. The
shots also span different object placements and do not supply repeated
same-session socket measurements. The 12 large `repre.pth` files remain under
`pending_foundpose/`, not canonical `AutoDex/foundpose_assets/`.

Before promoting any object for a live session:

1. Independently inspect raw image, SAM mask, projected CAD rim/axis, and
   refined pose per view; annotate wrong-object masks, flips and occlusion.
2. Measure multi-view pose residuals and repeated **fixed-socket** pose
   stability, including an independent board/fixture reference. For the
   cylinder, quotient out unobservable axial yaw but never a flipped rim.
3. Re-run the **exact live preprocessing/prompt** or implement and validate
   the ROI variant as a demo-local capture producer; don't use assisted
   results to certify an unassisted runtime.
4. Commission camera exposure-time metadata, the robot-clock relation and
   hand-eye/fixture uncertainty. Then install only QA-passing PTHs at the
   canonical v8 runtime paths with immutable hashes.

No grasp, guarded insertion, VLM retry or physical success is established by
these captures. The 1.5 mm square bring-up remains blocked by those gates and
by its missing endpoint-eligible runtime grasp pool.
