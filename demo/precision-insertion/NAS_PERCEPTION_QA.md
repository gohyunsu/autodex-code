# Real AutoDex-camera handoff: current evidence and limits

The NAS runtime addendum contains real AutoDex-rig images, not just CAD. The
read-only audit snapshot at
`/home/hyunsu/shared_data/AutoDex/precision_insertion/nas_perception_audit_20261010.json`
checked the selected capture index on 2026-10-10. The evaluation directory
has since been finalized; the 2026-10-11 audit below supersedes that initial
snapshot. Run it again whenever the evaluation changes:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/audit_perception_handoff.py \
  --capture-index /mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909/latest_capture_index.json \
  --candidate-index /mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909/existing_foundpose_candidates.json \
  --evaluation-root /mnt/paradex2/hyunsu/autodex_precision_insertion_perception_eval_20261010_652ac909 \
  --output /path/to/new_audit_snapshot.json
```

`--output` is exclusive. The audit reads NAS files but never promotes a
FoundPose candidate or modifies a capture. Use `--verify-candidate-hashes`
when a full ~9 GB NAS read is acceptable; the initial snapshot checked the
12 candidate PTHs' existence/expected size, **not** their SHA-256 bytes.
Each evaluation now includes `lowest_iou_views_for_manual_review` with up to
three camera IDs and existing mask/overlay paths. It ranks review work; it
does not declare a passing IoU or replace inspection of all relevant views.
The current re-audit at `/tmp/precision-perception-audit-review-20261011.json`
found 61 reports and no capture-integrity error, and placed camera
`24122734` first for the square 1.5 mm `pose_02` example below.

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

One concrete wrong-object example is square 1.5 mm key `pose_02`, camera
`24122734`: its ROI-assisted key view has IoU **0.04947** with the refined
CAD. The saved key mask has 10,224 pixels, 9,950 of which (97.32%) overlap
the *same-camera* socket-only mask captured earlier. In the raw frame the
white key is behind the socket, whereas the overlay places the CAD key over
the socket. This comparison is a diagnostic across two static-board captures,
not a synchronized live mask test or proof of the exact projected-socket
overlap used by `admit_key_capture()`. It does show why a numeric full-image
silhouette pass and even a high mean across other cameras cannot license that
view. Review its [raw frame](/mnt/paradex2/hyunsu/tmp/precision_insertion_multiview_20261010_652ac909/square/precision_key_1p5mm/pose_02/views/shot_20261010T131640Z_585435e4/24122734.jpg)
and [CAD overlay](/mnt/paradex2/hyunsu/autodex_precision_insertion_perception_eval_20261010_652ac909/results/key_roi_sam3_square/square/precision_key_1p5mm/pose_02/24122734_cad_overlay.jpg).

None of the 35 selected `shot.json` files provides a verified **per-camera
frame ID and exposure UTC interval**. Its overall `captured_at_utc` and
request timing cannot replace that evidence for the session bootstrap. The
shots also span different object placements and do not supply repeated
same-session socket measurements. The 12 large `repre.pth` files remain under
`pending_foundpose/`, not canonical `AutoDex/foundpose_assets/`.

## Finalized evaluation re-audit (2026-10-11)

The read-only report is
`/home/hyunsu/shared_data/AutoDex/precision_insertion/nas_perception_audit_final_20261011.json`.
It verified the hashes of all 700 selected JPEGs from 35 conditions and
found no capture-integrity error. More method variants were completed after
the initial snapshot, so the final report counts **61 evaluations, 56 numeric
silhouette passes**:

| Evaluator variant | Reports | Numeric silhouette pass |
| --- | ---: | ---: |
| Socket-only | 7 | 7 |
| Cylinder key, direct SAM3 | 12 | 12 |
| Key, registered-difference assisted | 27 | 23 |
| Square key, ROI-assisted SAM3 | 15 | 14 |

All selected object conditions now have at least one evaluation, but the
five planned square 0.5 mm poses were not captured. Every selected shot
still lacks a verified per-camera frame ID and exposure-time interval.
Consequently `perception_promotion_ready=false` and
`session_start_eligible=false`. The 56/61 figure is **not** a pose-accuracy
rate: methods overlap on the same shots, the key is tiny in a full frame,
and no independent metric key/socket ground truth was recorded. The
demo-local tabletop key gate now requires a separately commissioned minimum
selected-mask/refined-CAD IoU in addition to stock refinement and multi-view
pose consistency. This veto does not make the assisted ROI method equivalent
to the current live full-frame pipeline, nor does it certify sub-millimetre
pose accuracy. None of the candidate PTHs has been promoted.

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
