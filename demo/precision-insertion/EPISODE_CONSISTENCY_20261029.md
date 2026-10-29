# ZeroDex capture4 episode consistency review (2026-10-29)

This review uses the locally archived 5 mm cylinder-socket trials. The
Franka and Inspire hand were off during the analysis. A `result.json` visual
success flag is **not** an independent ground-truth annotation: four of the
six released-key trials disagree with the later human review. The human
review and its exact image hashes are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/released_vlm_dataset_20261029/manifest_six.json`.

| Attempt | Grasp | Original result | Human review of released key | Preinsert XY |
| --- | --- | --- | --- | --- |
| `30cbe1f3` | `table/0/20096` | inside | outside | no metric point |
| `30e9cd0b` | `table/0/20267` | outside | inside | no metric point |
| `c6141471` | `table/0/20096` | inside | outside | one-view planar hypothesis |
| `dc5958a9` | `table/0/20096` | inside | outside | no metric point |
| `5e08757b` | `table/0/20195` | inside (manual) | inside | no metric point |
| `4e26ec99` | `table/0/20195` | inside (manual) | inside | one-view planar hypothesis |

The separate `9542c1bd` and `0c787032` attempts grasped and transferred
`table/0/20096`, then made three measured lateral shifts each. Their
successive three-view XY residual norms were 32.0 → 21.5 → 12.6 → 1.27 mm
and 25.6 → 17.4 → 8.25 → 1.37 mm. Both stopped **before** an insertion
command: the measured hand geometry intersected the 5 mm socket at the
corrected 20 mm endpoint. The decreasing residuals show repeatable image
feedback in those two held-key trajectories; they do not demonstrate a
successful corrected insertion or eight independent trials.

## Method supported by these episodes

1. Measure the socket once per fixed-board session. Undistort each capture
   with its camera calibration. Project the *current* frozen socket pose into
   the image for a moving ROI and to distinguish the socket from the key.
2. Use an insertion-compatible grasp first. `table/0/20195` produced the two
   latest human-confirmed inside-socket results. `table/0/20096` produced
   three outside-socket releases and two corrected-hold endpoint collisions.
   The ZeroDex 5 mm launcher now prefers `20195` and excludes `20096` by
   default. Explicit environment overrides remain available; every selected
   grasp still passes the existing pickup and insertion preflight.
3. At pre-insertion, take simultaneous raw views before moving. The planned
   key pose chooses a **crop**, not a measured tip. Qwen boxes the key; boxes
   that almost wholly coincide with the projected socket are discarded;
   SAM3 supplies a silhouette; its distal half-ellipse centre is triangulated
   with calibrated cameras. Move the held grasp by the measured socket-frame
   XY residual, up to the existing 10 mm step, then capture again. Use only
   a consistent multiview metric point to command this same-trial correction.
4. Replan the 20 mm descent from the actually reached wrist and hand state,
   checking the full hand against the socket. A small XY residual alone does
   not imply the fingers can enter: this is the exact failure of `20096`.
5. After opening the hand and moving to clear view, classify the **final raw
   socket crop** with the short A/B prompt. Three independent camera answers
   (`25305462`, `25322639`, `26053248`) are batched locally and combined by
   majority. This is a visual inside/outside label, not a depth measurement.

The final VLM route reproduced all six human reviewed labels. A read-only
crop perturbation probe moved each socket projection by 10% of its width or
height in four directions, with the nominal crop as a fifth condition. The
majority stayed correct in all 30 episode/crop combinations; per-camera
correctness was 24/30 for `25305462` and 30/30 for each of `25322639` and
`26053248`. These are repeated images of **six** episodes across three
session-calibration hashes, not 30 independent episodes. The raw answers,
crop coordinates, and replay code are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_label_eval_20261029/perturbation_probe.json`
and `probe_final_crop_jitter.py` beside it.

## What is not yet consistent

- In the two latest successful insertions, the fixed camera geometry exposed
  the held key clearly mainly to `26053248`; two other views boxed the white
  socket as the key and another view had no SAM3 key mask. Thus neither
  episode produced a multiview metric correction. Both succeeded on the
  nominal target. More prompt variants cannot triangulate a hidden tip.
- In manual-labeled `0c787032`, the after-lift close-hand Qwen answers were
  `NO` in all four cameras even though the operator labeled the lift held and
  subsequent preinsert frames yielded three-view key points. Its lift crop
  and full-frame grids are `0c787032.png` and `0c787032_full.jpg` under
  `~/precision_insertion_project/development/zerodex_capture4/reviews/lift_crop_audit_20261029/`.
  A static hand crop can miss or hide the key.
- The current after-lift/preinsert coarse VLM prompt answered
  `coarse_match` even on archived duplicated empty-hand frames used as
  negative controls. Those controls are not real transfer-loss episodes;
  the method needs separately labeled real slips before it can replace a
  human transfer judgment. See `VLM_LABEL_REVIEW_20261029.md`.
- The 6/6 final visual result is in-sample for one 5 mm socket family. It
  does not establish generalization to another socket gap, lighting, or
  camera layout, and does not prove physical 20 mm penetration.

## Next discriminating experiment

Keep `20195` and the current 5 mm socket. At the preinsert hold, collect raw
four-view images while deliberately exposing the key end in **two calibrated
oblique views** (camera angle or collision-checked axial yaw gauge). Compare
triangulated points with manually marked end centres and record the measured
XY residual over successive corrections. Then run the corrected descent and
released-key judgment in several fresh episodes, alternating nominal and
same-trial correction when feasible. Label real post-lift and transfer slips
from the full recordings. This tests the missing link: the two episodes with
reliable XY feedback stopped at the hand/socket geometry, while the two
successful insertions had no reliable metric tip point.

The read-only replay does not modify robot or camera state. The perturbation
probe changes only the final VLM input crop; no new runtime failure threshold
or extra safety gate is introduced.
