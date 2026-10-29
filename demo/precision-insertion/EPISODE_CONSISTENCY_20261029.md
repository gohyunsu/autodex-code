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

## Archived attempt accounting

There are 19 local `attempt_*/result.json` files across the 5 and 10 mm
socket sessions. They include attempts that stopped before physical insertion,
so 19 is not an insertion-trial denominator. The files report five
`observed_success`, one `observed_failure`, one
`planning_stopped_before_insertion`, and 12 `in_progress`. The 12 include
two `lift_execution_failed`, one `pickup_execution_failed`, and one
`post_squeeze_preflight_failed`; `in_progress` is the compact result writer's
default when no final insertion label was recorded, **not** evidence that the
robot was still moving. Use `failure_code`, state events, and the human review
alongside `status` when counting outcomes. Five archived visual-success flags
become two 5 mm inside-socket results for `20195` after human review; the
10 mm `20267` visual-failure flag was judged inside by the operator. Neither
the recorded flags nor this small human review estimate a grasp success rate.

The recent `20195` pre-insertion crops expose a concrete perception mismatch.
The Qwen box/SAM3 masks on `25305462` and `25322642` in `5e08757b` follow
the white **socket rim**, whereas `26053248` shows the held key but returns
`sam_mask_unavailable`. In `4e26ec99`, only `25305462` yields a SAM mask and
its single-view, fixed-height XY hypothesis is about 105 mm; it must not be
used as a measured correction. The saved masks, per-view prompts, boxes and
images are in each attempt's `same_trial_xy/observations/000.json` and
`raw_captures/preinsert/`. This is why successful nominal insertion and
reliable multiview correction have not yet occurred in the same archived
episode.

A read-only local SAM3 experiment reused each archived Qwen box as a positive
geometric prompt on all eight saved `20195` views. A box alone and a box plus
the existing `white cylinder` text both recovered the **visible key** mask
on `26053248` in both successful attempts, where text-only SAM3 had returned
no mask. Neither prompt fixed the other views: it also segmented the socket
rim on `25305462`/`25322642` and white robot components on `25322639`.
All eight views produced a confident mask with a positive box, so SAM3's
score alone cannot identify the correct object. The exact masks, overlays,
per-view scores and reproducible script are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/sam_box_prompt_20261029/`
and `probe_sam_box_prompt_20261029.py` beside that directory. A verified
74-file mirror of the probe, images and script is under
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/sam_box_prompt_20261029/`.
The runtime segmentor was **not** switched to box prompting: the experiment proves a
one-view segmentation improvement, not a reliable multiview 3D tip.

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

## 29 October lift-input probe: keep the current runtime method

The archived lift reports contain three human-confirmed held keys: `0c787032`,
`5e08757b`, and `4e26ec99`. The production single-image hand crop produced
at least one positive view in two of these three. A read-only Qwen3-VL-2B
probe added the initial tabletop key-mask crop as image 1 and kept the exact
recorded post-lift hand crop as image 2. The prompt asked for `A` (visibly
held), `B` (visibly not held), or `C` (hidden). With the production-style
"any positive camera" combination, this pair produced a positive view in
all three human-confirmed lifts, including the missed `0c787032`. However,
the per-camera positive counts were **2/4, 1/4, 1/4**, versus **0/4, 3/4,
3/4** with the original single image. A majority-of-views rule would miss
all three. This is an unstable view dependence, not a demonstrated robust
improvement.

Three additional archived VLM-negative lifts (`687a4a49`, `804ef849`,
`ecb0c571`) produced zero `A` answers in the nine evaluable camera pairs.
They have **no independent human lift annotation**, so this is a diagnostic
control, not a measured false-positive rate. One further view was omitted
because its initial key mask was empty. The reproducible scripts and exact
per-camera answers are in the local development review directory:
`lift_pair_probe_20261029.py`, `lift_pair_probe_20261029.json`,
`lift_pair_controls_20261029.py`, and `lift_pair_controls_20261029.json`.
They access saved images only. The production prompt, crop, and gate were
left unchanged because the evidence does not yet establish a reliable
automatic negative lift label.

An additional saved-image probe shifted the same 650 px crop from the fixed
hand offset to the measured mean thumb/index/middle fingertip projection.
With the original single-image prompt, the three human-confirmed held lifts
had at least one `YES` each; however, two of the three unannotated control
attempts also gained a `YES` in a single camera. Pairing the initial-key crop
with this finger-centred crop likewise gave positive views in all three held
lifts **and** two control attempts. The white key or socket left on the board
can enter the enlarged finger-centred field, so the apparent sensitivity
gain does not establish better discrimination. The raw answers and exact ROI
coordinates are in `lift_finger_roi_probe_20261029.{py,json}` and
`lift_pair_finger_probe_20261029.{py,json}` in the same local review folder.
The controls have no independent human lift label; treat their `YES` results
as a reason to collect labels, not as a measured false-positive rate.

A simple initial-mask-versus-post-lift pixel difference also failed to
separate those groups. Across the four cameras, median absolute RGB changes
inside the initial mask were 6–206 levels for the three held lifts and
50–148 levels for the unannotated controls. The moving arm can cover the
initial key region, and a missed key may be displaced, so disappearance
alone is not evidence of a held key. No pixel-difference gate was added.
The saved-image paths, SHA256 hashes, and per-camera statistics are in
`lift_mask_difference_20261029.{py,json}` beside the Qwen probes.

For the next data collection, record a human held/miss/slipped label **at the
post-lift snapshot time**, and retain the later transfer result separately.
Evaluate camera-wise and episode-wise confusion on those independent labels.
The video should be used to mark slips occurring after the still image;
otherwise a correct snapshot label can be mistaken for a classifier error.
Until that dataset exists, use manual lift/transfer labels while logging the
Qwen answer in parallel. Final release can use the current three-camera
visual vote, with the explicit meaning `inside_socket` rather than measured
20 mm depth.

The cross-episode recipe supported by the available physical evidence is:
prefer the twice-inserted `20195` grasp for the 5 mm socket; locate the socket
once per fixed session; take fresh key images and run the usual pickup and
20 mm hand/socket preflight; retain the held key during transfer; at the
preinsert hold, run Qwen box + SAM3 half-ellipse + calibrated multiview
triangulation when at least two views resolve the tip; move the rigidly held
grasp by the measured socket-frame XY correction and recapture; replan the
descent from the reached wrist; release and classify the actual socket crop.
The two successful `20195` episodes followed the nominal branch because a
metric multiview tip was unavailable. The two episodes that did obtain a
sequence of metric XY points used `20096` and stopped before descent on
hand/socket collision. Therefore successful **closed-loop insertion** has
not yet been demonstrated in these records. A visual residual below the
configured tolerance must not be reported as physical insertion success.

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
