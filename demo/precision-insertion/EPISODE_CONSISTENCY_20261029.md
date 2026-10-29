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

The two success runs share the same calibrated camera geometry. Rays from
`25305462` and `25322642` to the socket origin differ by only **4.02°**,
below the current 5° minimum parallax for a metric tip. Each has a 38–39°
baseline with `25322639`, and an 82–83° baseline with `26053248`.
In the saved full frames, the hand occludes the tip from `25322639`;
`25305462` and `25322642` see almost the same socket-side aspect.
`26053248` shows the held key's side, but its mask did not resolve a
measured end centre. Thus two masks from the near-duplicate cameras could
not by themselves provide a metric correction even if both identified the
key. The next camera/hold arrangement must expose the **same physical end
centre** to at least two views with useful parallax. This is a geometric
observability issue in addition to the key/socket identity issue.

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

The after-lift and pre-insertion bundles for these two attempts both contain
four calibrated frames and measured robot joints. They permit a future
motion-based point reconstruction only if the **same physical key-end centre**
can be located at both times and the key stayed rigid in the hand. In the
saved `20195` images, `25322639` exposes the circular end after lift but the
arm covers it at pre-insertion; `26053248` exposes the key at pre-insertion
but mainly as an occluded side silhouette. The box-prompt mask there gives a
distal half-ellipse centre 71–77 px from the projected nominal tip. That
projection is an unverified grasp hypothesis, so this difference is **not**
a measured physical error; it does show that neither the mask score nor the
fitted point validates a corresponding end centre. Temporal triangulation
from these two snapshots therefore remains a proposal, not an established
shortcut for the current runtime. The existing two-view geometric fallback
in `tip_tracking.py` is already available when two real tip observations are
present; the third preferred camera helps reject mismatches rather than
supplying a fundamental degree of freedom.

Two shorter identity-focused Qwen box prompts were replayed on the same
eight pre-insertion crops. Both still boxed the socket in views where the
key was hidden. The more explicit key/socket prompt additionally returned
`null` for the **visible key** in `26053248` of `5e08757b`; the stock prompt
boxed it. Thus these prompt variants do not improve the archived success
cases, and the stock runtime prompt remains in place. The exact text,
per-view answers and replay script are mirrored at
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/qwen_key_identity_prompt_20261029/`
with matching SHA256 for both files. Camera visibility and object identity
need stronger evidence than a different wording of the same image question.

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

The final VLM route reproduced all six human reviewed labels (five 5 mm
episodes and one 10 mm episode). A read-only
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
- The 6/6 final visual result is in-sample for five 5 mm episodes and one
  10 mm episode. It does not establish generalization to other socket gaps,
  lighting, or
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

## Paired reference and geometric-mask probes: no reliable metric tip yet

We replayed four archived pre-insertion attempts (`5e08757b`, `4e26ec99`,
`0c787032`, `9542c1bd`) with the initial tabletop key-mask crop as image 1
and the original, undistorted pre-insertion crop as image 2. Qwen3-VL-2B was
asked to box the *same movable key* in image 2. On the two physically
successful `20195` attempts, the new boxes avoided the socket boxes returned
by the production single-image prompt on `25305462` and `25322642`, but
mostly boxed the black fingers or wrist instead. Applying SAM3 with either
the box alone or text plus box preserved these wrong identities. This is a
failure of image-object identity, not a syntax or batch-execution failure;
the paired prompt was **not** integrated into the runtime.

A second SAM3 replay skipped Qwen and prompted a 64 x 60 px box around the
distal 72% point of the *planned* key axis. On both successful attempts,
`25322642` segmented a small exposed white shaft below the hand, but
`25305462` and `26053248` segmented the fixed socket, while `25322639`
segmented robot hardware. SAM3 scores for these false masks were still
0.85-0.97. The planned position is useful for search but is not an identity
label; prompting it directly does not produce a robust multi-camera key mask.

The existing per-camera Lab-colour tracker was also replayed, using that
attempt's *own initial key mask* for colour. For the two `20195` successes,
it found a plausible key-side component in `26053248`, but found the metallic
wrist in `25322639` and returned no component in `25322642`. Its existing
admission rule accepted zero views in each success attempt. Simply removing
that rule would admit the wrong wrist point; it would not create a valid
two-view measurement. In a diagnostic pairing of the visually plausible
`25322642` SAM point and `26053248` SAM point, the two-view DLT estimates
were 18-21 mm higher than the nominal 85 mm key-tip height and reprojected
20-29 px away from the input points. Replacing the `26053248` SAM point by
the colour-track point still left 7-17 mm height error and 18-31 px
reprojection error. These are **inconsistent image features**, not a
physical key-tip offset. Do not turn them into a robot XY command.

The socket-only reference frame was compared with the pre-insertion frame
using raw pixel difference. Arm motion, lighting, and the patterned board
dominated the difference, so a static-background subtraction is not a key
segmentation result either. This probe did not change the runtime.

As a further diagnostic, forcing the `26053248` colour-track point onto the
nominal 85 mm tip-height plane would request 34.4 mm and 53.8 mm socket-frame
XY corrections in the **two physically successful nominal insertions**.
Those large false corrections are a direct counterexample to treating a
visible shaft pixel as the distal-face centre or promoting a single-view
planar hypothesis to an immediate correction. They were never sent to the
robot.

### Shaft-axis alternative tested on the same frames

The distal face can be hidden while part of the cylindrical side remains
visible. We fitted an image line through each candidate side mask and solved
for the socket-frame XY position where the two back-projected line planes
meet the nominal 85 mm pre-insertion tip-height plane. This method uses the
shaft axis rather than claiming to observe the hidden face centre. Its
read-only replay gave 2.87 and 1.54 mm from nominal on the two successful
`20195` frames, compared with 20.63 and 13.85 mm on the two displaced
`20096` frames. The first displaced case had a separately estimated
25.6 mm tip residual, so the axis-line method detects a gross offset there.

These four numbers do **not** validate a correction policy. In the successful
frames, the two key masks were selected manually from different prompt
routes: projected-region SAM3 on `25322642`, and Qwen-box SAM3 on
`26053248`. The former has only a small visible shaft patch with a fitted
aspect ratio of 1.21-1.50. Eroding or dilating those masks by 3-5 pixels
changed the second success's estimated XY from `[+7.74,+7.40]` to
`[-12.67,-16.98]` mm. Its original `[+1.50,-0.34]` mm estimate is therefore
not precise enough for an automatic millimetre-scale command. In the second
displaced case, the axis-line magnitude (13.85 mm) also disagreed with the
recorded distal-tip estimate (32.0 mm), which may reflect mask identity,
orientation, or end-centre error; the saved frames do not resolve which.
The script and strict JSON replay are `probe_two_view_shaft_axis_20261029.*`
in the same development evidence directory. No robot command or runtime
estimator change was made.

A follow-up replay applied the projected-region SAM3 prompt to all four
attempts, then ranked the available masks by overlap with the measured socket
projection, colour difference from that camera's initial key mask, and size
difference from that mask. It recovered the visually plausible key masks on
`25322642` and `26053248` in the two `20195` successes, and the existing
runtime masks remained the best choices on the displaced `20096` views.
However, on `25305462` it ranked a dark hand region above every socket mask;
on `25322639` of `4e26ec99` it ranked a large white robot region above the
dark hardware masks. A numerical rank alone therefore cannot certify key
identity. The complete per-camera ranking, input masks, and failed-attempt
contact sheet are in `mask_rank_probe_20261029/` beside this review's other
development evidence. This probe did not alter runtime selection or add a
new failure condition.

The reproducible saved-frame scripts, JSON replies, masks, and contact
sheets are grouped under
`~/precision_insertion_project/development/zerodex_capture4/reviews/method_probe_20261029/`
and mirrored below
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/method_probe_20261029/`.
The source episodes remain the original read-only evidence. The robot and
hand were off during this replay.

**Implication for a consistent pipeline.** Keep the physically successful
nominal `20195` branch for the current 5 mm standing-key task; lack of a
metric visible end must not be silently converted into a large correction.
For arbitrary grasps and placements, add an *observation action* before
insertion: use the cylinder's axial-yaw equivalence to present the held end
to a camera pair with useful parallax, capture again, and only then derive a
metric correction. The saved 0/150/180-degree plans establish that such a
change of held wrist yaw is kinematically possible for one recorded state,
but they do not establish that the real end becomes visible. That requires
new camera frames with the robot on. A narrow contrasting mark on the key
shaft above the insertion region is another testable appearance aid if a
different hold alone does not expose a common point; it is not part of the
current white-key experiment.

## Offline yaw-gauge experiment on the successful grasp

The cylinder has no meaningful axial yaw. This allows a held `table/0/20195`
key to approach the same socket at a different wrist orientation without
changing the key/socket fit. Using the frozen 5 mm world, saved pre-insertion
13-joint feedback and nominal key/hand transform from `5e08757b`, all 12
30-degree yaw samples passed the 20 mm hand/socket endpoint check and a
cuRobo path to the pre-insertion wrist pose. Four representative yaw choices
(0, 90, 150, 180 degrees) also passed the existing ten-waypoint socket-axis
Jacobian descent and `audit_held_joint_paths` for the dense transfer/descent
samples. These are offline plans from a recorded state, not four physical
trials or a proof about another tabletop key location.

A separate **predicted visibility** probe used the saved camera extrinsics,
the planned wrist joint endpoints and the actual FR3/Inspire visual meshes.
It ray-tested 72 distal cylinder-side samples per camera against the hand
mesh. At yaw 0, the visible/front-facing sample counts were `36/36` on
`25305462`, `30/33` on `25322639`, `36/36` on `25322642`, and `33/33` on
`26053248`. At yaw 180 all four counts equalled their front-facing totals.
The large counts at yaw 0 despite real-image tip failures are important:
visible **shaft side** does not establish a visible **bottom-face centre**.
The probe assumes the nominal key/hand transform, ignores real key slip,
key/self and socket occlusion, and does not run Qwen or SAM on a yaw-180
photo. It supports testing a different wrist orientation but does **not**
justify selecting one automatically yet. The reproducible scripts and JSON
are under `~/precision_insertion_project/development/zerodex_capture4/reviews/yaw_gauge_20261029/`.

The evidence-backed operating method remains: prefer the physically
successful `20195` grasp and nominal 5 mm insertion; at the pre-insertion
hold, use any calibrated camera pair that identifies the same key end and has
useful parallax for an actual metric correction; otherwise retain the nominal
target. The two corrected-hold runs that reached about 1.3 mm residual did
not insert because `20096` collided at the 20 mm endpoint. A future live
comparison of the **same `20195` grasp** at 0 and 150/180 degrees should save
all four raw pre-insertion views, the hand joints, per-view key boxes/masks,
accepted tip points and insertion outcome. Prefer a yaw only if that real
comparison improves end-centre detection without losing the complete
transfer/descent and release sequence. No VLM prompt can recover a centre
that remains hidden in every camera.

The ZeroDex launcher now resolves the FR3 planner's own cuRobo copy before
the BODex copy. In the old shell path, a debug `print` in BODex's
`world_mesh.py` can raise `operation not permitted when stream is capturing`
during a fresh CUDA-graph world initialization. The planner copy initializes
the recorded frozen world with default CUDA graphs, and the four yaw plans
above passed with it. This is a launcher path choice inside the precision
demo; neither vendored cuRobo tree was edited.

For the same saved `20195` grasp, an offline command-correction screen kept
the CAD key centered and shifted the modeled hand by eleven XY offsets:
zero, ±5 and ±10 mm along each socket axis, plus two ±5 mm diagonals. Every
aligned 20 mm endpoint passed the existing key/socket and whole Inspire
hand/socket screens; the smallest reported hand clearance was 12.97 mm.
**The same nominal-hypothesis control on the failed `20096` grasp also
passed all eleven offsets**, though its smallest clearance was only
2.45 mm. The recorded corrected-hold `20096` state was later rejected because
its image-inferred held-key relation put the hand into the socket (reported
minimum hand clearance 0, while the CAD key/socket fit itself was clear).
Nominal endpoint screening therefore cannot evaluate the changed held-key relation
after physical motion; the 20195 result is a planning margin, not evidence of a
successful real correction. It also does not establish a clear intervening
path or an accurately measured camera offset. The exact inputs and outputs
are archived as `precision_{20195,20096}_xy_screen.{py,json}` beside the
yaw-gauge probes.

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

### Corrected descent with the physically successful grasp

An additional read-only replay used the actual measured pre-insertion FR3
joints from **both** human-confirmed inserted `table/0/20195` episodes. For
each saved wrist state it hypothesized that the held key began with the
opposite socket-frame XY error, then planned a rigidly held wrist shift of
0, ±5, or ±10 mm on one socket axis. All **18/18** cases passed cuRobo's
lateral plan, the sampled full-key/whole-hand world collision check, the
full-hand/key 20 mm endpoint screen, and the existing socket-axis Jacobian
descent plus its sampled held-path audit. The smallest endpoint hand/socket
clearance reported across these hypothetical cases was **4.88 mm**. Once
the GPU planner was warm, each complete lateral-plus-descent replay took
about 3.2–3.8 s in this offline process. The scope is deliberately narrow:
the hypothesized starting key-in-hand XY error was *not measured in the
photographs*, no trajectory was executed, and no force/contact outcome was
observed. These results isolate the visual correspondence problem from a
general inability to plan a corrected `20195` descent. Reproducer and
per-offset JSON are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/corrected_path_20195_20261029/`.
The three files have SHA256-matched copies at
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/corrected_path_20195_20261029/`.

The saved-session startup warmup was also compared in separate offline GPU
processes using the same frozen 5 mm socket scene. Loading the measured world
directly took **13.22 s**; loading a provisional world followed by the
measured world took **13.19 s** (13.18 s plus 0.005 s). The provisional world
already performs the expensive cuRobo initialization, so replacing that
sequence does not establish a latency benefit. The proposed runtime edit was
removed. Exact probe records are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/warmup_saved_socket_20261029/`.
Its three files have SHA256-matched copies at the corresponding
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/warmup_saved_socket_20261029/` path.

The two physically inserted `20195` timing traces distinguish wall time from
parallel worker totals. Each trace reports 28 `evidence.png_write_hash`
worker spans summing to 4.61-4.65 s, but the four enclosing
`evidence.png_batch_write` spans sum to only 0.74-0.76 s of wall time.
Removing lossless frame evidence based on the worker sum would target the
wrong bottleneck. Their initial same-trial XY measurements took 2.05-2.13 s;
the 186/135/31 s and 109/80/65 s lift/preinsert/final label spans were
dominated by waiting for interactive human answers. One trace reports a
55.55 s `cleanup.integration_close`, but its timing event ended with
`KeyboardInterrupt()`; the other completed cleanup in 0.43 s. The interrupted
span is not a normal shutdown-latency measurement. For autonomous runs, the
first cuRobo world initialization and the extra raised-view capture/motion
remain the latency costs to measure on a new physical episode.

The new read-only observation-pose replay checked a second way to obtain
better photographs without committing to one key-end algorithm. Starting
from the measured pre-insertion FR3/Inspire joints in the four archived
episodes (`5e08757b`, `4e26ec99`, `0c787032`, `9542c1bd`), it planned
20 and 40 mm upward translations along each episode's frozen socket axis.
All **8/8** cuRobo arm plans succeeded (64–97 dense samples), and the
existing sampled full-key/whole-hand versus frozen-world collision check
reported no collisions at either height. With the measured camera
calibration and nominal rigid key pose, the projected key-tip to socket-rim
centre separation changes from about **27–47 px** at the current 30 mm
pre-insertion clearance to **41–80 px** after a 20 mm raise and **59–114 px**
after a 40 mm raise, depending on camera and the moved socket pose. This is
projected separation, not measured key visibility: raising may remove socket
mask confusion while the fingers could still hide the key. The audit also
assumes the nominal key-in-hand transform; it does not establish the actual
post-squeeze relation or a successful insertion from the raised pose. The
reproducer and exact eight-row report are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_observation_raise_plan_20261029.{py,json}`.
The same two files are SHA256-matched under
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/observation_raise_20261029/`.

The method choice for the next live comparison is therefore empirical:
photograph the same held `20195` grasp at its current pre-insertion pose,
then at one collision-planned raised pose; try a separately planned cylinder
yaw if raising still leaves fewer than two views with the *same visible key
end*. Evaluate the Qwen/SAM tip fit and shaft-axis fit against manually
marked key pixels on those actual new images, rather than ranking masks by
appearance alone. Execute an XY correction only when two calibrated views
support a corresponding point. If not, keep the already physically successful
nominal insertion branch. The ZeroDex profile now defaults to a conditional
40 mm observation raise **only after the initial view lacks a metric point**;
it can be disabled with `PRECISION_SAME_TRIAL_OBSERVATION_RAISE_MM=0`. This
is a prospective data-collection action, not evidence that the tip becomes
observable at that height. The robot was off during this implementation.

The additional read-only return replay planned a 20 or 40 mm upward move and
then planned back to the original pre-insertion wrist from the raised plan's
endpoint. All **8/8** raise/return pairs passed cuRobo and the sampled
full-key/whole-hand world audit. The exact script and per-attempt results are
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_observation_raise_return_20261029.{py,json}`.
For the two physically inserted `20195` attempts, a further replay began at
each **returned joint state**, screened the full hand/key at the 20 mm
endpoint, and planned the socket-axis Jacobian descent with the same sampled
held-path audit. All **4/4** variants (20 and 40 mm raise per attempt) passed.
This closes an offline planning gap between the observation action and the
subsequent insertion path; it does not prove that the raised photograph gives
a better key-tip estimate or that the real robot completes the route. The
script and four-row result are
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_observation_return_axial_20261029.{py,json}`;
both have copies under
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/`.

The same return-and-descent replay was also run on the archived **10 mm**
socket episode `30e9cd0b`, using its actual pre-insertion joints, frozen
10 mm socket pose and selected `table/0/20267` grasp. Both 20 and 40 mm
observation raises passed cuRobo up/return planning, sampled held-key and
whole-hand world collision audit, the 20 mm insertion endpoint screen, and
the socket-axis Jacobian descent with sampled path audit (**2/2**). This
extends the offline result to another gap and grasp; it is one saved episode,
not two independent physical insertions. Its original visual result said
outside, while the later operator review judged the released key inside.
The parameterized reproducer and 10 mm report are
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_observation_return_axial_20261029.py`
and `probe_observation_return_axial_gap10_20261029.json` beside it. As for
5 mm, no raised-view photograph or corrected physical insertion exists.

The runtime now accumulates the **measured socket-axis displacement** of
each held XY shift into its observation-height estimate. If a lateral
trajectory ends slightly higher or lower than commanded, the next image
projection and return move use that measured height. This fixes a bookkeeping
error without adding a new motion threshold or failure condition.

### Optional observation yaw: compare actual held-state paths

The initial yaw-gauge target probe replaced the pre-insertion target with a
new wrist pose. A subsequent replay instead started at the **recorded held
wrist joints**, rotated that wrist about the nominal held-key origin, and
planned the complete sequence 40 mm raise → yaw → return to pre-insertion
height → existing 20 mm socket-axis descent. Both human-confirmed 5 mm
`table/0/20195` states passed at 60°, 90° and 150° (**6/6** combinations):
cuRobo planned all three observation moves, the sampled held-key/whole-hand
paths cleared the frozen world, the 20 mm endpoint passed, and the Jacobian
descent passed its sampled audit. The archived 10 mm `table/0/20267` state
also passed the whole sequence at 30°, 45°, 60° and 75° (**4/4**), while its
90° and 150° raised turns did not plan. Thus 60° is a feasible candidate
for all three saved held states, but no rotated photograph has tested whether
it reveals the same key end in two cameras. A separate replay from the 10 mm
original pre-insertion height found that direct -90°, -45°, +30°, +45°, +60° and
+75° turns cleared the sampled observation path and passed the 20 mm
endpoint and descent; +90° and +150° had no endpoint IK. The -60° and -75°
paths hit the socket in the held-key audit, and -30° reached a clear yaw
endpoint but its axial continuation failed. This comparison shows why a
replaced target pose is insufficient evidence for the motion of an already
held object. The reachability of a yaw observation depends on the current
grasp and wrist state; 60° is one candidate shared by these three records,
not a proven universal observation policy.

The runtime exposes one **disabled-by-default** observation-yaw option. It
tries the configured turn only after the current pre-insertion photograph has
no metric multiview point; the 40 mm raise remains an independently
configurable first observation action. Its geometric pivot is the nominal
held-key origin, so actual post-squeeze key-in-hand displacement is still
unknown. The code records the commanded yaw, wrist-FK-inferred nominal
key-origin XY/axis displacements, fresh photograph and subsequent axial
plan; this is not a new observation of the key inside the hand. If the optional
yaw cannot be planned, it retains the current held pose and continues through
the existing return/insertion route. No rotated live image exists yet, so
no tested angle is established as a visibility improvement or a physical
closed-loop insertion. Read-only reproducer and exact JSON are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/` as
`probe_raise_yaw_return_axial_20261029.*` and
`probe_gap10_direct_yaw_20261029.*`.

The earlier `9542c1bd` timing trace charged about 1.82-1.90 s of exact
minimum-distance scanning to each held XY move. Its saved 67-sample first
shift was replayed with the **current** collision-audit implementation. After
the CAD/FK cache was warm, collision-only checking took 0.013 s, versus
1.79 s for an exact 28-pair distance report; both returned zero collisions.
The first collision-only call took 1.20 s while loading models, so 0.013 s
is a warm-path measurement, not session startup time. This confirms that the
existing collision-only path removes that repeated diagnostic scan from XY
correction while retaining the collision decision. It does not justify
removing exact distances from other preflights that use their margins. The
read-only benchmark and its four timings are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_saved_xy_audit_cost_20261029.*`.

### 10 mm saved-image key-tip replay

The actual four undistorted pre-insertion frames from `30e9cd0b` were also
replayed through the production Qwen box, socket-projection exclusion, SAM3
ROI and distal half-ellipse route. The first three cameras (`25305462`,
`25322639`, `25322642`) boxed the fixed socket: their projected-socket box
overlaps were 100%, 100% and 98.5%. Only `26053248` yielded a SAM3 point,
so there was **no multiview metric tip**. Projecting that one point onto the
nominal tip-height plane would suggest an 80.0 mm XY correction in an episode
whose released key the operator judged inside the socket. Its fitted ellipse
also had a near-zero minor radius, which can occur when a circular face is
viewed edge-on; this alone cannot identify whether the mask is correct. The
one-view plane result remains a
diagnostic and was never sent to the robot. This independent 10 mm recording
supports the same conclusion as the 5 mm successes: changing prompt wording
or trusting one visible white patch cannot replace a view where the actual
distal end is exposed. The read-only script, four camera responses, masks and
timing record are
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_gap10_saved_preinsert_xy_20261029.*`.
Rejecting every near-zero minor radius was tested and **not adopted**: it
removed one camera from each of the four `9542c1bd` feedback steps and made
all four previously metric estimates unobservable. Those steps had reduced
the recorded XY residual from 32.0 to 1.27 mm. A near-linear projected rim
can still contribute a useful image point; point identity and cross-view
consistency must be judged from the complete observation, not that radius
alone.

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
