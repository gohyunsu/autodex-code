# Released-key VLM review (2026-10-29)

## Current source and saved-data check

The capture4 four-view same-trial estimator now shifts its projected key ROI
by the **measured cumulative XY wrist motion in the frozen socket frame**.
Both Qwen native-pixel key-axis crops and Qwen-box/SAM3 crops use this
updated projection; each observation records the applied shift. Archived
attempt `9542c1bd` proves the old inconsistency: after three measured moves
totaling about (-8.4, +27.1) mm, its saved `nominal_tip_full_px` in camera
`25322639` was still `[761.56, 752.92]` in all four observations. The new
projection follows those moves while leaving the Qwen/SAM image measurement
free to disagree with the nominal held-key hypothesis. A rotated-socket
geometry regression, a camera-ROI regression, and the full nonmoving suite
passed: **1390 passed, 5 skipped**. The file-only 5 mm launch check reports
four cameras and 114 endpoint-eligible candidates.
For the archived fourth snapshot, the updated Qwen crop returned a parsable
key box in all four cameras; each box enclosed the previously saved SAM
key-face centre. The old 512 px crops happened to contain those same centres,
so this replay verifies that the updated ROI still sees the key, while the
stale prediction was a future repeated-shift risk rather than the cause of
that attempt's axial collision. The exact image hashes, old/new crop boxes,
Qwen answers and prior SAM points are in the development review file
`xy_shifted_roi_qwen_audit_20261029.json`.

For the capture4 key-face estimator, the first pass now runs Qwen box and
SAM3 ROI on cameras `25305462`, `25322639`, and `25322642`. They were the
three triangulation inliers in all four archived same-trial captures. The
fourth camera is queried only if the first pass lacks a consistent metric
point; it is still captured with the other cameras and remains available.
The alignment record lists processed and skipped cameras. This changes model
work, not a robot threshold or the insertion success criterion. A synthetic
four-camera test covers both early metric completion and fourth-view fallback;
the saved four-capture 3-view recomputation remained metric in every capture.

The local Qwen GPU replay of the current production final-socket prompt and
crop again matched the four reviewed released-key outcomes: one
`inside_socket`, three `outside_socket`, with no parse failures. This is a
four-case saved-image check. The preinsert transfer pair still has no real
captured slip-negative evaluation, and the `9542c1bd` corrected 20 mm path
still has a measured-hand/socket collision. No robot or hand command was sent
for this review.

## Same-trial arrival when the visual pair is occluded

The capture4 closed-loop route now lets an `unknown` pre-insertion VLM vote
reach the existing metric XY observation **only** when the transfer controller
reports completion, no abort, and a held grasp. The report keeps the VLM label
as unknown and records `held_arrival_visual_unknown_xy_refinement_pending` as
the stage reason. A clear `slip_or_miss` vote remains a negative transfer
label. This removes an occlusion-induced stop before the four-view Qwen/SAM3
tip measurement; it does not turn an unseen key into visual evidence. The
pre-insertion checkpoint's saved-report verifier replays the same decision.
The focused checkpoint suite passed 15 tests, and the file-only capture4
launch check still found four cameras and 114 eligible 5 mm grasp candidates.
The robot and hand were off, so no physical correction or insertion was
performed for this change.

The current compute split is appropriate for the measured machines: capture4
acquires four cameras and runs its existing SAM3/FoundPose perception; this PC
runs Qwen, cuRobo, and the persistent pre-insertion SAM3 ROI worker. A read-only
GPU query found capture4's RTX 5060 Ti had a separate GoTrack Python process
using 6,964 MiB, while the local RTX 3090 was nearly idle. Moving the ROI
worker to the shared capture PC would add a remote image/model dependency
without measured critical-path benefit. The saved physical traces instead
attribute 22.80 s of one 26.08 s first pickup plan to initial cuRobo world
setup; the archived four-view XY measurements took 2.35–2.69 s each before
the newer Qwen box batch optimization. Reassess the split from a future live
timing trace rather than changing it from GPU occupancy alone.

## Exact-input audit after a saved run

The capture4 temporal preinsert path now skips its separate GPU CAD overlay
by default. That overlay was generated before local Qwen inference but was
never a VLM input. The new report stores raw-frame hashes and the same
measured-joint hand/key hypothesis. Existing overlay reports retain their
original verifier. On archived 5 mm attempt `9542c1bd`, the four raw-image
hashes and hand/key transforms matched the earlier report when loaded with
the straight-mount capture4 profile. This is an offline consistency check,
not a measured end-to-end speedup or a new grasp observation.

`export_saved_vlm_inputs.py` reconstructs the RGB images submitted to the
classifier from a saved lift, preinsert, or current final checkpoint. It
reapplies each recorded ROI and Lanczos resize and compares the resulting
pixel SHA-256 with the original VLM input transform. The output is a per-view
PNG set, a contact sheet, and a JSON manifest with the saved model answers.
It uses no model, camera, or robot and adds no work to the live path. Example:

```bash
python demo/precision-insertion/export_saved_vlm_inputs.py \
  --report /path/to/attempt/preinsert_assessments/000/report.json \
  --output-dir ~/precision_insertion_project/development/zerodex_capture4/reviews/my_preinsert_review
```

The three-stage saved-frame audit is under
`~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_input_audit_20261029/`:
`lift_9542c1bd`, `preinsert_9542c1bd`, `preinsert_focused_9542c1bd`,
`final_30e9cd0b`, and `final_30cbe1f3`. The first preinsert report used full
frames; the focused report is an offline replay of the later crop policy. All submitted image
hashes reproduced exactly. In the focused four-view sheet, the white key is
visible below the hand in several views, while the overhead view gives much
weaker key evidence. This illustrates why a single preinsert camera or a
generic hand crop is not a dependable transfer label. The saved Qwen
`coarse_match` response is still only a coarse observation; the existing
empty-hand controls show that it can be falsely positive.

The offline 5 mm `run_zerodex_capture4.sh --check` path still passes with four
configured cameras and 114 eligible endpoint catalog entries. Focused
checkpoint, camera-selection, action-loop, lift, and same-trial tests pass
(108 tests in two groups). These are file/code checks; the robot and hand were
off during this review. The saved `9542c1bd` corrected 20 mm path still has
the measured thumb/socket collision documented below. The input exporter does
not change candidate screening, motion, or the live VLM decision rule.

For a relocated socket, start a new socket-only measurement session. The
measured pose is frozen and projected into each camera during key perception;
the final crop and preinsert hand/socket crop follow those new projected
pixels. Crops move with the socket between sessions. Neither crop nor the
collision world follows a socket moved during an active session.

This review uses saved capture4 frames only. It did not move the Franka or Inspire hand. The final label means **the key visibly occupies the socket opening after release and clear view**. It does not independently measure 20 mm depth.

The final classifier intentionally receives only the session-located socket
crop. The earlier initial-key/final-socket pair was wrong on all four reviewed
episodes with this Qwen model; both full frames remain linked for human audit.
For same-trial XY, a single visible key-face view remains a recorded planar
hypothesis, not an executed measured shift: an archived inserted 10 mm case
produced a false 35.7 mm single-view correction. The multiview criterion is
grounded in this observed error, not an additional arbitrary threshold.

## Repeatable local Qwen evaluation

`evaluate_saved_vlm.py` now accepts `released_insertion_visual` cases. It
reconstructs the **production** final-only socket crop from each case's
session-projected socket box, applies the same 384 px minimum and 768×576
classifier fit, sends the same prompt to Qwen, and records every camera token,
crop rectangle, source image hash and aggregate label. The initial tabletop
image remains in the case to bind the two capture times; Qwen sees the final
crop only. This is an offline visual-label evaluation; it never opens a camera
or robot connection.

The reviewed four-case manifest and separate visual annotations are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/released_vlm_dataset_20261029/`.
Run from the repository root with the local model already installed:

```bash
~/.venvs/precision-vlm/bin/python demo/precision-insertion/evaluate_saved_vlm.py \
  --manifest ~/precision_insertion_project/development/zerodex_capture4/reviews/released_vlm_dataset_20261029/manifest.json \
  --model-id ~/shared_data/AutoDex/models/Qwen3-VL-2B-Instruct \
  --max-input-width 768 --max-input-height 576 \
  --output ~/precision_insertion_project/development/zerodex_capture4/reviews/released_vlm_dataset_20261029/new_evaluation.json --profile
```

The local Qwen replay on 2026-10-29 classified all four reviewed final
frames as visually annotated: one `inside_socket` and three `outside_socket`.
The new per-camera breakdown of that saved report is 4/4 for `25322639`,
4/4 for `26053248`, and 3/4 for `25305462`: the last camera missed the
single visible insertion. The configured three-view vote still gives the
correct aggregate label on all four. These are four examples from one camera
layout, not a camera reliability estimate for every socket placement.
The source-hash-bound count table is saved as
`released_vlm_dataset_20261029/per_camera_summary_20261029.json` under the
development review directory.
An independent rerun with the current evaluation code and local Qwen model
saved `released_vlm_dataset_20261029/evaluation_camera_summary_20261029.json`;
it reproduced the same four aggregate labels and per-camera counts. The
paired initial images were retained and hashed, while Qwen again received
only the final session-socket crop.
An additional **socket-only negative control** used the production prompt,
session-projected socket crop, 384 px minimum long side, and 768×576 fit on
eight archived views: four from a 5 mm socket session and four from a 10 mm
session. Qwen answered `OUTSIDE` in all eight. The source frames, projection
paths, crop rectangles, exact input crops, answers, contact sheet, and replay
script are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/socket_only_negative_probe_20261029/`.
These are setup photographs, not eight additional released-key episodes.
The 10 mm side views `25305462` and `25322642` are overexposed; their empty
socket answers do not show that an inserted key would be visible there.
The positive-class evidence is still only one reviewed 5 mm episode.
The four-case benchmark's 24 raw PNGs are copied into its dataset folder and referenced by relative
path, so the evaluation does not depend on the old experiment tree. The
report is `evaluation_local_bundle.json` in that folder. This checks crop/prompt
replay on one camera layout and four recorded outcomes; it does not estimate
accuracy for a relocated socket or a new lighting condition. Review new
episodes with the same manifest format before treating their VLM labels as
reliable observations.
The evaluator's older `lift` and `insertion_visual` tasks exercise their
original generic prompts; they do not evaluate the capture4-specific
`visible_grip_v3` lift crop or the current after-lift/preinsert pair. Those
stages need their own labeled saved-image cases. The existing transfer-pair
negative control described below is synthetic and must not be counted as a
measured transfer-failure rate.

## What changed

The old final checkpoint sent each camera a two-image question: a crop of the loose tabletop key and a crop of the final socket. On the four available released-key episodes, the saved Qwen label disagreed with the visible final state every time. Varying resolution or placing the two images side by side did not resolve the observed inside case. The new checkpoint keeps the initial and final raw frames, timestamps and hashes paired for provenance, but sends **one final socket crop per camera** to Qwen. Its English question has three one-word answers: `INSIDE`, `OUTSIDE`, `UNCLEAR`. Camera votes are combined deterministically; the original tokens and input transforms are saved. Existing two-image reports still replay through their original verifier.

The exact, versioned prompt is in `precision_insertion/observer.py`. The
original `flat_top_final` wording was used for v1–v3 reports; v4 adds only
the offscreen-key clarification described below. No CAD overlay is added to
VLM pixels.

| Saved attempt prefix | Final frame, visually reviewed | Saved two-image Qwen label | Final-only Qwen, `25322639` / `26053248` |
| --- | --- | --- | --- |
| `30cbe1f3` | key lying outside; socket empty | `inside_socket` | `OUTSIDE` / `OUTSIDE` |
| `30e9cd0b` | key protruding within rim | `outside_socket` | `INSIDE` / `INSIDE` |
| `c6141471` | key lying outside; socket empty | `inside_socket` | `OUTSIDE` / `OUTSIDE` |
| `dc5958a9` | key lying outside; socket empty | `inside_socket` | `OUTSIDE` / `OUTSIDE` |

The archived `attempt_*/result.json` files still contain the original
two-image VLM decisions: `30cbe1f3`, `c6141471`, and `dc5958a9` say
`observed_success=true` although the reviewed final images show an empty
socket; `30e9cd0b` says `observed_failure` although the key visibly occupies
the opening. These historical records are immutable execution evidence,
not visual ground truth. Use the separate per-case human annotation JSONs in
`~/precision_insertion_project/development/zerodex_capture4/reviews/released_vlm_dataset_20261029/`
and the current final-only replay when comparing VLM labels. Neither the
annotation nor the replay independently verifies 20 mm depth.

These are four visually reviewed episodes, not a measured generalization rate. The first short-prompt probe is `precision_insertion_project/development/zerodex_capture4/reviews/vlm_prompt_probe_20261029.json`. An exact replay through the revised runtime observer with **all three originally selected cameras** produced the correct aggregate label on all four episodes: `vlm_released_runtime_replay_20261029.json`. Camera `25305462` answered `OUTSIDE` for the observed inside case, but the other two answered `INSIDE`, giving a 2-of-3 majority. The saved raw bundles also include `25322642`; a separate replay found that it answered `OUTSIDE` on the inside case, yielding a 2-to-2 tie if all four were used. Those fourth-view results are in `vlm_released_fourth_view_replay_20261029.json`. The capture4 profile now selects the reviewed three views **for the final label only**; four-view XY grounding and other stages are unchanged. Each full episode report is under `shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/attempt_<full-id>/released_insertion_assessments/000/report.json`.

The three final input crops and Qwen tokens for all four saved episodes are
collected in `~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_final_crop_grid_20261029.png`.

An offline camera-selection probe ranked the four views by projected socket
area and chose the largest three. On the reviewed inside case, Qwen returned
`INSIDE` for `25322639` but `OUTSIDE` for both `25322642` and `26053248`;
the aggregate was therefore wrong. The probe matched three of four reviewed
episodes (`vlm_auto_view_replay_20261029.json`). Projected size alone is not
a valid view-quality score for this classifier, so the live profile retains
its original reviewed three-camera subset.

Saved old three-view VLM decode times were 0.85–1.02 s per episode. The new replay reported 0.24–0.84 s; its first call included model warmup. This is an offline comparison, not a measured live-pipeline speedup.

## Image geometry and when it changes

| Stage | Source and crop | Update rule |
| --- | --- | --- |
| Key perception | Saved undistorted full camera image; key ROI is derived from the registered tabletop difference/SAM result. | Recomputed for every key capture. |
| Lift visual check | 650 px square around the **measured** Franka/Inspire hand projection, resized to at most 640 px. | Recomputed from current joints and calibration at each lift capture. |
| Transfer/arrival VLM | Per-camera raw `after_lift` + `preinsert` pair. The after-lift crop follows the measured thumb/index/middle contact region. The preinsert crop includes that region and the projected socket hull, with at least a 650 px side where the image permits. The normal capture4 384×288 classification ceiling then applies. | Recomputed for each pair from the two measured joint samples and this session's frozen socket projection. The uncropped raw frames and crop coordinates/hashes are retained. This is a coarse visual diagnostic, not metric XY. |
| Preinsert XY grounding | Lossless crop containing projected nominal key tip, rear and measured socket rim, with 96 px image margin and at least 256 px width/height; crop origin is recorded and Qwen/SAM pixel marks map back exactly. | Recomputed for each preinsert snapshot and correction from the current frozen socket pose and commanded held-key pose. |
| Final released-key VLM | Final raw RGB crop around the projected socket collision-mesh hull. Padding is 0.5× hull width on each side, 1.2× hull height above and 0.3× below, clipped to the image. For the classifier, crops with a long side below 384 px are enlarged with Lanczos, then all crops are fitted within 768×576. The raw crop is retained and hashed separately. If there is no valid projection, the full frame is used. | Socket projection is computed from **this session's measured socket pose** and calibration during key perception; final frames use that frozen position. A new socket location needs a new socket measurement/session. Pixel coordinates are not fixed across sessions. |

The final crop locates the socket but does not tell the model that a key is present. A fallen key may be outside the crop; the model can still see that the opening is empty. The full final frame is retained in the checkpoint for inspection. The frozen socket position is a session assumption: moving the socket without remeasurement makes the crop, collision world and XY target stale.

The tabletop key mask uses a separate, projected socket exclusion. This is
not a final-label crop or a motion-safety margin: it prevents a white socket
wall segmented by SAM3 from being admitted as the white key. In 29 archived
key captures (116 per-view proposals), four masks were rejected for socket
overlap; all four had overlap 1.0 in camera `25322642`. A saved overlay in
`socket_only_negative_probe_20261029/socket_mask_veto_example.png` visibly
shows SAM3 selected the socket wall. The configured 0.2 fraction is therefore
serving an observed object-identity problem in these captures. No new
overlap threshold or additional abort condition was added in this review.

The final-label profile prefers the three saved-frame-reviewed cameras
`26053248`, `25322639`, and `25305462`. A session-specific projection check
now substitutes another available camera only when a preferred camera's socket
hull is wholly outside that image. It does not rank cameras by projected socket
area: that ranking mislabeled the saved inserted-key example. If there is no
visible substitute, the existing full-frame input remains. The chosen cameras
and per-camera projected visibility are saved with each new final checkpoint;
the four saved final episodes retained their original three cameras in an
offline selection replay. This handles a socket moved across the cameras'
fields of view after a new session measurement. It cannot prove that the VLM
will classify a new view correctly without a labeled final image from that
position.

The transfer-pair crop is also location-adaptive. It uses measured hand joints
at the two exposures and the socket hull projected from the *current* frozen
session pose. The crop is only a way to present raw pixels at useful scale;
the projected hand/mesh is never treated as observed key pose or drawn into
the VLM image. Its input record can be replayed from the archived full frames.
Old preinsert reports without this crop record replay their original full-frame
inputs. When a camera lacks a socket projection, its preinsert VLM input
remains the full frame rather than a hand-only crop that could hide the socket.
All six saved preinsert reports still verified with their original inputs. A
new crop-policy report reconstructed from the four-camera attempt `9542c1bd`
also passed the complete checkpoint verifier in the development review folder.
Across those six saved preinsert captures, every selected camera's projected
socket hull lay fully within the new preinsert crop.
If one square cannot contain both the projected hand region and socket hull,
the preinsert input reverts to the full frame for that view; it never silently
cuts off the socket. The five saved comparison grids named
`*_full_vs_focus.png` and their per-camera rectangles are in the development
review folder above. Some camera views still occlude the carried key even in
the focused crop, so a projected hand position is not visible-key evidence.

For example, camera `25322639` used final ROIs `[554,444,1014,987]` in attempt `30cbe1f3` and `[372,432,838,969]` in attempt `30e9cd0b`. These saved rectangles demonstrate that the classifier is not tied to one fixed socket pixel location.
The capture4 final-label camera IDs are currently a fixed reviewed subset of
three; the **crop coordinates are not fixed**. If the socket moves to another
part of the board, a new socket-only session measurement updates its projected
hull in each camera. The three-view selection should be reviewed again if the
new physical layout changes occlusion substantially.

The read-only `replay_saved_live_vlm.py` now passes the report's recorded
camera IDs to the checkpoint. Without the capture4 profile, the old replay
silently used all four views and turned the reviewed inside case into a 2:2
`unobservable` vote. The corrected replay returned `inside_socket` on four
repeated inferences and logged each camera ID and inference stage in
`vlm_replay_instrumented_20261029/replay_summary.json` under the development
review folder. This fix does not change the live camera selection.

The small third-view final crop was also tested at its original size and at
1.5×/2× on all four reviewed episodes. At native 248×255 px, camera `25305462`
answered `OUTSIDE` for the visibly inserted key; at 1.5× and 2× it answered
`INSIDE`. Its three empty-socket answers stayed `OUTSIDE` at every tested
scale. The production final classifier now enlarges only crops whose long
side is below 384 px. A fresh end-to-end Qwen replay with the existing three
cameras returned the visually reviewed class on all four episodes, and all
four new reports passed the checkpoint verifier. The source frames, ROI and
hash are unchanged; the input resize is recorded as policy
`final_socket_crop_token_v2`. Saved v1 reports remain verifiable. Review data:
`~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_final_small_view_scale_20261029.json`
and `.../vlm_released_v2_checkpoint_replay_20261029/summary.json`. Four cases
from one camera layout are insufficient to estimate general accuracy.

A saved-frame batch probe passed the same three final crops to one Qwen
`infer_many` call. The four episode labels matched the independent calls, but
warm batch inference occupied 0.19–0.20 s versus roughly 0.21 s for the three
warm independent calls. The production per-camera calls remain in place.
Probe: `~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_final_batch_probe_20261029.json`.

The fourth capture4 view was retested with the v2 minimum-size crop on the
same four saved episodes. It still answered `OUTSIDE` for the visibly inserted
key while answering `OUTSIDE` for the three empty sockets. Upscaling does not
justify adding it to the final-label vote. Its crop coordinates and tokens
are saved in
`~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_final_fourth_view_v2_probe_20261029.json`.

## Pipeline review

- Socket-only capture and board-constrained socket pose come before key perception. The measured pose is frozen for repeated attempts and projected into each camera for key exclusion and final VLM ROI. The current crop is therefore location-adaptive, not a fixed pixel rectangle.
- In these four saved episodes the lift checkpoint recorded `grasp_success=true`, but only one camera was listed as a confirming close hand view in each. The preinsert checkpoint recorded `coarse_match` in all four. These labels describe the pictured moment and gross arrival; they do not establish that the key will remain held or enter the socket. The ZeroDex launcher currently waits 2 s at the lifted hold before its lift snapshot.
- An eight-attempt read-only lift-crop audit reproduced the live 650 px hand-centered crop from the archived joint samples, undistorted camera calibration, and raw after-lift PNGs. The four-view crop sheets, per-view ROIs, and the generating script are together under `~/precision_insertion_project/development/zerodex_capture4/reviews/lift_crop_audit_20261029/`. In the saved capture4 lift record, `evidence_views` lists cameras with an evaluated image; `confirming_views` lists cameras whose Qwen response was `YES`. The six recorded `held` labels had one or two confirming cameras, whereas the two recorded misses had none. White finger shells can resemble the white key in a side view, so these model labels should not be described as six independently human-verified grasps. The crop is recomputed from each exposure's measured joints; the 650 px side is fixed, but its location moves with the hand and camera projection.
- The active cylinder path uses the same-trial preinsert XY loop. Its four-camera metric grounding is separate from the final VLM label. The next-trial accumulated offset remains a separate retry path; do not apply one to the other implicitly.
- A saved-frame transfer prompt audit showed that the current Qwen3-VL-2B `coarse_match` response is biased: it answered `coarse_match` on all 12 original and all 12 focused positive-camera pairs from four recorded episodes. As a diagnostic hard negative, the two known empty-hand lift frames were duplicated as the two temporal images; all six full-frame and all six focused pairs still answered `coarse_match`. The duplicated frames are **not real transfer failures**, but demonstrate that the coarse pair prompt cannot presently be relied upon to catch a key dropped during transfer. A simpler one-image YES/NO prompt had unstable answers on the positive captures and one false positive on a known empty-hand lift. We therefore kept the existing coarse arrival class semantics, added only the raw-pixel focus, and did not promote this stage to proof of a held key. The first real transfer-loss images need manual labels for a meaningful classifier evaluation. Offline inputs, answers and comparison images are under `~/precision_insertion_project/development/zerodex_capture4/reviews/preinsert_vlm_crop_20261029/`.
- A further single-camera probe used the clearest saved hand view (`25322639`) and asked either whether the narrow key appeared in both frames (`BOTH/ONLY1/ONLY2/NEITHER/UNCLEAR`) or whether it was held in the second frame (`YES/NO/UNCLEAR`). At both 384×288 and 768×576, Qwen answered `NEITHER` or `NO` on all four real preinsert pairs **and** both duplicated empty-hand controls. Thus the alternative wording simply reversed the all-positive bias; higher classification resolution did not repair it. Exact prompts, saved-frame references, crop rectangles and responses are in `probe_temporal_tokens_{384,768}.json` in the same development folder. These controls are not actual transfer-loss episodes, so neither prompt has a defensible transfer-failure accuracy estimate. The production prompt was not changed on this evidence.
- A read-only two-side-view montage probe (`transfer_multiview_probe_20261029.py` and `.json` in the development review folder) combined cameras `25305462` and `25322642` at 1024×512 per time point. Both short prompts began with `UNCLEAR` for all four saved positive transfers and both duplicated empty-hand controls; some responses continued with explanatory text despite the requested one-word answer. Multi-view packaging alone did not yield a usable transfer label, so the live input and prompt remain unchanged. These six examples still do not include a recorded transfer-loss pair.
- A fingertip-centered crop was also tried for the *lift* classifier on seven saved lift episodes. It changed one known miss view from `NO` to `YES`, so the lift classifier and its established wrist-centered crop were left unchanged. The fingertip projection is used only to frame the transfer-pair images; it is not a new grasp-success label.
- The final checkpoint occurs after the guarded stroke, hand opening and clear-view exit. The VLM reads final pixels only; initial and preinsert captures still establish provenance and chronological order. `vlm_only` uses this visual `inside_socket` label for task success; a physical depth claim requires separate evidence.
- The saved four-view XY attempt `9542c1bd` executed three lateral corrections but its corrected axial preflight stopped at the ninth waypoint with `axis_segment_robot_collision` / `INVALID_START_STATE_WORLD_COLLISION`. No corrected insertion command was sent. This is a remaining motion-planning issue, independent of the final VLM change. Do not count that episode as a successful insertion.

The obstacle attribution in the saved development review isolates this to the
socket: the same robot state passes with the table alone, while both the
socket-only and full worlds reject it. The corrected key fits the opening, but
the measured right thumb tip intersects the socket at the intended 20 mm
endpoint. This is a candidate-specific hand/socket geometry conflict. The
configured 5 mm preferred candidate is now `table/0/208`; its nominal pickup,
transfer, 20 mm descent, and opening preflight passed offline, but no live
corrected descent or release has yet been observed for it.

When the first pre-insertion XY image has no metric multiview result, the live
same-trial branch now replans the axial path from the measured held joints
with zero observed residual. It does not reuse the earlier post-lift path,
whose first joint state could differ from the live arrival. This adds no new
confidence threshold and does not pretend to measure an unseen key point.

The current execution order was checked against
`run_robot_pc.sh`, `live_task_handlers.py`, and the session action map:
session socket measurement and frozen collision world → each attempt's
undistorted key capture → v8 pickup and measured lift → 2 s held lift image →
held transfer and raw `after_lift`/`preinsert` VLM pair → four-view Qwen-box,
SAM3-mask, half-ellipse XY measurement and repeated same-trial shifts →
corrected 20 mm axial planning → stroke → open hand and clear view → final
socket crop/VLM vote → session success or failed-trial recovery. Every
runtime launch created by `run_robot_pc.sh` has a unique evidence directory;
the fixed `evidence_dir` printed by file-only `--check` is the template path,
not the path used by `--run`. The 5 mm and 10 mm file-only checks each found
four cameras and 114 eligible endpoint catalog entries. They do not certify
camera, robot, physical success, or a new socket position.

The saved images do **not** establish dependable transfer-loss detection:
Qwen labeled all 12 original and all 12 focused positive transfer pairs,
plus all six original and six focused duplicated empty-hand controls, as
`coarse_match` in the earlier probe.
That result is the main unresolved visual-label issue. The duplicated frames
are diagnostic controls, not a measured physical transfer failure. Changing
the live prompt or adding a success/failure gate on these few examples would
substitute an unvalidated rule for the current label; retain the raw paired
frames and obtain real slip-after-lift examples before making that change.

A further per-camera probe on the same 4 saved transfer pairs and 2 duplicated
empty-hand controls used the production hand/socket crop and two short
`PEG`/`NO_PEG`/`UNCLEAR` prompts. For the 12 transfer camera pairs, the
final-image question returned 11 `UNCLEAR` and 1 `NO_PEG`; the temporal
question returned 12 `NO_PEG`. The six duplicate controls returned 4
`UNCLEAR`/2 `NO_PEG` and 5 `NO_PEG`/1 noncanonical response, respectively.
The exact inputs, crop rectangles, hashes, prompts, and replies are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/transfer_per_view_probe_20261029.{py,json}`.
Neither short prompt can replace the live label on this evidence. The
controls are not real transfer failures, and the recorded positive transfers
do not provide independent ground truth for key retention at every exposure.

## Verification and next camera session

Focused tests for the checkpoint, observer and distributed Qwen route passed. Four old checkpoint reports replay with their original labels. The exact new prompt and 768×576 crop were exercised through the revised runtime observer on the saved three-camera frames, with all four aggregate labels matching visual review. The complete `assess_released_insertion_checkpoint → write → verify` path also replayed all four saved episodes; its new reports are under `precision_insertion_project/development/zerodex_capture4/reviews/vlm_released_checkpoint_replay_20261029/<attempt-id>/report.json`. Their visual labels were `outside`, `inside`, `outside`, `outside`, respectively; only the visually inside case produced `vlm_only` task success. Old reports were not edited. GPU inference required an unrestricted read-only process because the default sandbox hid CUDA; the host RTX 3090 was available. A **new** live capture with the current socket location, lighting and three selected final views still needs evaluation. Record the per-camera token, aggregate label, final crop and manual visual label for each new episode. Keep raw frames and the observed physical outcome separate from the model label.

The released-key checkpoint now compares the sizes of the already decoded
initial/final frames. It no longer decodes the same six PNGs again solely for
this comparison. On one saved episode, reading three full-resolution PNGs
took 0.18–0.19 s, so removing six redundant reads should save roughly twice
that amount; this is a file-I/O estimate, not a measured live speedup. The
original frame hashes and replay verifier are unchanged.

## Corrected axial handoff found in the saved four-view attempt

The saved state of attempt `9542c1bd` reproduced its ninth-waypoint cuRobo
failure with the straight-mount FR3/Inspire URDF. Moving only the virtual
socket away made that state feasible; moving only the table away left it in
world collision. At the failed state, the nearest cuRobo hand sphere had about
1.1 mm socket-surface penetration. An independent exact hand-mesh check found
`right_thumb_4.STL` intersecting the socket. Reconstructing the corrected
held relation from the saved three XY shifts and last multiview residual made
the existing exact 20 mm key/hand/socket endpoint screen return
`endpoint_pass=false`: the key itself fit, but the measured hand pose did not.
The original post-lift endpoint report had passed with 8.0 mm minimum hand
clearance, so the corrected held relation changed this specific geometric
conclusion.

The same-trial corrected handoff now runs that **existing intended endpoint
screen** before its axial Jacobian plan. A rejected endpoint saves the
measured state, corrected targets, and hand/socket result as
`guarded_axial_handoffs/NNN/plan_failure.json`; no insertion command is sent.
This avoids spending planner time on a grasp whose corrected 20 mm goal is
already incompatible with the socket. The axial solver also records the
first invalid joint sample and cuRobo status when a later collision occurs.
Neither collision condition was disabled or relaxed. This saved grasp cannot
establish a successful corrected insertion; another grasp or a changed
physical key-in-hand relation is needed for that episode.

The corrected-plan failure is an explicit attempt outcome:
`preinsert_path_unavailable` with `status=planning_stopped_before_insertion`,
`success=null`, and the exact `plan_failure.json` reference. The direct
held-return planner could not reach the original tabletop target on the saved
state. A later implementation therefore reverses the *executed* XY, transfer
and lift trajectories, opens the hand at the inferred landing site, and uses
the existing key perception/restoration route. The saved attempt's 513 held
samples passed the existing sampled collision and cuRobo queries; its open-hand
clear-view plan had 103 samples. This recovery is now connected to the session
decision loop, but hand opening, landing and continuous physical execution
have not been tested. The exact sources and replay result are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/attempt_9542c1bd6c174baa9cc8a7f5b3a03211/`;
see `RECOVERY_REVIEW_20261021.md` for the implementation and limits.

The three executed XY shift archives in attempt `9542c1bd` contain 67, 64
and 68 joint samples. Each planned shift starts exactly at its saved measured
start; the largest planned-end versus measured-end joint difference was
0.0028 rad, and adjacent measured shift endpoints differ by at most
0.0006 rad. The original saved transfer trajectory, however, ends about
0.0207 rad (maximum joint difference) from the first shift's measured start.
Reversing the archived shift arrays is a plausible *planning input*, but it
is not yet a continuous return through tabletop release and clear-view exit.

An additional offline audit reversed those three shifts, the archived
transfer, and the archived lift while holding the Inspire finger joints fixed
at the last measured hold. All 510 sampled configurations passed the existing
key/hand versus fixed-world collision screen. Its `screened_pair_count=0`
means every moving/fixed bounding-box pair exceeded the clearance threshold;
the optimized collision screen therefore needed no exact mesh query. This
screen does **not** evaluate the whole Franka arm, path segments between
samples, release, or a future live key-in-hand relation. The reverse joins
had up to 0.0207 rad maximum joint difference at the shift/transfer boundary,
so a continuous executable path has not been assembled. Under the cylindrical
key's end-for-end symmetry, the reversed key center was about 3.7 mm in X,
21.8 mm in Y, and 4.7 mm in Z from its original tabletop center; its axis
differed by 1.34 degrees. A fresh visual landing check and the existing
original-pose restoration would still be needed after release. No reverse
motion command was installed or sent. Audit JSON:
`~/precision_insertion_project/development/zerodex_capture4/reviews/attempt_9542c1bd6c174baa9cc8a7f5b3a03211/reverse_executed_path_collision_audit_20261029.json`.

The final socket VLM crop now has policy `final_socket_crop_token_v3`. If a
measured socket projects partly beyond an image edge, the new policy crops
around its visible portion rather than reducing it to the entire frame. A
fully absent projection still uses the full raw frame. This changes neither
camera selection nor the one-word Qwen prompt. All 26 focused checkpoint tests
pass, and the four saved v2 reports verify unchanged. The older v1/v2 crop
algorithm remains selected by their recorded policy during replay.

The current input policy is `final_socket_crop_token_v4`. It keeps the v3
crop and selected three cameras. Its prompt clarifies that an empty opening
is `OUTSIDE` even if the fallen key lies beyond the socket crop. Two shorter
occupancy prompts were tested on all 12 saved camera images but returned
`OUTSIDE` for the visibly inserted episode in **every** view. They were not
adopted. The minimally changed prompt retained the visual labels on all
four reviewed episodes: three empty sockets `OUTSIDE`, one inserted key
`INSIDE` by two of three cameras. The exact probe inputs, prompts, tokens,
and crop rectangles are saved as
`~/precision_insertion_project/development/zerodex_capture4/reviews/probe_final_occupancy_prompt.py`
and `.json`. Legacy v1–v3 reports verify with their recorded original prompt;
the new prompt applies only to new v4 reports. These four examples do not
establish performance at a new socket position or under changed lighting.

The live final checkpoint now passes its just-captured undistorted BGR arrays
to the classifier input builder. The saved capture manifest still supplies
frame IDs, timestamps and PNG hashes; image pixel hashes are compared against
that manifest. Offline replay continues to decode and verify saved PNGs. On
the saved inside episode `30e9cd0b`, constructing the three final VLM inputs
from PNGs took a 0.214 s median across five calls; reusing already decoded
capture arrays took 0.029 s. This is a saved-data microbenchmark, not an
end-to-end live timing claim. The runner also stopped immediately replaying
the entire just-written final checkpoint before recording its label. That
full verifier remains callable after the run; in a warm saved-session-cache
replay it took about 0.59 s, while a cold offline replay took about 6.4 s.
No VLM pixels, crop policy or label rule changed in this latency pass.

### Socket-only baseline pair probe (offline)

I also tested whether showing Qwen the session's empty-socket capture as
image 1 and the final released-key socket crop as image 2 would improve the
visual label. Both images used the **same recorded final ROI** per camera;
the baseline was matched by the session's socket-capture request ID. On the
four visually reviewed episodes and three selected cameras, the model answered
`OUTSIDE` on all 12 pairs, including all three views of the visibly inserted
episode `30e9cd0b`. This temporal pair therefore cannot replace the current
final-only input. The read-only script, image hashes, ROIs, prompt and replies
are in `~/precision_insertion_project/development/zerodex_capture4/reviews/
probe_socket_baseline_pair_20261029.{py,json}`. The original socket-only
capture remains useful for measuring the socket, but the model's answer does
not become more reliable merely by adding that image.

### Current label and crop boundary

The final label's three-camera subset is configured separately from the
four-view metric XY observer. Its **pixel ROIs are recalculated at each
attempt's key capture** from the socket pose measured once at the beginning
of that socket session. A fixed padding formula is applied to each projected
socket hull; the rectangle itself is not fixed. A new socket position
requires a fresh socket measurement and fresh key capture before another
attempt. Moving the socket inside a frozen session invalidates the crop,
collision scene and target together. The saved final frames and ROIs allow
the exact visual input to be reviewed without rerunning a camera.

| Input | Spatial selection | Temporal selection | Model output and use |
| --- | --- | --- | --- |
| Socket and key perception | Socket-only ChArUco/body measurement opens the session. Each key capture uses the registered socket-only background and the projected socket exclusion; SAM's key text is `white plastic cylindrical key`. | One socket measurement per session; a new key image every attempt and after recovery. | Segmentation and FoundPose geometry, not a VLM success label. |
| Lift VLM | A 650×650 raw-pixel crop follows measured FR3/Inspire FK projected into each undistorted camera; resized to at most 640×640. | After a 2 s held dwell, the model receives the after-lift crop; the initial raw frame remains bound to the checkpoint. | Independent YES/NO in four configured views. One visible YES gives the experimental `held` visual class; `confirming_views` records which cameras actually said YES. |
| Transfer VLM | For each camera, an after-lift hand crop and a preinsert crop enclosing the projected hand and session socket. The hand crop moves with measured joints; the socket box moves with the session pose. | Same-camera after-lift/preinsert pair; no CAD overlay. | Coarse visual class. It cannot currently establish continued grip reliably because archived positive pairs and empty-hand controls both received `coarse_match`. Four-view SAM/Qwen tip grounding separately measures the XY residual. |
| Released-key VLM | Project the frozen socket hull into each undistorted final frame. The crop adds half a socket width on each side, 1.2 socket heights above and 0.3 below; clips at image edges; enlarges a crop whose long side is below 384 px, then fits within 768×576. | After the 20 mm motion, hand opening and clear-view exit, classify **only** the final socket crop. The initial tabletop and preinsert images are retained as trial evidence, not model input. | One-word INSIDE/OUTSIDE/UNCLEAR independently in the reviewed three-camera subset; majority gives the visual `inside_socket` label. It does not measure insertion depth. |

The socket is therefore allowed to be placed elsewhere on the board at the
**next session**. In the saved benchmark, camera `25322639` had socket hull
`[669,705,899,922]` in session `30cbe1f3` and
`[489,690,722,905]` in session `30e9cd0b`; the final VLM crop followed
those different projections. The crop padding constants are fixed; the
measured pose and resulting pixels are session dependent. The default launch
does not reuse an old socket session. Explicit `--reuse-session-evidence`
must only refer to an unchanged board/socket placement.

The remaining VLM limitation is at transfer arrival: archived positive
transfer frames and synthetic empty-hand controls received the same
`coarse_match` answer. The existing prompt, alternate short prompts, larger
input, and two-view montage have not separated those cases. A real captured
transfer-loss pair with a human label is needed before the arrival VLM can
be described as a reliable key-retention test. This limitation does not
invalidate the downstream four-view geometric XY calculation or the final
released-key label; those are different observations with different inputs.

### Local final-view GPU batching

The final classifier now uses `LocalQwenBackend.infer_many` to send its
three **independent one-image questions** through one local GPU decode.
Remote per-camera backends retain their parallel request path. The first
direct-model probe preserved its previous 12 camera tokens, including one
false negative on the inserted episode. A second probe used the **production
`LocalQwenBackend` image padding and the production observer** for both
serial and batch inference. The two paths returned identical tokens on all
12 saved views: `OUTSIDE` on all nine empty-socket views and `INSIDE` on
all three views of the inserted episode. Both paths gave the visually
reviewed aggregate label on all four episodes. The cold first pair took
0.776 s serial versus 0.201 s batched; over the three subsequent warm pairs,
the median was 0.254 s serial versus 0.196 s batched. These are saved-data
microbenchmarks, not full pipeline latency measurements.
Each view's raw token, crop hash and prompt remain separately recorded, so
the existing checkpoint verifier can still replay the majority vote. The
probe and individual timing values are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/
probe_final_gpu_batch_20261029.{py,json}` and
`probe_production_final_batch_vs_serial_20261029.{py,json}`. Two full saved
checkpoint replays, one inserted and one outside, also wrote and verified
new v4 reports under `batched_final_checkpoint_20261029/`.

### Hand-centered transfer prompt probe and measured latency boundary

One further read-only probe used a 650 px crop around the measured finger
region in **both** raw images, resized each to 640×640, and asked a short
`HELD`/`EMPTY`/`UNCLEAR` question about the second image. It used two
recorded preinsert pairs (`30e9cd0b`, `9542c1bd`) and two recorded empty-hand
lift images duplicated as a negative control (`687a4a49`, `ecb0c571`), in
two side cameras each. Qwen returned `UNCLEAR` for all eight requests. The
script and full answers are in the development review folder
`preinsert_vlm_crop_20261029/hand_pair_probe_640.{py,json}`; source image
bytes and input crop pixels are SHA-256 bound in the result.
The duplicated lift images are a synthetic transfer control, not a captured
transfer failure. This probe supplies no evidence for replacing the current
live transfer prompt or adding another stop condition. Actual dropped-key
transfer images with independent labels are still needed.

Two additional binary prompts on the same hand crops asked whether a separate
solid peg was visible at the hand in the second image. Both answered `NO` on
all eight requests; the second prompt also answered `NO` on all eight when
repeated with cameras `25322639` and `26053248`. Those scripts, source/crop
hashes, and responses are `hand_pair_probe_binary_640` and
`hand_pair_probe_attached{,_bestviews}_640` in the same development folder.
Together with the archived `coarse_match` responses, these probes show that
the current Qwen transfer wording is not a verified held-key classifier. In
the saved `9542c1bd` attempt, the later SAM half-ellipse estimator admitted
key-face observations in all four cameras at each of four preinsert
snapshots. That is key-tip localization evidence for this attempt, not a
validated empty-hand classifier. New labeled real transfer slips are required
to compare either method against a genuine negative case.

In the saved four-view same-trial run `20261028_233043_399147`, the four XY
measurements took 2.69, 2.35, 2.36 and 2.44 s. Each used sequential Qwen
key-box calls and SAM ROI calls in that historical run. The newer four-view
Qwen box batching was checked separately on saved frames, but has not been
timed in a new physical episode. The archived run also spent 7.45 s summed
across 44 PNG encode/hash jobs, while their six parallel batch-write spans
occupied 1.20 s wall time. Summing the child times as an extra 7.45 s of
serial latency would be incorrect. Startup SAM3 and fixed-world cuRobo warmup
were 23.71 and 20.36 s respectively in that run; these are session startup
costs, not costs repeated at every XY correction. No runtime threshold or
failure rule was changed based on this timing review.

### Archived production-input audit

The archived production inputs for eight lift, six pre-insertion, and four
released-key checkpoints are collected under
`~/precision_insertion_project/development/zerodex_capture4/reviews/vlm_presence_crop_20261029/`.

The six archived pre-insertion transfers were also replayed through the
**current** measured-hand plus session-socket crop policy, rather than the
historical full-frame input. `replay_current_preinsert_crop.py` writes one
`current_crop_<attempt>.jpg` sheet per attempt and
`current_crop_replay_results.json` with the exact per-camera ROI and Qwen
answer. The local Qwen backend used the live profile's native-pixel mode,
128-token limit, three or four available camera pairs, and the production
`observe_preinsert_temporal_views` prompt. All 6/6 transfers returned
`coarse_match` at both the configured 384×288 classifier fit and an offline
640×480 fit; all 19 per-camera answers matched at both sizes. Median warm
inference was 0.71 s versus 0.88 s per multiview checkpoint respectively.
The larger input did not improve these saved positive labels and costs more
inference time, so the runtime size remains 384×288. The replay does not
contain a genuine empty-hand pre-insertion frame: it verifies positive-case
reproduction and crop placement, not failure detection or future accuracy.
No runtime prompt or failure threshold was changed from this result.

All saved crop rectangles were 650–776 px squares inside the original
2048×1536 images; none fell back to the full frame. The after-lift ROI tracks
the measured hand, and the pre-insertion ROI encloses both that hand and the
socket projected from the session's frozen measured pose. The replay sheets
show why evidence differs by camera: in some views the hand hides much of the
key. These coordinates are recomputed per camera and per session, not fixed
pixel coordinates. The exact live crop procedure and its round-trip pixel
hash test are in `preinsert_checkpoint.py` and
`tests/test_preinsert_checkpoint.py`.

`export_saved_vlm_inputs.py` now reconstructs the older two-image released-key
reports as well as the current final-only policy, verifying each resized input
against its saved RGB hash. The old reports remain historical answers; the
export does not relabel them.

A key-local pre-insertion probe used the same 512 px nominal-axis crop as the
current XY detector. A short single-image presence question found at least
one positive view in all six archived arrivals, but only one of three views
for `30e9cd0b`, whose key is visibly present. A paired-image variant answered
`NO` in all four views of `9542c1bd` despite its visible key. On two archived
empty-lift controls, the single-image presence question gave one positive
among six views; the XY detector's Qwen box prompt returned a box in all six.
These controls are not real empty-hand pre-insertion examples. The probe
supports keeping a Qwen box, SAM3 mask, calibrated geometry, and a task label
as distinct evidence. It does not support replacing the live transfer
checkpoint with either new prompt or treating a box as a held-key label. The
exact images, answers, scripts, and per-attempt crop mappings are in the
development review directory. No extra live inference or abort criterion was
added.

### Closed-loop descent boundary

The same saved run spent 12.85 s on three measured XY motions and 4.04 s on
four pre-insertion captures. The corrected axial Jacobian replan lasted only
0.056 s before the hand/socket geometry conflict stopped it. The 10 mm move
cap and a fresh snapshot after each move account for the repeated cycles in
this 32 mm initial-error example; image writing is already parallel and was
not the main wall-time cost.

The live sequence is final XY observation, corrected held relation from the
measured wrist and observed residual, exact hand/socket endpoint screen,
socket-axis Jacobian waypoints with full robot collision checks, held path
audit, planned open-hand/axis-exit/clear-view route, stock force-stop stroke,
release, then final clear-view VLM. A failed endpoint or route is recorded as
a planning failure and sent to held-key recovery, never as an insertion.
For the archived `9542c1bd` grasp, the thumb/socket intersection is the
reason this sequence cannot reach the stroke. A separate nominal replay of
`table/0/208` passes pickup, transfer, 20 mm descent and release opening,
and an additional saved-scene replay passed the open-hand socket-axis exit
and clear-view retract (101 descent samples, 118 retract samples). The replay
is in `candidate_208_full_release_replay.json` beside the collision review.
That candidate has not been tested after a real XY correction.
