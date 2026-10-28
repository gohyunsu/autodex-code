# Released-key VLM review (2026-10-29)

This review uses saved capture4 frames only. It did not move the Franka or Inspire hand. The final label means **the key visibly occupies the socket opening after release and clear view**. It does not independently measure 20 mm depth.

## What changed

The old final checkpoint sent each camera a two-image question: a crop of the loose tabletop key and a crop of the final socket. On the four available released-key episodes, the saved Qwen label disagreed with the visible final state every time. Varying resolution or placing the two images side by side did not resolve the observed inside case. The new checkpoint keeps the initial and final raw frames, timestamps and hashes paired for provenance, but sends **one final socket crop per camera** to Qwen. Its English question has three one-word answers: `INSIDE`, `OUTSIDE`, `UNCLEAR`. Camera votes are combined deterministically; the original tokens and input transforms are saved. Existing two-image reports still replay through their original verifier.

The exact prompt is in `precision_insertion/observer.py`. It is the same `flat_top_final` prompt used in the saved offline probe. No CAD overlay is added to VLM pixels.

| Saved attempt prefix | Final frame, visually reviewed | Saved two-image Qwen label | Final-only Qwen, `25322639` / `26053248` |
| --- | --- | --- | --- |
| `30cbe1f3` | key lying outside; socket empty | `inside_socket` | `OUTSIDE` / `OUTSIDE` |
| `30e9cd0b` | key protruding within rim | `outside_socket` | `INSIDE` / `INSIDE` |
| `c6141471` | key lying outside; socket empty | `inside_socket` | `OUTSIDE` / `OUTSIDE` |
| `dc5958a9` | key lying outside; socket empty | `inside_socket` | `OUTSIDE` / `OUTSIDE` |

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
- The active cylinder path uses the same-trial preinsert XY loop. Its four-camera metric grounding is separate from the final VLM label. The next-trial accumulated offset remains a separate retry path; do not apply one to the other implicitly.
- A saved-frame transfer prompt audit showed that the current Qwen3-VL-2B `coarse_match` response is biased: it answered `coarse_match` on all 12 original and all 12 focused positive-camera pairs from four recorded episodes. As a diagnostic hard negative, the two known empty-hand lift frames were duplicated as the two temporal images; all six full-frame and all six focused pairs still answered `coarse_match`. The duplicated frames are **not real transfer failures**, but demonstrate that the coarse pair prompt cannot presently be relied upon to catch a key dropped during transfer. A simpler one-image YES/NO prompt had unstable answers on the positive captures and one false positive on a known empty-hand lift. We therefore kept the existing coarse arrival class semantics, added only the raw-pixel focus, and did not promote this stage to proof of a held key. The first real transfer-loss images need manual labels for a meaningful classifier evaluation. Offline inputs, answers and comparison images are under `~/precision_insertion_project/development/zerodex_capture4/reviews/preinsert_vlm_crop_20261029/`.
- A further single-camera probe used the clearest saved hand view (`25322639`) and asked either whether the narrow key appeared in both frames (`BOTH/ONLY1/ONLY2/NEITHER/UNCLEAR`) or whether it was held in the second frame (`YES/NO/UNCLEAR`). At both 384×288 and 768×576, Qwen answered `NEITHER` or `NO` on all four real preinsert pairs **and** both duplicated empty-hand controls. Thus the alternative wording simply reversed the all-positive bias; higher classification resolution did not repair it. Exact prompts, saved-frame references, crop rectangles and responses are in `probe_temporal_tokens_{384,768}.json` in the same development folder. These controls are not actual transfer-loss episodes, so neither prompt has a defensible transfer-failure accuracy estimate. The production prompt was not changed on this evidence.
- A fingertip-centered crop was also tried for the *lift* classifier on seven saved lift episodes. It changed one known miss view from `NO` to `YES`, so the lift classifier and its established wrist-centered crop were left unchanged. The fingertip projection is used only to frame the transfer-pair images; it is not a new grasp-success label.
- The final checkpoint occurs after the guarded stroke, hand opening and clear-view exit. The VLM reads final pixels only; initial and preinsert captures still establish provenance and chronological order. `vlm_only` uses this visual `inside_socket` label for task success; a physical depth claim requires separate evidence.
- The saved four-view XY attempt `9542c1bd` executed three lateral corrections but its corrected axial preflight stopped at the ninth waypoint with `axis_segment_robot_collision` / `INVALID_START_STATE_WORLD_COLLISION`. No corrected insertion command was sent. This is a remaining motion-planning issue, independent of the final VLM change. Do not count that episode as a successful insertion.

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

The corrected-plan failure is now an explicit attempt outcome:
`preinsert_path_unavailable` with `status=planning_stopped_before_insertion`,
`success=null`, and the exact `plan_failure.json` reference. The decision loop
ends at `stop_for_review` while the key is still held; it does not fabricate
an insertion label or try another candidate with the current held key.
An offline replay of the existing held-return planner from that saved corrected
joint state and inferred relation returned `held_transfer_unreachable`, so an
automatic return to the original tabletop pose is not established for this
attempt. The failed query was the return target's endpoint IK; no return
trajectory was produced. This is an unfinished recovery path before unattended
repeated trials.
The replay script and log are in
`~/precision_insertion_project/development/zerodex_capture4/reviews/attempt_9542c1bd6c174baa9cc8a7f5b3a03211/`.
The same saved corrected hold was also tried against the original tabletop
landing and 12 nearby XY landing points. All 13 held-return queries failed at
the return target endpoint IK before a release path existed. Automatic recovery
from this specific corrected-plan failure remains an implementation gap; these
offline probes did not command the robot.

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
