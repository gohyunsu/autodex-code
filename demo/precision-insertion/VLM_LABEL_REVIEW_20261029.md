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

Saved old three-view VLM decode times were 0.85–1.02 s per episode. The new replay reported 0.24–0.84 s; its first call included model warmup. This is an offline comparison, not a measured live-pipeline speedup.

## Image geometry and when it changes

| Stage | Source and crop | Update rule |
| --- | --- | --- |
| Key perception | Saved undistorted full camera image; key ROI is derived from the registered tabletop difference/SAM result. | Recomputed for every key capture. |
| Lift visual check | 650 px square around the **measured** Franka/Inspire hand projection, resized to at most 640 px. | Recomputed from current joints and calibration at each lift capture. |
| Transfer/arrival VLM | Per-camera raw `after_lift` + `preinsert` pair. The after-lift crop follows the measured thumb/index/middle contact region. The preinsert crop includes that region and the projected socket hull, with at least a 650 px side where the image permits. The normal capture4 384×288 classification ceiling then applies. | Recomputed for each pair from the two measured joint samples and this session's frozen socket projection. The uncropped raw frames and crop coordinates/hashes are retained. This is a coarse visual diagnostic, not metric XY. |
| Preinsert XY grounding | Lossless crop containing projected nominal key tip, rear and measured socket rim, with 96 px image margin and at least 256 px width/height; crop origin is recorded and Qwen/SAM pixel marks map back exactly. | Recomputed for each preinsert snapshot and correction from the current frozen socket pose and commanded held-key pose. |
| Final released-key VLM | Final raw RGB crop around the projected socket collision-mesh hull. Padding is 0.5× hull width on each side, 1.2× hull height above and 0.3× below, clipped to the image. The crop is resized to fit 768×576 for classification. If there is no valid projection, the full frame is used. | Socket projection is computed from **this session's measured socket pose** and calibration during key perception; final frames use that frozen position. A new socket location needs a new socket measurement/session. Pixel coordinates are not fixed across sessions. |

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

## Pipeline review

- Socket-only capture and board-constrained socket pose come before key perception. The measured pose is frozen for repeated attempts and projected into each camera for key exclusion and final VLM ROI. The current crop is therefore location-adaptive, not a fixed pixel rectangle.
- In these four saved episodes the lift checkpoint recorded `grasp_success=true`, but only one camera was listed as a confirming close hand view in each. The preinsert checkpoint recorded `coarse_match` in all four. These labels describe the pictured moment and gross arrival; they do not establish that the key will remain held or enter the socket. The ZeroDex launcher currently waits 2 s at the lifted hold before its lift snapshot.
- The active cylinder path uses the same-trial preinsert XY loop. Its four-camera metric grounding is separate from the final VLM label. The next-trial accumulated offset remains a separate retry path; do not apply one to the other implicitly.
- A saved-frame transfer prompt audit showed that the current Qwen3-VL-2B `coarse_match` response is biased: it answered `coarse_match` on all 12 original and all 12 focused positive-camera pairs from four recorded episodes. As a diagnostic hard negative, the two known empty-hand lift frames were duplicated as the two temporal images; all six full-frame and all six focused pairs still answered `coarse_match`. The duplicated frames are **not real transfer failures**, but demonstrate that the coarse pair prompt cannot presently be relied upon to catch a key dropped during transfer. A simpler one-image YES/NO prompt had unstable answers on the positive captures and one false positive on a known empty-hand lift. We therefore kept the existing coarse arrival class semantics, added only the raw-pixel focus, and did not promote this stage to proof of a held key. The first real transfer-loss images need manual labels for a meaningful classifier evaluation. Offline inputs, answers and comparison images are under `~/precision_insertion_project/development/zerodex_capture4/reviews/preinsert_vlm_crop_20261029/`.
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
