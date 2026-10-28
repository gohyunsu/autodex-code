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

Saved old three-view VLM decode times were 0.85–1.02 s per episode. The new replay reported 0.24–0.84 s; its first call included model warmup. This is an offline comparison, not a measured live-pipeline speedup.

## Image geometry and when it changes

| Stage | Source and crop | Update rule |
| --- | --- | --- |
| Key perception | Saved undistorted full camera image; key ROI is derived from the registered tabletop difference/SAM result. | Recomputed for every key capture. |
| Lift visual check | 650 px square around the **measured** Franka/Inspire hand projection, resized to at most 640 px. | Recomputed from current joints and calibration at each lift capture. |
| Transfer/arrival VLM | Per-camera raw `after_lift` + `preinsert` image pair; ordinary classification resize is configured separately (capture4 default 384×288). | Fresh images each attempt; this stage checks visible continuity, not metric XY. |
| Preinsert XY grounding | Lossless crop containing projected nominal key tip, rear and measured socket rim, with 96 px image margin and at least 256 px width/height; crop origin is recorded and Qwen/SAM pixel marks map back exactly. | Recomputed for each preinsert snapshot and correction from the current frozen socket pose and commanded held-key pose. |
| Final released-key VLM | Final raw RGB crop around the projected socket collision-mesh hull. Padding is 0.5× hull width on each side, 1.2× hull height above and 0.3× below, clipped to the image. The crop is resized to fit 768×576 for classification. If there is no valid projection, the full frame is used. | Socket projection is computed from **this session's measured socket pose** and calibration during key perception; final frames use that frozen position. A new socket location needs a new socket measurement/session. Pixel coordinates are not fixed across sessions. |

The final crop locates the socket but does not tell the model that a key is present. A fallen key may be outside the crop; the model can still see that the opening is empty. The full final frame is retained in the checkpoint for inspection. The frozen socket position is a session assumption: moving the socket without remeasurement makes the crop, collision world and XY target stale.

For example, camera `25322639` used final ROIs `[554,444,1014,987]` in attempt `30cbe1f3` and `[372,432,838,969]` in attempt `30e9cd0b`. These saved rectangles demonstrate that the classifier is not tied to one fixed socket pixel location.

## Pipeline review

- Socket-only capture and board-constrained socket pose come before key perception. The measured pose is frozen for repeated attempts and projected into each camera for key exclusion and final VLM ROI. The current crop is therefore location-adaptive, not a fixed pixel rectangle.
- The active cylinder path uses the same-trial preinsert XY loop. Its four-camera metric grounding is separate from the final VLM label. The next-trial accumulated offset remains a separate retry path; do not apply one to the other implicitly.
- The final checkpoint occurs after the guarded stroke, hand opening and clear-view exit. The VLM reads final pixels only; initial and preinsert captures still establish provenance and chronological order. `vlm_only` uses this visual `inside_socket` label for task success; a physical depth claim requires separate evidence.
- The saved four-view XY attempt `9542c1bd` executed three lateral corrections but its corrected axial preflight stopped at the ninth waypoint with `axis_segment_robot_collision` / `INVALID_START_STATE_WORLD_COLLISION`. No corrected insertion command was sent. This is a remaining motion-planning issue, independent of the final VLM change. Do not count that episode as a successful insertion.

## Verification and next camera session

Focused tests for the checkpoint, observer and distributed Qwen route passed. Four old checkpoint reports replay with their original labels. The exact new prompt and 768×576 crop were exercised through the revised runtime observer on the saved three-camera frames, with all four aggregate labels matching visual review. The complete `assess_released_insertion_checkpoint → write → verify` path also replayed all four saved episodes; its new reports are under `precision_insertion_project/development/zerodex_capture4/reviews/vlm_released_checkpoint_replay_20261029/<attempt-id>/report.json`. Their visual labels were `outside`, `inside`, `outside`, `outside`, respectively; only the visually inside case produced `vlm_only` task success. Old reports were not edited. GPU inference required an unrestricted read-only process because the default sandbox hid CUDA; the host RTX 3090 was available. A **new** live capture with the current socket location, lighting and three selected final views still needs evaluation. Record the per-camera token, aggregate label, final crop and manual visual label for each new episode. Keep raw frames and the observed physical outcome separate from the model label.
