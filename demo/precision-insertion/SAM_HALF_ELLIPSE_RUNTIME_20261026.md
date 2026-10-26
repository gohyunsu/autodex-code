# Optional same-frame SAM3 half-ellipse XY estimator

The cylinder live CLI accepts `--xy-estimator sam_half_ellipse`; the robot-PC
launcher accepts `PRECISION_XY_ESTIMATOR=sam_half_ellipse`. The default is
still `vlm`. This option changes **failed-trial next-episode XY advice**. It
does not replace lift/preinsert/final VLM labels or the stock snapshot camera
producer. No live robot motion was run while implementing or testing it.

## Data path

1. The existing preinsert checkpoint saves undistorted RGB PNGs with camera
   IDs, exposure frame IDs/timestamps, decoded-pixel hashes and the raw
   capture manifest. The insertion path uses the same checkpoint as before.
2. On an observed failed trial, `assess_saved_preinsert_xy_diagnostic` reads
   those verified PNGs and the frozen session calibration. The planned held
   key axis selects a lossless 512×512 image crop; it supplies only search
   location and distal-end direction.
3. The existing Qwen backend asks `KEY_BOX_PROMPT` on each selected camera's
   crop. Its normalized 0–1000 box is translated into crop pixels and padded
   25 px. Both the key and socket are white; the box keeps SAM3 focused on
   the key. A missing box leaves that camera unobserved.
4. `SamRoiBackend` keeps `sam_roi_worker.py` alive in the local `autodex`
   Python environment. The worker loads the installed AutoDex SAM3 source
   and checkpoint once, receives an exact PNG of the Qwen ROI, and segments
   with `white cylinder`. It receives no live camera or robot handle.
5. `distal_key_mask_half_ellipse` fits the distal mask boundary. Its center
   maps to the original full-frame pixels by adding the planned-crop and
   Qwen-ROI origins. The existing calibrated multiview tip XY estimator
   triangulates an approximate socket-frame point and bounds the next-trial
   increment. The approximation is not a measured physical face center.
6. The attempt's `xy_retry_assessments/NNN/` report records image identities,
   Qwen prompt/answer, crop/ROI, SAM3 prompt, fit, model latency/CUDA peak and
   PNG hashes. ROI masks are saved in `sam_roi_masks/`. The already saved raw
   preinsert bundle remains the image source; only the selected camera PNG
   byte hashes are rechecked, and no second full-frame PNG set is written.
   A subsequent episode consumes the advice through the same retry ledger.

The worker defaults to `/home/robot/anaconda3/envs/autodex/bin/python` and
`/home/robot/shared_data/AutoDex/weights/sam3/sam3.pt`. Override them with
`PRECISION_SAM_PYTHON` and `PRECISION_SAM_CHECKPOINT` if that current robot-PC
installation changes. The selected Python and checkpoint files are checked
before the command-capable FR3 core opens. The worker starts during session
startup and exits with the integration. `startup.sam3_roi_model_load`,
`vlm.infer` with stage `sam_key_box`, `sam3.roi_infer`, and
`xy_retry.sam_half_ellipse_alignment` are
timed separately.
The file-order regression and the full demo suite passed after this change:
`1207 passed, 5 skipped` in the planner environment. The checks did not open
a camera or robot connection.

## Saved-camera verification

Evidence is in
`/home/robot/shared_data/hyunsu/tmp/20261026_sam_half_ellipse_live_wiring/`.
It uses six real preinsert views across the visually screened 2026-10-22
10 mm socket sessions. `fullframe_sam_comparison.json` shows that direct
full-frame `white cylinder` SAM3 found the correct key ROI in only two of
six. `fullframe_prompt_comparison.json` tests four other full-frame text
phrases; none recovered all six. That route was removed before commit.

`roi_worker_replay.json` uses the new persistent worker on the **saved source
PNG and saved Qwen answer** for each of those six views. It reproduced all
six previously recorded half-ellipse centers; the largest center difference
was `0.000017 px`. The cached model load took `12.14 s` in this run (an
earlier cold load took `56.38 s`), ROI inference `0.25–0.62 s` per view, and
peak CUDA allocation was `5.67 GB`. These are replay times, not live
episode times. The replay preserves each source path, hash, camera/frame ID,
timestamp, Qwen response, ROI, SAM mask hash and resulting center. The
saved masks are in the same NAS folder. No GIF was generated.

`roi_worker_selected_five_replay.json` repeats the ROI check on the five
configured cameras from the accepted 10 mm run `20261022_103429_752726`.
All five centers match the earlier saved result within `0.000008 px`.
`pipeline_alignment_selected_five_replay.json` then runs the actual
multiview alignment function on those saved observations. It admits all five
views and reports an approximate socket-frame tip
`[16.843, 4.535, 83.036] mm`, a requested XY correction of
`[-16.843, -4.535] mm`, and the existing `10 mm` per-trial bounded increment
`[-9.656, -2.600] mm`. The 3D point differs from the earlier offline proxy
by at most `0.042 mm` per coordinate. This checks numerical wiring and
evidence binding, not the physical accuracy of that correction.
`saved_writer_replay_20261022_103429/report.json` exercises the new artifact
writer on the same real raw bundle and masks. It binds the five selected
source PNG hashes, saves five ROI masks, and creates no duplicate full-frame
PNG directory.

In the distributed live path, five Qwen box requests start on separate capture
PCs. The diagnostic can return after the first three completed SAM points
pass the existing multiview fit; if they do not, it waits for later views.
The saved raw snapshot still binds all selected camera IDs, frame IDs,
timestamps and hashes. Only completed inference views and their masks appear
in that diagnostic report. Remaining remote calls drain before the SAM worker
is closed. The five-view replay above deliberately evaluates every view and
is not a live latency measurement.

The saved comparison does not prove physical key-tip accuracy, fixture-pose
accuracy, or insertion success. A new file-only 20 mm recipient stage at
`/home/robot/precision_stage_20261026` passed `run_robot_pc.sh --check` with
the paired `20261023_040546` calibration; its launch is explicitly
`static_preflight_only`. See the report under
`/home/robot/shared_data/hyunsu/tmp/20261026_precision_cylinder_static_audit`.
That check does not test the live Qwen/SAM workers, cameras or robot motion.
