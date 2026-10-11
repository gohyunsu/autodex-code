# Registered tabletop-key ROI: saved-image diagnostic

The first key capture contains both the fixed socket and the small key. The
stock AutoDex init daemon asks SAM3 to segment the **full image**. In the NAS
square-key evaluation, a separate socket-only frame was ChArUco-registered to
the key frame to find a key ROI before cropped SAM3. That offline variant is
not the current live daemon path. `registered_key_roi.py` implements its
geometric ROI proposal and an additional mask QA gate entirely under this
demo; it does not change stock AutoDex or create a robot motion permit.

For each *same-camera*, undistorted socket-only/current image pair, it:

1. matches ChArUco corner IDs, checks board displacement, and estimates a
   RANSAC homography;
2. warps the baseline image/socket mask to the current frame, limits the
   difference to the board, and removes a dilated socket mask;
3. proposes a padded crop around the largest remaining component, abstaining
   if a second component is comparably large;
4. optionally runs the existing SAM3 segmentor **on that crop**, maps its
   mask back to original pixels, and rejects masks with excessive socket
   overlap or insufficient support from the registered difference.

The exploratory thresholds are in
[`configs/registered_key_roi_exploratory.json`](configs/registered_key_roi_exploratory.json).
They came from the saved NAS diagnostic, **not** physical commissioning. The
ROI is only a proposal: even a passing mask still needs per-view FoundPose,
silhouette/identity checks, cross-view agreement, tabletop-pose
classification and a fresh state-bound v8 trial preflight. A baseline from a
different camera, a moved board/socket or a different session is invalid.

To replay one saved view without loading SAM3, supply its previously saved
evaluator mask via `--saved-evaluator-mask`. The CLI checks that mask against
the new ROI; it does not generate or authenticate it:

```bash
cd /home/hyunsu/autodex-code
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/probe_registered_key_roi.py \
  --current-jpeg /absolute/key_shot/CAMERA.jpg \
  --socket-only-jpeg /absolute/socket_only_shot/CAMERA.jpg \
  --socket-only-mask /absolute/socket_only_masks/CAMERA_mask.png \
  --saved-evaluator-mask /absolute/key_eval/CAMERA_mask.png \
  --calib-dir /absolute/cam_param/20260921_213318 \
  --camera-id CAMERA --board-id 9 \
  --board-config-json /absolute/paradex/system/paradex2/charuco_info.json \
  --limits-json demo/precision-insertion/configs/registered_key_roi_exploratory.json \
  --output /absolute/new_roi_probe.json
```

The explicit board config is essential on this workstation:
`paradex/system/current/charuco_info.json` lists only boards `1–3`, while the
saved AutoDex rig uses board `9` from `system/paradex2`. The first replay
using the workstation's global detector correctly abstained for missing board
corners. The demo probe now constructs board `9` directly from the supplied
config without altering that global symlink. The AutoDex PC must likewise
verify its **actual** board config at session startup.

On the real saved square 1.5 mm `pose_02` key shot, camera `24122734`:

- the proposed crop was `[1135, 820, 1398, 1043]`, within one pixel of the
  NAS evaluator's `[1136, 821, 1397, 1042]`;
- all 24 common board corners were homography inliers, but the evaluator's
  10,224-pixel SAM mask had **100% overlap** with the dilated registered
  socket exclusion and **0% support** from the selected key difference;
- the new mask QA abstained with `sam_mask_overlaps_socket`. The earlier
  evaluator's CAD fit for this view had IoU 0.04947, consistent with a
  wrong-object mask rather than a usable key observation.

For comparison, saved camera `25322644` from the **same** key shot had 38
registration inliers; its 9,643-pixel evaluator mask had 0 socket overlap
and 90.3% difference support, so this *mask-only* gate passed. It remains an
offline image/mask result, not a key pose or a physical-grasp result. Hashes,
exact source paths and geometry are recorded in the generated JSON reports:
[`24122734`](/home/hyunsu/shared_data/AutoDex/precision_insertion/registered_key_roi_saved_probe_20261011/24122734.json)
and
[`25322644`](/home/hyunsu/shared_data/AutoDex/precision_insertion/registered_key_roi_saved_probe_20261011/25322644.json).
Identical readback-verified copies for the AutoDex PC are under
`/mnt/paradex2/hyunsu/autodex_precision_insertion_registered_key_roi_20261011/`.

This module is **not yet installed in the live key capture producer**. To do
that safely, first bind one verified socket-only baseline image and mask per
camera from the *same frozen session*, retain the current frame's acquisition
identity, run the cropped SAM/FoundPose chain with full-frame pixel mapping,
and compare it against independently annotated key masks/poses. Do not treat
the saved NAS mask pass or the exploratory thresholds as evidence that the
live pipeline is ready to grasp.
