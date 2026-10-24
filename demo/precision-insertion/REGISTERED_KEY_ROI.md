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

The explicit board config is essential. At the original saved-image replay,
the then-current `paradex/system/current/charuco_info.json` listed only
boards `1–3`, while the saved AutoDex rig used board `9` from
`system/paradex2`; that first replay correctly abstained. On this robot PC
as checked on 2026-10-13, `system/current` includes board `9`. The demo
probe still constructs the selected board from the supplied config without
changing that system directory. Current precision-insertion session startup
must explicitly select board `9` for the board-9 fixture.

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
`/mnt/paradex2/hyunsu/handoff/20261011_registered_key_roi/`.

`session_key_baseline.load_session_key_baseline(...)` reads
one **explicitly selected** socket capture from the frozen session bundle. It
verifies the complete bundle, demands acquisition-bound frames and at least
two admitted calibrated views, and binds each socket-only image/mask to its
camera, frame ID, request ID and manifest digest. The preferred helper also
checks that this digest matches the **currently opened run**. For example:

```python
from precision_insertion.session_key_baseline import load_bound_session_key_baseline

baseline = load_bound_session_key_baseline(
    opened=opened_precision_session,
    socket_capture_id="socket_001",
    calibrated_camera_ids={"CAMERA_A", "CAMERA_B"},
)
```

The opt-in `registered_key_capture` path now connects that baseline to
`prepare_next_live_key(..., registered_key_assistance=assistance)`. A caller
first constructs `assistance` with `prepare_registered_key_assistance(...)`,
passing the currently `OpenedPrecisionSession`, the explicitly chosen socket
capture ID, the *actual rig's* ChArUco board config, explicit ROI thresholds,
and already-loaded stock `Sam3ImageSegmentor` and `FoundPoseInit` objects for
the selected key. No model is loaded or camera request made by the factory.
The caller may keep both model objects alive across key trials.

```python
from precision_insertion.registered_key_capture import prepare_registered_key_assistance
from precision_insertion.live_key_trial import prepare_next_live_key

assistance = prepare_registered_key_assistance(
    opened=live_session.opened,
    socket_capture_id="socket_001",
    calibrated_camera_ids=calibrated_camera_ids,
    board_config_path=rig_charuco_config_path,
    board_id="9",
    limits=commissioned_roi_limits,
    sam_segmentor=loaded_stock_sam3,
    foundpose_model=loaded_stock_key_foundpose,
)
prepared = prepare_next_live_key(
    **commissioned_trial_arguments,
    registered_key_assistance=assistance,
)
```

The supplied FoundPose model's `obj_name` must equal the session mode's key
object ID. `prepare_next_live_key` also checks the reference against the
run's saved `source_evidence_binding.json` *before* issuing a camera request.

After the normal AutoDex key request, the optional path reuses its exact saved
undistorted frames and acquisition IDs. It registers each view to its own
socket-only frame, runs SAM3 on the crop, maps the mask back to full pixels,
and calls stock FoundPose with the **full frame and full-frame camera matrix**.
The existing key-view quality, fixed-socket exclusion, multi-view agreement,
stock IoU/silhouette refinement and v8 preflight then run unchanged. A failed
ROI, SAM mask or FoundPose view is rejected; stock full-frame outputs are never
used as a fallback. Its evidence records the original stock result as an
unselected summary, the ROI/mask QA, frame provenance, baseline source,
board config bytes and thresholds. The verification step rechecks the session
baseline and refuses mixed perception producers.

The optional producer writes an exclusive
`CAPTURE_DIR/registered_key_roi/` diagnostic **before** the key-pose admission
gate. It contains the original same-request frames, derived full-frame masks,
pose payloads, per-view refusal reasons and hashed source manifest. Thus a
failed perception attempt remains inspectable even when there is no admitted
key pose or v8 trial preflight. A passing trial additionally gets the normal
`key_evidence_dir` bundle; the diagnostic does not itself grant motion.

This is an **experimental opt-in** producer, not the default `run_auto.py`
path. The socket-only procedure must still be checked physically: hashes do
not prove that the key or a moving hand was absent. The exploratory limits
in the supplied JSON are not commissioned for robot motion. Independent
same-session key-mask/pose annotations, camera-time validation, GPU placement
and real trial checks remain before relying on this path to grasp.
