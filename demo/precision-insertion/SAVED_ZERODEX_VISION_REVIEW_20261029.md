# Saved ZeroDex episode vision review (2026-10-29)

This review uses only archived capture4 images and the operator's final
labels. The Franka and Inspire hand were off. Five released-key cases use the
5 mm cylinder socket and one uses the 10 mm socket, all on the same
four-camera rig; there are three `inside_socket` and three `outside_socket`
labels. They are a replay set, not
an estimate of performance after the socket, lighting, or camera layout moves.

## Final visual label

The initial tabletop image and the final released, clear-view image remain
paired and saved for every camera. The model receives the **final** raw socket
crop only. The crop follows the session-measured socket CAD projection, with
0.5 socket widths at the sides, 1.2 socket heights above, and 0.3 below.
Small crops are enlarged to a 512 px long side, then fitted within 768×576.
The exact English prompt is:

> Classify the large white socket: B = narrow solid peg inserted in its
> opening; A = empty hollow opening. Reply A or B.

Qwen3-VL-2B answers independently for `25305462`, `25322639`, and
`26053248` in one GPU batch. A strict A/B parser and the existing three-view
majority produce the visual label. No CAD overlay is shown to Qwen. The
label means visible socket occupancy, **not** independently verified 20 mm
penetration. Manual mode retains both operator and VLM labels.

| Attempt prefix | Operator | Qwen 25305462 / 25322639 / 26053248 | Majority |
| --- | --- | --- | --- |
| `30cbe1f3` | outside | A / A / A | outside |
| `30e9cd0b` | inside | A / B / B | inside |
| `c6141471` | outside | A / A / A | outside |
| `dc5958a9` | outside | A / A / A | outside |
| `5e08757b` | inside | B / B / B | inside |
| `4e26ec99` | inside | B / B / B | inside |

This is **6/6 episode labels** with the production crop, native-pixel
padding, default 512-token output allowance, and three-view batch. A
different direct-model probe without the production padding was less stable,
which is why the official evaluator now replays that padding and the v5
prompt. The raw answers and source hashes are in the development directory's
`released_vlm_dataset_20261029/evaluation_v5_six.json`; the six-case
`manifest_six.json` preserves the older four-case manifest and adds the two
operator-confirmed insertions. The previous `INSIDE/OUTSIDE` prompt and
two-image question remain readable only for historical reports.

As a separate **diagnostic**, the checkpoint compares the same small socket
mouth region in the initial and final undistorted frames. It records the
fraction of pixels whose mean RGB change exceeds 15, per camera and as a
three-view median. The outside cases have medians 0.001, 0.003, 0.003;
inside cases have 0.279, 0.238, 0.246. These six happened to separate, but
illumination, shadows, or motion can change the same pixels. This score is
never substituted for a VLM label or used to stop an episode. Its replay is
`vlm_label_eval_20261029/temporal_change_eval.json` in the development
directory.

## Pre-insertion key-face point

The current chain is a 512×512 crop around the **planned** key axis, Qwen
key box, SAM3 mask inside the box, distal half-ellipse centre, and calibrated
multi-view triangulation. The planned axis only chooses a crop and which
silhouette end is distal. It is not an observed key pose.

The archived failure was more basic than a triangulation threshold: Qwen
sometimes boxed the **fixed white socket** as the white key. In 12 saved
pre-insertion observations, 39 camera boxes were recorded. Nine boxes had
at least 90% of their own area inside the projected frozen-socket bounding
box, and eight of those nine had already been accepted as SAM3 key-face
points. Visual review of the
two latest inserted episodes confirms socket masks in cameras `25305462`
and `25322642`. The image and source-bound box audit is at
`endpoint_roi_eval_20261029/endpoint_box_overlap_audit.json` and the
two-episode crop grid is
`endpoint_roi_eval_20261029/preinsert_key_box_grid_two_successes.jpg`.

The endpoint chain now projects the already-frozen socket into the same
undistorted camera frame and skips SAM3 when at least 90% of Qwen's proposed
key box lies in the broad socket bounding box. This is an object-identity
check backed by the saved boxes, not a new robot failure condition. In the
four earlier `9542c1bd` and four `0c787032` snapshots, the three reported
inlier boxes in each snapshot had less than 90% overlap. The filter leaves
their usable views available while removing the clearly wrong socket boxes.
It also avoids a SAM3 call for each rejected box.

The two latest inserted episodes **did not yield a consistent metric key-face
point**: the remaining view `26053248` shows the key, but its saved SAM3
result had no key mask. Increasing the ROI padding or changing `white
cylinder` to `white cylindrical key` did not consistently recover a mask in
the saved-image probe. Consequently, neither episode demonstrated a measured
same-trial XY correction; both inserted on the nominal target. A single
visible tip view plus an assumed tip height can yield a planar XY hypothesis,
but previous archived data include a false 35.7 mm single-view estimate. It
must not silently become a metric 3D correction. When geometry is
unobservable, the current action loop can continue its nominal insertion;
it does not invent an XY offset.

For repeatable metric correction, the next data collection should first make
the key end independently visible in at least two calibrated oblique views
at pre-insertion, then compare triangulated image points and residuals with
the saved raw frames. Moving the hold farther above the socket or changing a
camera angle are scene changes to test with new images; a prompt alone cannot
recover an occluded physical point. Keep the frozen socket projection in the
image audit so a socket mask cannot masquerade as the key again.

## Reproduce the final-label replay

From the repository root, with the local Qwen weights installed:

```bash
~/.venvs/precision-vlm/bin/python demo/precision-insertion/evaluate_saved_vlm.py \
  --manifest /mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/released_vlm_dataset_20261029/manifest_six.json \
  --model-id /mnt/paradex2/hyunsu/precision_insertion/runtime/active_shared_data/AutoDex/models/Qwen3-VL-2B-Instruct \
  --max-input-width 768 --max-input-height 576 \
  --output /tmp/released_v5_replay.json --profile
```

The existing four-case historical manifest is unchanged. The full raw
capture and manual-label records remain in their original attempt folders.
