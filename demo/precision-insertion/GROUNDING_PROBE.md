# Saved-image local VLM alignment probe

`probe_grounded_alignment.py` exercises the **existing** local ZeroDex/Qwen
backend and cylinder tip-plus-visible-axis multi-view estimator on saved,
undistorted AutoDex views. It is a read-only way to see the VLM's raw pixel
marks, rejected cameras, 3D fit and continuous socket-frame XY correction
before wiring any robot command. It does not certify the model's pixel
accuracy or provide a live insertion permit. Square-key yaw is not solved by
this two-landmark probe.

The manifest has schema `precision_insertion_saved_grounding_probe_v1`:

```json
{
  "schema": "precision_insertion_saved_grounding_probe_v1",
  "max_camera_skew_s": 0.02,
  "socket_rim_z_m": 0.055,
  "verification_depth_m": 0.02,
  "alignment_limits": {
    "pixel_sigma_px": 1.0,
    "max_reprojection_px": 3.0,
    "min_parallax_deg": 5.0,
    "max_axis_tilt_deg": 4.0,
    "max_20mm_axis_sweep_m": 0.001,
    "max_lateral_uncertainty_95_m": 0.0005,
    "systematic_lateral_sigma_m": 0.0001,
    "cad_spacing_sigma_m": 0.0003,
    "minimum_views": 3
  },
  "views": [
    {
      "camera_id": "CAMERA_SERIAL",
      "image": {
        "path": "preinsert_CAMERA_SERIAL.png",
        "sha256": "64 lowercase hex digits of the image file bytes",
        "timestamp_s": 1791600000.0
      },
      "intrinsics": [[1000, 0, 1024], [0, 1000, 768], [0, 0, 1]],
      "roi_xyxy": [800, 500, 1200, 900],
      "T_camera_socket": [[1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0.5], [0, 0, 0, 1]]
    }
  ]
}
```

This is a **format example**, not a measured rig calibration or recommended
threshold set. Supply at least `minimum_views` distinct cameras, the true
full-resolution intrinsics for each **undistorted** image, and the frozen
session's `T_camera_socket` transform. Add one `views` row per camera; the
single row above is illustrative only. `timestamp_s` must be the corresponding
camera exposure time on one common clock, not the AutoDex publisher `ts`.
The probe checks claimed skew and image hashes, but cannot prove timestamp or
calibration provenance from this manifest. Obtain commissioned values from
held-out localization errors and the actual AutoDex session; do not copy the
example numbers into runtime safety gates.

`roi_xyxy` is **optional**, but if used it is required for every view. It is
an integer `[x0,y0,x1,y1]` box in the original undistorted image, with the
last coordinates exclusive. Choose it from an independent key/socket region
proposal with enough margin to include the *actual* visible tip and shaft;
a crop based only on a rigid squeeze assumption can hide a slipped key. The
probe copies these raw pixels without resampling and pads only the right and
bottom to the local Qwen processor's 32-pixel grid (at least 256 pixels per
side). It rejects marks on that padding. The model's raw JSON and parsed
marks remain crop-local, while `pixel_mapping.mapped_full_pixels` and the 3D
estimator use exact full-frame coordinates `u_full=u_crop+x0`,
`v_full=v_crop+y0` with the original camera intrinsics. The source-image hash
still binds the *whole* original frame. This avoids treating a resized pixel
as a metric observation; it does not establish that the VLM's mark is
accurate. The same explicit ROI map may be supplied to the *read-only* live
`SessionRunner.prepare_grounded_xy_diagnostic(..., roi_xyxy_by_camera=...)`
for a cylindrical key after a documented failed insertion and guarded
withdrawal. It records the full raw frames and replays the crop mapping at
observation time. If its saved diagnostic is passed to the lateral hold
preflight, that binder replays every raw answer, crop geometry, full-frame
pixel mapping, camera/frame identity and saved image hash. The ROI must be
independently selected from the live key region; a predicted squeeze pose
alone is not sufficient. The square-key two-landmark route does not support
this ROI option because insertion yaw remains unobservable. Lift, arrival,
and insertion checkpoint verifiers still use their existing full-frame
inputs. This option does not approve motion or contact.

From the repository root, after the [local VLM setup](README.md):

```bash
export PYTHONPATH="$PWD/demo/precision-insertion:$HOME/realtime_vlm"
~/.venvs/precision-vlm/bin/python \
  demo/precision-insertion/probe_grounded_alignment.py \
  --manifest /path/to/new_grounding_manifest.json \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --output /path/to/new_grounding_report.json
```

The script refuses to overwrite a report or load the model if hashes,
transforms or claimed cross-camera times fail. Local metric grounding uses
`require_native_pixels=True`: Qwen/ZeroDex resizing is rejected, since
resized pixel coordinates cannot be triangulated against original camera
intrinsics. For an ROI the padded input is checked against the model's actual
processor geometry as well. The inspected Qwen processor requires image dimensions to be
multiples of 32; latest *saved* AutoDex calibration lists 2048×1536, but a
live session must check its actual frames. `--allow-cpu` is only for a slow
offline smoke test.

The report preserves source paths/hashes, exact prompts and raw responses,
per-view pixel marks, inlier/rejected cameras, an estimated continuous XY
offset, and an at-most-1-mm *diagnostic* next increment. A malformed answer,
hidden tip, inconsistent projection or excessive uncertainty yields an
abstention. Even a non-abstaining result is not a measured 20 mm insertion
outcome or permission to move: validate the local model against independent
real held-key annotations, current camera-time provenance, fixture
calibration, trajectory/collision checks and guarded force control first.

In a 2026-10-11 RTX 3090 transport smoke test, cached Qwen3-VL-2B received a
256×256 crop of a **presentation panel**, not a real held-key frame. It
returned `[256,256]` for the tip and both line points—exactly outside the
displayed pixel range. The parser rejected all marks and produced no metric
landmark. This confirms model/crop transport and fail-closed bounds, **not**
localization ability; real synchronized AutoDex frames with independent
tip/axis labels remain necessary.
