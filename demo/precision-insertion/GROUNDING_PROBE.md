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
intrinsics. The inspected Qwen processor requires image dimensions to be
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
