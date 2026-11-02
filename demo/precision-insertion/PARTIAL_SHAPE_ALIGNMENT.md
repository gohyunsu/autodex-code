# Occlusion-aware partial-shape alignment

## Why it exists

The original closed loop triangulated one estimated distal-face centre per
camera.  In a saved 2026-11-02 failure, camera `26053248` repeatedly selected
a hand-occluded white fragment.  Its image feature followed the wrist motion,
but it was not the key end.  Temporal consistency therefore cannot establish
feature identity by itself.

The runtime now requires a new metric multi-view observation before a
single-view probe can accumulate.  In parallel, it records a partial-shape
fit that uses both the visible distal centre and the visible shaft direction.

## Symmetry-reduced model

The circular key has no observable or task-relevant axial yaw.  At the fixed
pre-insertion height, its visual pose is represented by

```text
q = [x, y, tilt_x, tilt_y]
```

For each calibrated camera, the CAD key tip and rear are projected into the
undistorted image.  The robust optimization residual contains:

- the 2-D difference between the projected and observed distal centres; and
- the angular difference between the projected and observed shaft axes,
  expressed in pixels over the projected CAD shaft span.

The solver minimizes all view residuals directly in image space.  It does not
triangulate every face centre independently.

## Occlusion-aware information weights

No camera ID is hard-coded and no view is rejected by a new threshold.
Measurements from the SAM mask determine complementary information weights:

- distal-centre information: projected face minor/major radius ratio;
- shaft-axis information: silhouette eccentricity;
- common support: largest-component solidity.

Thus an end-on view contributes a strong centre and little shaft direction,
while a long side view contributes a strong shaft direction.  A concave
hand-occluded fragment remains in the report but receives less influence.

The recorded `visible_shape_profile` includes area, solidity, fill fraction,
axial/lateral span and occupied axial bins.  `partial_shape_alignment`
includes the fitted pose, per-view residuals, Jacobian rank and condition.

## Current commissioning status

The result is currently **diagnostic only** and does not command the robot.
This is intentional: saved attempts show that two inconsistent views can
still determine a mathematically full-rank but physically incorrect tilt.
The XY part is already more stable in replay, but orientation must first be
validated after camera recalibration and against more human-labelled runs.

Replay one saved attempt without camera or robot access:

```bash
cd ~/autodex-code
PYTHONPATH=$PWD/demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/review_partial_shape_alignment.py \
  --attempt-dir /mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/precision-insertion/inspire/precision_key_cylinder_r15_h80/attempt_<id> \
  --session-run-dir /mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/<run_id> \
  --output /tmp/partial_shape_review.json
```

Before promoting this estimate to live control:

1. recalibrate the capture4 cameras;
2. replay all labelled success/failure attempts;
3. compare predicted motion against measured motion and final tip-entry label;
4. render observed masks, projected CAD contour and per-view residuals; and
5. enable the result only after the fitted XY/tilt is repeatable across
   distinct grasps and socket locations.
