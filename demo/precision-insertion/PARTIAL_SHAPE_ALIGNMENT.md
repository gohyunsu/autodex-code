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

No camera ID is hard-coded and no camera is permanently blacklisted.
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

The commissioned default remains the endpoint/temporal controller.  An
explicit A/B profile can promote the saved partial-shape fit to live XY and
tilt control without rerunning segmentation:

```bash
export PRECISION_EXPERIMENT_PROFILE=fixed_grasp_partial_shape_alignment_5mm.env.sh
```

Promotion requires two distinct calibrated views, a full-rank four-variable
image Jacobian, and an estimated tilt inside the stage policy's existing
`alignment_limits.max_axis_tilt_deg` range (currently 10 degrees).  This does
not add a task failure: if the fit is missing, rank-deficient, or outside that
commissioned model range, the same observation continues through the
endpoint/temporal controller.  `alignment_selection` records both the reason
and the preserved endpoint result.

This distinction matters because saved attempts show that two inconsistent
occluded views can still make the optimizer converge to a formally full-rank
but physically implausible 34--55 degree pose.  Those estimates remain in the
diagnostic record instead of becoming wrist commands.  The controlled A/B
experiment is still pending; no success-rate improvement is claimed.

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

Replay every labelled attempt under the canonical object run root:

```bash
PYTHONPATH=$PWD/demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/review_partial_shape_alignment.py \
  --attempt-root /mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/precision-insertion/inspire/precision_key_cylinder_r15_h80 \
  --output /tmp/partial_shape_all_attempts.json
```

On 2026-11-02 this discovered 12 saved same-trial attempts. Two labelled
successful attempts contained enough current-format shape evidence; their
last fitted XY values were approximately `(0.8, 1.0) mm` and
`(5.8, -0.5) mm`. The latest wall-contact failure remained approximately
`(26.0, 22.6) mm` with mutually inconsistent shaft directions. These values
are correlations in saved evidence, not physical key-pose ground truth.

Replaying the exact live selector over the same 57 observations selected the
partial-shape fit 14 times, fell back because shape geometry was unavailable
39 times, and fell back because a converged fit was outside the existing
10-degree alignment-model range four times. Those four cases are precisely
the saved 33.9--55.3 degree occlusion fits; the selector did not turn them
into wrist commands. The replay is stored at
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/partial_shape_alignment_20261102/all_attempts_live_selection_v2.json`.

The batch report now includes a `summary` block.  For every attempt it keeps
the final iteration status and, separately, the last earlier iteration with a
valid multiview fit.  This distinction matters when the last view becomes
occluded after an otherwise measurable sequence.  It also groups the saved
XY norm and axis tilt by the recorded physical outcome.  No statistic in this
block is read by the live controller or converted into a new execution gate.

Before making partial shape the default controller:

1. recalibrate the capture4 cameras;
2. replay all labelled success/failure attempts;
3. compare predicted motion against measured motion and final tip-entry label;
4. render observed masks, projected CAD contour and per-view residuals; and
5. compare the opt-in partial-shape profile with the endpoint profile using
   the recorded runtime-configuration identity; and
6. change the default only after the fitted XY/tilt is repeatable across
   distinct grasps and socket locations.
