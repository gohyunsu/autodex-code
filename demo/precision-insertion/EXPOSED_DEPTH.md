# Optional VLM-assisted exposed-length depth diagnostic

At final/abort hold, the insertion tip can be hidden inside the socket.
`observe_exposed_key_rear_axis` asks a local or API VLM for the **visible rear
face centre** and two pixels on the projected straight key axis in each raw,
undistorted AutoDex view. It never asks the VLM to guess the hidden tip or
read a CAD overlay as an observation. The existing multi-view axis-line
triangulator estimates the rear centre and downward unit axis in the frozen
socket frame. `estimate_exposed_depth_for_mode` reads the same v8 task
geometry used by endpoint planning and computes:

```text
tip_est = rear_est + L_CAD * axis_est
depth_est = socket_rim_z - tip_est_z
depth_bound = rear_position_bound
            + 2 * L_CAD * sin(axis_angle_bound / 2)
            + rim_height_bound + CAD_tip_projection_bound
interval = [depth_est - depth_bound, depth_est + depth_bound]
```

The triangle inequality makes the interval conservative **only if** each
supplied error is a valid worst-case bound covering actual VLM landmark,
triangulation, camera/hand-eye, socket pose, and CAD/print deviations. A
reprojection residual or model confidence is not such a bound. The present
synthetic tests establish the calculation and fail-closed visibility gates;
they do not establish any real-world error bound. At a nominal *exactly*
20 mm pose, any positive bound makes the lower endpoint less than 20 mm, so
this method cannot honestly certify `>=20 mm` without observed extra depth.

The diagnostic abstains when too few views show the rear/axis, exposures
are not synchronized, camera parallax/axis-plane separation is insufficient,
the axis is too tilted, or pixel reprojection is inconsistent. The hand may
hide the rear face in many Inspire grasps; this route is optional and must
not silently fall back to a predicted key pose. Square-key yaw, contact
forces and grasp retention are separate gates even if its axial depth is
visible.

`assess_saved_exposed_depth` now performs the **read-only saved-image handoff**:
it verifies a `final_or_abort` raw capture, checks its exposure interval
against the supplied guarded-stroke completion, reloads the frozen session
camera/socket frame, calls the VLM on full undistorted images, and saves the
original text, parsed pixels, image/calibration/CAD hashes and interval.
`verify_saved_exposed_depth` rechecks these files and replays the saved VLM
text through the same parser and geometry without calling the model again.
This is source consistency, **not** an independent physical measurement.

The report deliberately sets `depth_source_admissible_for_task_label=false`.
It is **not yet** a `guarded_execution_v1` key-depth source: the final camera
producer and the worst-case landmark/camera/socket/CAD error bounds still
need rig-specific, held-out commissioning. The guarded execution log also
needs to bind its actual completion time and attempt ID to this report.
Until those gates are established, `insertion_checkpoint.py` must not use
the interval for a physical success label. Local VLM metric points require
native-size images (`require_native_pixels=True`); resized coordinates are
rejected.

## Evaluate against independent depth measurements

`evaluate_exposed_depth.py` compares saved visual-depth reports with a separate
instrument measurement made during the **same final hold**. This is the next
commissioning input, not a way to promote a visual interval to a task label.
Collect both failures (short insertion/rim jams) and apparent successes,
including hand occlusion; keep data used to choose VLM prompts/bounds separate
from the held-out evaluation set. The evaluator checks source hashes, the
frozen session/CAD/camera IDs, attempt/candidate/final-request identity,
instrument calibration file, timestamp clock and maximum acquisition skew.
These checks cannot verify the external instrument's physical accuracy or
that the key stayed motionless between exposure and measurement.

An `independent_depth.json` reference has this shape (replace all example
IDs/hashes and use real absolute instrument file paths):

```json
{
  "schema": "precision_insertion_independent_depth_v1",
  "attempt_id": "trial_001",
  "candidate_id": "table/0/3",
  "mode": {"family": "cylinder", "gap_mm": 15.0},
  "session_calibration_sha256": "<64 hex characters>",
  "camera_calibration_sha256": "<64 hex characters>",
  "task_geometry_sha256": "<64 hex characters>",
  "final_manifest_sha256": "<64 hex characters>",
  "final_capture_id": "final_001",
  "final_request_id": 71,
  "measurement_method": "calibrated_depth_gauge",
  "measurement_time_s": 1790000000.0,
  "clock_domain": "unix_utc",
  "depth_interval_m": [0.0192, 0.0196],
  "raw_evidence": [
    {"path": "/abs/path/to/raw_gauge_log.csv", "sha256": "<64 hex characters>"}
  ],
  "instrument_calibration": {
    "path": "/abs/path/to/gauge_calibration.json",
    "sha256": "<64 hex characters>"
  }
}
```

The depth interval is the instrument's bounded *key penetration below the
socket rim*, not commanded wrist travel. An external optical-metrology system
can instead use `measurement_method: "external_optical_metrology"` and its raw
pose/images. Its error interval must include the instrument, fiducial,
hand-eye, socket-rim and timing errors; do not substitute a confidence score
or the same VLM output as independent evidence. `clock_domain` must match the
raw camera capture. A measurement after release/reset is not the same hold,
even when file hashes and IDs match.

Place a manifest beside the per-trial files, with relative paths that remain
inside its directory:

```json
{
  "schema": "precision_insertion_exposed_depth_eval_manifest_v1",
  "max_reference_capture_skew_s": 0.02,
  "samples": [
    {"depth_report": "trial_001/depth/report.json",
     "independent_depth": "trial_001/independent_depth.json"}
  ]
}
```

Then run from the repository root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/evaluate_exposed_depth.py \
  --manifest /path/to/evaluation.json --shared-root ~/shared_data \
  --mode cylinder --gap-mm 15 --session /path/to/session_calibration.json \
  --output /path/to/new_depth_evaluation.json
```

The report includes abstention rate, visual intervals that do not contain the
independent interval, definite/possible false 20 mm successes, and the largest
observed overestimation bound required by these samples. A sample maximum is
**not** a deterministic bound for future trials. The evaluator always emits
`depth_source_admissible_for_task_label=false`; the insertion checkpoint still
records `unknown` unless another separately verified source is commissioned.
