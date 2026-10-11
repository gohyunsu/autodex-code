# 20 mm depth label in the live pipeline

The guarded controller's commanded wrist stroke is **not** key penetration.
The first and retry checkpoints therefore retain a tri-state
`insertion_success`: `True`, `False`, or `None`. The controller's unverified
depth claim alone still yields `None` for a visually normal insertion.

An optional completion path now reuses the existing
`saved_exposed_depth.py` estimator. After the final/abort exposure, Qwen marks
the visible rear-face centre and shaft centreline in each original AutoDex
view. Calibrated multi-view geometry reconstructs the rear point and directed
axis in the frozen socket frame; the CAD rear-to-tip length extrapolates the
hidden tip. The estimated 20 mm penetration is an interval widened by
commissioned worst-case landmark, axis, rim and CAD errors. Invisible rear
face or insufficient parallax means **abstain**, not success.

`LiveTaskHandlers` and `LiveGroundedRetryHandlers` accept the optional fields
`depth_limits`, `depth_capture_parameters`, and
`depth_commissioning_path`. `depth_limits` is an `ExposedDepthLimits` instance;
`depth_capture_parameters` supplies `max_capture_skew_s`, `max_final_gap_s`
and `max_frame_age_s`. If limits are supplied, each handler writes a
replayable report under the current attempt's `depth_assessments/first/` or
`depth_assessments/retry/`. With no commissioning path, that report remains
diagnostic and the success depth stays unadmitted. Configure the same reviewed
commissioning path for both handlers if both strokes should be eligible to
produce a positive task label.

The commissioning JSON schema is
`precision_insertion_exposed_depth_commissioning_v1`. It must contain the
selected `{family, gap_mm, key_object, socket_object}` mode, exact camera and
task-geometry SHA-256 hashes, VLM model ID, the exact serialized
`ExposedDepthLimits`, `reviewed_by`, a positive `minimum_heldout_trials`, and
`heldout_evaluation: {path, sha256}`. The referenced
`precision_insertion_exposed_depth_eval_v1` audit must use the same cameras
and mode, meet the declared sample count, contain the independent instrument
depth references, and show no interval-containment failures or possible
false-positive 20 mm labels. The reviewer remains responsible for deciding
whether those trials cover the actual rig and expected failure distribution;
a finite held-out sample cannot mathematically prove future worst-case error.

At observation time `depth_admission.py` replays the saved images, VLM text,
geometry and bounds; checks the commissioning/evaluation binding; and admits
the interval only for the exact final capture, attempt and session. The
insertion checkpoint then combines it with the VLM appearance, alignment,
held-grasp state and abort status. A positive lower depth bound of at least
20 mm plus those positive observations yields `insertion_success=True` and
the session ends immediately. The same route works after one grounded XY
retry. Any missing report, occlusion, mismatched review or inconclusive bound
leaves the outcome unknown or failed according to the other evidence.

No real AutoDex-camera held-out depth audit or reviewed commissioning JSON is
available in this checkout. The synthetic tests demonstrate the software
contract only; they are **not** physical calibration or a real 20 mm success.
