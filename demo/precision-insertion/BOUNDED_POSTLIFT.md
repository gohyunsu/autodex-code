# Hidden-key post-lift planning with a physical grasp calibration

After Inspire closes, the key can be largely occluded. A positive **raw**
two-view lift checkpoint establishes only that visible key pixels followed
the hand. It does not provide a 6D key pose or the achieved key–hand transform.
The demo now has a second post-lift *planning* branch that does not demand a
runtime held-key FoundPose estimate:

1. Use the unchanged v8/object-processing trial and exact socket endpoint
   catalog to choose the pickup. Save the raw `after_lift` capture and call
   `SessionRunner.prepare_raw_lift_label(...)`; this must yield a positive
   `grasp_success` and `held_relation_evidence_required`.
2. During a separate commissioning campaign, independently observe the key
   and measured wrist in at least five **distinct physical pickups** of this
   exact candidate. Hash the original per-pickup source JSON, including
   measurement-error bounds and measured Inspire joints. Use
   `calibrate_physical_held_relation(...)` to produce a grasp-specific medoid
   `T_key_hand` and descriptive scatter. MuJoCo squeeze states and the v8
   nominal `wrist_se3.npy` cannot masquerade as these physical samples.
3. Independently commission future-trial bounds on the displacement of
   **every moving key and hand surface relative to the frozen socket** over
   the whole relevant motion. These must cover grasp variability/slip,
   calibration, Franka tracking/FK, finger geometry and socket/CAD errors.
   Empirical scatter from step 2 is only a necessary lower bound, never the
   future bound itself. Represent the inputs as `SurfaceDeviationBounds` with
   source `commissioned_future_trial_surface_bound`.
4. At a stationary post-lift hold, collect a new `LiveRobotState` and call
   `SessionRunner.prepare_bounded_postlift_transfer(...)` with the physical
   calibration file, bounds, and commissioned timing/drift limits. The method
   re-verifies the lift images and calibration source files, checks measured
   Inspire joints against the calibration range, re-screens the actual held
   hand and medoid relation at centered 20 mm, replans transfer/axial waypoints
   from the measured 13-DOF state, and rejects insufficient sampled clearance
   after the supplied surface bounds.
5. A `sampled_postlift_preflight_pass` yields only
   `transfer_execution_gate_required`. After separately controlled transfer,
   `prepare_observed_preinsert_label(...)` compares fresh raw multi-view images
   with the **predicted** physical-calibration overlay and measured wrist pose.
   The overlay is not a measured key pose; occlusion remains unknown. The
   subsequent 20 mm task label still needs guarded depth, force and visual
   evidence.

The API deliberately requires all limits as arguments rather than embedding
uncommissioned robot thresholds:

```python
from precision_insertion.uncertainty_margin import SurfaceDeviationBounds

bounds = SurfaceDeviationBounds(
    key_surface_m=commissioned_key_surface_bound_m,
    hand_surface_m=commissioned_hand_surface_bound_m,
    source="commissioned_future_trial_surface_bound",
)
plan = runner.prepare_bounded_postlift_transfer(
    planner=fr3_inspire_planner,
    physical_calibration_path=exact_candidate_calibration_json,
    joint_sample=fresh_stationary_feedback,
    bounds=bounds,
    max_state_age_s=commissioned_state_age_s,
    max_arm_hand_skew_s=commissioned_arm_hand_skew_s,
    max_hand_command_error_raw=commissioned_hand_error_raw,
    max_arm_velocity_rad_s=commissioned_hold_velocity_rad_s,
    max_postlift_arm_drift_rad=commissioned_postlift_arm_drift_rad,
    max_postlift_hand_drift_raw=commissioned_postlift_hand_drift_raw,
    max_calibration_hand_excess_rad=commissioned_hand_range_excess_rad,
    limits=commissioned_path_audit_limits,
    axial_waypoint_step_m=commissioned_axial_step_m,  # <= 0.005 m
)
assert plan.status == "sampled_postlift_preflight_pass"
# This assertion does not authorize Franka motion or insertion contact.
```

Reports are saved under
`attempts/<attempt_id>/postlift_preflights/<index>/`. The report records the
raw lift and physical-calibration paths/hashes, measured state, medoid,
descriptive lower bound, supplied future bound, exact endpoint result, path
and sampled uncertainty margins. `physical_relation_binding.json` binds these
to the session and catalog. The supervisor re-verifies source bytes before
offering the transfer execution gate; the pre-insertion checkpoint checks them
again. This verifies file consistency, **not** physical authenticity or
statistical coverage of the supplied bound.

No grasp-specific physical calibration or commissioned future-trial bound is
present in the current local/NAS handoff. There is also no commissioned live
camera provenance adapter, fail-closed transfer/contact controller, or
bounded-relation **live** XY retry. After a logged safe withdrawal, the
cylinder-only `prepare_grounded_xy_diagnostic(...)` can now re-verify this
post-lift report and use native-pixel multi-view VLM tip/axis observations to
calculate a continuous socket-frame XY correction capped to a 1 mm increment.
It saves a metric hypothesis only; no lateral or renewed insertion plan is
recorded. A separate `lateral_preflight.py` now plans and sampled-audits a
socket-plane hold shift using the unchanged cuRobo planner, the full held
key/hand meshes and explicit future surface bounds. It does not yet consume
the saved diagnostic or physical calibration through a verified session
binding, and never plans a renewed insertion. The existing observed-key
**live** XY retry still requires a fresh
held-key FoundPose after withdrawal. A square key also needs observable yaw,
so its two-collinear-landmark route abstains. Do not use a test fixture or a
nominal/MuJoCo transform in place of the missing physical records.
