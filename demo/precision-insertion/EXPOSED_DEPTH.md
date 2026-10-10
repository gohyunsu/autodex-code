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
