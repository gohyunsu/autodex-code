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

This is **not yet** a `guarded_execution_v1` key-depth source: there is no
commissioned final-camera producer/held-out landmark error study or saved
source binding. To integrate it, build `CalibratedXYFrame` objects from the
verified final/abort capture and frozen session camera/socket transforms,
run `observe_exposed_key_rear_axis`, estimate the interval using approved
`ExposedDepthLimits`, then save raw image hashes, prompts, responses,
landmarks, calibration/CAD hashes, timestamps and the computed interval.
The insertion checkpoint must verify that complete source before using it
for a physical success label. Local VLM metric points require native-size
images (`require_native_pixels=True`); resized coordinates are rejected.
