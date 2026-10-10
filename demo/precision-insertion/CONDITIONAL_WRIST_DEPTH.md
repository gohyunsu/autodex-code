# Conditional wrist-to-key depth

`precision_insertion.conditional_wrist_depth.conditional_key_tip_depth()`
computes a *hypothesis* for the insertion-tip centre, not observed physical
penetration. It uses the exact planned CAD entry pose, the fixed socket frame,
a measured wrist SE(3), the held `T_key_hand`, and a commissioned worst-case
key-surface displacement bound. The CAD file's SHA-256 must match the target
plan; both square and cylinder geometry use the actual local key-tip centre.

If `T_robot_hand = T_robot_key T_key_hand`, the nominal tip pose is obtained
from `T_robot_key = T_robot_hand (T_key_hand)^-1`. With insertion direction
`a` and CAD tip at the socket rim on entry, the nominal axial depth is
`d = a · (p_tip_live - p_tip_entry)`. A worst-case key-surface error `e`
gives the conditional interval `[d-e, d+e]`; lateral tip residual relative
to the *planned entry*, actual tip offset from the **socket axis**, and axis
tilt are separately reported. The first two lateral quantities differ for an
intentional XY retry offset. This interval is valid only if the grasp relation
really persists and the supplied bound covers **grasp slip, wrist/FK error,
socket calibration and CAD** for the entire relevant motion. Its source
label is intentionally `null`, so this diagnostic cannot be fed directly
to the insertion-success resolver as an independently measured key depth.

At a planned 20 mm endpoint with any positive error bound, the lower depth
bound is below 20 mm. Thus endpoint joint feedback alone cannot certify the
task threshold. A physical positive label still needs a commissioned
independent key-depth observation (or a validated cross-check of the held
relation with an adequate depth margin), time-bound force/grasp evidence,
and the saved multi-view final-camera assessment. The current AutoDex rig
has not yet supplied that commissioning evidence.
