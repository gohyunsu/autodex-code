# Cylinder: next-trial XY learning (current live contract)

`run_live.py` uses AutoDex cameras and the selected full-key v8 grasp. It does
**not** measure or substitute a post-squeeze key–hand transform. The nominal
v8 relation is an explicit hypothesis throughout transfer and the centered
20 mm endpoint collision preflight. Square remains an offline-only mode with
no XY learning in the live entry point.

For attempt `n`, synchronized calibrated AutoDex views are saved while the
held key is at the pre-insertion position. They are captured **before** the
guarded stroke. If the later released-key label is `outside_socket`, those
same saved raw views—not a newly lifted key and not a CAD overlay—are sent to
local Qwen (Gemini is optional) to mark the visible insertion-tip center and
cylinder-axis line. Camera intrinsics/extrinsics and the frozen session socket
pose—not VLM text—produce a socket-frame metric XY estimate.
The local Qwen path frames each native-pixel input with a lossless rectangular
crop containing the projected commanded tip, key rear and frozen socket rim,
plus 96 px of margin. It pads to the model's 32 px patch size without resizing
source pixels. The crop only locates the image region; its projected points
are never promoted to observed landmarks. The saved original frame, crop
translation and Qwen marks remain linked so returned pixels map exactly into
the undistorted full frame. Each capture PC owns at most one persistent Qwen
worker; clean session shutdown releases that model before the next session.

- Two or more visible tips: triangulate in 3D. If the axis-line fit is usable,
  minimize mean XY error at the rim and 20 mm depth; otherwise use tip XY with
  the explicit vertical-axis assumption. A metric tip estimate must have
  adequate camera parallax and reprojection agreement, and its depth must be
  within the existing 10 mm held-key translation budget of the commanded
  preinsert tip height. Inconsistent VLM marks cannot become a metric XY step.
- One visible tip: intersect its calibrated camera ray with the **commanded
  nominal pre-insertion tip-height plane**. This supplies XY but does not
  measure tip depth.
- No usable tip: do **not** report a fabricated measurement or abstain from
  the trial loop. Record an explicitly named deterministic 2 mm XY search
  step. It is exploration, not multi-view triangulation.

The diagnostic status is `diagnostic_metric_xy_correction` for a multi-view
measurement, `diagnostic_planar_xy_hypothesis` for a single-view ray intersected
with the commanded height plane, or `diagnostic_exploratory_xy_step` when no
usable tip was observed. `next_trial_xy_advice.json` and the attempt event
retain this status, the exact `xy_source`, and the supporting camera IDs. An
exploratory step has no supporting camera and must never be described as a
measured XY residual.

The continuous correction `d_n` is clipped to a **10 mm Euclidean norm per
failed attempt**. It is applied only to the *next* pickup:

`p_0 = (0,0);  p_(n+1) = p_n + w · d_n`, with default `w = 1.0`.

There is **no separate cumulative XY bound** and no default count cap.
`--max-xy-retries` can opt into a count cap. Every advice record retains its
raw/weighted increment, previous and next cumulative offsets, evidence
images, camera IDs, source class (`multiview`, `single_view_nominal_height`,
or `exploratory_search`), calibration/catalog binding, and VLM replies. The
correction is bound to the same candidate and tabletop pose; if the key
lands in another pose or the grasp changes, it is not silently transferred.

The failed attempt has **no lateral retry**. The hand was fully opened after
the 20 mm stroke and Franka/Inspire already moved to clear view before the
final task label. SAM/FoundPose now captures the released key. If its pose is
misplaced, a new v8 pickup→held transfer→release/exit plan attempts to restore
the original pose within the **same physical attempt**, followed by another
clear-view recapture. This requires a robot-PC `execute_restoration` callback
and independent execution log. The current handler allows three such
placements before labeling reset failure; it does not claim a planned drop
landed correctly. The next attempt starts only after observed reset success.

The command transform is
`T_robot_hand = T_robot_socket · T_socket_key_centered · T_key_hand_command`.
The cumulative socket XY correction changes only the *commanded wrist*;
the modeled key stays on the socket centerline. The next corrected trial
plans the same v8 grasp and tabletop class. If that combination has no path,
the correction is not silently tested with another grasp. A changed tabletop
class at the next insertion attempt invalidates the previous correction;
an intermediate reorientation retains it. Each new trial rechecks the
whole hand against the socket at the intended 20 mm endpoint and replans
from its fresh tabletop key and measured robot joints. A candidate rejected
by that collision/path check is not executed merely because the VLM asked
for it. XY-only learning cannot compensate nonrepeatable slip, key tilt or
unmodeled yaw; those are deliberately outside this experiment.

Current implementation: `grounded_alignment.py`, `retry_session.py`,
`live_retry_handlers.py`, `session_runner.py`, `next_trial_xy.py`,
`original_pose_restoration.py`, and `live_recovery_handlers.py`. Legacy
same-held-attempt 1 mm voting/preflight files remain for historical replay
but are not assembled into `run_live.py`.
