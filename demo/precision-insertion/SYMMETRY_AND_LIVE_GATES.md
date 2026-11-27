# Cylinder symmetry and live gate audit

This note covers the board-9, white cylindrical key and fixed white 20 mm
socket route. It describes the active v8 path in this checkout, not a claim
that a planned or visually labelled insertion physically succeeded.

## Object frames

The current `object_processing` key declares D∞ symmetry about local Z,
with local center `[0, 0, 0.04]` m. The `000` v8 tabletop frame has its
representative insertion end at local `z=0.08` m and its origin at the top
of an upright key. The `001` frame represents the key on its side. A
FoundPose end-for-end flip is the **same physical key**, but its CAD origin
moves by 80 mm. The socket declares C∞ symmetry about local Z: axial yaw
is unobservable, while exchanging its open rim and closed base is invalid.

| Stage | Input | Symmetry use | Output |
| --- | --- | --- | --- |
| Socket freeze, `calibration.py` | Per-capture refined socket poses | Compares directed C∞ axes, ignoring axial yaw; one capture is permitted | Frozen socket pose and collision fixture |
| Key multiview, `key_perception.py` | SAM masks, raw FoundPose pose/class evidence, calibrated cameras, metric key mesh and frozen support plane | Selects end-down or side-down geometry under D∞, makes the full mesh touch the plane, and fits support-plane XY to all admitted masks; axial yaw and identical ends do not count as disagreement | Support-constrained observed key pose plus retained raw/fitted diagnostics |
| Tabletop class, `pose_selection.py` | Fresh observed key pose | Matches local-Z D∞ axis to v8 `000` or `001` | Candidate catalogue stem |
| Trial scene, `symmetry.py` and `world.py` | Observed pose, v8 tabletop frame, D∞ center | Rotates an exchanged-end pose **about the key center** into the candidate's end convention, preserving the mesh center | v8 key target pose for pickup planning |
| Held relation, `held_relation.py` | Post-lift key pose, measured wrist, selected v8 grasp | Searches axial-yaw and end-exchange branches around the physical center | Equivalent key frame and observed key-to-hand relation |
| Retry/restoration, `grasp_fidelity.py` and `original_pose_restoration.py` | Observed/simulated key and target tabletop pose | Measures cylinder center and unoriented axis; restoration offsets the target origin from the physical center | Pose-drift diagnostics and original-center placement target |
| Socket insertion, `targets.py` | Frozen socket, task geometry, held relation | Uses the socket's directed insertion axis; the C∞ socket is never flipped | Preinsert and 20 mm axial wrist targets |

The post-shift tip/axis fit in `postshift_pose.py` uses a **directed** axis:
the observed tip is the end aimed at the socket. It keeps the unobservable
axial-yaw gauge from the measured wrist and prior grasp. An antiparallel
directed tip/axis contradicts that observation; it is not silently treated
as an end exchange after a key end has been assigned for insertion.

For a tabletop cylinder, a single RGB silhouette does not independently fix
metric depth along its camera ray. `key_perception.py` therefore never sends
the raw one-view FoundPose translation directly to pickup planning. FoundPose
still supplies object identity and raw axis evidence. `board_socket_fit.py`
uses the known cylinder mesh and the session support plane to recover exact
support-normal placement, then estimates the two tangential coordinates from
the calibrated SAM-mask projections. The identical corrected
`KeyPoseObservation.pose_world` is consumed by both the trial scene/planner
and MV-GoTrack initialization; there is no parallel pose convention.

## Runtime checks

Stock `autodex/executor/real.py` reads the live arm/hand state and replans a
lift if the saved start no longer matches. It does not impose a second
one-second deadline while evidence files are written after squeeze. This
demo now reads fresh feedback for the measured lift and plans from that
state. Its nominal route plans only the lift before motion, then the
transfer/axial path after the lift observation.

The active route no longer stops solely because:

- a saved post-squeeze sample aged during log writing, or the measured
  fingers/arm settled before a new plan from current joints;
- a post-lift sample aged during catalogue/report verification, or the hand
  differed from the preceding lift snapshot;
- an axial handoff took longer than an arbitrary wall-clock budget while
  the live arm still matches the planned start;
- an object-blocked Inspire finger differed from the planned hand columns
  at the transfer or insertion boundary;
- a coarse preinsert VLM match had an extra wrist-target threshold. The
  wrist error remains recorded for diagnosis; the insertion boundary
  still checks the arm against the axial plan.

The robot-state reader still rejects absent or stale FR3 packets and
inconsistent FR3/Inspire feedback. Motion boundaries still require the
selected saved trajectory, matching **arm** start, controller completion,
and measured final arm position. The v8 planner and path audit retain
collision, joint and target checks. The stock force-stop insertion retains
its force stop and reverse-axis ascent. Camera labels do not certify 20 mm
physical penetration.

The D∞ tests cover both v8 tabletop stems, several axial yaws and both end
frames; they check that the transformed physical center and candidate axis
are unchanged. The complete demo suite is run with this checkout on
`PYTHONPATH` so subprocess CLI tests import the same AutoDex package.

## 2026-11-27 ZeroDex streamed endpoint convergence

The first physical Robotiq rerun reached the planned squeeze and reported a
real grasp, but its measured-state lift replan stopped at
`jacobian_start_robot_collision`. Robot-command-free replay with
`diagnose_saved_lift.py` isolated that collision to the table: removing the
socket did not change the result, removing the table passed, and substituting
the planned arm state while retaining the measured jaw passed. The arm was
about 0.019 rad L2 from the planned squeeze endpoint even though the stock
0.04-rad follower tolerance considered it complete.

Interpolating between the planned and measured arm states placed the
collision boundary near a 0.0038-rad endpoint residual. The ZeroDex factory
therefore sets its process-local streamed follower tolerance to 0.002 rad.
The follower keeps sending the final reference with zero feed-forward until
that residual is reached. It does not use the daemon's blocking landing
motion, add a new grasp/insertion acceptance gate, or change the AutoDex and
Inspire defaults. The physical rerun with this tighter endpoint convergence
is still required before claiming that the measured lift is resolved on the
robot.
