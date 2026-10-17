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
| Key multiview, `key_perception.py` | SAM/FoundPose per-view poses and stock refined pose | Compares physical centers and unoriented key axes; axial yaw and identical ends do not count as disagreement | Observed key pose and frame evidence |
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
