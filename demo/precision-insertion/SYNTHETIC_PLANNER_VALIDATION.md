# Synthetic v8 pickup-to-insertion planner check (2026-10-11)

This is an **offline diagnostic**, not a calibrated AutoDex session or a
robot-executable plan. It uses real square 1.5 mm v8 candidate/mesh files,
but invents a level 40 mm table, key center `(0.4, 0, tabletop pose 004) m`,
socket center `(0.6, 0, 0.04) m`, identity camera-to-robot transform, and the
stock FR3/Inspire initial joint state. Its path-audit tolerances are exploratory,
not commissioned safety limits. The exact source file hashes and hypothetical
poses are in each run's `synthetic_context.json`.

The default AutoDex Cartesian route passed pickup for a later candidate, but
its transfer trajectory moved held Inspire joints by **0.0372496
rad**. The demo's fixed-hand invariant rejected that path rather than
relabeling it as an insertion success. The preserved traceback is
`/tmp/precision-synthetic-square-pose004-all-r3-20261011/failure.json`.

A subsequent default-mode rerun now treats this specific fixed-hand path
failure as a **candidate rejection** and scans the remaining eligible grasps.
The seven-candidate synthetic report is
`/tmp/precision-synthetic-square-default-hand-drift-evidence-20261011/report.json`.
Four grasps failed AutoDex pickup preflight; the other three passed pickup but
their nominal transfer paths changed Inspire joints by 0.0354, 0.0206 and
0.0369 rad, respectively. Each attempt records its measured maximum drift.
None of the seven passed the complete planning chain in default mode. These
figures are from one hypothetical scene and do **not** establish that no
physical pose can work or that the experimental mode is hardware-safe.

The same scene with `AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS=1` and
`--planner-mode native-locked-experimental` produced:

| v8 candidate | Pickup preflight | Insertion preflight |
| --- | --- | --- |
| `table/4/104` | passed | sampled held path rejected |
| `table/4/35` | failed | not attempted |
| `table/4/5102` | passed | sampled planning pass |

The selected candidate's saved trajectory and report are under
`/tmp/precision-synthetic-square-native-lock-20261011/`. Its sampled audit
reported no key/hand collisions with the frozen table/socket world, and
maximum hand-joint drift from commanded hold was `3.6e-8 rad` over 855 lift,
94 transfer, and 621 axial samples. Ten Cartesian axial goals were planned
over the preinsert-to-20 mm motion. These are **sampled numerical** checks;
an independent FK calculation from the saved final 13-joint state and the
Franka/Inspire URDF gave **19.9979 mm nominal depth beyond socket entry**,
2.13 µm lateral deviation from the planned axis, and 2.98 µm translation
residual to the 20 mm hand goal. New preflight reports also persist those
rigid-model endpoint metrics rather than relying on a single pass/fail bit.
They assume the key remains fixed in the hand after squeeze. They do not
prove swept-volume safety between samples, squeeze stability,
rigid key retention, force/contact safety, camera/hand-eye accuracy, or a
physical insertion. The native locked-hand AutoDex route is explicitly
experimental and requires independent hardware-stack validation. No live
robot command should use these artifacts.

The unchanged original AutoDex planner and vendored cuRobo have two checkout
compatibility mismatches exposed by repeated candidate planning: a default
sample-count typo and a singleton `WorldConfig` passed to the batch IK world
updater. The demo installs process-local, source-guarded adapters for these
two exact versions. Unknown implementations fail closed. The third issue,
default-route finger motion, is **not** suppressed by a compatibility patch:
the path is rejected, or the explicitly selected native locked-hand mode
must itself pass the fixed-hand and sampled-world checks.

## Current-tree recheck

On 2026-10-11 the same hypothetical scene and seven-candidate cap were
re-run against the current local v8 catalogue. The [saved report](/tmp/precision-synthetic-square-pose004-20261011-recheck/report.json)
again selected `table/4/5102` after the same three attempted candidates.
The saved paths have 193 pickup, 855 held-lift, 107 transfer and 621 axial
samples. The transfer sample count differs from the older run (94); numerical
planning is not assumed bitwise deterministic. The new
`run_pipeline.py verify-saved-preflight` command independently confirmed the
scene/NPZ hashes, selected-candidate and sampled-audit array digests, stage
continuity, at-most-0.01494 rad held joint steps, and at-most-3.6e-8 rad held
hand drift. This is artifact integrity and a fixed-hand contract check, **not**
a repeated collision audit or a physical/robot-ready result.

## Cylinder 20 mm-gap path-audit consistency check

An additional hypothetical scene used the *current* complete 20 mm-gap
cylinder endpoint catalogue, tabletop pose `000`, key XY `(0.4, 0)` m,
socket XY `(0.6, 0)` m, a 40 mm table and the stock FR3/Inspire home state.
The native locked-hand v8 planner passed pickup for `table/0/194`, but the
first sampled-path audit rejected exactly **one** descent sample as a
key/socket collision. The diagnostic report preserved stage, sample 417,
pair and minimum surface distances under
`/tmp/precision-synthetic-cylinder20-pose000-native-audit-20261011/`.

The inconsistency exposed two offline checks applying different *solid
occupancy* methods to the same validated cylinder fixture: endpoint screening
used the geometry-checked analytic cylinder bore, whereas path auditing used
generic triangle-ray parity. The latter can report an ambiguous/occupied
vertex in this concave socket despite the surface being well separated.
The earlier rejected trajectory was not retained, so this comparison does
not prove bitwise identity of both planner runs or the exact ray that failed.
The sampled path, reset/repose path and release-footprint audits now reuse
the same `CylinderSocketOccupancy` as the endpoint screen; the exact socket
mesh remains in the Coal surface-intersection check, so this does not waive
wall or floor contact. Square sockets retain generic watertight-mesh
occupancy. The path audit also hashes the cylinder task geometry it used.

With this consistency fix, the otherwise unchanged synthetic plan selected
`table/0/194` and passed all sampled stages. The [saved report](/tmp/precision-synthetic-cylinder20-pose000-native-analytic-20261011/report.json)
records 226 pickup, 501 held-lift, 77 transfer and 621 axial samples;
no sampled audit failure; **19.86 mm** minimum sampled key/socket surface
distance; **20.0035 mm** rigid-model depth past entry; and **5.94 µm** lateral
FK residual. Independent `verify-saved-preflight` checked the saved scene,
trajectory hashes, stage continuity, candidate identity, and maximum held
hand drift of **4.77e-8 rad**. It did **not** rerun collision checking.

This is a useful cross-check of the nominal planning and collision contracts,
not evidence of real grasp retention, calibrated geometry, closed-loop contact
control, or physical insertion. The 20 mm radial gap is a very loose
bring-up condition; smaller gaps still need their own complete preflights
and physical calibration. No synthetic report is a motor permit.

## Cylinder 1 mm radial-gap nominal check

The same offline diagnostic was also run for the **1 mm radial-gap** socket,
using its complete endpoint catalogue, tabletop pose `000`, the same
hypothetical key/socket XY positions, 40 mm table and stock FR3/Inspire home
state. With the explicitly experimental native locked-hand planner, candidate
`table/0/194` passed pickup, held lift, transfer and sampled 20 mm axial
descent after an earlier candidate failed pickup. The [saved report](/tmp/precision-synthetic-cylinder01-pose000-native-20261011/report.json)
records 211 pickup, 501 lift, 82 transfer and 621 axial samples. Its sampled
audit found no collision and a **0.915 mm minimum key/socket surface
distance**. Rigid-model FK at the endpoint gave **20.0035 mm nominal depth**
past the socket entry and **6.12 µm lateral residual** from the socket axis.
`verify-saved-preflight` independently checked the saved trajectory hashes,
candidate, stage continuity and maximum held-hand drift of **4.77e-8 rad**;
it did not repeat the collision audit.

This is an idealized, nominal-geometry pass, **not** a 1 mm physical insertion
result. The 0.915 mm sampled clearance can be consumed by fixture/hand-eye
calibration, print tolerances, camera localization, squeeze-induced key pose
error and robot tracking. No measured uncertainty bound or contact-force
validation is available. The test also does not establish continuous
swept-volume safety between samples or commission the experimental planner.
