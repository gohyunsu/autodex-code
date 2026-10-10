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
