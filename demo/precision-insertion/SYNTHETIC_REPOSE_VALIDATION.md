# Synthetic v8 reset/repose path diagnostic

`diagnose_synthetic_repose.py` reuses the demo's real
`preflight_v8_repose_transition` and the unchanged AutoDex v8 pickup,
lift and Cartesian planners. It consumes a **staged, non-runtime** reset
candidate directory and an endpoint catalogue. The board rectangle, table,
key/socket placement and stock FR3/Inspire home state are hypothetical. It
does not connect to the robot, certify a measured session, promote a reset
seed, or authorize motion.

Run from the repository root with the AutoDex BODex environment. For the
current cylindrical `000→001` handoff and 1 mm radial-gap socket:

```bash
AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS=1 \
PYTHONPATH="$PWD/demo/precision-insertion/compat" \
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/diagnose_synthetic_repose.py \
  --shared-root /home/hyunsu/shared_data \
  --catalog /home/hyunsu/shared_data/AutoDex/precision_insertion/endpoint_catalogs/cylindrical/precision_socket_cylinder_gap_01mm/scan_20261011_nominal_0p2mm_r2.json \
  --reset-candidate-dir /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff/reset_12 \
  --mode cylinder --gap-mm 1 \
  --from-pose-stem 000 --to-pose-stem 001 --height-cm 12 \
  --table-z-m 0.04 --key-x-m 0.4 --key-y-m 0 \
  --socket-x-m 0.6 --socket-y-m 0 \
  --release-x-m 0.45 --release-y-m 0.08 \
  --board-x-min-m 0.2 --board-x-max-m 0.8 \
  --board-y-min-m -0.25 --board-y-max-m 0.25 \
  --max-reset-drift-mm 3 --max-reset-axis-tilt-deg 8 \
  --min-rest-socket-clearance-mm 10 \
  --min-board-edge-clearance-mm 10 \
  --max-seed-attempts 1 --planner-mode native-locked-experimental \
  --diagnose-collision-obstacle \
  --output-dir /tmp/precision-synthetic-repose-cylinder01-0to1-new
```

The 3 mm/8° reset-seed fidelity bounds and all path-audit tolerances in
this example are **illustrative, not commissioned**. Choose distinct output
paths for repeated runs. The script saves synthetic source inputs, file
hashes, planner logs, trial world and a fail-closed report. It returns exit
code 2 if no complete held reset path is available. Optional
`--retreat-goal-q-npy` (seven arm joints) plus
`--min-release-key-clearance-mm` requests the later open-hand/retract
preflight; without both, a held-path pass still means release is unplanned.

## 2026-10-11 observed offline result

The first diagnostic stopped before pickup because the reset path had not
installed the same source-guarded vendored-cuRobo compatibility adapter as
the insertion trial path. That demo-local omission is now fixed; stock
AutoDex/cuRobo files remain unchanged.

With the fix, staged reset seed `0_1/191` passed v8 pickup, held lift and
transfer, but its 12 cm release-height descent failed in cuRobo with
`jacobian_segment_robot_collision`. This happened for both hypothetical
release sites `(0.45, 0.08)` m and `(0.35, -0.12)` m on the same broad
synthetic board. Reports are
[`r2`](/tmp/precision-synthetic-repose-cylinder01-0to1-r2-20261011/report.json)
and [`r3`](/tmp/precision-synthetic-repose-cylinder01-0to1-r3-20261011/report.json).
The second location is not a systematic reachability search. Neither run
reached release/open-hand/retract or a physical landing observation.
The finished diagnostic was rerun at the first site in
[`r4`](/tmp/precision-synthetic-repose-cylinder01-0to1-r4-20261011/report.json):
it reproduced the same rejection, returned exit code 2, and saved hashes
for all files in the attempted reset seed alongside its synthetic inputs.
An instrumented [repeat](/tmp/precision-synthetic-repose-cylinder01-0to1-r5-20261011/report.json)
identified the failed descent as a **world-collision check**, not self
collision: `MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION` at Jacobian
waypoint 17 (requested wrist Z approximately 0.125 m in this hypothetical
scene). The status does not identify which obstacle or robot link was involved;
neither the table nor socket may be removed from the real preflight on this
basis. A measured scene and targeted collision-pair inspection are needed
before changing the release policy.

The optional obstacle-isolation probe captures the **same rejected 13-joint
state** from the failed stroke, then asks cuRobo to evaluate it with both
obstacles, only the table, only the socket and neither. Obstacle masks are
restored after each query; the actual reset preflight is never rerun with an
obstacle removed. In the [r7 diagnostic](/tmp/precision-synthetic-repose-cylinder01-0to1-r7-isolation-20261011/collision_isolation.json),
the joint state was infeasible with both obstacles and with the table alone,
but feasible with the socket alone and with neither. Thus **the synthetic
table obstacle is necessary for this particular collision verdict**; the
socket is not. This does not identify the contacting robot link, validate
the hypothetical table geometry against the measured ChArUco board, or
make a higher drop safe. The failed `r6` experiment attempted to rebuild
the cuRobo world with removed obstacles and hit a vendored world-update
error; it is not collision evidence. The working probe toggles the existing
obstacle masks and still leaves this reset candidate rejected.

Therefore seed `191` is **not a validated reset path**, despite its prior
MuJoCo grasp-stability result. The staged seed stays outside the canonical
runtime reset pool. Next validation needs measured ChArUco bounds, a
commissioned release location and a successful fixture-aware pickup → held
lift → transfer → descent → release/retreat preflight, followed by observed
landing/reorientation. A synthetic planner pass, if found, would still not
establish physical reset success.
