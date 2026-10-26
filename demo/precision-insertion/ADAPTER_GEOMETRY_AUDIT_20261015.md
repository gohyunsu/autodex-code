# FR3–Inspire mount recovered from October v8 executions

This audit used saved AutoDex v8 episodes and the October 15 camera snapshot.
No robot moved. `/home/robot/AutoDex` and the NAS assets were not edited.

## Source and asset path

The October 12 run manifest `20261012T014048_304553Z_fdb70403` records
`run_pipeline.py --obj pepsi --arm franka --hand inspire --grasp_version v8
--auto`. The inspected stock checkout is `448732d` (September 30); the manifest
does not record its source commit or a robot-model hash. These source paths
therefore describe the checked-out implementation, while the saved trajectory
and FK below establish the **model that actually ran**.

1. `run_pipeline.py` enters `run_auto.py`, which chooses
   `GraspPlanner(hand="fr3_inspire")` and `FrankaExecutor(hand_name="inspire")`.
   The planner's `HAND_CONFIGS` maps `fr3_inspire` to `fr3_inspire.yml` and
   `inspire_floating.yml`. `robot_configs_path` is
   `/home/robot/shared_data/AutoDex/content/configs/robot`; `project_dir` is
   `/home/robot/shared_data/AutoDex`. The planner normally sets cuRobo's
   `external_asset_path` to that NAS content tree.
2. `fr3_inspire.yml` names `robot/fr3_inspire_description/fr3_inspire.urdf`,
   `ee_link: base_link`, and `spheres/fr3_inspire.yml`. Arm IK and trajectory
   FK use this combined robot model. Candidate scene collision first uses the
   separate floating Inspire hand model and its spheres.
3. `load_candidate` reads the v8 Inspire grasp files from
   `/home/robot/shared_data/AutoDex/candidates/inspire/v8/<object>`, with the
   object mesh/tabletop from `/home/robot/shared_data/object_processing`.
   `wrist_se3.npy` is transformed to robot space as
   `T_robot_object @ T_object_hand`. The precision loader preserves that
   calculation and passes an explicit pool into the same `GraspPlanner.plan`.
4. The selected episode `20261012_104059_820384_attempt_0001` used
   `shelf/3/n200_s123_27@variant04`: 1,336 candidates, 199 collision-valid,
   132 IK-valid, two approach attempts, one successful lift preflight. The
   recorded arm/hand sequence is pregrasp, grasp/squeeze, and lift. The raw
   image `label_at_lift/raw/images/25322645.png` clearly shows the hand
   holding the Pepsi can off the board; `result.json` also says success.
   These establish a successful grasp, not just a successful planner label.

## The effective October 12 mount

`plan/lift_preflight.npz` stores the selected 13-joint configuration and the
planner's FK pose of `base_link`. For each of 15 October 12 episodes with that
artifact, I computed

`T_link8_hand = inverse(FK_URDF(fr3_link8, saved_arm_q)) @ saved_start_wrist_se3`.

All 15 recover the same fixed transform (maximum spread: 0.00033 mm and
0.000044°). Their mean is approximately:

```text
fr3_link8 → Inspire base_link
xyz = (-0.010000, -0.000004, +0.040000) m
rpy = (+1.57079642, -1.04719752, -1.57039831) rad
```

This is a right-angle hand orientation with a 10 mm lateral offset. The
current NAS `fr3_inspire.urdf` instead says `(0, 0, 40 mm)` and
`rpy=(π, 0, +45°)`; the checkout's generic URDF says `(0, 0, 40 mm)` and
`rpy=(π, 0, -135°)`. Neither reproduces the saved October FK. Direct cuRobo
replay of the pictured success episode with the current NAS file differs
from its saved FK by 10.000 mm and 169.425°. Replacing only the fixed joint
with the recovered values reproduces that FK within 0.000031 mm and
0.000003° on this machine.

The exact October 12 URDF file is not in the current NAS or checkout, so its
original filename and source commit remain unknown. The effective transform
is identified by the saved joint states and FK across episodes. The raw
success image independently supports a side-mounted hand. It does not measure
adapter surface CAD or submillimetre physical flange-to-hand error.
The selected URDF has the same 24 links, 23 joints and 29 mesh references as
the current NAS model; all 29 referenced local meshes have byte-identical NAS
counterparts. The stock `spheres/fr3_inspire.yml` adds five spheres on
`fr3_link7` near the flange, but has no separately named elbow-adapter mesh.

The old `FR3_INSPIRE_LINK_TO_WRIST` constant in `robot_config.py` remains the
generic -135° mount. It disagrees with both the current NAS model and the
October 12 effective model; the stock executor uses it for live wrist
references. This explains why copying only stock classes did not copy the
successful episode's geometry.

## Precision checkout application

`fr3_inspire_oct12_mount.urdf` in the separate checkout copies the existing
FR3/Inspire meshes and joint tree, changing only `flange_to_hand` to the
recovered October transform. `run_robot_pc.sh` selects it with
`PRECISION_FR3_INSPIRE_URDF` for `--check`, `--plan-only`, and `--run`.
`GraspPlanner` accepts that URDF as an optional robot-model path while keeping
its stock FR3 hand config, seeds, and other planning parameters. The
`FrankaExecutor` uses that URDF's `fr3_link8 → base_link` fixed transform for
live wrist references, because ParaDex exposes libfranka `O_T_EE` at
`fr3_link8`. `AssetPaths.robot_urdf` supplies
that same file to hand-mesh collision and camera overlays. All three consume
one model; no NAS asset or original AutoDex source file is rewritten.

### October 26 measured-EE frame correction

The earlier precision executor passed the composed `fr3_link7 → base_link`
transform to its measured `O_T_EE` path. That counted the fixed 107 mm
`fr3_joint8` segment twice. A read-only replay of 21 evenly spaced raw arm
samples from each of three October 12 Pepsi v8 episodes compared saved
`O_T_EE` with FK from the recovered mount URDF. Its maximum translation
residual to `fr3_link8` was 0.259 mm and maximum orientation residual was
0.000014°. Applying only `fr3_link8 → base_link` kept the hand-base residual
below 0.260 mm; applying the old composed transform put it about 107 mm away.
The episode paths, source hashes, sample indices, and per-episode maxima are
in `/home/robot/shared_data/hyunsu/tmp/20261026_fr3_otee_mount_frame_audit/report.json`.
The precision factory now passes `link8_to_hand` into the executor, and the
executor's generic FR3 default also removes the known 107 mm fixed segment.
The stock `/home/robot/AutoDex` checkout remains untouched. This corrects
live wrist feedback, endpoint comparisons and the wrist-distance fallback;
the cuRobo planning URDF and joint trajectories already used the recovered
mount. Saved data does not prove the current robot's EE configuration or
physical hand-eye calibration, so this is a source/frame correction with
historical state evidence rather than a new live verification.

A four-camera read-only overlay of the October 15 snapshot is under
`/tmp/precision-current-overlay-20261015/adapter_compare_inferred_*.jpg`.
The recovered model follows the photographed sideways hand much better than
the two current generic URDFs. It is still a visual check, not physical mount
metrology or a successful precision insertion. The endpoint catalog was
screened before this mount correction; live cuRobo planning uses the corrected
model, but catalog eligibility remains historical until it is regenerated.

## October 21 clear-view correction

The earlier `FR3_CLEAR_VIEW` rotated only the base joint from calibrated home.
With the October 12 right-angle mount, URDF FK placed the Inspire palm normal
at approximately `(0.739, 0.137, 0.660)` in robot coordinates, pointing up.
The first correction put the palm down but pointed the finger axis back toward
the robot (`-X`). The operator rejected that physical orientation. The live
run was interrupted during planning, before approach or grasp; the arm is
stationary at that first correction. The replacement keeps the same base
heading and almost the same hand-base position as the original clear view,
while aiming both the fingers and the link7-to-fingertip vector outward in
robot `+X`. Its `q2..q7` are `(0.157, -0.233, -1.574, 0.837, 1.351,
-0.521)` rad. Oct-12 URDF FK gives hand base `(0.427, -0.367, 0.533)` m,
palm normal `(0.010, -0.192, -0.981)`, finger direction
`(0.904, -0.417, 0.091)`, and link7-to-fingertip direction
`(0.842, 0.289, -0.456)`. The hand remains approximately where it was for
camera clearance. At that time this changed the target shared by initial
`home(clear_view=True)`, release retracts, and endpoint checks.

The measured stationary joints at the first correction and the saved
board-9/20-mm scene from run `20261021_092625_546425` were checked with the
same Oct-12 URDF and cuRobo model. A planner path to the replacement exists
(133 waypoints); all 201 samples of the direct joint interpolation pass
joint, self-collision, and world-collision constraints. A replacement-to-home
path also exists (146 waypoints). These are offline checks of the saved scene,
not a physical trial of the replacement. Startup still uses AutoDex's blocking
free-space `home` command. This paragraph describes the offline review before
the later pose-only move.

Saved real camera frames `25322649` and `25305461` from run
`20261021_100220_060896` were used to render a read-only comparison. The
measured-current CAD aligns with the visible robot; the right-hand comparison
panels show only the proposed joints, not a second physical capture. The
source paths and SHA-256 hashes are recorded in
`/tmp/precision-clear-view-review-20261021/manifest.json` alongside the two
`*_hand_closeup.jpg` comparisons. `src/execution/franka_home.py --clear_view`
also reads this configured target, so the standalone parking command no
longer silently uses the previous base-only rotation.

### Comparison with the stock AutoDex parking pose

The checked-out stock AutoDex source (`/home/robot/AutoDex`, September 30,
commit `448732d`) loads `FR3_INIT` from the Franka hand-eye calibration's
`home_qpos.npy` and constructs `FR3_CLEAR_VIEW` by changing **only joint 1**
by -40 degrees. `run_auto.py` calls `FrankaExecutor.home(clear_view=True)`
before perception; `FrankaExecutor.execute()` returns to `FR3_INIT` before a
grasp trajectory. The right-angle hand mount affects the planner's hand URDF
and wrist transforms, but this stock clear-view joint target does not solve
an adapter-aware hand orientation. The October 12 run's source commit and
clear-view joint telemetry were not saved. Its real `init_capture` images
show a compact parked arm with the sideways adapter; they do not establish
exact clear-view joint values. The episode's raw arm recording starts later,
at execution capture setup, so its first sample is not clear-view evidence.

With the calibration file currently installed on the robot PC, stock
`FR3_CLEAR_VIEW` would be approximately
`[-0.702, 0.023, 0.003, -2.329, -0.000, 3.958, -2.304]` rad. The new
precision target is `[-0.702, 0.157, -0.233, -1.574, 0.837, 1.351,
-0.521]` rad. In the October 21 pose-only physical move, the measured arm
reached the new target and the operator confirmed the palm direction, but
reported that the arm was too extended. Joint 4 is 0.755 rad (43.3 degrees)
less flexed than the stock target; this agrees with that observation. No
approach or grasp was commanded after this feedback. Simply reverting to stock
would also restore its upward palm direction with the recovered October 12
mount.

### First compact palm-down clear view, October 21

An offline search held joint 1 at the AutoDex clear-view heading, kept joint 4
near the stock calibrated elbow bend, and matched the operator-approved
palm-down/outward-finger orientation with the recovered mount. The first
compact candidate was `[-0.701714, -0.4070, -0.1478, -2.3300, 0.7928,
1.6279, -0.5650]` rad. The Oct-12 URDF FK places the hand base near
`(0.325, -0.228, 0.477)` m, palm normal near `(0.020, -0.201, -0.979)`,
and finger axis near `(0.875, -0.471, 0.115)` in robot coordinates. The hand
base is about 10 cm farther inboard and 6 cm lower than the previous target;
its wrist is therefore not held at the same Cartesian position merely to
rotate the palm. Joint 4 differs from the calibrated stock clear view by
approximately 0.001 rad.

On the saved board-9/20-mm key/socket world from run
`20261021_092625_546425`, cuRobo planned 79 waypoints from the measured
extended posture to the compact target and 121 waypoints from the compact
target to `FR3_INIT`. All 201 sampled states of the direct joint
interpolation used by the stock blocking `home()` command passed cuRobo's
joint/self/world collision rollout in that saved scene. These are discrete
checks on saved geometry, not a swept-volume or live-obstacle guarantee.
Synthetic CAD projections on preserved real frames are at
`/tmp/precision-compact-clear-view-review-20261021/`; their manifest names
the source images, calibration and URDF. The operator selected the candidate
and confirmed the board/socket/key arrangement and on-site monitoring before
the pose-only move. The measured arm reached
`[-0.70172, -0.40018, -0.14875, -2.32028, 0.79325, 1.62481, -0.56459]`
rad, at most 0.0098 rad from target, and later feedback showed at most
0.0025 rad/s joint velocity. The executor was closed after that move. The
operator confirmed the palm direction but found the overall arm shape still
too extended, then prioritized a clear view that feeds later planning smoothly
and stays away from singularity. No approach, grasp or insertion was commanded.
The full precision-insertion test suite passed: 1066 passed, 4 skipped;
`run_robot_pc.sh --check` passed its static file check.

### Planning-conditioned clear view, October 21

The live precision pipeline passes the measured clear-view joint state to
`plan_admitted_key_trial`, and the stock Franka executor receives the resulting
pickup with `start_from_current=True`. Unlike stock `run_auto.py`, it does not
return to `FR3_INIT` before the pickup. Therefore the relevant comparison is
from clear view to each saved v8 pregrasp, not only from clear view to home.

Using the Oct-12 mount URDF and the saved real scenes from runs
`20261020_100938_490865`, `20261021_090945_053190`, and
`20261021_092625_546425`, a read-only cuRobo replay tested the stock clear
view, the physically reached compact candidate, and two nearby palm-down
targets. Each saved scene and selected pregrasp endpoint was held fixed while
the clear-view start changed. All candidate starts and all three selected
pregrasp paths planned successfully. The chosen target is
`[-0.701714, -0.3248, -0.1545, -2.2500, 0.7929, 1.6115, -0.5782]` rad.
It keeps the palm normal about `(0.018, -0.201, -0.980)`, the finger axis
about `(0.880, -0.462, 0.111)`, and the hand base about
`(0.340, -0.247, 0.478)` m in the robot base frame. Its hand is only about
26 mm from the physically reached candidate. Synthetic CAD overlays on two
saved camera frames show the key and socket unobstructed:
`/tmp/precision-planning-clear-view-review-20261021/` (synthetic projections,
not live verification).

| Saved scene | Stock clear to pregrasp (arm joint path L2) | First compact | Chosen mid |
| --- | ---: | ---: | ---: |
| Oct 20 10:09 | 4.33 rad | 4.35 rad | 4.26 rad |
| Oct 21 09:09 | 4.96 rad | 3.94 rad | 3.84 rad |
| Oct 21 09:26 | 3.73 rad | 1.19 rad | 1.13 rad |

cuRobo can select different paths on repeat solves, so those single-run path
lengths are not deterministic guarantees. Five extra replays of the Oct 20
scene gave the first compact target one long 5.40 rad solution with a sampled
minimum wrist Jacobian singular value of 0.017, followed by four 3.56–3.57
rad solutions near 0.052. The chosen target gave five 3.49–3.51 rad solutions,
all near 0.052. The Jacobian metric scales rotational rows by 0.10 m/rad,
matching the repository's `JacobianStrokeOptions`; it is a conditioning
comparison, not a probability of singularity or proof of motor smoothness.
In the latest Oct 21 scene, the chosen target's sampled minimum was 0.0723,
with a minimum cuRobo joint-limit margin of 0.80 rad; every saved approach
waypoint passed the cuRobo world and self-collision check. A sampled direct
joint interpolation from the measured parked pose to the chosen target also
passed. With the operator confirming the empty hand, fixed board/socket/key,
clear surroundings, and monitoring, a pose-only `home(clear_view=True)` moved
to the chosen target. Measured FR3 joints after the move were
`[-0.701719, -0.334555, -0.153736, -2.259167, 0.792979, 1.613225,
-0.576440]` rad (maximum target error 0.00976 rad, maximum joint velocity
0.00284 rad/s). The executor was closed. No pickup, insertion, or other task
motion was commanded in this pose-only check. The operator then confirmed
that the arm and hand direction and key/socket camera view were satisfactory,
with no contact or change in the board/socket/key arrangement.

After the target change, `python -m pytest demo/precision-insertion/tests -q`
in the planner environment passed (1066 passed, 4 skipped, 31.16 s), the
static `run_robot_pc.sh --check` passed, and `git diff --check` was clean.

Read-only verification:

```bash
cd /home/robot/autodex-precision-insertion
/home/robot/anaconda3/envs/planner/bin/python -m pytest \
  demo/precision-insertion/tests/test_mount_geometry.py \
  demo/precision-insertion/tests/test_franka_core.py -q
bash demo/precision-insertion/run_robot_pc.sh --check
```
