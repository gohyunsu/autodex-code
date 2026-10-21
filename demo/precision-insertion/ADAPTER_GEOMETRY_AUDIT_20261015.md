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
`FrankaExecutor` accepts the same URDF's composed `fr3_link7 → base_link`
fixed transform for live wrist references. `AssetPaths.robot_urdf` supplies
that same file to hand-mesh collision and camera overlays. All three consume
one model; no NAS asset or original AutoDex source file is rewritten.

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
camera clearance. This changes the target shared by initial
`home(clear_view=True)`, release retracts, and endpoint checks.

The measured stationary joints at the first correction and the saved
board-9/20-mm scene from run `20261021_092625_546425` were checked with the
same Oct-12 URDF and cuRobo model. A planner path to the replacement exists
(133 waypoints); all 201 samples of the direct joint interpolation pass
joint, self-collision, and world-collision constraints. A replacement-to-home
path also exists (146 waypoints). These are offline checks of the saved scene,
not a physical trial of the replacement. Startup still uses AutoDex's blocking
free-space `home` command. No further robot motion was commanded after the
operator's pause request.

Saved real camera frames `25322649` and `25305461` from run
`20261021_100220_060896` were used to render a read-only comparison. The
measured-current CAD aligns with the visible robot; the right-hand comparison
panels show only the proposed joints, not a second physical capture. The
source paths and SHA-256 hashes are recorded in
`/tmp/precision-clear-view-review-20261021/manifest.json` alongside the two
`*_hand_closeup.jpg` comparisons. `src/execution/franka_home.py --clear_view`
also reads this configured target, so the standalone parking command no
longer silently uses the previous base-only rotation.

Read-only verification:

```bash
cd /home/robot/autodex-precision-insertion
/home/robot/anaconda3/envs/planner/bin/python -m pytest \
  demo/precision-insertion/tests/test_mount_geometry.py \
  demo/precision-insertion/tests/test_franka_core.py -q
bash demo/precision-insertion/run_robot_pc.sh --check
```
