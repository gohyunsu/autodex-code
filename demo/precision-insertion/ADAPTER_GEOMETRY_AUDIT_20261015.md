# FR3–Inspire mount geometry: 2026-10-15 code and capture audit

Scope: read-only inspection of the 2026-10-12 stock AutoDex v8 recordings and
the 2026-10-15 precision-insertion snapshot. No robot motion was used for this
audit. The original `/home/robot/AutoDex` and NAS assets were not edited.

## What the October 12 v8 run did

The saved manifest at
`/home/robot/shared_data/AutoDex/experiment/v8_demo_lg/inspire/pepsi/_pipeline_runs/20261012T013743_214544Z_4d352074/manifest.json`
records `run_pipeline.py --obj pepsi --arm franka --hand inspire --grasp_version
v8 --auto`. Its code path in `src/execution/run_auto.py`:

1. Loads the current camera calibration, distributed FoundPose result, and
   object-processing mesh. `save_arm_C2R` saves the latest hand-eye transform;
   this episode's `C2R_meta.json` identifies `fr3_link8` and session
   `20261012_060559`. This calibrates camera to arm flange, not flange to hand.
2. Builds `GraspPlanner(hand="fr3_inspire")` and `FrankaExecutor(hand_name="inspire")`.
   `GraspPlanner` reads `fr3_inspire.yml` and sets cuRobo's external asset path
   to `/home/robot/shared_data/AutoDex/content/assets`. The effective arm and
   hand URDF is therefore the **NAS** `fr3_inspire.urdf`.
3. `load_candidate` reads v8 Inspire `wrist_se3.npy`, `pregrasp_pose.npy`, and
   `grasp_pose.npy`. It forms the target hand-base pose as
   `T_robot_object @ T_object_hand`. This is exactly the transform supplied by
   precision-insertion's direct `planner_candidate_override` loader. The
   direct loader avoids a full NAS candidate-tree walk; it does not change
   the pose convention. The candidate-pool argument is `hand="inspire"`, while
   the robot model remains `fr3_inspire`.
4. Candidate world/self collision uses the **floating Inspire hand** model
   (`inspire_floating.yml`, `spheres/inspire.yml`) against the scene. The arm
   IK, joint trajectory, lift preflight, and arm/hand collision model use
   `fr3_inspire.yml`, the effective NAS URDF, and
   `spheres/fr3_inspire.yml`. A candidate that passes those model checks is
   executed with `FrankaExecutor.execute`: approach trajectory, pregrasp,
   grasp ramp, squeeze, then lift. The 2026-10-12 selected Pepsi candidate
   `shelf/11/n200_s123_46@variant04` passed 245/1648 model collision filters
   and 167 IK checks before approach/lift planning, according to its saved
   planner report. These counts prove model feasibility, not physical fit.

## Mount geometry actually represented

The effective NAS URDF (SHA-256
`62dbe03d03b2c9e8589584d3552167c5c92e1fda677c711d71bd33cc7065b134`)
has one fixed `fr3_link8 → base_link` joint:

```xml
<origin rpy="3.14159 0.00000 0.78540" xyz="0 0 0.04" />
```

The checkout URDF (SHA-256
`e5ecd9466883d15c5691895db3e5126dbd76844d02ff7972c8d8996c8492f5f3`)
uses `rpy="3.14159 0.00000 -2.35619"` with the same 40 mm axial offset.
Their hand orientation differs by 180 degrees about the flange axis. Neither
fixed joint has a lateral translation or a 90-degree bend. Both reference the
same generic Inspire hand meshes; neither has a separately named adapter link
or adapter mesh. The NAS collision-sphere file does include extra spheres on
`fr3_link7`; their existence does not change the hand-base kinematics.

`autodex/utils/robot_config.py` defines `FR3_INSPIRE_LINK_TO_WRIST` as
`fr3_link7 → base_link`: 147 mm along link7 Z (107 mm `fr3_joint8` + 40 mm
mount) and the **checkout** -135-degree orientation. `FrankaExecutor` uses
that constant for its live wrist reference. Thus the stock executor's wrist
reference and the effective NAS cuRobo model disagree by 180 degrees in
orientation. This discrepancy is already in stock AutoDex; reusing stock
classes does not remove it.

The current precision-insertion runtime also constructs the stock
`GraspPlanner(hand="fr3_inspire")` and the completion-checked subclass of the
stock `FrankaExecutor`. Its candidate loader supplies the same
`T_robot_object @ T_object_hand` pose to the unchanged `GraspPlanner.plan`.
Pickup calls the stock executor's `execute(..., skip_lift=True,
start_from_current=True)`; measured lift is separately planned and executed.
`AssetPaths.robot_urdf` points to the same NAS URDF as cuRobo. The adjusted
`plan(..., hand="inspire", candidate_override=...)` argument matches the
stock candidate-pool name; because the explicit candidate arrays bypass
`load_candidate`, this edit does not alter the arm model or pose mathematics.

## Physical evidence and conclusion

- Raw October 12 lift snapshot:
  `/home/robot/shared_data/AutoDex/experiment/v8_demo_lg/inspire/pepsi/20261012_104059_820384_attempt_0001/label_at_lift/raw/images/25322645.png`.
  The side-mounted white connector, hand, and Pepsi can are visible together.
- The inspected October 12 episodes also have `videos/exec/*.avi`, but these
  clips have 53–54 frames (about 1.8 seconds); the inspected footage does not
  include the whole approach/grasp. Their `videos/place/*.avi` decode as empty.
  The still lift capture is the useful physical evidence here.
- The October 15 current snapshot and saved-model comparison are in
  `/tmp/precision-current-overlay-20261015/adapter_compare_25322651.jpg`
  and sibling camera views. The NAS +45-degree hand projection visibly
  disagrees with the real hand. The checkout -135-degree projection is closer
  in some views, but also disagrees in others. A projection match has not
  been calibrated or certified.

**The current right-angle physical adapter is not represented by a measured
flange-to-hand transform in either URDF.** Hand-eye calibration at
`fr3_link8` cannot supply the missing mount transform. A stock `success`
record is its own task label; the still image alone does not establish a
measured mount transform or validate a 30 mm precision key and 20 mm socket.
Switching only the URDF yaw to -135 degrees would leave the lateral offset and
bend unknown, and would not be a justified physical correction.
