# Cylinder precision-insertion presentation visuals

## Passed full-robot offline trajectory set (recommended)

The newest set is under:

```text
~/shared_data/AutoDex/precision_insertion/presentation_assets/
  cylinder_passed_trial_20261012/
```

The identical media plus its five-file source-preflight record is mirrored at
`/mnt/paradex2/hyunsu/handoff/20261012_passed_visuals_870f7aa/`.

It uses all 41 original FR3/Inspire URDF visual geometries (479,710 robot
mesh faces), the exact cylinder key/socket meshes, and saved joint paths from
one current precision-insertion preflight.  All MP4 files are 1280x720 at
24 fps; every MP4 has a 960x540, 10 fps GIF derivative below 50 MB.

| Stage | MP4/GIF stem | Motion/evidence shown |
| --- | --- | --- |
| 0 | `00_socket_pose/00_socket_pose` | Static session socket-pose checkpoint; not a camera measurement. |
| 1 | `01_key_perception/01_key_perception` | Static tabletop-key checkpoint; not a camera measurement. |
| 2 | `02_pickup_grasp/02_pickup_grasp` | Saved AutoDex v8 pickup approach followed by the commanded pregrasp-to-grasp-to-hold hand transition. |
| 3 | `03_held_lift/03_held_lift` | Saved locked-hand lift path; key pose is derived from exact URDF FK and the fixed v8 `T_key_hand`. |
| 4 | `04_transfer/04_transfer` | Saved native Cartesian transfer path with the same fixed hand/key relation. |
| 5 | `05_preinsert_hold/05_preinsert_hold` | Static pre-insertion checkpoint at zero XY retry offset. |
| 6 | `06_axial_insertion/06_axial_insertion` | Saved axial path to the nominal 20 mm verification transform. |
| 7 | `07_success_judgement/07_success_judgement` | Static observation checkpoint; it does not claim that a VLM verdict was evaluated. |
| Full | `08_full_pipeline/cylinder_passed_full_pipeline_overview` | Clean full-arm sequence. |
| Full | `08_full_pipeline/cylinder_passed_full_pipeline_labeled` | The same sequence with concise phase labels for slides. |

Use `passed_trial_storyboard_16x9.png` for a one-slide overview.  `MANIFEST.json`
binds the stage trajectories to the source report and hashes; `gif_manifest.json`
records every rendered MP4/GIF's dimensions, duration, and size.

### What actually passed

Candidate `table/0/184` passed the current `plan_fresh_key_trial` offline
preflight in explicitly selected `native-locked-experimental` mode:

- stock v8 pickup preflight;
- locked-hand lift, transfer, and ten-waypoint axial planning;
- sampled held-key/whole-hand collision audit against the frozen world; and
- the nominal 20 mm endpoint, with 0.00552 mm translation error and
  0.000369 degree rotation error in the saved result.

The saved planner arrays contain 123 pickup, 694 lift, 90 transfer, and 621
axial samples.  The maximum hand-joint drift after grasp is
`4.77e-8 rad`.  The presentation exporter keeps path endpoints and only
subsamples intermediate frames for playback speed.  It derives every
post-grasp key pose as
`T_robot_key = T_robot_hand(q) @ inverse(T_key_hand)`, so the key cannot slide
or rotate independently of the hand in the animation.

This is an **offline preflight trajectory**, not real execution evidence.
The scene uses hypothetical socket/key poses, stock diagnostic home joints,
and exploratory planning limits.  It does not validate camera estimates,
controller tracking, the physical post-squeeze hand/key relation, contact
force dynamics, VLM accuracy, or real insertion success.  The ordinary
planner mode did not produce a full pass in the first 20 candidates (18 hand
drift rejections and two unreachable transfers); the passing set therefore
must not be described as a default live-runtime result.

### Rebuild the passed set

Generate a new source report in a new output directory.  The environment flag
and matching `--planner-mode` declaration are both required:

```bash
cd ~/autodex-code
AUTODEX_ENABLE_NATIVE_POSE_CONSTRAINTS=1 \
PYTHONPATH=demo/precision-insertion/compat:. \
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/diagnose_synthetic_full_chain.py \
  --shared-root ~/shared_data \
  --catalog ~/shared_data/AutoDex/precision_insertion/cylindrical/endpoint_catalog_full_length_collision_only_gap_20mm_20261011.json \
  --mode cylinder --gap-mm 20 --pose-stem 000 --table-z-m 0 \
  --key-x-m 0.45 --key-y-m 0.08 \
  --socket-x-m 0.45 --socket-y-m -0.10 \
  --max-candidates 20 --planner-mode native-locked-experimental \
  --output-dir ~/shared_data/AutoDex/precision_insertion/cylindrical/NEW_SOURCE_DIR
```

The presentation exporter refuses a failed, modified, default-mode, or
non-cylinder source and refuses to overwrite an existing output root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/prepare_passed_cylinder_trial_visuals.py \
  --shared-root ~/shared_data \
  --report-dir ~/shared_data/AutoDex/precision_insertion/cylindrical/NEW_SOURCE_DIR \
  --output-root ~/shared_data/AutoDex/precision_insertion/presentation_assets/NEW_RENDER_DIR
```

Convert each resulting trajectory with
`prepare_blender_actual_mesh_animation.py`, render it through
`render_blender_actual_mesh_animation.py`, and build GIF derivatives with:

```bash
python3 scripts/precision_insertion/render_gif_derivatives.py \
  --root ~/shared_data/AutoDex/precision_insertion/presentation_assets/NEW_RENDER_DIR \
  --fps 10 --width 960 --colors 128 --force
```

The synthetic diagnostic and native planner switch are deliberate offline
tools.  They do not authorize a robot command.

## Four additional full-chain grasp passes

Four more pose-000 cylinder grasps passed the same native-locked offline
pickup/lift/transfer/20 mm axial audit:

```text
~/shared_data/AutoDex/precision_insertion/presentation_assets/
  cylinder_passed_grasp_comparison_20261012/
```

The identical 124-file presentation tree plus the four five-file source
preflights is mirrored at
`/mnt/paradex2/hyunsu/handoff/20261012_passed_grasp_visuals_41df7cc/`.

The passed keys are `table/0/194`, `table/0/20071`, `table/0/20140`, and
`table/0/20141`.  `four_grasp_full_pipeline_grid.{mp4,gif}` synchronizes the
four full-arm sequences; `four_grasp_20mm_insertion_grid.{mp4,gif}` shows the
insertion strokes at the task-focused view; and
`four_grasp_20mm_endpoint_grid.png` compares the four terminal grasps.  Each
`grasp_ID/` directory also contains its individual full-pipeline and
close-insertion MP4/GIF.

Candidate `table/0/20096` was deliberately excluded after its endpoint-eligible
grasp failed the full sampled axial path audit.  Candidate `table/0/20097` was
excluded after pickup preflight failure.  Thus the comparison does not relabel
endpoint-only candidates as full-chain passes.  The top-level `MANIFEST.json`
hashes all four reports and output media and records the same synthetic/robot-
ready limitations as the first passed visualization.

For reproducible candidate-specific diagnostics, repeat
`--exclude-candidate table/SCENE_ID/GRASP_ID` on
`diagnose_synthetic_full_chain.py`.  Excluded keys are treated exactly as
already-attempted candidates by `plan_fresh_key_trial` and are written to the
synthetic context record.

## Earlier composed presentation set

The presentation set is generated under:

```text
~/shared_data/AutoDex/precision_insertion/presentation_assets/
  cylinder_pipeline_20261012/
```

All delivered stills and animations use a 16:9 frame. The Blender renders use
the exact cylindrical key/socket CAD meshes and the Inspire visual-link meshes
exported at a current AutoDex v8 grasp posture. The older rendered blue key
and red socket are visualization colors only. The physical key and every socket
are white; active SAM and VLM prompts use white objects and shape/location cues.

## Recommended slide assets

| Slide purpose | Asset | What it establishes |
| --- | --- | --- |
| Task family | `01_socket_family/cylinder_socket_family_compatibility.{mp4,gif}` | The same cylindrical key mates with the 1, 3, 5, 10, 15, and 20 mm radial-clearance sockets. |
| Offline endpoint | `02_success_20mm/cylinder_20mm_insertion_success.{mp4,gif}` | Candidate `table/0/194` maintains a rigid screened key/hand transform and clears the exact 20 mm-deep nominal endpoint for the current 20 mm-gap bring-up socket. |
| Pipeline overview | `03_full_pipeline/cylinder_full_pipeline.{mp4,gif}` | Socket pose, key perception, grasp, lift/transfer, multi-view XY alignment, guarded 20 mm insertion, and success judgement in execution order. |
| One-slide overview | `cylinder_pipeline_storyboard_16x9.png` | Nine-panel summary of the same presentation set. |
| Individual checkpoints | `03_full_pipeline/stage_00.png` through `stage_06.png` | A clean 16:9 still for each execution checkpoint. |

The 1/3/5/10/15/20 mm labels are **radial socket clearances**. The 20 mm in
the endpoint and pipeline assets is the **verification insertion depth**. Do
not conflate these quantities in captions.

## Evidence boundary

These files are presentation visualizations, not physical robot evidence. The
source candidate has current AutoDex v8/MuJoCo grasp evidence and passes a
fresh exact-CAD nominal 20 mm endpoint collision check. The animation does not
claim:

- a continuous collision-free Franka transfer plan;
- the physical post-squeeze key/hand transform;
- guarded-contact dynamics or force-limit qualification;
- VLM success-classification accuracy; or
- real-robot insertion success.

The exact provenance and exclusions are machine-readable in
`cylinder_pipeline_20261012/MANIFEST.json`.

## Rebuild

Prepare the audited animation bundles:

```bash
python demo/precision-insertion/prepare_cylinder_presentation_bundles.py \
  --shared-root ~/shared_data \
  --catalog \
  ~/shared_data/AutoDex/precision_insertion/cylindrical/endpoint_catalog_full_length_collision_only_gap_20mm_20261011.json \
  --output-root \
  ~/shared_data/AutoDex/precision_insertion/presentation_assets/cylinder_pipeline_20261012
```

The builder deliberately refuses to overwrite an existing output root. Use a
new dated directory for a new presentation revision. Render the family clip
with `render_blender_cylinder_compatibility.py`; render the two bundle clips
with `scripts/precision_insertion/render_blender_actual_mesh_animation.py`.
Both Blender entry points expose their complete command-line usage with
`--help` after Blender's `--` separator.
