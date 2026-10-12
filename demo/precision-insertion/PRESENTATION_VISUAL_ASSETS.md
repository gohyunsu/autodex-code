# Cylinder precision-insertion presentation visuals

The presentation set is generated under:

```text
~/shared_data/AutoDex/precision_insertion/presentation_assets/
  cylinder_pipeline_20261012/
```

All delivered stills and animations use a 16:9 frame. The Blender renders use
the exact cylindrical key/socket CAD meshes and the Inspire visual-link meshes
exported at a current AutoDex v8 grasp posture. Blue denotes the key and red
denotes the socket.

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
