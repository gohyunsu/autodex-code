# Precision key/socket FoundPose onboarding

This is an **asset-generation** workflow, not a pose-accuracy or robot-readiness
test. It reuses the MV-GoTrack/FoundPose generator and the exact v8 metric
`raw_mesh/<object>.obj`; no stock AutoDex execution code is changed.

## Current workstation result (2026-10-11)

The following objects were generated with 57 viewpoints × 14 in-plane
rotations (798 templates), DINOv2-ViT-S/14-reg, 256 PCA components, and 2048
visual words. The generator saved each `repre.pth` and reloaded it successfully.

| Object | Purpose | Status |
| --- | --- | --- |
| `precision_key_1p5mm` | Square 1.5 mm baseline key | Generated, camera validation pending |
| `precision_socket_unified` | Fixed square socket | Generated, camera validation pending |
| `precision_key_cylinder_r15_h80` | Shared cylinder key | Generated, camera validation pending |
| `precision_socket_cylinder_gap_15mm` | Cylinder 15 mm radial-gap socket | Generated, camera validation pending |

All four completed the generator's own representation reload. Other gap
variants are **not** implied to be onboarded by these files.

For every object, the canonical runtime path is:

```
<shared-root>/AutoDex/foundpose_assets/<object>/object_repre/v1/<object>/1/repre.pth
```

The actual source mesh is in
`<shared-root>/object_processing/<object>/raw_mesh/<object>.obj`.
`AssetPaths.foundpose_repre` and stock `FoundPoseInitializer` both use this
directory layout. Do not copy another object's `repre.pth` under a new name.

## Provenance and geometry choices

The source snapshot is the NAS
`autodex_precision_insertion_runtime_addendum_20261010_652ac909/source_snapshot/MV-GoTrack`;
the `onboard_custom_mesh_for_foundpose.py` version is the NAS
`autodex_precision_insertion_handoff_20261010_652ac909/source/isolated_onboarding`
copy, which correctly converts AutoDex's solid OBJ/MTL color to vertex colors.
The DINOv2 weight was already cached locally from the NAS snapshot.

The reference calibration was
`shared_data/cam_param/20260921_213318/intrinsics.json`, camera `25322651`,
2048×1536 undistorted pixels. The generated PLY uses `mesh_scale=1000`
(metric OBJ metres → BOP millimetres). The template distance interval was
explicitly set to 600–1200 mm, one viewsphere at 900 mm. This is a working
choice based on the 2026-09-21 calibrated camera positions relative to the
board origin, **not** a measurement of the key/socket at the future fixture
location. The generator's default 1125–1875 mm for the 80 mm cylinder key was
rejected as potentially too far for the near views. The default SSAA 4 was
changed to SSAA 1 to make the full-resolution 798-view run tractable; assess
the resulting image quality on real captures before acceptance. An interrupted
SSAA-4/default-depth run was preserved under
`shared_data/AutoDex/archive/precision_foundpose_default_depth_ssaa4_interrupted_20261011`;
it has no `repre.pth` and must not be installed as an asset.

## Isolated environment

On this workstation, `~/.venvs/precision-foundpose` was created with
`--system-site-packages` from the `autodex_bodex` Python 3.10 environment.
Only the venv received `faiss-cpu==1.8.0`, `pyrender==0.1.45`,
`omegaconf==2.3.0`, `scikit-learn==1.5.0`, `kornia==0.7.2`,
`pyglet==2.0.15`, `pypng`, `pytz`, and `einops`. The production AutoDex
environment was not modified. GPU and EGL rendering were required.

To generate an additional object, first verify its exact CAD, active camera
calibration, reference distance, and that no completed/partial output already
exists. Then run the generator in the isolated environment. For example:

```bash
gotrack_root=/mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909/source_snapshot/MV-GoTrack
onboard_script=/mnt/paradex2/hyunsu/autodex_precision_insertion_handoff_20261010_652ac909/source/isolated_onboarding/onboard_custom_mesh_for_foundpose.py
shared_root=/home/hyunsu/shared_data
object_name=precision_key_1p0mm
env PYTHONPATH="$gotrack_root:$gotrack_root/external/bop_toolkit:$gotrack_root/external/dinov2" \
  PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless \
  /home/hyunsu/.venvs/precision-foundpose/bin/python "$onboard_script" \
  --mesh-path "$shared_root/object_processing/$object_name/raw_mesh/$object_name.obj" \
  --object-id 1 --dataset-name "$object_name" \
  --output-root "$shared_root/AutoDex/foundpose_assets/$object_name" \
  --reference-intrinsics-json "$shared_root/cam_param/20260921_213318/intrinsics.json" \
  --reference-camera-id 25322651 --reference-image-scale 1.0 \
  --mesh-scale 1000 --depth-min-mm 600 --depth-max-mm 1200 \
  --min-num-viewpoints 57 --num-inplane-rotations 14 \
  --ssaa-factor 1.0 --pca-components 256 --cluster-num 2048
```

Never use `--overwrite` on a previously accepted asset without archiving it
and recording the reason. A failed run can leave partial templates even
without `repre.pth`; preserve and separate those before retrying.

## Handoff format

The complete four-object tree was archived locally as
`/tmp/autodex_precision_foundpose_4objects_20261011.tar` (3,217,797,120
bytes, SHA-256
`3198df432a5beb7d3a3470b6021512045171d27d3db85fec50a444f330b67c96`).
The NAS handoff is
`/mnt/paradex2/hyunsu/autodex_precision_insertion_foundpose_20261011/autodex_precision_foundpose_4objects_20261011.tar`.
Its SHA-256 was read back from the NAS on 2026-10-11 and **matched** the
local digest above; a same-sized file alone would have been insufficient.
The archive has
`AutoDex/foundpose_assets/<object>/...` paths relative to a shared-data root
and includes the representation, templates, model PLY and `summary.json`.
It does **not** replace the matching `object_processing` raw meshes from the
earlier handoff. The summary's source workstation absolute paths are
provenance, not portable runtime paths.
An interrupted, incomplete directory-by-directory NAS transfer is retained
as `partial_directory_copy_interrupted/` in the same handoff directory.
Never install that partial tree.

Expected representation SHA-256 digests:

| Object | `repre.pth` SHA-256 |
| --- | --- |
| `precision_key_1p5mm` | `f01c27669fbb721c326002eb5e809fc75fc21bcc5081a6d7280de77c3b00d8b2` |
| `precision_socket_unified` | `f9ce9b733aec9853ada959a0ac69781db0435e03bcb5fd9813f89bc9c40302c2` |
| `precision_key_cylinder_r15_h80` | `e7a0cff2181a85aafe5b26c865b5d9752c9c18aeb8567b59e6a4f06537145597` |
| `precision_socket_cylinder_gap_15mm` | `aa5894bc7637b672388c1c7f073acd490c6335e37ed7f183ee90e3624b40861f` |

On the recipient, list and hash the tar before extracting it into the
**intended** shared-data root. Do not unpack over a previously validated
object without checking what would be replaced. The files are not a
substitute for the camera validation below.

## Acceptance still required

1. Verify the generated representation loads with the **runtime**
   `FoundPoseInitializer`, using its matching metric mesh and the session's
   actual camera configuration. The generator's own CPU reload only proves
   file-format integrity.
2. Capture real empty-board/socket and tabletop-key views from the intended
   AutoDex camera set. Check segmentation, pose residuals, inter-view
   consistency and repeated measurements against independent calibration.
3. For the rotationally symmetric cylinder, evaluate centre/axis rather than
   treating an arbitrary FoundPose yaw as a physical error. Test the observable
   end-up/side-down cases and partial occlusion.
4. Commission the acquisition timestamps and hand-eye/fixture calibration;
   a `repre.pth` alone cannot authorize a robot trial.

The demo's `audit_assets` checks file presence only. In particular, a green
asset audit does **not** certify pose accuracy, filtered grasps, a Franka
transfer/insert plan, contact safety, or physical insertion success.
