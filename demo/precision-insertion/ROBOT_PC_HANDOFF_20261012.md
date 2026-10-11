# AutoDex-PC handoff: offline assets and NAS (2026-10-12)

This is a file-backed handoff, **not** evidence of camera accuracy, a complete
Franka trajectory, contact-control commissioning, or physical insertion. The
live demo currently selects only the cylinder mode. Square remains an offline
planning mode. Keep this checkout separate from the stock AutoDex checkout.

## Local inventory

Reproduce the all-mode audit on a machine where `shared_data` is installed:

```bash
python3 demo/precision-insertion/audit_offline_asset_profile.py \
  --shared-root /absolute/shared_data \
  --output /absolute/new_asset_profile.json
```

On the producer PC, all ten configured mode pairs had key/socket meshes,
exact socket collision meshes, task geometry, FR3/Inspire URDF, and a
FoundPose representation file. All ten endpoint catalogs passed their
source-bound audit. Counts of nominal 20 mm endpoint-eligible grasps were:

| Family | Radial gap (mm) | Eligible | Covered tabletop poses |
| --- | ---: | ---: | --- |
| Square | 0.3 | 31 | 000–004 |
| Square | 0.5 | 24 | 001–004; **000 has none** |
| Square | 1.0 | 24 | 000–004 |
| Square | 1.5 | 41 | 000–004 |
| Cylinder | 1 | 109 | 000, 001 |
| Cylinder | 3 | 109 | 000, 001 |
| Cylinder | 5 | 112 | 000, 001 |
| Cylinder | 10 | 112 | 000, 001 |
| Cylinder | 15 | 112 | 000, 001 |
| Cylinder | 20 | 110 | 000, 001 |

The current cylinder candidates are a union of 12 historical and 112 new
full-length proposal IDs. The seven newly generated FoundPose representations
have only generation/reload and exact-source-mesh binding checks; do not
silently replace those with the older `pending_foundpose` candidates. Neither
file availability nor these counts establishes accurate pose estimation or
successful physical insertion. For reset/reorient, the two directed cylinder
v8 seeds exist, but full measured-scene motion and release have not been
qualified. Square directed reset coverage is incomplete and remains staged.

## NAS bundles

Use the following distinct bundles, preserving their manifests. Never merge
an `.incomplete` or `partial_directory_copy_interrupted` directory into an
installed tree.

| Bundle under `/mnt/paradex2/hyunsu/` | Contents | Recipient note |
| --- | --- | --- |
| `autodex_precision_insertion_handoff_20261010_652ac909` | Base object/fixture assets and older pending FoundPose candidates | Pending representations are **not** canonical. |
| `autodex_precision_insertion_runtime_addendum_20261010_652ac909` | SAM3/DINOv2 and optional perception code/weights | Install the relevant runtime dependencies on the recipient. |
| `autodex_precision_insertion_perception_eval_20261010_652ac909` | Captured-image evaluation/evidence | Not a live-calibration pass. |
| `autodex_precision_insertion_tabletop_v8_handoff_20261011_955e133` | Historical cylinder v8 candidates | Install before the append-only cylinder addendum. |
| `autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993` | 112 new cylinder v8 IDs | Verify, rehydrate, then rebuild all six endpoint catalogs on the recipient. |
| `autodex_precision_insertion_square_fullkey1000_handoff_20261011_10c455d` | Four square candidate pools | Offline only; rebind source paths and rebuild catalogs. |
| `autodex_precision_insertion_foundpose_generated_20261011` | Seven generated FoundPose objects in a tar and manifest | Extract to staging and install only exact-mesh-matched files. |
| `autodex_precision_insertion_foundpose_20261011` | Earlier generated FoundPose files, including cylinder 20 mm | Verify the intended archive before staged installation. |
| `autodex_precision_insertion_reorient_curated_20261011` | Curated reset seeds | Staging; do not infer physical repose success. |
| `autodex_precision_insertion_cylinder_reset1687_addendum_20261012` | Current selected cylinder `001→000/1687` seed, source scenes and audit | Rehydrate into its **own** staging root, then install; older reorient bundle has only `001→000/631`. |
| `autodex_precision_insertion_qwen3_vl_2b_20261012_89644892` | Dereferenced local Qwen3-VL-2B-Instruct snapshot | Set `vlm.model_id` to `payload/model` after verifying the manifest. |

The Qwen bundle is produced and checked with:

```bash
python3 demo/precision-insertion/export_local_qwen_handoff.py \
  --verify-root /mnt/paradex2/hyunsu/autodex_precision_insertion_qwen3_vl_2b_20261012_89644892
```

Its `payload/model` directory can be passed directly as `vlm.model_id` in
the robot-PC JSON config. The export contains no API key or ZeroDex source.
The model is available from the [official Qwen model page](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

## Recipient order

1. Clone `gohyunsu/autodex-code` branch `feat/precision-insertion` into a
   **new** checkout, not over the stock AutoDex repository; pin its commit.
2. Verify each bundle's manifest/hash **on the mounted NAS**. Copy assets
   into a **staging** `shared_data` tree first. Keep existing robot-PC assets
   untouched until exact object IDs, meshes, and file destinations match.
3. Install the base object/fixture/perception/FoundPose assets, historical
   cylinder candidate pool, then the full-length append-only 112-ID addendum.
   Use the bundle-specific rehydrate scripts and rebuild catalogs against
   the recipient's absolute `shared_data` root; producer catalogs under
   `audit_only` are not runtime inputs on a different root.
4. Run the all-mode file audit and `run_pipeline.py audit-reorient` without
   camera/robot. Then commission the actual camera calibration/perception
   and only later the FR3/Inspire and guarded-contact callbacks described in
   [LIVE_RUNTIME.md](LIVE_RUNTIME.md).

For the current reverse reset seed, verify the separate bundle with
`export_current_cylinder_reset_handoff.py --verify-root BUNDLE`. Then use
`rehydrate_reset_handoff.py` with `--source-shared-root BUNDLE/payload/shared_data`,
`--source-origin-shared-root /home/hyunsu/shared_data`,
`--source-candidate-root BUNDLE/payload/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12`,
`--source-audit BUNDLE/audit_only/SOURCE_AUDIT.json`, and a **new** target
`SHARED/AutoDex/precision_insertion/cylindrical/reorient_handoff_seed11012/reset_12`.
Set `--target-shared-root SHARED`. Finally call
`install_cylinder_reset_seeds.py --gap-mm 20 --from-pose-stem 1 --to-pose-stem 0
--seed-id 1687` with that target staging root as `--source-reset-root` and a
new `--output-manifest`. The older forward `000→001/191` seed can be installed
from its separate curated handoff. Do not substitute the old reverse `631`
for `1687` without deliberately changing the selected reset policy.

Do not treat a successful file audit as a live dry run. In particular,
single-scene Franka planning, socket pose repeatability, key masks, actual
grasp-to-hand relation, force/contact behavior and physical labels remain
unproven. Square 0.5 mm tabletop pose `000` has no eligible offline grasp in
the current finite scan.

## GitHub connectivity check on the producer host

```bash
getent ahostsv4 github.com
git -C /home/hyunsu/autodex-code ls-remote --exit-code origin refs/heads/feat/precision-insertion
gh auth status
```

An isolated sandbox may block DNS even when the host shell resolves GitHub.
Run these in the local terminal; a successful `ls-remote` proves GitHub
fetch connectivity but not that every current local commit has been pushed.
