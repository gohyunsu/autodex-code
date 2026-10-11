# Unrestricted full-length cylindrical grasp proposals

The old cylindrical proposal mesh covered only the first 25 mm of the key.
That was an extra search-space prior, not an AutoDex v8 or 20 mm insertion
requirement. For new proposal runs, use
`precision_key_cylinder_r15_h80_full_length_proposal`: a separate 64-sided,
radius-15 mm, height-80 mm proxy with **no axial contact-region restriction**.
The 64-sided approximation is needed because BODex's contact convex-hull
seeder failed on the 256-sided exact cylinder. Its maximum radial sagitta is
0.0181 mm. The exact full physical key—not the proxy—is used for the stock
scene collision, squeeze-contact and MuJoCo checks, and later for the exact
20 mm key/socket and whole-hand/socket screen. A proposal on the lower or
inserted part of the key can therefore be generated but may fail that 20 mm
screen. We do not silently reinstate a grip-region policy.

The old 25 mm proxy and all historical raw/candidate files remain untouched.
The new asset manifest is
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/full_length_proposal_asset_manifest.json`.
The proposal object lives under
`/home/hyunsu/shared_data/object_processing/precision_key_cylinder_r15_h80_full_length_proposal/`.
Its AutoDex table and `reorient_12` scenes live under the corresponding
`shared_data/AutoDex/scene/inspire/` object ID. The scenes copy the exact
full-key poses and obstacle layouts and only change the BODex target mesh.

From the repository root, on a machine with the AutoDex BODex environment
and GPU, create the proxy and a **new** raw proposal run:

```bash
PYTHONPATH=demo/precision-insertion:src \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/build_full_length_cylinder_proposal.py \
  --shared-root /home/hyunsu/shared_data

PYTHONPATH=demo/precision-insertion/compat \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 2 \
  --obj_list_file demo/precision-insertion/configs/cylinder_full_length_proposal.txt \
  --obj_root_dir /home/hyunsu/shared_data/object_processing \
  --scene_filter_file demo/precision-insertion/configs/cylinder_tabletop_scenes.json \
  --scene_type table --seed_num 1000 --seed 123 \
  --exp_name NEW_full_length_1000 \
  --output_dir /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_full_length_1000

PYTHONPATH=demo/precision-insertion:src \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/stage_cylinder_tabletop_proposals.py \
  --raw-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_full_length_1000 \
  --stage-root /home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/NEW_full_key_filter_input \
  --shared-root /home/hyunsu/shared_data --expected-per-scene 1000
```

The same full-length object is the new default for
`stage_cylinder_reorient_proposals.py`. Every output directory is exclusive;
use a new run name instead of overwriting prior evidence. Continue with the
unchanged `run_sim_filter()` on the *physical key object* and then
`screen_cylinder_tabletop_batch.py --minimum-hand-clearance-m 0`. The
screen accepts `--workers 6` to process distinct socket gaps in parallel.

## 2026-10-11 local result

The new full-length BODex run generated 2,000 raw candidates; native BODex
`success` was 0/2,000 but is not the stock v8 filter gate. The unchanged
full-key filter produced:

| Tabletop pose | Raw | Scene clear | Squeeze contact | MuJoCo stable |
| --- | ---: | ---: | ---: | ---: |
| `000` end-down | 1,000 | 912 | 700 | 115 |
| `001` side-down | 1,000 | 912 | 499 | 8 |
| Total | 2,000 | 1,824 | 1,199 | 123 |

The saved BODex target-contact coordinates confirm that this is not just a
renamed 25 mm proxy: **1,697/2,000** raw proposals contain at least one
target point at local `z > 25 mm`; among the 123 stock-MuJoCo passes this is
**115/115** in pose `000` and **7/8** in pose `001`. Target contact is a BODex
proposal field, not proof that a physical finger actually maintained that
contact during squeeze. Many upper-key contacts may be rejected by the
separate socket endpoint screen.

The centered, axis-aligned 20 mm endpoint scan with **zero extra clearance
margin** passed 97/123, 97/123, 100/123, 100/123, 100/123 and 98/123
MuJoCo-stable candidates for radial gaps 1, 3, 5, 10, 15 and 20 mm,
respectively. The union across the six independently screened gaps contains
112 distinct candidates. This still assumes the *nominal* BODex key-in-hand
transform and does not test a continuous Franka path or physical insertion.

The raw source root is
`/home/hyunsu/shared_data/AutoDex/bodex_raw/inspire/precision_insertion_cylinder_full_length_1000_20261011`;
the full-key simulation input is the sibling
`precision_insertion_cylinder_full_length_fullkey_eval_1000_20261011`;
stock passes are under
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_full_length_sim_pass_1000_20261011`.
The endpoint reports are under
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_full_length_endpoint_screen_1000_20261011`.
The 112 source-verified candidates were appended, **without overwriting old
IDs**, at ID offset 20,000 under the existing v8 physical-key candidate tree.
The [promotion manifest](/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/full_length_1000_promotion_20261011.json)
records each original seed, installed ID and independently eligible socket
gaps. Regenerate a complete per-gap endpoint catalogue after installing;
older catalogue files become stale when candidate IDs change. All results
remain **offline-only**, not robot or insertion successes.

From the repository root, repeat the source-verified non-overwriting promotion
on another shared-data checkout after producing the stage, stock and endpoint
roots above:

```bash
PYTHONPATH=demo/precision-insertion:src \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/promote_full_length_cylinder_candidates.py \
  --shared-root /home/hyunsu/shared_data \
  --screen-root /home/hyunsu/shared_data/AutoDex/precision_insertion/cylinder_full_length_endpoint_screen_1000_20261011 \
  --manifest /home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/NEW_full_length_promotion.json \
  --install
```

## AutoDex-PC addendum handoff

The new 112 installed candidate directories have a separate, append-only
handoff at
`/mnt/paradex2/hyunsu/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993`.
Its `MANIFEST.json` binds all payload bytes, source key meshes, both physical
key tabletop scenes and tabletop-pose files. The source-machine catalogues
and promotion manifest are retained under `audit_only/`; they are **not**
live catalogues on a PC with a different `shared_data` root. Existing key,
socket, fixture, URDF and FoundPose assets must be installed from prior
handoffs first. The proxy asset need not be installed to *use* already
filtered v8 candidates; regenerate it on that PC with
`build_full_length_cylinder_proposal.py` only if reproducing BODex generation.
The bundle's `eb38993` hash identifies the candidate-generation source;
pull the **latest** `feat/precision-insertion` branch for the export/import
utilities documented here.

From the recipient checkout, check the addendum and inspect the dry-run
install report before appending the new candidate IDs:

```bash
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/export_full_length_cylinder_handoff.py \
  --verify-root /mnt/paradex2/hyunsu/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993

PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/rehydrate_full_length_cylinder_handoff.py \
  --bundle-root /mnt/paradex2/hyunsu/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993 \
  --target-shared-root /ABSOLUTE/AUTODEX_PC/shared_data

# Only after the dry run reports the expected 112 new IDs and no conflicts:
PYTHONPATH=demo/precision-insertion:src python \
  demo/precision-insertion/rehydrate_full_length_cylinder_handoff.py \
  --bundle-root /mnt/paradex2/hyunsu/autodex_precision_insertion_cylinder_full_length_addendum_20261011_eb38993 \
  --target-shared-root /ABSOLUTE/AUTODEX_PC/shared_data --install
```

Then run `run_pipeline.py screen-catalog` once per actual cylinder gap using
the recipient's `--shared-root`, `--min-hand-clearance-mm 0` and a **new**
output path. The recipient's catalogue must show the complete candidate tree
and pass `preflight_endpoint_catalog_file`. If the 12 older IDs have not yet
been installed there, its new tree will contain 112 rather than 124 rows;
the import is valid but the legacy baseline remains absent. Do not copy the
source-root-bound `audit_only` catalogue into a live session.
