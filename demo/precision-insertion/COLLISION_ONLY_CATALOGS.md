# Collision-only endpoint catalogues (2026-10-11)

`minimum_hand_clearance_m=0` means **no added numerical clearance margin**.
It does not disable key/socket or whole-hand/socket collision detection at
the centered, axially aligned 20 mm pose. The unchanged v8 source/MuJoCo
grasp evidence remains required. This is a nominal CAD baseline, not a
physical uncertainty or insertion-success certificate. Positive margins
remain selectable for separately justified experiments.

All paths below are under `/home/hyunsu/shared_data/AutoDex/precision_insertion/`.
The first table was complete **before** the new 112 full-length cylinder candidates
were appended. Their six cylinder files are now preserved *historical*
12-row scans and will deliberately fail current-pool provenance checks;
do not use them to launch a session. The square files are unaffected.
Use the new full-length cylinder files in the second table. Rebuild on another
PC using its local `shared_data` root.

| Mode / radial gap | Relative catalogue path | Full-pool rows | Eligible |
| --- | --- | ---: | ---: |
| Cylinder 1 mm | `cylindrical/endpoint_catalog_collision_only_gap_1mm_20261011.json` | 12 | 12 |
| Cylinder 3 mm | `cylindrical/endpoint_catalog_collision_only_gap_3mm_20261011.json` | 12 | 12 |
| Cylinder 5 mm | `cylindrical/endpoint_catalog_collision_only_gap_5mm_20261011.json` | 12 | 12 |
| Cylinder 10 mm | `cylindrical/endpoint_catalog_collision_only_gap_10mm_20261011.json` | 12 | 12 |
| Cylinder 15 mm | `cylindrical/endpoint_catalog_collision_only_gap_15mm_20261011.json` | 12 | 12 |
| Cylinder 20 mm | `cylindrical/endpoint_catalog_collision_only_gap_20mm_20261011.json` | 12 | 12 |
| Square 0.3 mm | `endpoint_catalogs/square_0p3mm_20261011_collision_only.json` | 32 | 31 |
| Square 0.5 mm | `endpoint_catalogs/square_0p5mm_20261011_collision_only.json` | 25 | 24 |
| Square 1.0 mm | `endpoint_catalogs/square_1p0mm_20261011_collision_only.json` | 25 | 24 |
| Square 1.5 mm | `endpoint_catalogs/square_1p5mm_20261011_collision_only.json` | 42 | 41 |

The current source-bound cylinder pool has the 12 historical IDs plus 112
new full-length-proposal IDs. Each gap was rescanned independently with zero
extra margin; all six files have `complete_scan=true`, zero errors and passed
`preflight_endpoint_catalog_file` against the 124-directory current tree.

| Cylinder radial gap | Current catalogue path relative to this directory | Full-pool rows | Eligible |
| --- | --- | ---: | ---: |
| 1 mm | `cylindrical/endpoint_catalog_full_length_collision_only_gap_1mm_20261011.json` | 124 | 109 |
| 3 mm | `cylindrical/endpoint_catalog_full_length_collision_only_gap_3mm_20261011.json` | 124 | 109 |
| 5 mm | `cylindrical/endpoint_catalog_full_length_collision_only_gap_5mm_20261011.json` | 124 | 112 |
| 10 mm | `cylindrical/endpoint_catalog_full_length_collision_only_gap_10mm_20261011.json` | 124 | 112 |
| 15 mm | `cylindrical/endpoint_catalog_full_length_collision_only_gap_15mm_20261011.json` | 124 | 112 |
| 20 mm | `cylindrical/endpoint_catalog_full_length_collision_only_gap_20mm_20261011.json` | 124 | 110 |

These are counts of **nominal offline** v8 grasp and aligned 20 mm endpoint
passes, not 124 physical attempts, Franka path passes or insertion successes.

Each file was complete, error-free and reopened with
`preflight_endpoint_catalog_file` when its 12-candidate cylinder pool (or
unchanged square pool) was current. The unchanged eligible counts versus the
previous positive-margin scans meant the extra margin rejected no candidate
in those pools. They do **not** imply that the margin was calibrated or that
a physical grasp, path or contact insertion will succeed.

Example reproduction, from the repo root:

```bash
PYTHONPATH=demo/precision-insertion:src \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root /absolute/path/to/shared_data \
  --mode cylinder --gap-mm 20 --min-hand-clearance-mm 0 \
  --output /absolute/path/to/new_collision_only_catalog.json
```

The CLI refuses to overwrite existing catalogues. This regeneration only
updates the offline candidate screen; full physical relation, transfer,
force/contact and visual-label commissioning are separate tasks.
