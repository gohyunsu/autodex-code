# Current local cylinder endpoint catalogs (2026-10-11)

The earlier `endpoint_catalog_staged{,_verified}_gap_*mm_20261011.json`
files fail `preflight_endpoint_catalog_file`: their saved
`endpoint_implementation_sha256["candidates.py"]` differs from the current
demo implementation. They remain untouched as historical evidence. Do not
edit their hashes or use them for a session; the candidate/endpoint screen
must be rerun after a bound implementation changes.

The current **local-only** full-pool screens are in
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/`:

| Cylinder radial gap | Current catalog file | v8 candidates | Nominally eligible |
| --- | --- | ---: | ---: |
| 1 mm | `endpoint_catalog_current_gap_1mm_20261011.json` | 12 | 12 |
| 3 mm | `endpoint_catalog_current_gap_3mm_20261011.json` | 12 | 12 |
| 5 mm | `endpoint_catalog_current_gap_5mm_20261011.json` | 12 | 12 |
| 10 mm | `endpoint_catalog_current_gap_10mm_20261011.json` | 12 | 12 |
| 15 mm | `endpoint_catalog_current_gap_15mm_20261011.json` | 12 | 12 |
| 20 mm | `endpoint_catalog_current_gap_20mm_20261011.json` | 12 | 12 |

Each catalog has `complete_scan=true`, no screen errors, 11 pose-`000`
rows and one pose-`001` row. All six passed the same source-hash and
pose-conditioned checks used by `open_verified_session`. The explicit
minimum whole-hand/socket clearance was **0.001 mm**, copied from the prior
catalogs to reproduce their threshold. It is *not* an approved physical
uncertainty margin. A CAD endpoint pass means only the centered, nominal
key-in-hand 20 mm insertion pose and both recorded hand hold configurations
passed the offline check. It does not establish post-squeeze key fidelity,
Franka IK/path safety, contact success, or robot readiness.

From the repository root, rerun on the **recipient AutoDex PC's own**
`shared_data` after installing the matching demo code and v8 candidate files.
Use a new output filename: the CLI refuses to overwrite old evidence.

```bash
PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src" \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root /absolute/path/to/shared_data \
  --mode cylinder --gap-mm 15 --min-hand-clearance-mm 0.001 \
  --output /absolute/path/to/new_endpoint_catalog_gap_15mm.json

PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src" \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py select-catalog \
  --mode cylinder --gap-mm 15 \
  --catalog /absolute/path/to/new_endpoint_catalog_gap_15mm.json \
  --pose-stem 000
```

Repeat for the desired gap and pose `001`. A copied catalog may fail on a
different host because its absolute `shared_root`, v8 candidate paths, mesh
hashes, and implementation hashes are bound to its producer. A successful
`select-catalog` is still only offline eligibility; the live session must
run its own source-bound preflight. The `/mnt/paradex2` mount on this machine
is currently **read-only**, so these refreshed local files have not been
uploaded to NAS.
