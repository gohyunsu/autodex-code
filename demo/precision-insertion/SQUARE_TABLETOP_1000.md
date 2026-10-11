# Square full-key tabletop v8 candidate refresh

This refresh uses the existing `sim_inspire/precision_insertion.yml` and
unchanged AutoDex `BODex/generate.py` and `run_sim_filter()` implementations.
It creates **1,000 full-key proposals for each of five tabletop scenes and
each of four square-key gaps**. No object-side contact-region mask or separate
project BODex numerical quality threshold is used. The precision config
still chooses four Inspire contact fingers; this is not a claim that all five
fingers or all physical key contacts were tested.

Stock v8 filtering checks pregrasp scene/self collision, squeeze contact,
then MuJoCo squeeze and tabletop-gravity stability. The demo's additional
screen checks the centered, axis-aligned 20 mm CAD key/socket fit and the
whole Inspire hand/socket endpoint, including the stock simulated squeeze
pose and nominal AutoDex controller hold pose. It **does not** screen the
continuous Franka path, contact-rich insertion, or physical success.

From the repository root, use the isolated `autodex_bodex` environment and
RTX GPU for BODex and the stock filter:

```bash
env PYTHONPATH=demo/precision-insertion/compat \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file demo/precision-insertion/configs/square_tabletop_full_keys.txt \
  --obj_root_dir /home/hyunsu/shared_data/object_processing \
  --scene_type table --seed_num 1000 --seed 12011 \
  --exp_name NEW_SQUARE_TABLETOP_RAW --output_dir /absolute/NEW_SQUARE_TABLETOP_RAW

env PYTHONPATH=demo/precision-insertion/compat \
  /home/hyunsu/miniconda3/envs/autodex_bodex/bin/python -c \
  'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_1p0mm", "/absolute/NEW_SQUARE_TABLETOP_RAW", "/absolute/NEW_SQUARE_TABLETOP_STOCK_PASS", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'
```

Repeat the stock filter call for `precision_key_0p3mm`, `0p5mm`, and
`1p5mm`. Use new output roots; do not overwrite a previous batch. The stock
single-process function is intentional: its separate multiprocessing CLI
mode can skip the contact check.

```bash
/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/screen_square_tabletop_batch.py \
  --shared-root /home/hyunsu/shared_data \
  --raw-root /absolute/NEW_SQUARE_TABLETOP_RAW \
  --pass-root /absolute/NEW_SQUARE_TABLETOP_STOCK_PASS \
  --output-root /absolute/NEW_SQUARE_1P0_SCREEN \
  --expected-per-scene 1000 --minimum-hand-clearance-m 0.0002 \
  --gap-mm 1.0

/home/hyunsu/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/promote_square_stock_batch.py \
  --shared-root /home/hyunsu/shared_data --gap-mm 1.0 \
  --screen-root /absolute/NEW_SQUARE_1P0_SCREEN \
  --candidate-id-offset 12011000 \
  --output-manifest /absolute/NEW_SQUARE_1P0_PROMOTION.json
```

The promotion command is **read-only by default**. Add `--install` only after
reviewing `summary.json`: it appends eligible grasps as v8 numeric IDs
`candidate-id-offset + original-seed-id`, refuses to replace an existing
candidate, and records source, scene, mesh and endpoint hashes. A complete
current-pool endpoint catalog must then be rebuilt with
`run_pipeline.py screen-catalog`; an older catalog becomes stale when the
candidate pool grows. The 0.2 mm hand clearance here is an uncommissioned
numerical screen, not a measured safety margin.
