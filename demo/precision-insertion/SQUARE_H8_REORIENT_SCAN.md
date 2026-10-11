# Square-key 8 cm v8 reorientation scan (2026-10-11)

This is a **proposal/stability scan**, not an executable reset. It explores
whether a shorter AutoDex v8 release height supplies directed grasps for the
square 1.5 mm key's tabletop poses `001` and `002` toward insertion-eligible
pose `004`. It does not alter the default 12 cm runtime selection, install
canonical reset candidates, select physical fidelity limits, or verify a
socket-aware Franka/release trajectory.

The existing `prepare-reorient-scenes` command produced and checked all 20
directed square-key `reorient_8` scene pairs, each at both the BODex and stock
sim-filter paths. Its file/hash record is
[`reorient_scene_manifest_v8_h8_20261011.json`](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_scene_manifest_v8_h8_20261011.json).

Using the unchanged AutoDex BODex generator, `precision_insertion.yml`,
`--task_gamma 180`, seed 11021, and **1,000 proposals per scene**, the
unchanged AutoDex scene-collision, squeeze-contact and MuJoCo-gravity filter
returned:

| Directed cell | Raw | Scene clear | Squeeze contact | MuJoCo stable |
| --- | ---: | ---: | ---: | ---: |
| `001→004` | 1,000 | 881 | 233 | 1 (seed `66`) |
| `002→004` | 1,000 | 858 | 204 | 0 |

[`stage_square_reorient_passes.py`](stage_square_reorient_passes.py) now takes
`--height-cm` (default 12 for existing runs). For the selected 8 cm height it
checked all raw seeds, matched the copied stock-filter arrays, bound the
full-key mesh and both v8 scenes by hash, and saved separate non-runtime
handoffs: [cell `1_4`](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_h8_handoff_1_4_1000_20261011/manifest.json) and
[cell `2_4`](/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_h8_handoff_2_4_1000_20261011/manifest.json).

Seed `66` changed its key–hand relation by **5.72 mm center displacement and
12.95° full relative rotation** by the end of MuJoCo gravity. It therefore
fails the previously used *illustrative, uncommissioned* 5 mm / 10° fidelity
gate. It is not a verified reset grasp, let alone a socket-aware planned or
physical repose. Changing the fidelity gate to admit this seed would require
real pickup/release evidence; the scan does not justify doing so. `002→004`
remains without a MuJoCo-stable seed in this finite batch.

Reproduce from the repository root using fresh output roots:

```bash
PYTHONPATH=demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  src/grasp_generation/BODex/generate.py \
  -c sim_inspire/precision_insertion.yml -w 1 \
  --obj_list_file demo/precision-insertion/configs/square_full_key.txt \
  --obj_root_dir /home/hyunsu/shared_data/object_processing \
  --scene_type reorient_8 \
  --scene_filter_file demo/precision-insertion/configs/square_reorient_8_to_4.json \
  --seed_num 1000 --seed 11021 --task_gamma 180 \
  --exp_name NEW_SQUARE_H8_RAW --output_dir /absolute/NEW_SQUARE_H8_RAW

PYTHONPATH=demo/precision-insertion/compat \
  ~/miniconda3/envs/autodex_bodex/bin/python -c \
  'from src.grasp_generation.sim_filter.run_sim_filter import run_sim_filter; print(run_sim_filter("inspire", "v8", "precision_key_1p5mm", "/absolute/NEW_SQUARE_H8_RAW", "/absolute/NEW_SQUARE_H8_STOCK_PASS", obj_root_dir="/home/hyunsu/shared_data/object_processing"))'

~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/stage_square_reorient_passes.py \
  --shared-root /home/hyunsu/shared_data --gap-mm 1.5 \
  --raw-root /absolute/NEW_SQUARE_H8_RAW \
  --stock-candidate-root /absolute/NEW_SQUARE_H8_STOCK_PASS \
  --height-cm 8 --cell 1_4 --expected-seed-count 1000 \
  --output-root /absolute/NEW_SQUARE_H8_HANDOFF_1_4
```

Repeat the final command with `--cell 2_4` and a second output root. These
commands require the same v8 scenes and GPU/MuJoCo environment; they do not
authorize robot motion.
