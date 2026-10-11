# Curated v8 reorient seed handoff (2026-10-11)

The original AutoDex v8 reset seeds are **not** in this workstation's
canonical `AutoDex/candidates/inspire/reset_12` tree. That path resolves to
the ParaDex2 NAS and currently audits as zero precision-key reset candidates.
The existing offline runs stored valid full-key MuJoCo-pass seeds in separate
handoff folders. `run_pipeline.py assemble-reorient-handoff` now copies
**explicitly selected directed cells** into a new staging root, checking each
source seed and copied seed with the demo's direct-v8 loader. It never edits
stock AutoDex code or canonical candidates.

Local staging roots:

- Square 1.5 mm:
  `/home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_curated_handoff_20261011`
- Cylinder 20 mm gap:
  `/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_curated_handoff_20261011`

Readback-identical copies are at:

- `/mnt/paradex2/hyunsu/autodex_precision_insertion_reorient_curated_20261011/square`
- `/mnt/paradex2/hyunsu/autodex_precision_insertion_reorient_curated_20261011/cylinder`

The local and NAS copies passed `diff -qr` after copying. Neither is an
installed canonical runtime pool or a verified robot trajectory.

| Key | Directed pose | Verified MuJoCo-pass seeds | Exploratory pose-fidelity pass |
| --- | --- | ---: | ---: |
| Square | `000 → 004` | 10 | 6 at 10 mm / 30° |
| Square | `003 → 004` | 1 | 0 at 10 mm / 30° |
| Cylinder | `000 → 001` | 1 | 1 at 5 mm / 10° |
| Cylinder | `001 → 000` | 1 | 0 at 5 mm / 10° |

These thresholds are **diagnostic examples, not commissioned acceptance
limits**. Square rotation is the full relative rotation; cylinder rotation
is symmetry-reduced axis tilt. The source evidence binds the unchanged v8
key mesh, both reset scene JSONs, BODex grasp arrays, MuJoCo `sim_eval.json`
and squeeze/gravity trajectory by SHA-256. A seed still needs the caller's
real pose-fidelity limits and a socket-aware Franka pickup–lift–repose–release
preflight; landing must be observed. MuJoCo gravity success alone does not
authorize motion.

To audit the local staging root, use:

```bash
cd /home/hyunsu/autodex-code
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py audit-reorient \
  --shared-root /home/hyunsu/shared_data \
  --mode square --gap-mm 1.5 \
  --candidate-root /home/hyunsu/shared_data/AutoDex/precision_insertion/square/reorient_curated_handoff_20261011 \
  --max-reset-drift-mm 10 --max-reset-axis-tilt-deg 30
```

For the cylinder, change `--mode cylinder --gap-mm 20`, the candidate root
to the cylindrical root above, and example limits to 5 mm / 10°. The saved
local audit reports are `exploratory_fidelity_audit.json` inside each staging
root. In code, `SessionRunner.preflight_repose` accepts the **parent of**
`reset_12` as `reset_candidate_root`; the offline CLI `preflight-repose`
accepts the exact `reset_12` folder as `--reset-candidate-dir`. All normal
key observation, socket collision world and measured-start gates remain.

Important limitations:

1. There are no verified staged reset seeds for square poses `001` and `002`
   toward `004`, nor for the other directed square cells. An exhausted pose
   without a valid directed reset path must stop for review or use an
   independently commissioned alternative; do not assume that every pose
   can reorient automatically.
2. The NAS copy is a **handoff**, not immediately runnable on another PC.
   The v8 reset scene JSONs contain absolute mesh/URDF paths from this
   workstation, and candidate source hashes bind those exact scene bytes.
   On the AutoDex PC, verify the shared-data mount and path resolution. If
   the paths differ, regenerate and revalidate the scenes and provenance on
   that host; do not edit path strings while retaining old hashes.
3. No physical grasp, drop/repose landing, socket-aware full-chain motion or
   actual insertion has passed. The runtime must require fresh pose and
   measured Franka/Inspire state, then run its complete preflight and
   external motion-controller interlocks before any command.
