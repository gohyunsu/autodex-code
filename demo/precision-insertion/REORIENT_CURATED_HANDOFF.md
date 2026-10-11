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

The local and NAS seed copies passed `diff -qr` after copying. Each NAS mode
folder now also has `source_shared_data/`, a hash-audited snapshot of its
original key object-processing files and v8 reset scenes. The square snapshot
verified 11 seeds; the cylinder snapshot verified two. Both snapshots were
relocated from NAS into temporary target roots and reaccepted by the direct
v8 loader. Neither is an installed canonical runtime pool or a verified robot
trajectory.

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
   the paths differ, use the explicit rehydration workflow below; do not
   edit path strings while retaining old hashes.
3. No physical grasp, drop/repose landing, socket-aware full-chain motion or
   actual insertion has passed. The runtime must require fresh pose and
   measured Franka/Inspire state, then run its complete preflight and
   external motion-controller interlocks before any command.

## Rebind a curated handoff on another host

The demo-only `rehydrate_curated_reorient.py` accepts either mode and only
the directed cells listed in its `handoff_manifest.json`. The recipient must
already have **byte-identical non-scene key object-processing assets**. The
source shared-data tree must include the original key and both v8 reset
scene JSONs for those cells; the NAS `source_shared_data/` folder provides
them. If copied to a different mount point, keep the historical source root
in the curated manifest: the tool verifies that
origin against the embedded scene paths and checks every source seed's
MuJoCo trajectory, key geometry and scene hashes before writing. It then
regenerates the same scene geometry with recipient paths, preserves the
original seed evidence, writes rebound hashes into a new staging root and
rechecks the exact v8 loader. It does **not** install canonical candidates.

Example on the AutoDex PC, after installing byte-identical square key
object-processing files into its active shared-data root:

```bash
cd /path/to/autodex-code
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/rehydrate_curated_reorient.py \
  --mode square --gap-mm 1.5 \
  --source-shared-root /mnt/paradex2/hyunsu/autodex_precision_insertion_reorient_curated_20261011/square/source_shared_data \
  --source-handoff-root /mnt/paradex2/hyunsu/autodex_precision_insertion_reorient_curated_20261011/square \
  --target-shared-root /path/to/active_shared_data \
  --target-handoff-root /path/to/active_shared_data/AutoDex/precision_insertion/square/reorient_curated_rehydrated
```

For the cylinder use `--mode cylinder --gap-mm 20` and replace both NAS
`square` components with `cylinder`. The curated seed folder without its
`source_shared_data/` companion is insufficient. If recipient scene
files were copied verbatim with the old host paths, the tool refuses them by
default. `--archive-source-bound-scenes` explicitly archives **only** files
byte-identical to the verified source scenes before writing regenerated
ones; changed scenes are never replaced. A filesystem error during staging
can leave a partial new staging root, so inspect it before retrying under a
new name. Neither a successful relocation nor the loader's stable-seed pass
is a socket-aware Franka path, commissioned pose-fidelity threshold, or
physical reset/insertion result.
