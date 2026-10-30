# Precision insertion development artifacts

The canonical ZeroDex capture4 diagnostic root is
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/`.
The ZeroDex machine profile writes new diagnostics there directly. The older
`/home/hyunsu/precision_insertion_project/development/zerodex_capture4/`
path is a compatibility symlink for saved references from earlier experiments.
On 2026-10-30, all 2,701 local regular files were copied and checked against
the NAS bytes before the redundant local copy was removed. The receipt is
`/mnt/paradex2/hyunsu/precision_insertion/development/LOCAL_MIGRATION_20261030.json`.
Runtime source assets live under the NAS project's `assets/`, the active asset
view under `runtime/active_shared_data/`, and recorded episodes under `runs/`.

| Folder under the diagnostic root | Contents |
| --- | --- |
| `camera_only/` | New camera-only runs and their timing reports |
| `timing/` | New run timing profiles |
| `probes/` | Explicit socket/key SAM3 and FoundPose probes |
| `planning/` | Offline path and grasp diagnostics |
| `legacy_stage_runs/YYYYMMDD/` | Earlier camera, pose, planning and latency diagnostics created inside the runtime stage's `runs/` |
| `legacy_tmp/YYYYMMDD/` | Earlier local logs, JSON, images and temporary timing results |
| `MIGRATION_MANIFEST.json` | Original and canonical path for each moved artifact |

`organize_zerodex_development.py` is a one-time migration tool for older
runtime-stage and `/tmp` diagnostics. It leaves symlinks at those old paths
for saved evidence. New development output needs no post-run copy because
`PRECISION_DEV_ROOT` already points into the NAS project. Full recordings are
still uploaded separately after a run.

New camera-only runs default to `camera_only/`; timing defaults to `timing/`.
Other diagnostic tools that require an explicit `--output-dir` should use a
date-named directory under `probes/` or `planning/`.

Historical NAS diagnostics from `hyunsu/tmp/20261023_prev`, the 24–26 October
precision experiments, and socket pose reviews are collected under
`development/historical_nas/` by:

```bash
python demo/precision-insertion/organize_zerodex_development.py \
  --organize-nas-history
```

The original NAS paths become relative symlinks, including when viewed from
capture4's different mount point. `development/HISTORICAL_NAS_MANIFEST.json`
records every move. Older handoffs and the explicit backup remain provenance
archives and are listed there; live runtime assets do not depend on them.
