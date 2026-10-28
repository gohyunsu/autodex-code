# Precision insertion development artifacts

The ZeroDex capture4 project keeps diagnostic outputs under one local root:

`/home/hyunsu/precision_insertion_project/development/zerodex_capture4/`

The matching NAS archive is
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/`.
These are development outputs. Runtime inputs live under the NAS project's
`assets/`, and recorded episodes live under `runs/`.

| Local folder | Contents |
| --- | --- |
| `camera_only/` | New camera-only runs and their timing reports |
| `timing/` | New run timing profiles |
| `probes/` | Explicit socket/key SAM3 and FoundPose probes |
| `planning/` | Offline path and grasp diagnostics |
| `legacy_stage_runs/YYYYMMDD/` | Earlier camera, pose, planning and latency diagnostics created inside the runtime stage's `runs/` |
| `legacy_tmp/YYYYMMDD/` | Earlier local logs, JSON, images and temporary timing results |
| `MIGRATION_MANIFEST.json` | Original and canonical path for each moved artifact |

Run `organize_zerodex_development.py` once to move existing diagnostics. It
leaves symlinks at their old paths so saved evidence and reports containing
absolute paths still open. It keeps `session_evidence`, `recordings`, and the
current clean 5 mm socket session in the runtime stage.

After development work, publish without adding NAS I/O to the pipeline:

```bash
source demo/precision-insertion/config/zerodex_capture4.env.sh
rsync -a --no-owner --no-group --no-perms \
  "$PRECISION_DEV_ROOT/" \
  "$PRECISION_NAS_PROJECT_ROOT/development/zerodex_capture4/"
```

New camera-only runs default to `camera_only/`; timing defaults to `timing/`.
Other diagnostic tools that require an explicit `--output-dir` should use a
date-named directory under `probes/` or `planning/`.
