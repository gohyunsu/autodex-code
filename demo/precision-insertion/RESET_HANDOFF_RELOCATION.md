# Rehydrate cylinder v8 reset handoff on the AutoDex PC

The local 12 cm reset handoff contains two offline full-key MuJoCo passes,
`0_1/191` and `1_0/631`. Its scene JSON and candidate evidence bind the
original `/home/hyunsu/shared_data` paths. Copying the scene JSON directly to
another host leaves those paths stale; editing the JSON text alone breaks the
recorded scene hashes. `rehydrate_reset_handoff.py` regenerates the v8 scenes
from byte-identical key assets, checks that all numeric scene geometry is
unchanged, and writes new evidence while preserving each original evidence
file byte-for-byte.

The original archive on this workstation is
`/home/hyunsu/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff_bundle_20261010.tar.gz`
(SHA-256 `6b83a94026074f925c57567dd754771e01cbfda16491d2e8a10ba647894acc98`).
It has **not** been copied to NAS; transfer it to the AutoDex PC as a file,
then extract it into a fresh staging directory. The extracted directory must
contain `object_processing/` and `AutoDex/` immediately below it.

From the personal fork's `feat/precision-insertion` checkout on that PC:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/rehydrate_reset_handoff.py \
  --source-shared-root /path/to/extracted_handoff \
  --source-origin-shared-root /home/hyunsu/shared_data \
  --target-shared-root /path/to/AutoDex_shared_data \
  --source-candidate-root /path/to/extracted_handoff/AutoDex/precision_insertion/cylindrical/reorient_handoff/reset_12 \
  --target-candidate-root /path/to/AutoDex_shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff/reset_12 \
  --source-audit /path/to/extracted_handoff/AutoDex/precision_insertion/cylindrical/reorient_filter_audit_1000_20261010.json \
  --install-missing-key-assets
```

`--install-missing-key-assets` copies only absent key files other than the
source-bound scenes. A changed pre-existing asset is rejected, never
overwritten. If old directed scene JSONs have already been copied into the
recipient tree, add `--archive-source-bound-scenes`. This moves only files
whose bytes exactly match the handoff originals into
`AutoDex/precision_insertion/cylindrical/reorient_handoff/source_bound_scenes_archive/`
before writing regenerated scenes. Unrelated or modified scenes are rejected.
The output `reset_12/` staging directory must not exist. No candidate is
installed into the canonical AutoDex/NAS reset tree.

Check the result from the target shared-data root:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py audit-reorient \
  --shared-root /path/to/AutoDex_shared_data \
  --mode cylinder --gap-mm 20 \
  --candidate-root /path/to/AutoDex_shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff \
  --max-reset-drift-mm 3 --max-reset-axis-tilt-deg 8
```

The 3 mm / 8° example is diagnostic, **not commissioned**. For an offline
`preflight-repose`, point `--reset-candidate-dir` to the exact staged
`reset_12/` directory; the `audit-reorient` option instead wants its parent.
The target side's `RELOCATION.json` records original and new evidence hashes,
key-asset hashes, regenerated-scene count, and any archives. The original
source-evidence bytes are kept next to each rebound `source_evidence.json`;
`SOURCE_AUDIT.json` and both original directed scene pairs are also retained
inside the staged `reset_12/` directory so that deleting the temporary
extraction does not erase their byte-level provenance.

This tool transfers reset **grasp seeds only**. It does not install socket
assets, validate a physical fixture or camera calibration, create a Franka
repose trajectory, commission a motion controller, or prove a successful
physical reset. At illustrative 3 mm / 8° limits, only `0→1/191` passes the
post-squeeze fidelity gate; `1→0/631` does not. Its best status remains
`robot_ready=false`.
