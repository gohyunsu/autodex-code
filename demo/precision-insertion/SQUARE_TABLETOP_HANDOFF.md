# Square 1.5 mm v8 handoff (offline assets only)

The local bundle is
`~/shared_data/AutoDex/precision_insertion/handoff/square_tabletop_v8_20261011_8cb11ae.tar.gz`.
Its SHA-256 is
`b47e8c8315d7879d4aabe282f461941108596279dcddf9b1c633faa564ad2ed9`.
It contains 130 hashed files (361 KB compressed): full-key and unified-socket
object-processing assets, the five v8 tabletop scene JSONs, the square
fixture, eight v8 grasp directories, and audit-only source catalog/fidelity
reports. Seven pose-`004` candidates pass the **nominal** 20 mm endpoint;
the original pose-`000` candidate fails. `robot_ready` is false throughout.

Create the bundle from the source workstation with
[`export_square_tabletop_handoff.py`](export_square_tabletop_handoff.py).
It refuses to overwrite an output directory and checks that the catalog,
promotion manifest, seven selected IDs, fidelity report and current source
bytes agree. Verify any copied or extracted directory before installation:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/export_square_tabletop_handoff.py \
  --verify-root /path/to/square_tabletop_v8_20261011_8cb11ae
```

Extract into a **staging directory**, never directly over an existing
`shared_data` tree. The source v8 scene and socket fixture JSONs contain
`/home/hyunsu/shared_data` paths. The seven promoted candidates' validation
records also bind the source scene SHA-256. Therefore copying the payload to
a different root and running `screen-catalog` immediately will correctly
reject those candidates. Use the demo-local, non-overwriting
[`rehydrate_square_tabletop_handoff.py`](rehydrate_square_tabletop_handoff.py)
to verify and rebind only these known fields. It preserves the source
validation files in the handoff and adds original-scene and manifest hashes
to each recipient validation record. Other candidate paths remain historical
provenance, not runtime inputs. It does **not** copy the audit-only catalog or
fidelity report into runtime paths.

From the fork checkout on the recipient AutoDex PC, with the archive already
extracted under a staging directory:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/rehydrate_square_tabletop_handoff.py \
  --bundle-root /path/to/staged/square_tabletop_v8_20261011_8cb11ae \
  --target-shared-root /path/to/recipient/shared_data

# After reviewing the dry-run report, repeat the same command with --install.
```

The recipient must already have the **same Franka/Inspire URDF bytes**; the
script does not install robot assets. It rejects package tampering, a changed
URDF, target symlinks in destinations and any conflicting existing file. It
never overwrites candidate, scene, object or fixture files. An interrupted
installation can be rerun: completed identical files are accepted. Verify
that `files_to_install` becomes zero after installing. No FoundPose PTH or
calibration is invented.

Then rebuild the endpoint catalog at the **recipient** root; never use the
source-host JSON in `audit_only/` as a live catalog:

```bash
~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/run_pipeline.py screen-catalog \
  --shared-root /path/to/recipient/shared_data \
  --mode square --gap-mm 1.5 --min-hand-clearance-mm 0.2 \
  --output /path/to/new_recipient_catalog.json
```

The `0.2 mm` clearance is only the existing offline screening example;
actual clearance and grasp-fidelity bounds need physical commissioning.
The installer was smoke-tested at a different temporary root: all 127 payload
files installed and a fresh full scan returned eight screened / seven
nominally eligible / zero errors. This is not robot or insertion validation.

The `/mnt/paradex2` mount on this workstation is NFS read-only, so this
archive has **not** been uploaded to NAS. A writable AutoDex-side machine
must perform the transfer. The bundle omits QA-approved key/socket FoundPose
representations and does not supply camera timestamps, socket calibration,
measured post-lift hand/key pose, Franka execution, guarded insertion or
physical outcome labels. These are separate remaining requirements.
