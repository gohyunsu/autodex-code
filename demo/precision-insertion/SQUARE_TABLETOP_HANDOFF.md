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
reject those candidates. A recipient must compare existing assets without
overwriting, rebind only the known scene/fixture absolute-path fields,
preserve the original source validation bytes, bind the new scene hash in a
relocation record, and then rebuild the endpoint catalog at the recipient
root. **That relocation installer is not implemented yet**; do not bypass
the hash check or treat the source catalog in `audit_only/` as a live catalog.

The `/mnt/paradex2` mount on this workstation is NFS read-only, so this
archive has **not** been uploaded to NAS. A writable AutoDex-side machine
must perform the transfer. The bundle omits QA-approved key/socket FoundPose
representations and does not supply camera timestamps, socket calibration,
measured post-lift hand/key pose, Franka execution, guarded insertion or
physical outcome labels. These are separate remaining requirements.
