# Cylinder socket handoff: relocate all six fixtures

`rehydrate_cylinder_socket_handoff.py` installs the six cylindrical socket
object-processing directories and their fixed-fixture CAD/template files at a
*different* `shared_data` root. It changes only four host-bound JSON fields:

| File | Field | New value |
| --- | --- | --- |
| `task_geometry.json` | `socket_pose_mesh` | target socket raw OBJ |
| `fixture_pose.template.json` | `pose_object_mesh` | target socket raw OBJ |
| `fixture_pose.template.json` | `pose_object_frame_contract` | target socket frame contract |
| `fixture_pose.template.json` | `pose_estimator_asset` | target FoundPose location, **not a claim that it exists** |

The tool checks every source socket's identity, the 20 mm aligned insertion
transform, uncalibrated pose template and byte-identical fixture/object
collision OBJ. It compares every pre-existing recipient file before writing;
changed recipient CAD or geometry aborts the whole preflight. It preserves
source template bytes and per-file hashes under
`AutoDex/precision_insertion/cylindrical/socket_fixture_handoff_relocation/`.
The final `RELOCATION.json` is a completion marker. An interrupted install
without that file is **not** accepted as complete; inspect the staged
`.socket_fixture_relocation.*` directory and target files before retrying.

For the handoff archive produced on the source PC, extract it to an ordinary
temporary directory. The source root below is the directory containing
`object_processing/` and `AutoDex/`. The historical origin is the absolute
shared-data root stored in its JSON; in the currently prepared bundle it is
`/home/hyunsu/shared_data`. Replace the paths with the actual PC paths:

```bash
python demo/precision-insertion/rehydrate_cylinder_socket_handoff.py \
  --source-shared-root /path/to/extracted/source \
  --source-origin-shared-root /home/hyunsu/shared_data \
  --target-shared-root /path/to/robot/shared_data
```

That is a read-only preflight. Review its counts, then install:

```bash
python demo/precision-insertion/rehydrate_cylinder_socket_handoff.py \
  --source-shared-root /path/to/extracted/source \
  --source-origin-shared-root /home/hyunsu/shared_data \
  --target-shared-root /path/to/robot/shared_data \
  --install
```

If the recipient already contains *byte-identical* source-bound fixture JSONs,
add `--archive-source-bound-templates`. This moves only those exact historical
copies into the relocation report before installing rebound JSON. An existing
file with different content still fails closed; nothing is overwritten. The
script does **not** move key assets, reset seeds, square-socket assets, source
`asset_manifest.json` or `socket_family.json`. The latter two are historical
builder/audit manifests with source-PC paths, not live runtime inputs. Use
[`RESET_HANDOFF_RELOCATION.md`](RESET_HANDOFF_RELOCATION.md) separately for
cylindrical reset candidates.

`fixture` here means the fixed socket obstacle in the robot collision world,
not a mandatory mechanical jig. A socket taped to the ChArUco board still
needs an accurate pose and must pass a session repeatability check before the
frozen collision world is trusted. This relocation supplies CAD only: it does
not create a FoundPose `repre.pth`, measure socket/hand-eye calibration, or
certify robot motion. The pose template remains `calibrated: false` and
`T_robot_socket: null` by design.
