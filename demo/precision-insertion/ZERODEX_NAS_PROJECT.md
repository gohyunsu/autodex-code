# Precision insertion project bundle on ParaDex2 NAS

The project root is `/mnt/paradex2/hyunsu/precision_insertion`. `ZeroDex` is
the rig profile, not the project name. The AutoDex fork supplies implementation
code; the old AutoDex experiment directory is not this project's archive.

| Path under project root | Contents |
| --- | --- |
| `code/` | Exact tracked AutoDex fork and ParaDex source archives, with commits and SHA256 in `PROJECT_MANIFEST.json` |
| `assets/shared_data/` | Object meshes, all six cylindrical sockets, FoundPose representations, v8 grasps, reset candidates, scenes, fixture geometry, Qwen, SAM3, capture4 calibration, launch/catalog templates, robot dependency bundle |
| `assets/capture4_cache/` | capture4's offline SAM3 Hugging Face cache and two DINOv2 checkpoints |
| `rigs/zerodex_capture4/runtime` | Project-local link to the capture4 calibration, robot profile and launch/catalog templates in `assets/shared_data` |
| `environments/` | Local planner and capture4 Python package lists |
| `runs/zerodex_capture4/YYYY/MM/DD/RUN_ID/` | Post-run uploaded full-resolution camera recording, frame ID index, log and run receipt |
| `PROJECT_MANIFEST.json` | Code commits, asset paths, sizes and SHA256 hashes |

The asset tree uses the existing `AssetPaths` layout. One project bundle covers
gap sizes 1, 3, 5, 10, 15 and 20 mm for the cylindrical key. The current
launcher selects 5 mm by default. The archived square-mode code remains in
the source snapshot; square hardware assets are outside this ZeroDex run
profile.

## Build or refresh the project bundle

Run after committing code changes:

```bash
cd /home/hyunsu/autodex-code
python demo/precision-insertion/prepare_zerodex_nas_project.py
```

Large unchanged NAS source files are hardlinked where available, and local
runtime files are copied into the project tree. Hardlinks remain readable
from the project root even if the older handoff pathname is removed. The
manifest hashes the resulting files. The builder does not move source files or
start the robot.

## Restore on a comparable PC

```bash
python /mnt/paradex2/hyunsu/precision_insertion/install_project.py \
  --destination /home/hyunsu/precision_insertion_runtime
source /home/hyunsu/precision_insertion_runtime/activate_zerodex_capture4.sh
cd /home/hyunsu/precision_insertion_runtime/autodex-code
bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

The installer copies assets to local storage and rewrites old absolute
`shared_data` paths in JSON catalogs, scenes and launch profiles. It extracts
the two code archives and writes an explicit machine profile. It does not
install the operating system, NVIDIA driver, Python environments, RT kernel,
Franka FCI, or camera hardware. The saved calibration is valid for the
capture4/FR3 setup only while camera and robot geometry remain unchanged.

Capture4 itself must have the project's source adapter deployed under its
own namespace and its offline model caches available. The captured package
lists and model caches in this bundle supply those inputs; the launch profile
points capture4 at the project's `assets/shared_data` path on its NAS mount.

## Record each pipeline run without NAS I/O

`run_zerodex_capture4.sh --run` and `--plan-only` start a local capture4
recorder for camera `26053248`, then run the pipeline. The recorder reads
the existing camera shared memory without claiming the camera. It writes a
full-resolution 10 fps MJPG AVI, source frame IDs and read timestamps to
`/home/capture4/precision_insertion_capture4_20261027/recordings/YYYYMMDD/RUN_ID/video/`.
The local PC writes `runs/recordings/RUN_ID/recording_ref.json` under its
ZeroDex stage. A recording failure is visible there but does not change the
pipeline's grasp or insertion decisions. The capture4 recorder is deployed
once with `manage_zerodex_capture4_services.py deploy`.

**No NAS upload happens while the pipeline runs.** After it exits, use:

```bash
python demo/precision-insertion/upload_zerodex_recording.py \
  --receipt /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/recordings/RUN_ID/recording_ref.json
```

The uploader copies the finished recording into
`runs/zerodex_capture4/YYYY/MM/DD/RUN_ID/`, checks the AVI hash against the
recorder manifest, and writes `run_manifest.json`. It is never invoked by
the runtime wrapper. A long session therefore does not add upload traffic to
camera inference or robot planning.

## Current camera-only evidence

The 5 mm socket-only clean session is locally saved under
`zerodex_capture4/runs/socket_session_gap5_clean_20261028/socket_evidence/`.
The separate key capture under
`zerodex_capture4/runs/key_with_clean_gap5_session_20261028/` reused that
socket session and classified the upright key as pose `000`; it found 108
endpoint-eligible catalog candidates. These are perception/catalog results,
not Franka path or physical insertion results. The earlier one-off
`camera_only_gap5_key_present_20261028` run has a key in its socket baseline;
its report records this and it should not be reused as the key-free baseline.
