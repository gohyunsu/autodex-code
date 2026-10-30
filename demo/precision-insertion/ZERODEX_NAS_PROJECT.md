# Precision insertion project bundle on ParaDex2 NAS

The project root is `/mnt/paradex2/hyunsu/precision_insertion`. `ZeroDex` is
the rig profile, not the project name. The AutoDex fork supplies implementation
code; the old AutoDex experiment directory is not this project's archive.

| Path under project root | Contents |
| --- | --- |
| `code/` | Exact tracked AutoDex fork and ParaDex source archives, with commits and SHA256 in `PROJECT_MANIFEST.json` |
| `assets/shared_data/` | Object meshes, all six cylindrical sockets, key and extracted 5 mm socket FoundPose representations, v8 grasps, reset candidates, scenes, fixture geometry, Qwen, SAM3, capture4 calibration, launch/catalog templates, robot dependency bundle |
| `runtime/active_shared_data/` | Active path-correct `AssetPaths` tree: heavy files link into `assets/shared_data`, small JSON files contain project-local paths. This is the live `PRECISION_SHARED_ROOT`. |
| `runtime/session_inputs/` | Frozen input JSON for the 2026-10-30 socket remeasurement. Retained because its evidence ledger records these exact paths. |
| `restore/foundpose_packages/` | Six socket FoundPose installation TARs. The installer extracts a socket only when its complete representation is absent from copied assets. Live episodes do not read these packages. |
| `assets/capture4_cache/` | capture4's offline SAM3 Hugging Face cache and two DINOv2 checkpoints |
| `assets/capture4_foundationpose/` | FoundPose weights formerly reached through capture4's old absolute symlink |
| `rigs/zerodex_capture4/runtime` | Project-local link to the capture4 calibration, robot profile and launch/catalog templates in `assets/shared_data` |
| `rigs/zerodex_capture4/main_pc_system/current/` | ParaDex system configuration snapshot for this PC's robot adapter |
| `rigs/zerodex_capture4/capture4_source_20261028.tar.gz` | capture4 perception source snapshot for our namespaced adapter |
| `environments/` | Local planner and capture4 Python package lists |
| `runs/zerodex_capture4/precision-insertion/inspire/KEY/_pipeline_runs/RUN_ID/` | Main-PC pipeline ledger, snapshots, timing, labels and planning results written directly to the project NAS. |
| `runs/zerodex_capture4/YYYY/MM/DD/RUN_ID/` | Separately uploaded full-resolution camera recording, frame ID index and run receipt. |
| `development/zerodex_capture4/` | Camera, planning, timing and implementation diagnostics written directly to NAS; never read by the live pipeline |
| `development/LOCAL_MIGRATION_20261030.json` | Receipt for the byte-verified migration of earlier local diagnostics; the old local development path is a compatibility symlink |
| `PROJECT_MANIFEST.json` | Code commits, asset paths, sizes and SHA256 hashes |

The live pipeline consumes `runtime/active_shared_data` and the selected rig profile.
Create or refresh this compact tree with
`python3 demo/precision-insertion/link_zerodex_nas_runtime.py` after adding
assets to `assets/shared_data`. The command rewrites project paths in small
JSON files and links large immutable directories without copying model data.
`restore/`, `code/`, and `environments/` are for installation and reproduction;
`runs/` and `development/` contain outputs and are never input asset roots.

The asset tree uses the existing `AssetPaths` layout. One project bundle covers
gap sizes 1, 3, 5, 10, 15 and 20 mm for the cylindrical key. The current
launcher selects 5 mm by default. The archived square-mode code remains in
the source snapshot; square hardware assets are outside this ZeroDex run
profile.
All six complete socket representations are preserved as TARs in `restore/`.
The 5 mm representation is also fully extracted in the active NAS asset tree:
all 2,402 archive files are present at the expected sizes and `repre.pth`
matches its provenance SHA256. The current socket route uses masks and a
board-constrained body fit; it does not load socket FoundPose. Extract the
selected TAR before switching another gap to a socket FoundPose route, or use
`install_project.py` when restoring the complete bundle on another PC. The
installer flattens the handoff TAR's extra `models/` wrapper into the canonical
FoundPose object layout.
Development output rules and the migration of earlier scattered diagnostics
are described in `development/README.md`.

The live ZeroDex profile reads no NAS path outside
`/mnt/paradex2/hyunsu/precision_insertion`. This PC imports ParaDex from a
project-owned local checkout with a copied `system/current`; it does not
follow the old `/home/hyunsu/paradex/system/current` link into NAS `/paradex`.
On capture4, the existing `/home/capture4/paradex` installation remains an
explicit exception for camera acquisition and shared-memory reading. Our
FoundPose/SAM3 source and weights use the namespaced capture4 adapter and this
project's assets. No shared capture4 source is modified.

A 2026-10-30 file-open trace of the **5 mm `--check` path** recorded 2,935
successful opens, including 19 opens of 14 distinct project paths. It opened
no NAS file outside this project and no old `~/shared_data` file; resolving
the opened project paths also stayed inside the project. This establishes the
static preflight's file dependencies, not those of a live camera/robot run.
On capture4, the namespaced cache contains the key representation at
`cache/shared_data/AutoDex/foundpose_assets/precision_key_cylinder_r15_h80/object_repre/v1/precision_key_cylinder_r15_h80/1/repre.pth`
and raw meshes for all six socket gaps. The current board-constrained socket
measurement does not load a socket representation. The trace and summary are
kept under `development/zerodex_capture4/audits/filesystem/`.

## Build or refresh the project bundle

Run after committing code changes:

```bash
cd /home/hyunsu/autodex-code
source demo/precision-insertion/config/zerodex_capture4.env.sh
python3 demo/precision-insertion/prepare_zerodex_nas_project.py
```

After a source-only commit, refresh the tracked code snapshots without
recopying runtime assets:

```bash
python3 demo/precision-insertion/prepare_zerodex_nas_project.py --refresh-code-only
```

Large unchanged NAS source files are hardlinked where available, and local
runtime files are copied into the project tree. Socket representations are
packed locally and transferred as one file per socket. Hardlinks remain readable
from the project root even if the older handoff pathname is removed. The
manifest hashes the resulting files. The builder does not move source files or
start the robot.
It sources the capture4 perception snapshot from local
`~/precision_insertion_project/restore/` and ParaDex system settings from
`~/precision_insertion_project/vendor/paradex/system/current`. Use
`--capture4-source-archive` or `--paradex-system` to override those locations
when rebuilding on another machine.

## Restore on a comparable PC

```bash
python3 /mnt/paradex2/hyunsu/precision_insertion/install_project.py \
  --destination /home/hyunsu/precision_insertion_runtime
source /home/hyunsu/precision_insertion_runtime/activate_zerodex_capture4.sh
cd /home/hyunsu/precision_insertion_runtime/autodex-code
bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

The installer copies assets to local storage and rewrites old absolute
`shared_data` paths in JSON catalogs, scenes and launch profiles. Its profile
sets `PRECISION_SHARED_ROOT` to the restored copy. The complete 5 mm socket
representation is copied once; other socket packages are extracted as needed.
It extracts
the two code archives, restores ParaDex `system/current` into the local
checkout, and writes an explicit machine profile. It does not
install the operating system, NVIDIA driver, Python environments, RT kernel,
Franka FCI, or camera hardware. The saved calibration is valid for the
capture4/FR3 setup only while camera and robot geometry remain unchanged.

Prepare capture4's namespaced adapter and local model cache once:

```bash
cd /home/hyunsu/autodex-code
python3 demo/precision-insertion/manage_zerodex_capture4_services.py deploy
python3 demo/precision-insertion/manage_zerodex_capture4_services.py prepare-assets
```

Preparation copies source, model weights, the key FoundPose representation,
and key/socket perception meshes from NAS `hyunsu/precision_insertion`
to capture4 local storage. The resulting source and caches are reused on
later starts. `manage_zerodex_capture4_services.py start` launches the
namespaced services; there is no source or model transfer per episode. The
launch profile points capture4 FoundPose at its local cached copy of these
project assets. The NAS copy is read only during one-time preparation.

## Record each pipeline run without NAS I/O

`run_zerodex_capture4.sh --run` and `--plan-only` start a local capture4
recorder for camera `26053248`, then run the pipeline. The recorder reads
the existing camera shared memory without claiming the camera. It writes a
full-resolution 10 fps MJPG AVI, source frame IDs and read timestamps to
`/home/capture4/precision_insertion_capture4_20261027/recordings/YYYYMMDD/RUN_ID/video/`.
The local PC writes
`runs/zerodex_capture4/_recording_receipts/RUN_ID/recording_ref.json`.
A recording failure is visible there but does not change the pipeline's grasp
or insertion decisions. Launch files and socket-session evidence for the same
ID live under `runs/zerodex_capture4/_control/RUN_ID`; the episode ledger uses
that same ID below the normal `_pipeline_runs` tree. None of these mutable
files is written into `assets/` or `runtime/active_shared_data/`. Same-request
inline source frames and registered-ROI diagnostics are grouped under
`runs/zerodex_capture4/_control/RUN_ID/raw_captures`; the launch file binds
that directory explicitly. Evidence bundles hard-link an already persisted
identical PNG when both paths are on the project filesystem, rather than
encoding and writing the full-resolution frame repeatedly. This changes only
storage I/O, not perception admission or task outcomes. The capture4
recorder is deployed once with `manage_zerodex_capture4_services.py deploy`.

**No full-video NAS upload happens while the pipeline runs.** The main-PC run
ledger and small snapshots are written to the NAS project during the trial;
the camera video remains on capture4 until the separate upload command. The
high-frequency timing trace is written to `/tmp/precision-insertion-timing`
during the run and copied once beside the NAS run ledger on orderly exit.
After the pipeline exits, use:

```bash
python3 demo/precision-insertion/upload_zerodex_recording.py \
  --receipt /mnt/paradex2/hyunsu/precision_insertion/runs/zerodex_capture4/_recording_receipts/RUN_ID/recording_ref.json
```

The uploader copies the finished recording into
`runs/zerodex_capture4/YYYY/MM/DD/RUN_ID/`, checks the AVI hash against the
recorder manifest, and writes `run_manifest.json`. It is never invoked by
the runtime wrapper. A long session therefore does not add upload traffic to
camera inference or robot planning.

## Restart on the current kds-Z790 rig

The 2026-10-30 5 mm socket-only measurement is saved under
`development/zerodex_capture4/camera_only/socket_remeasure_20261030_03/socket_evidence`.
Its accepted cameras were `25322639`, `25322642`, and `26053248`, with the
socket center at approximately `(0.72148, 0.03303, 0.06292)` m in the robot
frame. Reuse this evidence only while the board/socket and camera geometry
have not moved. The key may be placed at a new tabletop location.

The latest robot-free preflight command is:

```bash
cd /home/hyunsu/autodex-code
PRECISION_GAP_MM=5 bash demo/precision-insertion/run_zerodex_capture4.sh --check
```

The check passed with 126 catalogued, 114 endpoint-eligible candidates. It
checks files and catalog structure; it does not certify a Franka trajectory
or physical insertion. The current env profile sends run outputs to this
project's `runs/zerodex_capture4` tree. A new live session still needs the
robot, hand, FCI daemon and namespaced camera services running; no robot
motion is part of the static check.

Two 2026-10-30 live attempts ended before grasp. The first exposed a
capture4 asset-path mapping error after the NAS migration; the namespaced
`precision_init_daemon.py` now resolves key mesh and FoundPose representation
from its local cache by object name, and its focused tests pass. The second
received four camera responses but rejected the key capture because too few
views remained after excluding a SAM mask that covered the known socket.
Camera-only probes showed that the old prompt sometimes selects the hollow
socket as the key. `small white solid peg` selected the actual key in three
views of the same tabletop layout. This is now the primary key prompt, with
one alternate-prompt recapture only when the first capture has too few views.
The failed request's same-frame masks are saved for diagnosis. This change
has not yet been tested in a new robot episode. A camera-only end-to-end run
with the powered-off robot reused the frozen socket session and admitted the
upright key on three cameras (`25322639`, `25322642`, `26053248`) using the
new prompt, classified tabletop pose `000`, and found 108 endpoint-eligible
candidates. Its report is in
`development/zerodex_capture4/camera_only/key_admission_primary_prompt_retry_20261030/report.json`.

The local Qwen checkpoint in this NAS project's active asset view was also
replayed against the six saved, independently reviewed released-key cases.
The three-view aggregate matched all six visual labels (three inside, three
outside); individual cameras scored 5/6, 6/6 and 6/6. The input manifest and
output are both in this project's `development/zerodex_capture4/reviews/`:
`released_vlm_dataset_20261029/manifest_six.json` and
`nas_vlm_replay_20261030/evaluation_six.json`. This checks the archived visual
label route and NAS model path, not a new physical insertion or generalization
to unseen scenes.

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

After switching the namespaced capture4 service from the NAS asset root to
its local perception cache, the same saved 5 mm socket session and upright-key
camera-only route completed again. `camera.foundpose_request` took **69.81 s**
in `20261028_212003_949258273_183682` with the NAS asset root, versus
**4.21 s** in `20261028_212831_958031007_186856` with the local cache. Both
reports selected pose `000` and found 108 eligible catalog candidates. This
is an observed end-to-end request-time comparison; it does not isolate file I/O
from model initialization or GPU cache effects. No robot motion was involved.

## 2026-10-30 powered-off audit

With Franka and Inspire off, the current 5 mm `--check` completed without
opening either device: 126 catalogued grasps, 114 endpoint-eligible grasps,
four configured capture4 cameras, and the project-owned NAS paths above. A
repeat of the same static command for every cylinder gap also completed:

| Radial gap (mm) | Catalogued | Endpoint-eligible |
| ---: | ---: | ---: |
| 1 | 126 | 111 |
| 3 | 126 | 111 |
| 5 | 126 | 114 |
| 10 | 126 | 114 |
| 15 | 126 | 113 |
| 20 | 126 | 111 |

These counts are file/catalog checks, not a new trajectory or physical pass.
A walk of the project tree found 94 symlinks, zero broken targets, and zero
targets outside `/mnt/paradex2/hyunsu/precision_insertion`. A scan following
those links found 136 JSON files containing historical absolute paths:
126 `simulation_validation.json` records, two reset source records, two
FoundPose template metadata files, three FoundPose summaries, two onboarding
provenance files, and one generation marker. These are retained source or
onboarding metadata, not launch/profile paths. The live candidate loader
reads the simulation validation status and file hashes, without opening its
`source_candidate` value; the capture4 FoundPose wrapper points at the
project-cached `repre.pth`. This static audit does not prove every live file
open, so external-path independence still needs a complete runtime trace.
No camera, planning, or robot trial was part of this audit.

The most recent saved failed-start timing trace is
`runs/zerodex_capture4/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/20261030_151321_204391.timing/timing_summary.json`.
Its camera FoundPose request took 3.92 s. SAM3 loading (9.85 s), Qwen
loading (6.28 s), and provisional cuRobo world warmup (18.74 s) overlapped
other startup or camera work; their inclusive durations must not be added
to infer session wall time. This attempt stopped at key pose admission before
pickup, so the trace cannot establish closed-loop or insertion latency.
