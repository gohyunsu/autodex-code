# Saved-camera timing replay, 2026-10-15

All commands below were run in `/home/robot/autodex-precision-insertion` on
`feat/precision-insertion`. The branch was clean before a successful
`git pull --ff-only origin feat/precision-insertion`; pulled HEAD was
`6f5a8030bef26c302ec98cc572dc6ed0a3b1ab65`, a descendant of `6f5a803`.
`/home/robot/AutoDex` was not edited. `run_robot_pc.sh --check` passed with
`scope: static_files_only_no_camera_cuda_or_robot_connection`, 20 cameras,
112 catalog candidates and 98 eligible. The first check exposed an existing
`session_evidence_001` directory; static preflight now reports its existence
without rejecting it. Live launch still rejects an existing output directory.
No `--run`, camera service, or robot command was executed.

## Actual saved inputs

Let `STAGE` be
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical`.
`/home/robot/shared_data` is the mounted NFS4 NAS on this PC; temporary PNG
outputs were on local `/tmp` (ext4/NVMe).

| Item | Location | What is present |
| --- | --- | --- |
| Session board/socket capture | `$STAGE/live_board9_gap20_20261013/session_evidence_001` | 100 actual PNGs: 20 board images and two 20-camera socket image/mask sets; `evidence_manifest.json`, `session_calibration.json`, per-socket `payloads.json`. This 188 MB session ended before a key trial. |
| Separate key QA capture | `$STAGE/strict_camera_only_static_interval_20261013_143002` and `$STAGE/key_socket_20261013_143036_request_1615929368/images` | 20 actual camera PNGs with `key_socket_qa/report.json`; source file hashes, decoded pixel hashes, camera/frame IDs and bounded acquisition timestamps. It is camera-only QA, **not** an admitted trial key pose. |
| Other actual camera QA | `$STAGE/precision_frame_bound_stock_qa_20261013_retry1` | Key/socket QA reports, raw request PNGs, SAM masks and FoundPose payloads. Its recorded key request took 3.129 s from dispatch to collection; this is historical request timing, not replay timing. |
| Frozen session and screened candidates | `/home/robot/shared_data/AutoDex/experiment/precision-insertion/inspire/precision_key_cylinder_r15_h80/_pipeline_runs/20261013_165557_728987` | `frozen_session_calibration.json`, `endpoint_catalog.json`, `source_evidence_binding.json`, `session_run.json`; no attempt, key observation or planner report. |
| Catalog source | `$STAGE/recipient_gap20_endpoint_catalog_20261013.json` | Screened v8 endpoint catalog, 112 candidates, 98 eligible. It is an offline screening result, not a fresh motion plan. |
| Qwen model | `/home/robot/shared_data/hyunsu/handoff/20261012_qwen3_vl_2b_89644892/payload/model` | Local Qwen3-VL-2B model files, including 4.26 GB `model.safetensors`. Present, but no valid live VLM pair to infer from this stopped session. |
| Scene snapshot | `$STAGE/current_scene_snapshot_20261013_123549` | One `preview.jpg` and manifest; a separate snapshot, not a motor-trial video or an input to this replay. |
| Other real AutoDex video | `/home/robot/shared_data/AutoDex/experiment/v8_video/inspire/pepsi/20260906_225030/videos/exec` | 20 camera-ID AVI recordings from a separate v8 Pepsi episode. For example `25305467.avi` is H.264, 2048×1536, 55 frames. Its `init_capture/images` and `label/raw/images` contain saved camera PNGs. Different object/task, so excluded from white cylinder insertion timing. |
| Other real perception images | `/home/robot/shared_data/AutoDex/experiment/perception_check/pepsi/20260911_004845/capture/images` | 20 camera-ID PNGs from Pepsi perception QA, with separate masks/visualizations; excluded from the cylinder replay. |

No actual `after_lift`, `preinsert`, `final_clear_view`, failed-trial XY
capture, `key_observation.json`, per-attempt planner report, or live
`timing_summary.json` was found for the board-9/gap-20 run. The MP4 files in
the staged `visualizations` and `presentation_assets` trees are rendered
motion previews, not camera videos. The `qwen_synthetic_smoke` PNGs are also
synthetic and were not used.

## Method and measurements

The current live path and this diagnostic replay have different verification
frequencies. The replay deliberately repeats a full audit so that its cost can
be measured; that repeated audit is not part of every live trial.

| Procedure | Current live path | This saved-data replay |
| --- | --- | --- |
| v8 endpoint catalog | Parse once before camera capture; reuse the in-memory snapshot. Candidate source files are not rehashed. | Parse once per replay pass to time catalog preflight; no path planning. |
| Board/socket session evidence | Write the bundle and pass its small producer manifest into the ledger; full readback is an offline audit. | Fully reread/hash the saved bundle on every pass as an offline audit. |
| Session identity bindings | Hash three small JSON files once when opening the ledger: evidence manifest, calibration and catalog. | Hash the same three files on every pass to isolate their cost. |
| Key and later raw-camera evidence | Reuse the producer-validated manifest within the live session; offline/handoff verification still checks original bytes. | Reread/verify separate key-QA PNG bytes and decoded pixels on every pass. |
| PNG creation | Production encoder writes and hashes encoded PNG bytes as it creates each capture. | Run the same encoder twice on the same saved pixels, with 1 and 4 workers. |

There is no measured live trial timing in this dataset; the rows below are
offline replay measurements only. In particular, the repeated full-bundle
audit time must not be added to a live per-attempt estimate.

`replay_saved_timing.py` deliberately performs an **offline full audit** with
the production `verify_session_evidence_bundle`, loads the catalog with
production `preflight_endpoint_catalog_file`, hashes only the three small
live ledger bindings, and verifies each QA key image by original
file SHA-256, decoded-pixel SHA-256, SAM/FoundPose frame ID and acquisition
timestamp. It then calls the production PNG encoder/hash/worker function on
those **same 20 decoded 2048×1536 images**, writing copies into separate
`/tmp` directories. Source files are read only. The session bundle and key QA
frames were acquired separately; this composite replay is a throughput check,
**not** a measured end-to-end trial. Each of four runs gets a
stock AutoDex `PipelineTrace`, `timing_summary.json` and `report_timing.py`
compatible event log. Run 0 is the first cold data replay; runs 1–3 are warm.
The PNG worker order alternates between repeats. All timings below are wall
seconds and exclude model loading, since no model was loaded. "Cold" here is
the first data pass after Python imports, not a measured Qwen cold start.

| Procedure | What it does | First run | Warm median (3 runs) | Warm range |
| --- | --- | ---: | ---: | ---: |
| Composite saved replay | Sequentially audits evidence, loads catalog, hashes small bindings, verifies key QA and writes both PNG batches | 6.869 | 7.145 | 6.770–7.189 |
| Full session evidence audit | Reads/hashes the saved board/socket bundle; **offline-only in this repeated replay** | 3.022 | 2.942 | 2.894–2.945 |
| Catalog preflight | Parses/validates the immutable v8 endpoint catalog; **does not plan paths** | 0.0070 | 0.0093 | 0.0071–0.0098 |
| Live-style small bindings | Hashes only session manifest, session calibration and endpoint catalog JSON | 0.0013 | 0.0015 | 0.0013–0.0015 |
| Key QA source audit | Reads/decodes 20 separate QA PNGs; checks hashes, camera/frame IDs and timestamps; **offline-only, no SAM/FoundPose** | 0.988 | 1.035 | 0.972–1.039 |
| PNG batch, 1 worker | Encodes/hashes/writes the same 20 frames to one temp directory | 2.266 | 2.562 | 2.250–2.565 |
| PNG batch, 4 workers | Encodes/hashes/writes the same 20 frames to another temp directory | 0.584 | 0.631 | 0.590–0.645 |

The last two rows are substeps of the complete replay; do not add them to
its total again. Each worker setting produced byte-identical PNG hashes.
For the 4-worker warm median, 20 worker spans sum to **2.359 s of work**
while their enclosing batch occupies **0.631 s wall time**. For one worker,
work and occupancy are 2.559 s and 2.562 s. The batch's own non-child
time is about 0.00043 s (4 workers) or 0.00274 s (1 worker). These measurements
used `/tmp` storage and do not establish NAS write latency. The trace has no
operator wait, network, motion or contact span because replay performs none;
their live durations are **측정 불가**, rather than zero. There are no VLM calls:
per-call camera count, image size, model, CUDA memory and inference time are
**측정 불가** for this data. The model's cold-load time is also **측정 불가**.

The historical stock QA key request's 3.129 s dispatch-to-collected value
includes its own request and result collection and comes from a different
capture. It must not be added to the replay totals. Another older key QA
request records 66.519 s; its conditions differ, so it is not a warm-run
comparison.

| Requested stage | Saved input needed | Result |
| --- | --- | --- |
| Fresh key perception | Admitted key capture plus original SAM/FoundPose producer and model | **측정 불가**; only camera-only QA images/payloads exist. |
| Candidate motion planning | Key pose, measured robot start state and planner invocation/report | **측정 불가**; catalog load above is not planning. |
| Lift VLM | Same-attempt before-grasp/after-lift pairs | **측정 불가**. |
| Preinsert VLM | Same-attempt after-lift/preinsert pairs | **측정 불가**. |
| Final live `inside_socket` VLM | Initial tabletop plus post-release final-clear-view pairs and checkpoint records | **측정 불가**. |
| Failed-trial XY | Failed attempt plus saved preinsert images, frozen calibration and postlift report | **측정 불가**. |
| Motor, camera network, contact | Live controller/camera events | **측정 불가** from saved images. |

`replay_saved_live_vlm.py` is a separate read-only harness for a future
source-verified released checkpoint. It invokes the exact live
`assess_released_insertion_checkpoint` function; with a same-failed-attempt
XY context it invokes the production `_compute_grounded_alignment` core on
the saved preinsert frames. It records the first model load separately,
three warm repeats, every call's camera, source image sizes, model, prompt
hash, CUDA allocated/peak memory and inference wall time. It refuses a
missing or altered source before loading Qwen. Its XY path measures the
grounded computation core, not the whole failed-trial controller/ledger gate.
It was **not run** because this session has no released checkpoint or failed
trial frames; filling it with rendered or unrelated images would misstate
the requested live task latency.

## Commands and artifacts

```bash
cd /home/robot/autodex-precision-insertion
bash demo/precision-insertion/run_robot_pc.sh --check
STAGE=/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical
/home/robot/anaconda3/envs/planner/bin/python demo/precision-insertion/replay_saved_timing.py \
  --session-evidence "$STAGE/live_board9_gap20_20261013/session_evidence_001" \
  --key-report "$STAGE/strict_camera_only_static_interval_20261013_143002/key_socket_qa/report.json" \
  --catalog "$STAGE/recipient_gap20_endpoint_catalog_20261013.json" \
  --output-root /tmp/precision-saved-replay-20261015-v4
python demo/precision-insertion/report_timing.py \
  demo/precision-insertion/timing_results/20261015/replay_01.json.timing
/home/robot/anaconda3/envs/planner/bin/python -m pytest -q \
  demo/precision-insertion/tests/test_replay_saved_timing.py \
  demo/precision-insertion/tests/test_replay_saved_live_vlm.py \
  demo/precision-insertion/tests/test_raw_lift_checkpoint.py \
  demo/precision-insertion/tests/test_live_runtime.py \
  demo/precision-insertion/tests/test_session_launch.py \
  demo/precision-insertion/tests/test_timing_profile.py
/home/robot/anaconda3/envs/planner/bin/python -m pytest -q \
  demo/precision-insertion/tests/test_runtime_evidence_cache.py \
  demo/precision-insertion/tests/test_checkpoint_camera.py \
  demo/precision-insertion/tests/test_insertion_checkpoint.py \
  demo/precision-insertion/tests/test_preinsert_checkpoint.py \
  demo/precision-insertion/tests/test_retry_insertion_checkpoint.py
```

The small trace/summary files from all four repeats are copied to
`timing_results/20261015/`; temporary PNG copies remain under `/tmp` and
are excluded from version control. Focused tests passed **78/78** across two
groups.

The measured **offline audit** cost is repeated full session evidence validation
(~2.94 s warm). This audit was still in the live path when the replay was
recorded; it has since been removed from the newly produced live bundle path.
The live producer now passes its small manifest in memory. The 1-worker
PNG batch is ~2.56 s; four workers reduce its elapsed time to ~0.63 s on this
`/tmp` filesystem. No inference or motion
bottleneck can be ranked until a real profiled trial supplies those events.

## Comparison with the branch before the pull

The pre-pull commit was `8231d85` and the pulled commit is `6f5a803`.
`8231d85` was exported to a separate `/tmp` source tree; neither the current
checkout's live devices nor `/home/robot/AutoDex` were used. Both versions
read the same saved board-9/gap-20 evidence and 112-row endpoint catalog. The
read-only comparison replayed this exact sequence four times per version:
catalog preflight, producer's full bundle verification, calibration reload,
ledger opening to a new `/tmp` output, pose `000` candidate selection and a
single candidate's NPY override load. The override used identity key pose
solely to time file loading; it did **not** plan a path. These are CPU/NAS
substeps, not a complete episode. The current checkout's subsequent
`write_png_jobs` extraction has the same encoder/worker behavior as the
pulled commit.

| Saved-data substep | Pre-pull `8231d85`, warm median | Current, warm median | Relevant code change |
| --- | ---: | ---: | --- |
| Catalog preflight | 0.0825 s | 0.0066 s | Old code rechecked source hashes across the v8 pool; current live code parses the immutable catalog once. |
| Producer's full board/socket verification | 2.829 s | 2.918 s | Both replayed versions performed this once after capture. A subsequent local change removed it from the active live path; the offline verifier remains. |
| Calibration reload | 0.0091 s | 0.0094 s | Essentially the same work. |
| Open session ledger | 3.999 s | 0.0714 s | Old code reread the full evidence bundle, reran catalog preflight and hashed two FoundPose representations as well as three small JSONs. The pulled code receives the producer manifest/calibration/catalog and hashes only the three JSONs. |
| Select pose `000` candidates | 0.932 s | 0.0009 s | Old code rescanned the candidate tree and checked matching source hashes. Current code selects from the in-memory catalog. |
| Load one selected candidate override | 0.118 s | 0.0020 s | Old stock loader walked the object candidate tree; current code reads the selected three NPY files. This row includes the selection invoked by the override. |
| **Measured subset total** | **7.983 s** | **3.008 s** | **4.975 s less in this saved-data subset**; not an end-to-end insertion time. |

The first pass of this comparison was 9.746 s before and 3.007 s after. The
versions were run sequentially, so their first passes had different NAS cache
states and are **not** a controlled cold-start speedup comparison. The warm
table uses the median of the following three passes. The total adds the six
listed sequential calls; it does not include camera capture, SAM/FoundPose
inference, Qwen, cuRobo planning, robot movement, contact or operator waiting.

A second comparison used the exact old `cv2.imwrite`-then-file-SHA loop and
the current encoder/worker function on the **same 20 real 2048×1536 key-QA
frames**, writing distinct directories under local `/tmp`. All output PNG
hashes matched. Warm medians were **2.249 s old serial**, **2.250 s current
one worker**, and **0.578 s current four workers**. Thus this local-disk
comparison shows a benefit from parallel encoding/writing, not a meaningful
one-worker benefit from dropping the immediate file reread. NAS write latency
and full derived captures with both original and undistorted images remain
unmeasured.

Other pulled changes have no saved-data timing here: separate concurrent
service management, one-time key model setup, immutable fixed-world model
caching, and more detailed timing traces. Local single-GPU Qwen remains
serial across views. No old or current complete trial trace exists for this
scene, so a full old-pipeline or full current-pipeline episode duration cannot
be computed or reliably inferred from these substeps.

The exact measurements and one-off scripts are in
`timing_results/20261015/{before_comparison.json,after_comparison.json,png_comparison.json,compare_pre_pull_timing.py,compare_pre_pull_png.py}`.
