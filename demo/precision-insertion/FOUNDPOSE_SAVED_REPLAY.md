# Saved AutoDex-camera replay of new FoundPose representations

This is a **read-only diagnostic** of the four representations generated on
2026-10-11, not an accepted session calibration. The four full JSON reports
are under
`shared_data/AutoDex/precision_insertion/foundpose_saved_replay_20261011/`.
The replay uses the existing `autodex.perception.foundpose_init.FoundPoseInit`
with each object's canonical v8 asset path. No stock source file, live camera
or robot was changed or called.

Each test used three selected views from **one** real AutoDex-rig shot, not
the complete 20-camera set. It verified the JPEG against `shot.json` and
the 2026-09-21 intrinsics/extrinsics against their shot hashes. The saved
SAM mask is separately hashed but its provenance to that shot is not
cryptographically authenticated by the replay. Square-key masks came from
the NAS **ROI-assisted** offline method, which the live full-frame pipeline
does not yet implement. Cylinder-key and socket masks came from the earlier
direct-SAM/socket-only evaluator. No exposure-time evidence or independent
metric ground-truth object pose was supplied.

| Object / selected condition | Saved-mask CAD IoU range, per view | Maximum pairwise CAD-bounds-centre distance | Maximum axis difference |
| --- | ---: | ---: | ---: |
| Square socket, socket-only | 0.827–0.925 | 72.8 mm | 8.0° |
| Square 1.5 mm key, tabletop pose 01 | 0.797–0.864 | 133.2 mm | 15.2° |
| Cylinder 15 mm-gap socket, socket-only | 0.900–0.979 | 23.1 mm | 10.0° |
| Cylinder key with 15 mm socket, tabletop pose 01 | 0.648–0.903 | 115.5 mm | 11.9° |

The values above are computed from **independent per-camera initial
FoundPose estimates**, before stock AutoDex's best-candidate selection and
multi-view silhouette refinement. They do not establish the accuracy or
failure of that later refinement. IoU is only agreement with the *supplied
mask* in that same view; even high IoU can coexist with severely inconsistent
3D poses. The square key also has a large full-rotation discrepancy; for the
axially symmetric cylinder the raw yaw/full-rotation discrepancy is not
physically meaningful. Cylinder-key axis difference quotients out its
end-for-end symmetry and uses the CAD-bounds centre, not the arbitrary end
chosen as its object-frame origin. Cylinder-socket axis remains directed:
open and closed ends are not exchangeable.

## Replay of the actual stock multi-view method

On the **same three saved views and the same newly generated PTHs**, the
demo-only [`probe_foundpose_stock_refinement.py`](probe_foundpose_stock_refinement.py)
rechecked source hashes and called the unchanged
`InitOrchestrator.refine_from_payloads(selection_mode="iou")` without its
network constructor. It used the stock 100-iteration, `lr=0.002`, antialiased
silhouette optimizer and its `sil_loss_threshold=0.003`. The demo-local
`SolidMtlSilhouetteOptimizer` only converts a no-texture solid-colour MTL to
vertex colour for the CUDA renderer; it does not alter the mesh geometry.

| Object | Best initial candidate: mean cross-view IoU | Refined: mean cross-view IoU | Stock silhouette loss | Stock silhouette gate |
| --- | ---: | ---: | ---: | --- |
| Square socket | 0.816 | 0.974 | 0.000373 | Passed |
| Square 1.5 mm key | 0.490 | 0.843 | 0.000663 | Passed |
| Cylinder 15 mm-gap socket | 0.925 | 0.975 | 0.000512 | Passed |
| Cylinder key | 0.688 | 0.959 | 0.000257 | Passed |

Full per-view scores, selected camera IDs, poses, source-report hashes and
stock timings are in the adjacent durable `foundpose_stock_refine_20261011`
JSON directory under `shared_data/AutoDex/precision_insertion/`, also copied
to the FoundPose NAS handoff. A high IoU or a stock `0.003` loss pass means
only that the CAD fits the supplied saved masks. These masks are not
independently authenticated to the camera exposure, and no independent
metric socket/key pose ground truth or repeatability experiment exists. In
particular, **none of these four poses is an accepted fixture calibration**.

The socket-only evaluator retained more masks from each shot than the first
three-view probe. Replaying **all available saved masks** from the same shot
(still not all 20 rig cameras) produced:

| Socket | Saved mask views | Maximum initial CAD-centre disagreement | Selected candidate mean IoU | Refined mean IoU | Weakest refined view |
| --- | ---: | ---: | ---: | ---: | ---: |
| Square | 12/20 | 266 mm | 0.897 | 0.942 | `25305463`: 0.717 |
| Cylinder 15 mm gap | 10/20 | 387 mm | 0.908 | 0.972 | `25322640`: 0.964 |

Both still passed the stock full-frame silhouette-loss threshold. In the
square replay the optimizer's own outlier filter excluded one of twelve views
from its gradient, and the final pose remained a poor match for camera
`25305463` when independently scored against **all** twelve masks. Thus a
low aggregate loss must not be treated as an all-view identity/pose pass.
Per-view minimum refined IoU, wrong-object visual review, and independent
fixture truth must be commissioned before session acceptance. The reports
record the exact full saved-view set and source hashes; no mask was invented
for the other cameras.

The demo `start_precision_session()` now runs this same stock selection and
silhouette refinement separately for each admitted socket capture. It checks
every admitted camera's post-refinement IoU against a **commissioned**
`minimum_socket_refined_iou`, then checks repeatability between one pose per
independent capture. Raw per-view estimates and the stock selection report
remain in the saved evidence bundle. The table above is still a saved-image
diagnostic, **not** proof that a future live session will produce those poses
or that either pose has sub-millimetre absolute accuracy. Independent physical
fixture truth, camera calibration and real acquisition clocks remain mandatory.

To reproduce one read-only replay, after creating the isolated environment
described below and installing [NVlabs nvdiffrast](https://github.com/NVlabs/nvdiffrast/blob/main/docs/index.html)
at tag `v0.3.3` against the existing CUDA 12.1 toolkit:

```bash
~/.venvs/precision-foundpose/bin/python \
  demo/precision-insertion/probe_foundpose_stock_refinement.py \
  --source-report /home/hyunsu/shared_data/AutoDex/precision_insertion/foundpose_saved_replay_20261011/cylinder_socket.json \
  --shared-root /home/hyunsu/shared_data \
  --calib-dir /home/hyunsu/shared_data/cam_param/20260921_213318 \
  --output /tmp/new_cylinder_socket_stock_refine.json
```

On a fresh environment, nvdiffrast's JIT also needs `CUDA_HOME` pointing at
the CUDA 12.1 toolkit and include paths for the PyTorch-bundled CUDA headers
(`cusparse`, `cublas`, `cusolver`, `cuda_runtime`). The test workstation used
the `autodex_bodex` conda toolkit path **only for compilation**, while every
package installation went into `~/.venvs/precision-foundpose`. The stock
runtime environment and stock source files were not modified. This script
refuses to overwrite an output report and never contacts capture PCs or a
robot.

The current `freeze_fixture_pose` rejects a socket capture if any admitted
view exceeds the *commissioned* translation or angle limit relative to its
observed medoid. These saved results would therefore fail any sub-23 mm
translation limit for the tested cylinder views, and any sub-73 mm limit for
the tested square views. No permissive threshold should be chosen merely to
make this data pass. A separate fixed-fixture reference is needed to know
which estimate is right; repeated multi-view agreement alone still cannot
prove absolute pose accuracy.

## Reproduce a read-only view

The isolated `~/.venvs/precision-foundpose` environment needs PyTorch,
MV-GoTrack/FoundPose's local dependencies, EGL, and the NAS MV-GoTrack source
snapshot. After verifying the source paths and the selected `shot.json`:

```bash
~/.venvs/precision-foundpose/bin/python \
  demo/precision-insertion/probe_foundpose_saved_views.py \
  --shared-root /home/hyunsu/shared_data \
  --family square --gap-mm 1.5 --part socket \
  --shot-dir /mnt/paradex2/hyunsu/tmp/precision_insertion_multiview_20261010_652ac909/square/socket_only/views/shot_20261010T130233Z_ff46a919 \
  --calib-dir /home/hyunsu/shared_data/cam_param/20260921_213318 \
  --gotrack-root /mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909/source_snapshot/MV-GoTrack \
  --view 25322651 /mnt/paradex2/hyunsu/autodex_precision_insertion_perception_eval_20261010_652ac909/results/socket_only/precision_socket_unified/25322651_mask.png \
  --output /tmp/new_socket_probe.json
```

Repeat `--view CAMERA_ID SAVED_MASK_PNG` for synchronized views from that
same shot. The output includes exact source/calibration/mesh/repre hashes,
per-view initial pose and mask IoU, plus pairwise centre/axis diagnostics.
It refuses to overwrite an existing report. Its `robot_ready` is always
`false`.
For all masks supplied by one offline evaluator, `--mask-dir` may be used
instead of repeated `--view`; it reads only `*_mask.png` and verifies every
camera against that same shot and calibration.

Next, evaluate the exact *live* SAM prompt/preprocessing and all admitted
camera views, visually reject wrong-object masks, and compare a repeated
fixed socket with an independent fixture/board measurement. Then test the
stock multi-view selection/refinement with these **same** new PTHs on the
complete capture rather than only the three views replayed here. Commission
the camera acquisition clocks
and set repeatability limits from measured uncertainty before freezing a
socket collision pose. None of those steps is replaced by this replay.
