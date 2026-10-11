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

Next, evaluate the exact *live* SAM prompt/preprocessing and all admitted
camera views, visually reject wrong-object masks, and compare a repeated
fixed socket with an independent fixture/board measurement. Then test the
stock multi-view selection/refinement with these **same** new PTHs and record
post-refinement per-view residuals. Commission the camera acquisition clocks
and set repeatability limits from measured uncertainty before freezing a
socket collision pose. None of those steps is replaced by this replay.
