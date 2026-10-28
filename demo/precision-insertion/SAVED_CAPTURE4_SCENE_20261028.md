# Saved four-camera scene replay (robot-free)

This replay uses the lit October 27 **lying cylindrical key** and socket
photographs. It runs the same board-9 triangulation, current socket body
fitter, cylinder tabletop classifier and v8 endpoint-catalog selection as the
live demo. It does **not** infer Franka joints from a photograph, run a
current-scene cuRobo trajectory, or claim a physical insertion.

The key and socket masks were captured in separate requests. Their source
images each give an independent ChArUco board pose. The script records the
two request IDs and the difference between the board origins instead of
silently treating them as one synchronized camera instant. It does not add
a new live acceptance threshold.

```bash
cd /home/hyunsu/autodex-code
PYTHONPATH="$PWD:/home/hyunsu/paradex:$PWD/demo/precision-insertion:$PWD/src" \
  /home/hyunsu/.venvs/precision-vlm/bin/python \
  demo/precision-insertion/replay_zerodex_saved_scene.py \
  --launch-config /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/launch.json \
  --socket-images-dir /mnt/paradex2/hyunsu/zerodex_capture4_runtime/socket_probe_1cb67be54180_request_727786403/images \
  --socket-mask-dir /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/socket_lying_key_probe_hollow_20261027_1 \
  --key-images-dir /mnt/paradex2/hyunsu/zerodex_capture4_runtime/key_probe_227a659f8b9b_request_2107810824/images \
  --key-refinement-report /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/lying_key_probe_20261027_1/current_stock_refined_20261028.json \
  --endpoint-catalog /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/recipient_gap20_endpoint_catalog_straight.json \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/my_saved_scene_replay.json
```

Choose a new `--output` path each time; the tool never overwrites a report.
The saved images are already undistorted, so the board routine is given their
undistorted intrinsic matrix and zero distortion. The live camera path still
uses original JPEGs with the original intrinsic/distortion calibration. Both
paths register the same board-9 definition through
`register_profile_charuco_board` without editing ParaDex's shared board file.

Recorded result:

| Measurement | Result |
| --- | ---: |
| Board corners triangulated in each capture | 37/54 |
| Board rigid-fit RMS, socket / key capture | 0.450 / 0.452 mm |
| Difference between separately fitted board origins | 0.018 mm |
| Socket mask-body mean IoU with current fitter | 0.953 |
| Socket centre spread, two usable views | 3.425 mm |
| Key mask-silhouette mean IoU after current refinement | 0.871 |
| Key tabletop stem and axis residual | `001` / 2.38° |
| 20 mm-gap, straight-mount endpoint-eligible grasps for `001` | 5 |

The report is
`/home/hyunsu/shared_data/AutoDex/precision_insertion/zerodex_capture4/runs/saved_scene_replay_20261028_2.json`.
The earlier October 27 socket probe reported a much lower body IoU because
it used the old wall-only fit; the current region-aware fitter selects the
flange in both usable views. A good silhouette score is **not** an independent
millimetre-accuracy ground truth. The two-view 3.425 mm centre spread matters
especially for smaller socket gaps. The 5 grasp entries have passed the
offline v8 stability/20 mm endpoint screen, not a path from the measured
robot joint state. The powered-off robot supplies no such state for this
replay. A new lit socket-only session followed by a key capture remains the
live procedure when the experimental objects are placed again.
