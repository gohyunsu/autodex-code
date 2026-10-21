# Post-lift VLM replay, 2026-10-21

This was a read-only replay of six real AutoDex camera captures. No camera or robot command was issued. The source is each attempt's `lift_assessments/000/report.json` and the PNGs named in its `frames` records. The replay preserved each frame's camera ID and timestamp; the observer's saved record includes the exact prompt, crop coordinates, input transform, model answer, and session calibration/URDF digests.

## Live method

- Cameras: `24122734` (capture1) and `25322638` (capture3), inferred concurrently by their persistent Qwen3-VL-2B workers.
- Project the post-lift measured FR3/Inspire hand from its 13 joint values through the Oct12 right-angle adapter URDF into each undistorted camera image using the frozen session calibration. Center a 650 × 650 source-pixel crop on the finger workspace point `[0, 0.07, -0.08]` metres in the Inspire base frame. Resize that crop to 640 × 640 RGB pixels. Only raw camera pixels enter the VLM.
- Exact English prompt: `Is the small solid white cylindrical peg visibly held by the robot's black fingers in this image? The robot also has white finger shells. Reply only YES or NO.`
- `held` when either close view answers `YES`; `miss` when both answer `NO`; otherwise `unobservable`. Both camera answers and the confirming view IDs are saved. This is a visual label, not a measured key–hand transform.

The previous live observer centered a 900 × 900 crop on the **before-grasp key mask** and reduced it to 384 × 384 or 512 × 512. It asked whether the key was "between" the fingers. That method labeled the visibly held key in `47444` as `miss`. Enlarging the same mask-centered crop to 900 × 900 did not fix the tested held cases. A second prompt asking only whether a white object touched a finger answered `YES` for empty hands as well, so it was removed.

## Saved-image results

The labels in the second column were assigned by inspecting the original after-lift images. `974ad` shows an empty hand and a key lying on the board, although its old automated report says success.

| Attempt prefix | Visual review | New label | 24122734 | 25322638 | Observer elapsed, model already loaded |
| --- | --- | --- | --- | --- | ---: |
| `1a371` | miss | miss | NO | NO | 0.926 s |
| `f8cb4` | held | held | YES | YES | 0.313 s |
| `62f25` | held | held | YES | YES | 0.285 s |
| `974ad` | miss | miss | NO | NO | 0.300 s |
| `a5325` | held | held | NO | YES | 0.251 s |
| `47444` | held | held | YES | NO | 0.256 s |

Thus the saved set has 4/4 true held and 2/2 true miss labels. These six examples are too few to estimate future accuracy. The model-load time, raw capture, image storage, robot motion, and network camera latency are excluded from the elapsed column. Across the final 12 Qwen calls, median GPU inference was about 0.120 s and peak allocated CUDA memory about 4.36 GB per worker. The first observer call after worker readiness took longer than the following five.

The replay output is `/tmp/precision_grip_v2_replay.json` and worker timing is `/tmp/precision_grip_v2_worker_timing.jsonl` on the robot PC. Old `visible_grip_v1` reports remain verifiable without changing their historical labels. The new live observer writes `visible_grip_v2` records.
