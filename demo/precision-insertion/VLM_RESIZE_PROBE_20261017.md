# Saved-image lift VLM resize probe (2026-10-17)

Source: run `20261017_142332_429150`, `initial_tabletop` and `after_lift`
raw undistorted 2048×1536 PNGs. The same source capture manifests retain each
camera's frame ID, timestamp and SHA-256. This probe did not connect to the
robot. Model: local Qwen3-VL-2B-Instruct on one RTX 3090. Each request used one
same-camera before/after pair and the active lift prompt. The resized probes
used `max_new_tokens=128`; the preceding live full-resolution calls used 512.

| Camera | Pair size per image | Inference time | Parsed class |
| --- | ---: | ---: | --- |
| 25305463 | 2048×1536 | 117.898 s | Interrupted run: answer not saved |
| 25305463 | 1024×768 | 31.735 s | held |
| 25305463 | 768×576 | 18.172 s | held |
| 24122734 | 768×576 | 17.282 s | held |
| 25305463 | 768×576 | 17.298 s | held |
| 25322651 | 768×576 | 17.484 s | held |

The three 768×576 calls took 52.064 s in total, excluding a separate 4.163 s
model load. All three responses parsed successfully and agreed with the
operator's observation that the key was lifted in the hand. This is one saved
grasp scene, so it does not establish accuracy for other stages or viewpoints.

The live configuration now limits classification VLM input to 768×576 and
128 generated tokens. Source PNGs remain full resolution. Per-view records
include source and resized RGB pixel hashes, dimensions and the resize method.
The coordinate-grounding VLM path keeps its original native-pixel behavior.
These measurements describe the former robot-PC three-view route. The active
distributed route selects 24122734, 25305463, 25322638, 25322649 and
25305461, one per capture PC; the 20-camera capture and FoundPose path are
unchanged. Classification still uses 768×576 inputs and records each view's
actual input dimensions and inference duration. See
`LAST_RUN_AUDIT_20261020.md` for its saved-image replay.
