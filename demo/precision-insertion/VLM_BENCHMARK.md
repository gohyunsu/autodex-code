# Evaluate local or API VLM visual labels on saved AutoDex trials

`evaluate_saved_vlm.py` loads one VLM and replays multiple **paired,
time-ordered, two-or-more-camera** trials through the same read-only lift and
insertion-appearance observers used by the demo. Source image bytes and a
separate human annotation file are SHA-256 checked before model loading and
again immediately before inference. A changed input stops the run. No camera
or robot is opened, and no output is a motion or physical-success permit.

Each case requires an annotation file prepared by an independent reviewer:

```json
{
  "schema": "precision_insertion_visual_annotation_v1",
  "case_id": "trial_001",
  "task": "lift",
  "label": "miss",
  "source": "independent_human_review",
  "reviewer_id": "reviewer_1",
  "evidence": "The key remains on the tabletop in both camera views."
}
```

The input manifest has schema `precision_insertion_vlm_visual_benchmark_v1`
and an explicit maximum within-phase camera skew in seconds. Paths may be
absolute or relative to the manifest directory. Hash *file bytes*, not the
decoded pixel array. The same camera ID must be present before and after:

```json
{
  "schema": "precision_insertion_vlm_visual_benchmark_v1",
  "max_phase_skew_s": 0.02,
  "cases": [{
    "case_id": "trial_001",
    "task": "lift",
    "annotation": {
      "path": "trial_001_annotation.json",
      "sha256": "<sha256sum of annotation file>"
    },
    "views": [{
      "camera_id": "front",
      "before_grasp": {
        "path": "trial_001_front_before.png",
        "sha256": "<sha256sum of PNG>",
        "timestamp_s": 1791600000.000
      },
      "after_lift": {
        "path": "trial_001_front_after.png",
        "sha256": "<sha256sum of PNG>",
        "timestamp_s": 1791600001.000
      }
    }, {
      "camera_id": "side",
      "before_grasp": {
        "path": "trial_001_side_before.png",
        "sha256": "<sha256sum of PNG>",
        "timestamp_s": 1791600000.005
      },
      "after_lift": {
        "path": "trial_001_side_after.png",
        "sha256": "<sha256sum of PNG>",
        "timestamp_s": 1791600001.005
      }
    }]
  }]
}
```

For insertion-appearance cases, use task `insertion_visual`, phases
`preinsert` and `final_or_abort`, and one of `normal_appearance`, `partial`,
`rim_jam`, `slip`, `unobservable` in the annotation. Lift labels are `held`,
`miss`, `slip`, `unobservable`. The insertion `normal_appearance` class is
**not** a measured 20 mm insertion-success label.

From the repository root after the [local-VLM setup](README.md):

```bash
export PYTHONPATH="$PWD/demo/precision-insertion:$HOME/realtime_vlm"
~/.venvs/precision-vlm/bin/python \
  demo/precision-insertion/evaluate_saved_vlm.py \
  --manifest /path/to/held_out_manifest.json \
  --backend local --model-id Qwen/Qwen3-VL-2B-Instruct \
  --output /path/to/new_local_vlm_report.json
```

`--backend gemini` uses the same manifest but additionally requires
`--allow-external-images`, `GEMINI_API_KEY` and the optional Gemini SDK.
The report retains exact prompts/responses, parse errors, source hashes,
per-task confusion counts and target-class false-positive counts. The
reported target classes are `held` and `normal_appearance`, respectively;
they do **not** grant insertion motion or certify final task success.

This tool checks only that the manifest *claims* independent review and
timestamp provenance. It cannot prove reviewer independence, image capture
time, calibration, or ground-truth correctness. Commission the dataset from
real AutoDex camera and physical outcome records before using any resulting
metrics to choose a runtime VLM or threshold.
