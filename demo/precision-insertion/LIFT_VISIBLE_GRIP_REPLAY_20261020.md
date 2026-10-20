# Lift VLM replay, 20 October 2026

The active distributed lift label uses two independently routed capture PCs:
`24122734` on capture1 and `25322638` on capture3. Each worker receives one
key-centred crop of its saved raw `after_lift` frame. The crop is a 900×900
square anchored by that camera's saved `before_grasp` key mask. The first view
is resized to 384×384 and the second to 512×512. The raw before/after frames,
frame IDs, timestamps and PNG hashes remain bound to the checkpoint; the
model receives the after-lift crop because the question concerns visible grip.
No key-axis direction is used as a success or failure condition.

The exact English questions are in
[`lift_visible_grip.py`](precision_insertion/lift_visible_grip.py). Both YES
answers mean `held`; both NO answers mean `miss`; disagreement or an answer
other than YES/NO means `unobservable`. This is a visual label, not measured
key–hand contact or proof of rigid grasp.

The saved-image replay used Qwen3-VL-2B-Instruct from the local October 12
bundle. Each source report and raw PNG was hashed before inference. The four
cases were repeated three times after model load; the table shows the three
serial robot-PC replay times in seconds. The original checkpoint verifier also
accepted a temporary report with each new answer and unchanged source frames.
The exact answers, source-report SHA-256 values, crop mapping and per-view
inference times are in
[`timing_results/20261020/lift_visible_grip.jsonl`](timing_results/20261020/lift_visible_grip.jsonl).

| Saved attempt | Visual reference | New result (3/3) | Serial replay, s |
| --- | --- | --- | --- |
| `974ad7ad` 09:16 | Key lies loose on board; old VLM said held | miss | 7.97, 7.49, 7.74 |
| `1a371459` 09:39 | Key lies loose on board | miss | 7.69, 7.70, 7.76 |
| `f8cb41a1` 09:53 | Operator confirmed held | held | 7.64, 7.94, 9.30 |
| `62f250bd` 10:10 | Operator confirmed held | held | 7.61, 7.81, 9.46 |

These are offline, single-GPU serial times. They exclude model loading and
do not measure camera network, robot motion or the parallel capture-PC latency.
The 09:16 reference is a review of the saved image; the other operator labels
come from the session. No saved, operator-labelled successful grasp of a key
that *started lying down* was available in this set. The prompt and code allow
either key orientation, but that physical case still needs a live observation.
