# Calibrate a directed v8 reset grasp from physical pickups

`calibrate_reset_grasp.py` reuses the demo's existing
`physical_grasp_calibration` algorithm and the **full-key v8 reset seed**
provenance verifier. It does not generate camera observations, grasp the key,
measure a future-trial worst-case error, install a runtime reset candidate or
command the robot. The current staged reset grasp pool has no such physical
calibration yet.

After rehydrating a reset handoff on the AutoDex PC, collect at least five
**distinct physical pickups of the same exact seed**. An independent,
calibrated tracker must observe the key and wrist together, and measured
Inspire joints must be recorded. Each source JSON needs, for example:

```json
{
  "trial_id": "reset_pickup_001",
  "source": "physical_independent_key_and_wrist",
  "candidate_key": ["reset", "0_1", "191"],
  "T_robot_key_observed": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "T_robot_hand_measured": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "hand_q_measured": [0, 0, 0, 0, 0, 0],
  "key_translation_error_bound_m": 0.001,
  "key_rotation_error_bound_deg": 1.0,
  "wrist_translation_error_bound_m": 0.001,
  "wrist_rotation_error_bound_deg": 1.0
}
```

The matrices and bounds above illustrate the **schema only**. They are not
valid measurements or recommended thresholds. The CLI hashes each original
JSON file, rejects duplicate trial IDs/files, rejects MuJoCo or nominal
sources, and reconstructs the reported medoid from the source files.

Example for the staged cylinder `0→1/191` cell (replace the paths and
selection scales with values from the actual AutoDex PC):

```bash
PYTHONPATH=demo/precision-insertion \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/calibrate_reset_grasp.py \
  --shared-root /path/to/shared_data \
  --family cylinder --gap-mm 20 \
  --candidate-root /path/to/shared_data/AutoDex/precision_insertion/cylindrical/reorient_handoff/reset_12 \
  --height-cm 12 --from-pose 000 --to-pose 001 --seed-id 191 \
  --sample /path/to/physical_pickup_001.json \
  --sample /path/to/physical_pickup_002.json \
  --sample /path/to/physical_pickup_003.json \
  --sample /path/to/physical_pickup_004.json \
  --sample /path/to/physical_pickup_005.json \
  --max-nominal-translation-drift-mm 3 \
  --max-nominal-rotation-drift-deg 8 \
  --output-dir /path/to/new_reset_calibration_0_1_191
```

The two drift parameters above are only **medoid selection scales**. They do
not certify a future-trial bound or approve a reset trajectory. Output is an
exclusive directory with `physical_grasp_calibration.json` and `binding.json`;
both retain `robot_ready=false`. The summary is candidate- and socket-mode
specific. If the seed's grasp, scene, key mesh or source evidence changes,
the run fails instead of silently calibrating a different candidate.

A subsequent measured-state repose replan must still combine the verified
physical relation with a separately commissioned worst-case surface bound,
sampled socket/table clearance, new robot feedback and supervised pickup.
The current nominal repose preflight is not a substitute for those steps.
