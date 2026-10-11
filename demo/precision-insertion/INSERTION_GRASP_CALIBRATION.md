# Physical key–hand relation for an insertion grasp

The default live physical-held-relation route needs one calibration **per
eligible v8 grasp and socket mode**. The demo now provides
[`calibrate_insertion_grasp.py`](calibrate_insertion_grasp.py) for this missing
commissioning step. It checks the selected grasp against the current full-key
MuJoCo + 20 mm endpoint catalog, then reuses the same physical-sample
calibration algorithm as the reset-grasp tool. It does not drive the robot.

Collect at least five distinct pickups of the **same** grasp. For each pickup,
record an independently observed key pose, measured robot wrist pose, measured
Inspire joint state, and the measurement error bounds in an immutable JSON
file. The sample fields and example schema are in
[`RESET_GRASP_CALIBRATION.md`](RESET_GRASP_CALIBRATION.md); use the insertion
candidate key, such as `["table", "0", "184"]`, instead of a reset seed key.
Do not substitute the MuJoCo post-squeeze pose or the nominal BODex relation.

For commissioning with a visible lifted key, set
`held_relation_route="observed"` and give the `held_relation` stage a
`calibration_sample_bounds` mapping with all four fields:
`key_translation_error_bound_m`, `key_rotation_error_bound_deg`,
`wrist_translation_error_bound_m`, and `wrist_rotation_error_bound_deg`.
After a successful post-lift key observation and transfer preflight, the demo
writes one non-overwriting
`<run-output>/physical_pickup_samples/<attempt-id>.json`. Its key pose comes
from fresh multi-view SAM3/FoundPose images; its wrist pose comes from the
synchronized measured FR3 state and planner FK. It also records the source
camera manifest. The four error bounds are **operator-supplied**, not inferred
by the exporter. An occluded or rejected key produces no sample. These sample
files can be passed directly as repeated `--sample` arguments below.

From the repository root on the AutoDex PC, for example:

```bash
PYTHONPATH="$PWD/demo/precision-insertion:$PWD/src" \
  ~/miniconda3/envs/autodex_bodex/bin/python \
  demo/precision-insertion/calibrate_insertion_grasp.py \
  --shared-root /path/to/shared_data \
  --family cylinder --gap-mm 5 \
  --endpoint-catalog /path/to/current_gap_5mm_catalog.json \
  --candidate-id table/0/184 \
  --sample /path/to/pickup_001.json \
  --sample /path/to/pickup_002.json \
  --sample /path/to/pickup_003.json \
  --sample /path/to/pickup_004.json \
  --sample /path/to/pickup_005.json \
  --max-nominal-translation-drift-mm 3 \
  --max-nominal-rotation-drift-deg 8 \
  --output-root /path/to/shared_data/AutoDex/precision_insertion/physical_grasp_calibrations
```

The candidate ID above illustrates the current catalog layout; verify its
eligibility in the catalog on the robot PC before collecting trials. The two
drift options are medoid-selection scales, **not** acceptable physical
insertion tolerances. The output path is deterministically
`<output-root>/<family>/<key-object>/<socket-object>/<scene-type>/<scene-id>/<grasp-id>/physical_grasp_calibration.json`.
`binding.json` records the catalog hash and grasp identity. A second run
cannot silently overwrite an existing calibration.

If the stage factory does not set `held_relation.calibration_for_candidate`,
`assemble_live_components` uses this canonical directory beneath the active
session's `shared_root`. A robot PC may still inject a different callback for
an explicitly managed calibration store. The handler rechecks the source
pickup files and current candidate wrist transform on every use.

The output describes measured scatter, not a certified bound on the next
pickup. The physical route still needs a separately chosen future-trial
surface bound and a new post-lift measured state. A positive Qwen lift label
alone cannot provide the key-to-hand transform. If the held key is visible,
the existing `held_relation_route="observed"` uses fresh FoundPose instead
of this calibration; a failed observation does not silently become nominal.
