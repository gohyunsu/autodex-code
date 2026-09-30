# Pipeline timeline and external-video workflow

`run_pipeline.py` writes one timestamp schema at two scopes. The run timeline
contains the complete process; every episode timeline is an exact filtered
view of those same event records and therefore keeps the same `seq`,
`monotonic_ns`, `pipeline_time_s`, `utc_ns`, and `utc` values. Legacy episode
timing trees and `plan/timing.json` are not written.

## Output

```text
experiment/<exp>/<scene>/<hand>/<object>/_pipeline_runs/<run_id>/
  manifest.json
  events.jsonl                 # crash-tolerant canonical run timeline
  timeline.json                # finalized run view
  episodes.json                # episode index, including recovery attempts
  episodes/<episode_id>/
    manifest.json
    events.jsonl               # same records, episode-filtered
    timeline.json
  artifacts/
    coverage/                  # uncovered scenes and gain-ranked grasps
    planner/                   # candidate pass/fail stack and selection
    recovery/                  # rotate grids and reorient height attempts
    scene/                     # exact planning scene
  sync/external_video_sync.json
  edit/
    markers.csv
    storyboard.json
    storyboard.html
    pipeline_video.json         # render inputs, mapping, layout, and cut audit
    pipeline_video.mp4          # final external-camera composite
    grasp_pose_renders/         # one fixed, pose-matched 3-D PNG per selected grasp
    episodes/<episode_id>/      # episode-only markers and storyboard
```

Each episode's `result.json` contains links to both timelines, not another
timing aggregate.

## External camera sync

Start the external camera before the pipeline and stop it after the pipeline.
For frame-level automatic alignment, split the UTGE900 TTL signal to a small
LED fixed near the edge of the external camera frame and run:

```bash
python src/execution/run_pipeline.py --obj <object> --external-sync-cue ...
```

The pipeline records start/end cue events. After copying the external video,
select a tight ROI around the LED and align it:

```bash
python scripts/sync_external_video.py \
  --run experiment/<...>/_pipeline_runs/<run_id> \
  --video /path/to/external.mp4 \
  --roi x,y,width,height
```

If automatic LED detection is ambiguous, inspect the video and provide the
two cue onset times directly with `--cue-video-seconds <start> <end>`. Two
anchors estimate both offset and clock drift. The command rewrites edit
markers with external-video timestamps.

## Visual layout

The final renderer keeps the external video full-frame and puts every overlay
flush against a canvas edge. There are no decorative outer margins:

- The currently selected grasp is a fixed object+hand 3-D view in the top-left
  corner. Its object orientation is the world-frame planning pose estimated for
  that episode, and the hand uses the exact selected grasp transform. Its view
  is fixed to calibrated camera `25322649`: each episode's `extrinsics.json`
  (Charuco world→camera) and `C2R.npy` are combined as
  `inv(C2R) @ inv(world_to_camera)`. The renderer preserves that camera's
  forward/up axes and calibrated vertical field of view, while translating the
  virtual camera only along its viewing axis so the object and hand fill the
  square card.
- The right edge contains one narrow, opaque-white history column above the
  pipeline bar. Four square grasp cards fill that region exactly, without gaps
  or overlap with the black bar. All judged executed grasps enter the same
  chronological stack without changing pose. Render backgrounds remain true
  RGB `(255,255,255)` from the original PNG render (Open3D post-processing is
  disabled), and cards are never stretched. The column shows only the newest four;
  older cards slide past the bottom and disappear, without an overflow count.
  Object meshes use the same neutral untextured gray material in every card.
  A green check/strip or red cross/strip on each card carries the result.
- Planner-rejected candidates and unjudgeable validations do not enter the
  executed-grasp history.
- A fixed neutral diagram spanning the entire bottom edge uses the paper's
  terminology: `POSE EST. → SELECT → APPROACH → GRASP → LIFT & HOLD → LABEL →
  PLACE → RESET`; only the current stage becomes
  brighter. This replaces transient text for ordinary pipeline actions.
- Only physical rotation and reorientation receive the full-frame translucent
  black overlay and centered message. A recovery message is held for at least
  1.5 output seconds unless another recovery event supersedes it.
- The current grasp has no status-colored border. Green and red are reserved
  for result marks and result strips.

The chosen plan now saves `grasp_pose.npy`, `pregrasp_pose.npy`, and
`wrist_obj_local.npy` in addition to `wrist_se3.npy`. This preserves the exact
post-symmetry grasp that the robot executed. Older runs are reconstructed from
the selected world wrist and the exact planning object pose in
`artifacts/scene/<attempt>.json`; perception-frame `pose_world.npy` is not used
for this conversion.

## Render a complete run

The normal workflow is available as one command. A bare video filename is resolved under `~/Downloads`, and the newest matching run is selected unless `--run-id` is supplied:

```bash
python scripts/render_pipeline_experiment.py \
  --exp-name v8_video_13 \
  --obj apple \
  --video 1000085551.mp4
```

The wrapper rebuilds the edit package, force-renders every grasp from calibrated camera `25322649`, composites at 5x, performs per-grasp calibration and white-background checks, verifies the Enter-anchor residual, fully decodes the MP4, and writes `edit/render_verification.json`. Use `--reuse-grasp-renders` only when the existing PNGs are known to be current, or `--verify-only` to check an existing output. Manual result corrections remain explicit through `scripts/correct_pipeline_outcome.py` so a human decision is never inferred automatically.

The normal external-camera convention uses only the placement Enter press as
the origin and an explicitly known playback rate. It never infers speed from
the end of the camera file, because the camera and pipeline need not stop
together. The default is 5x and can be overridden with `--speed`:

```bash
/home/robot/anaconda3/envs/autodex/bin/python \
  scripts/render_pipeline_video.py \
  --experiment /path/to/experiment/<exp_name> \
  --video /path/to/external.mp4 \
  --speed 5
```

Before composition, the command renders every `grasp.selected` episode to
`edit/grasp_pose_renders/` using the real object mesh, selected hand URDF,
episode-specific object world pose, exact wrist transform, and exact hand pose
when available. Camera-specific cache names include `cam25322649`; use
`--force-grasp-renders` only when geometry or calibration inputs changed. The
renderer refuses to substitute a catalog standing pose when the episode
planning pose or camera calibration is missing.

For fixed-speed external footage, `initial_object_placement.end` (the placement
Enter press) is mandatory and maps to video time zero by default. The explicit
`--speed` value supplies the slope; the video endpoint never changes it.
`pipeline_video.json` records the anchor UTC time, the exact affine equation,
its residual, and mapped selection/result checkpoints under `sync_audit`.

By default, a consecutive group of interrupted final episodes is removed from
the output at the first interrupted episode boundary. Here “interrupted” means
`aborted`, `pipeline_closed`, an unhandled exception, or a keyboard interrupt.
A normally completed Charuco/grasp failure is retained and moves to the shared
history column. Use `--keep-trailing-interruptions` to preserve interrupted tail
footage, or `--min-trailing-interruptions N` to require a longer interrupted
tail before cutting.

`edit/storyboard.json` contains exact pixel rectangles, colors, segment
in/out times, and artifact references. `edit/storyboard.html` is a quick
timeline audit; `markers.csv` is the handoff for an NLE or rendering script.
