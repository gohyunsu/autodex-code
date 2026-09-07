# box_to_tray — pick out of a container, put it somewhere else

Two operator turns, one script.

```
turn 1   (goal steps) place the object where it should END UP
                                     -> Enter -> its pose is recorded, then you
                                                 move it to its start position
turn 2   everything on the table     -> Enter -> containers measured, then the
                                                 steps run: grasp, lift, carry,
                                                 place, retreat, home
```

With `--auto-steps` the containers are measured in that same pass, so there is
no container-only turn. Without it, the containers are measured first, in a turn
of their own, and each step then asks you to place its object.

Each step names an object, where it is picked from, and where it goes. The
sequence repeats until you quit, so one container measurement, one camera
session, one planner and one robot connection serve the whole video.

## Run

```bash
# apple back onto the pose it was shown at, then banana off the table into the bowl
/home/robot/anaconda3/envs/planner/bin/python src/demo/box_to_tray/run_demo.py \
  --fixture bowl=smallbowl \
  --step apple:bowl:goal --step banana:table:bowl \
  --auto-steps --execute

# single-step shorthand: one object, out of open_box, onto the tray
/home/robot/anaconda3/envs/planner/bin/python src/demo/box_to_tray/run_demo.py \
  --obj attached_container --execute
```

`SOURCE` is a fixture name or `table`; `TARGET` is a fixture name or `tray`.
Per step, `Enter` runs it, `s` skips it, `r` re-measures the containers, `q`
quits.

With `--auto-steps` the whole sequence runs from a single Enter instead. Use it
when every object is already on the table at the start and nothing is placed
mid-take.

```bash
... --fixture bowl=smallbowl --step apple:bowl:tray --step banana:table:bowl \
    --auto-steps --execute
```

### the three targets

| | `goal` | container (`bowl`, `box`, ...) | `tray` |
| --- | --- | --- | --- |
| aiming | the pose measured in turn 1b | measured bearing of that container | relative `--place-turn-deg` sweep of J0 |
| height | the contact height it was measured at + `--goal-clearance` | its measured rim + `--into-gap` (10 cm), reached by the lift | `--tray-top-z` + `--lay-down-clearance` |
| motion | carried onto the goal xy/yaw, then set down | direct release from the carry height | laid down on the surface |

A container step never calls `place()`: the hand opens at the carry height. That
also removes the fixed 10 cm pre-place ascent + descent `FrankaExecutor.place()`
performs around every release (`PLACE_VERTICAL_TRAVEL_M`), which is what made
the bowl drop look like a pointless up-down jog. A `goal` or `tray` step still
uses `place()`, because there the object is set down precisely rather than
dropped.

A container step lifts higher than `--lift-height` when the rim needs it
(`rim + --into-gap - table`, capped by `--max-lift-height`), because the drop
happens *from the carry height*: the hand never descends into a container it is
not picking from.

A `goal` step reproduces the recorded rotation about z and keeps the attitude
the object was actually picked with (`--goal-pose-mode full` demands the whole
recorded rotation instead, which is harder to reach). The tray is deliberately
not perceived: it is thin, it sits where the turn points, and only its surface
height matters.

Every step ends the same way regardless of target: release, straight up
`--retreat-h`, then home over a raised virtual floor.

| flag | meaning |
| --- | --- |
| `--fixture NAME=ASSET` | a container to measure, e.g. `bowl=smallbowl`. Repeatable |
| `--step OBJ:SOURCE:TARGET` | one pick, run in order. Repeatable |
| `--prompt NAME=TEXT` | SAM3 text for an object or container (default: its name with underscores as spaces) |
| `--into-gap 0.10` | release height above a target container's rim — the object drops the last 10 cm in |
| `--goal-pose-mode yaw` | how closely a goal step reproduces the measured attitude (`full` = the whole recorded rotation) |
| `--goals-json <path>` | reuse recorded goal poses and skip turn 1b |
| `--auto-steps` | run the whole sequence from one Enter (everything is already on the table); the containers are measured in the same pass |
| `--reuse-fixtures` | with `--auto-steps`, measure the containers once instead of every round |
| `--exclude-grasp PATTERN` | drop success-library grasps whose episode path or source contains PATTERN (a bad episode timestamp, or `selected_100` for a whole store). This run only; the shared dataset is untouched |
| `--rim-tolerance 0.02` | a probe landing this far below the rim counts as interior; lower it for a shallow bowl |
| `--fixture-sil-loss-max 0.01` | silhouette-loss limit for a container pose (objects use 0.003; a big open container scores worse) |
| `--fixture-sil-iters 0` | skip silhouette refinement and keep the IoU-selected container pose |
| `--fixture-selection quality` | skip rendering entirely: take FoundPose's own best-scoring view |
| `--place-turn-deg -30` | relative J0 sweep from the measured post-lift pose (negative = clockwise). Not a joint or bearing target |
| `--place-bearing-deg X` | alternative: release at an absolute robot-frame bearing (+x forward, +CCW) and let J0 work out its own sweep — a different number from the turn |
| `--tray-top-z 0.05` | the thin tray's top surface height; the lay-down descends to it |
| `--lay-down-clearance 0.01` | height above that surface at which the hand opens |
| `--max-descend 0.35` | upper bound on the descent; the real one is measured per take |
| `--fixtures-json <path>` | reuse an earlier measurement and skip turn 1 |
| `--no-source-check` | accept a pose outside the source container (off by default) |
| (no `--execute`) | dry run: perceive and plan, send no motion |

The tray is deliberately **not** perceived: it is thin, it sits at the release
bearing, and only its surface height matters.

## Turn 1 — what is measured

`open_box`, `smallbowl` and the other containers are normal AutoDex objects
(mesh under the v8 asset root, FoundPose representation under
`foundpose_assets/`), so a pose comes from the same distributed init pipeline
the picked object uses, with silhouette refinement on. `fixtures.py` then reads
geometry off the posed mesh:

* **interior** — vertical rays are cast down onto the posed mesh. A ray inside
  lands on the box floor, a ray on the rim lands near the top, a ray outside
  misses. That gives the floor height and the interior footprint exactly; a
  silhouette hull could not, because it cannot see a concavity.
* **collision geometry** — by default (`--container-model mesh`) the container's
  own mesh goes into the planning world, which is exact: a round bowl is a round
  bowl. cuRobo drops every *mesh* obstacle from the worlds built with
  `include_obj_obstacle=False` (lift, carry, retreat), so in mesh mode the
  container is not an obstacle during those phases — they are a vertical lift, a
  transfer above the rim and a vertical retreat. `--container-model cuboid` uses
  fitted wall/floor boxes that survive those phases but over-approximate a round
  container; `both` adds each.
* **floor height** — the *low* percentile of the interior probe heights
  (`--floor-percentile`, 5 by default), not their median. A bowl's cavity is
  curved, so the median sits halfway up the curve; using it put the modelled
  floor 2.4 cm above the real bottom, buried the apple resting there inside the
  floor obstacle, and made every approach unplannable.
* **interior region** — the oriented cavity, inset by `--interior-inset`. For a
  step that picks *out of* that container, a pose estimated outside it is
  rejected *before the arm moves*.

The footprint frame is fitted to the posed geometry, never read off the pose
rotation: the box assets stand on their local z and the bowl assets on their
local y, so a pose-derived yaw would rotate a bowl's walls into nonsense. A
round bowl gets a square footprint whose corners are empty air — modelled as
wall, which is conservative rather than wrong.

Written to `fixtures/NN/<name>/` (capture, `pose_world.npy`, `fixture.json`)
and collected into `fixtures.json` at the session root.

## How it relates to the other demos

The motion is `src/demo/inference/run_demo.py::run_once` — unforked. This demo
passes it two optional arguments that default to the old behaviour:

* `scene_cfg_hook` — adds every measured container's walls/floor, puts an object
  picked out of a container on that container's floor instead of the table, and
  gates the pose to its interior;
* `place_surface_z` — replaces the fixed `--drop-h` release with a lay-down: the
  descent is measured from the held object's own mesh bottom down to the target
  surface (tray top, or a container rim), so the object is set down rather than
  dropped from a fixed height;
* `joint0_turn_deg` — commands the J0 sweep itself for a tray step, instead of a
  bearing to end up facing;
* `place_object_pose` / `place_pose_mode` — carries the held object onto a pose
  measured in turn 1b. It overrides the transfer mode, because a goal pose needs
  the xy and yaw control a J0-only sweep cannot express.

Layout: `~/shared_data/AutoDex/experiment/box_to_tray/inspire/<objects>/<stamp>/`
with `fixtures.json`, `goals.json`, `fixtures/`, `goals/`, and one
`takes/NNN_<obj>/result.json` per step.

## When turn 1 says "no reliable <container> pose"

The message names which of the three rejections happened and
`fixtures/NN/<name>/perception.json` holds the full timing/diagnostics.

| reason | fix |
| --- | --- |
| `no_candidates_or_masks` | the SAM3 prompt found nothing — try another `--prompt NAME=...` ("container", "cardboard box", "bowl"), check the container is visible and alone |
| `iou_select_failed` | no candidate view rendered a usable IoU; re-check the pose assets |
| `sil_loss_too_high (...)` | the refinement rejected an otherwise plausible pose — raise `--fixture-sil-loss-max`, or `--fixture-sil-iters 0`, or `--fixture-selection quality`. Always eyeball the pose before trusting one accepted this way |

## Bring-up order

1. `bash scripts/init_daemons.sh start` (FoundPose init daemons).
2. Dry run first: the same command without `--execute` — it measures the
   containers, perceives, plans, and sends no motion. Check `fixtures.json`:
   floor/rim heights and interior footprints must match the real containers
   within a centimetre.
3. Then `--execute`, first take with a clear vertical path over the tray.
