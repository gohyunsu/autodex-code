# Post-lift route review (2026-10-21)

This review compares the active board-9, white-cylinder, 20 mm path in this
checkout with local stock AutoDex `main` at `448732d`. It uses saved runs and
source code only. The original `/home/robot/AutoDex` checkout, cameras,
daemons, and robot were not changed or started.

## Last measured boundary

The 10:10 run `20261020_100938_490865` has a completed pickup and lift but
its saved episode labels `grasp_success=false` and has no post-lift plan,
transfer, preinsert, axial stroke, release, or final image. The operator
reported that the key was physically held and lifted. The new two-view lift
classifier labels this saved scene `held` on three read-only replays, but it
has not been used in a new physical run. The earlier 09:16 run reached a
transfer after a false-positive `held` label and stopped before preinsert.
Neither run supplies live downstream timing or a proven insertion outcome.

## Active code path against stock

| Step | Stock AutoDex | Current cylinder route | Decision |
| --- | --- | --- | --- |
| After-lift relation | The selected v8 grasp's `wrist_se3.npy` remains the nominal object-to-hand relation during place. Stock `run_auto.py` has no fixed-socket relation check. | `live_nominal_relation_handlers.py` reads measured FR3/Inspire joints. `nominal_postlift.py` uses the selected v8 relation as an explicit hypothesis, screens the measured finger shape against the socket, and replans transfer and axial paths. The observed FoundPose and physical-calibration handlers are not selected for live cylinder mode by `runtime_assembly.py`. | Retain measured start and socket geometry. A visually held key does not measure its hidden six-DoF pose. |
| Transfer | `FrankaExecutor._follow` streams a planned path with the held-object speed cap. | `transfer_execution.py` uses the same FR3 connection through the measured adapter, binds the selected path, and checks completion and measured endpoint before another move. This checkout's follower also has the 20 October reference-progress fix; it is therefore not byte-identical to stock. | Retain completion and endpoint checks: the 09:16 trace actually stopped short of its transfer endpoint. |
| Preinsert image | Stock places an object and has no socket preinsert decision. | `live_task_handlers.py` captures the hold; `preinsert_checkpoint.py` compares the raw `after_lift` and `preinsert` image for each of five assigned camera IDs, using first-three decisive agreement when available. The separate measured-hand/socket CAD overlay is a human diagnostic and never enters Qwen. | Keep chronological exposure and measured-state binding. No absolute photo-age veto remains. Start remote Qwen while this PC renders the diagnostic overlay. |
| Axial stroke | Stock place uses a 12 N wrist-Z stop and returns early on contact. | `stock_force_insertion.py` calls that follower along the planned socket axis and reverses the reached path immediately on contact. `guarded_insertion_execution.py` checks the stopped pose and records force abort or completion; no separate contact-commissioning record gates this active adapter. | Retain the planned-path corridor, contact stop, and measured endpoint. Stock has no 20 mm socket insertion or reverse-axis recovery to substitute here. |
| Release and final | Stock opens the hand and retracts after place; it has no inside-socket label. | `live_task_handlers.py` uses a preplanned release/exit, clear-view capture, and initial/final same-camera VLM comparison. `outside_socket` can use the saved pre-stroke image for next-trial XY diagnosis. | Keep the task-specific final observation. Force abort and unknown visual evidence must not invent an XY correction. |

The prior active-route audit already removed demo-only 120/180-second image
age limits, a separate 3 Nm/20 N held-lift cutoff, hand-command transition
abort, and redundant transfer hand-drift limit. No remaining active
downstream numeric gate was shown unnecessary by the saved runs. In
particular, weakening the measured endpoint check would hide the 09:16
short transfer, and dropping the reverse-path corridor could command an
unplanned movement while the key is near the socket. Historical external
controller and square-mode checks are outside this live cylinder path.

## Latency change and verification

The preinsert VLM consumes only saved raw camera pairs. Previously it began
after `build_held_scene_comparison` had loaded the FR3/Inspire visual model,
rendered five CAD overlays, and computed image digests. It now starts on the
capture PCs as soon as the paired raw frames are loaded, while the robot PC
builds that separate diagnostic. The same frames, prompts, verdict rules,
overlay report, and verification remain. Local Qwen still runs serially with
the renderer to avoid competing for the robot PC's GPU memory.

A regression test synchronizes a fake remote VLM and fake renderer to prove
the two jobs overlap, and the existing preinsert tests check the checkpoint
and replay result. There is **no saved live preinsert bundle** in the inspected
runs, so no physical-stage latency reduction is claimed. Per-stage timing
spans will measure the benefit on a later run. Robot motion, camera network
round trip, contact control, and downstream success remain unmeasured here.
