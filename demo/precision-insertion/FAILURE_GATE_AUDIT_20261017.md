# Board 9 / white cylinder / 20 mm: active failure gates

This review covers `run_robot_pc.sh --run` with the built-in FR3 integration
and the board-9 stage factory. It compares the live path with the current
AutoDex v8 perception, planner, and FR3 follower. Offline square and the
historical external guarded-controller APIs are outside this live route.

The Oct 17 trials reached pickup and held lift. They did **not** demonstrate
20 mm insertion or a completed failure return. An operator reported that the
key was physically lifted during the run that ended at the demo's 3 Nm torque
threshold. That statement is a physical observation, not a VLM success label.

| Stage | Stock AutoDex behavior | Former precision-only gate | Current live decision |
| --- | --- | --- | --- |
| Tabletop key | FoundPose candidates selected by cross-view IoU, then silhouette refinement; v8 candidate planning tries the next candidate if needed | Raw cylinder pose spread, absolute tabletop IoU, and pose symmetry could veto a valid refined result | Tabletop cylinder spread and IoU are diagnostics; D-infinity cylinder symmetry is used. The first candidate planning pass is a latency budget, with a continuation over the remaining candidates. |
| Pickup and lift path | v8 approach, squeeze, measured-start lift replan, collision and joint checks; no extra wrist-torque cutoff for the held lift | An independent 3 Nm torque, 20 N force, and 50 raw hand tracking abort; a commanded hand transition could also abort | Removed those three thresholds and the command-transition abort. Actual wrench and hand readings remain in the trace. FR3 reflex, controller stream progress, completion, and measured endpoint still apply. |
| Lift visual | AutoDex compares before-grasp and after-lift images | A 120 s absolute before-image to after-lift-image age cap could reject a long planning interval | Removed the elapsed-time cap. The captures must still bracket the physical lift in timestamp order and use paired camera views. |
| Transfer visual | Stock follower uses the planned held path and measured arrival; it does not invalidate a later photo solely because planning or saving took time | 120 s transfer-to-preinsert photo cap; unused 10 mm / 10 degree diagnostic hand-error limits; dead `measured_hold_pose_outside_limits` recovery branch | Removed the cap, unused limits and dead branch. Transfer completion, new exposure order, aligned robot state, and visible held-key VLM evidence still decide arrival. |
| Insertion path | Stock FR3 `stop_wrench_z=12 N` contact stop, then hold | Added full axial path audit and completion check | Retained because the 20 mm inclined socket stroke and immediate reverse-path ascent require a valid measured starting pose, planned path and a stopped endpoint. Contact is failure, not successful insertion. |
| Final visual | AutoDex saves observation images and a VLM label | 180 s preinsert age and 120 s release-to-final photo cap | Removed both wall-clock caps. Initial, preinsert, contact stroke, release and final frames must remain in causal order; the final label uses paired initial/final raw views. |
| Failure XY | No stock socket XY diagnostic | Extra live-frame age limit in historical retry helpers | Active cylinder path reuses its **saved pre-stroke** snapshot. It does not apply the historical live-frame age gate; camera capture skew and geometry quality still govern whether a metric correction is available. |
| Evidence | Stock run records images and timing | Extensive SHA256 and source-binding checks | Kept source identity checks needed to replay the named camera frames and separate attempts. They are not numerical motion or visual-success thresholds; repeated hashing remains an I/O cost to profile separately. |

## Conditions still capable of stopping a live run

- A real FR3 reflex, daemon stream loss, stalled stock follower, missing
  measured state, or a measured arm endpoint that differs from the commanded
  path. These prevent the next physical command after an unverified motion.
- An actual collision or invalid joint/hand path in the v8 planner and the
  20 mm axial path audit. The axial path check is task geometry, not an
  independent contact controller.
- A camera frame that predates its motion, a frame ID reused across a paired
  comparison, or robot feedback that cannot be paired with the exposure.
  Absolute time since the previous stage is no longer a veto on the active
  cylinder checkpoints.
- VLM `miss`/`slip`, visible gross preinsert misalignment, or a final
  `outside_socket`/`unobservable` result. Those are task observations; neither
  a positive visual label nor a completed 20 mm joint path proves physical
  seating depth.

The stock follower's 90 s motion timeout, its held-object speed cap and
the daemon's 200 ms command expiry were not loosened. The completed endpoint
check is demo-local because stock `_follow` can exit before exact arrival and
then issue a blocking landing; this check stops before another held-key
action if the arm did not reach the planned endpoint. A 0.5 s controller-state
progress check is also retained to avoid driving a stale feedback stream.

Regression tests cover a 200 s planning gap at lift, a 200 s transfer-to-
preinsert observation gap with aligned feedback, a 200 s release-to-final
capture gap, and the removed torque abort. The experimental follower catch-up
change was reverted after an operator-stopped approach; no smoother motion
has been demonstrated. These are offline tests; the next
physical run must report its own result.
