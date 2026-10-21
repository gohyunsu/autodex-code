# Directed v8 repose: execution evidence contract

`SessionRunner.begin_repose_attempt` opens a record only after a complete
pickup, held transfer/descent, opening and post-release exit **preflight**. It
does not command Franka or Inspire. `observe_repose_landing` now rejects an
arbitrary release-file existence check. The caller must supply a JSON file
with schema `precision_insertion_repose_execution_evidence_v1` and a new
tabletop key capture after the entire release/exit sequence. The current
cylinder FR3 integration can produce this record through the stock AutoDex
executor; an external controller may still supply the older producer format.

The stock producer uses `source: "stock_autodex_franka_executor"`. It runs the
saved selected-v8 pickup with `start_from_current=True` and `skip_lift=True`,
checks measured arm arrival after squeeze against the saved held-lift start, then follows the
preflighted held lift, transfer, descent, hand opening, upward exit and
clear-view retract. A bound `stock_feedback` JSON records seven measured
waypoints and a stopped stream acknowledgement. The verifier replays their
order, source hashes and arm endpoints. While a key is held, measured Inspire
finger joints may differ from the nominal closure; fresh hand feedback is
still recorded and validated. After release, measured hand opening and all
thirteen joints are checked. A missed arm waypoint stops before the next
motion; no physical key landing is inferred from these checkpoints. The
verified `stock_feedback` reference is carried into the landing report.

For an external producer, the following contract remains available.

The execution file must identify `source: "commissioned_external_controller"`,
the exact `attempt_id`, selected seed record, frozen session hash, SHA-256 of
the saved repose `report.json` and `planned_trajectories.npz`, plus
`safety_abort: false` and `grip_loss_before_release: false`. It must contain
these seven phases **in order**, each with finite `started_at_s`, later
`completed_at_s`, `complete: true` and `safety_abort: false`:

1. `pickup_squeeze`
2. `held_lift` — `grasp_held: true`
3. `held_transfer` — `grasp_held: true`
4. `held_descent` — `grasp_held: true`
5. `release_open` — `hand_open_feedback: true`
6. `post_release_lift`
7. `post_release_retract`

`source_records` must provide distinct absolute `{ "path", "sha256" }`
references for `trajectory_feedback`, `safety`, `grasp_state` and
`hand_feedback`. These are independent producer records, not the summary
file or preflight archive. The verifier checks their current hashes and
rejects changed preflight inputs, scene or trajectory archive. The supplied
`release_completed_at_s` must equal the `release_open` completion time. The
first exposure in the landing capture must be **after**
`post_release_retract` completes, so an image taken while the hand still
occludes the key cannot count as a landing observation.

The file and source hashes prevent accidental evidence mix-ups or edits; they
do **not** prove that controller assertions are honest, that the robot-side
watchdog works, that the key detached, or that the dynamic drop was safe.
Only a fresh, admitted key pose supported by the board and clear of the
socket can produce the separate `reorient_success` label. A wrong tabletop
class is a failure; a missing release phase or ambiguous capture leaves the
label unknown. The stock callback has offline test coverage only; physical
execution and landing still require staged robot-PC checks and a fresh camera
observation.
