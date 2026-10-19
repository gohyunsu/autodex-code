# Operator-stopped trial, 2026-10-17 14:45 KST

Run `20261017_144530_121497` reused the fixed board-9/socket session evidence.
The operator reported that the FR3 approach moved in visible start-stop steps
and used the emergency stop. The operator subsequently released the stop.
The live process was interrupted and exited with code 130. Its final log says
`stop_streaming done`; no `run_live.py` process remained. No further robot
motion was commanded.

The trial reached key perception (20 camera mask/pose responses) and candidate
`table/0/20237`. It stopped during the pickup approach, before grasp, lift,
VLM classification, transfer or insertion. The episode recorded
`pickup_execution_failed` with `FR3 stream did not finish at a measured
stationary endpoint`; its controller stop acknowledgement was true. The
attempt's physical key state must be checked by the operator before any future
trial. This run does not test the new three-view VLM live.

The saved timing trace at
`/tmp/precision-insertion-timing/20261017_144530_121497.timing` reports:

| Operation | Elapsed |
| --- | ---: |
| Total live session | 203.371 s |
| Geometry path audit | 105.210 s |
| v8 pickup planning | 15.409 s |
| Pickup approach, incomplete | 40.417 s |
| VLM inference | 0 s; not reached |

The controller's periodic print showed `ref_rate` around 1–3 planned
waypoints/s while the nominal rate was 40–100/s. The reference was held back
by the existing `max_lead=0.12 rad` rule. The planned path itself had smooth
neighboring joint samples (maximum 0.031 rad in 7D norm), but this does not
prove that the executed motion was smooth. The old `arm 53/158` print was a
hand-command progress heuristic, not a measured arm path index; the print is
now labeled `hand_idx` to prevent that confusion. The uncommissioned
reference/hand catch-up helper added in this checkout was reverted to the
stock follower logic after this event. It could jump the reference or hand
index across waypoints, and there is no evidence it improved this run.

The emergency stop and interruption overlap with the terminal stall, so the
saved logs do not isolate the cause of the final 0.102 rad endpoint residual.
No claim of a repaired controller or safe restart is made from this replay.

## Operator-requested shutdown of the restart

The restart launched as `20261017_145731_580842` completed the startup
clear-view motion (4.113 s), opened all five camera transports, and loaded
the Qwen model (9.003 s). The operator then requested immediate shutdown
while `key.model_init` was still running. The live process was terminated;
there is no completed timing summary or episode for this interrupted run.
No pickup approach, grasp, lift, transfer, or insertion was recorded in its
timing events. The absence of later events is a limit of the record, not a
measurement of those stages.

After process exit, `manage_robot_pc_services.sh stop` stopped the five
capture hosts' camera, snapshot, and precision daemons. The separately
started Franka Docker container was also stopped. All corresponding capture
ports (5480, 6894, 15006, 15007, 16893), Franka ports (5555, 5556), and
running Docker containers were checked absent.
