# One-session action loop

`run_live_precision_session()` joins the already implemented AutoDex-camera
ChArUco→socket startup to `SessionRunner`'s existing trial policy. It keeps
the same initialized cameras available across key captures, attempts and
reobservations. It does **not** duplicate v8 grasp ranking, FoundPose,
cuRobo planning, label policy or the guarded execution boundaries.

Its handler factory receives the live session after the socket/collision
world is frozen. It returns an action-name → callable mapping. Each callable
receives `(runner, decision)` and completes the entire indicated boundary
using the existing demo functions before returning. For example:

| Runner action | Handler's existing boundary |
| --- | --- |
| `capture_fresh_key`, `reobserve_key_and_preflight` | `prepare_next_live_key()` with the open camera context, exposure-time robot feedback and the same frozen runner. A failed physical grasp appears in `runner.attempted_candidates`, so the next preflight excludes it. |
| `continue_candidate_preflight` | `runner.preflight_next_key()` on the *same* saved key capture and measured state, continuing a budget-limited prefix rather than falsely declaring pose exhaustion. |
| `execution_gate_required` | `runner.begin_selected_attempt()` and the separately commissioned `execute_bound_pickup()` boundary. A planning pass itself never writes a positive grasp label. |
| `await_lift_observation` | Complete the measured lift, paired-camera Qwen lift checkpoint and `runner.observe_stage("grasp_success", ...)`. |
| `postlift_observed_preflight_required`, `held_relation_evidence_required` | Obtain the post-squeeze key/hand relation or its explicit uncertainty bound, then call the matching `runner.prepare_*postlift_transfer()` method. |
| `transfer_execution_gate_required` | Execute the bound held transfer, capture the pre-insertion hold, and record the separate `preinsert_reached` label. |
| `await_guarded_insertion_and_observation` | Prepare the bound axial handoff, run a commissioned guarded stroke, then use paired Qwen views plus measured depth/alignment/force in `runner.prepare_observed_insertion_label()`; only its checkpoint may set `insertion_success`. |
| `guarded_withdrawal_then_xy_assessment` | Withdraw the still-held key and use the existing native-pixel multi-view cylinder landmark/XY diagnostic; replan and record a bounded retry. Do not turn a VLM suggestion directly into a motor command. |
| `await_retry_execution_and_observation` | Execute the bound retry path and record a *new* insertion checkpoint. |
| `preflight_repose`, `continue_repose_seed_preflight`, `preflight_repose_release`, `preflight_alternative_repose_cell` | Use the same captured key and `runner.preflight_repose()` with directed v8 reset seeds; a staged seed alone is not a plan. |
| `repose_execution_gate_required`, `await_repose_observation` | Execute a commissioned reset plan, observe the landed tabletop pose, and record `reorient_success` before a fresh key capture. |

`stop_for_review` and `hold_for_supervised_completion` are terminal for this
loop. Recovery/reset actions not listed above still need explicit handlers;
otherwise the result is `needs_handler`. There is no default robot-motion
handler. A `max_steps` limit returns `step_budget_exhausted` instead of
silently continuing forever. If a handler returns without changing the
decision, the loop stops with `decision_unchanged` rather than repeating a
physical action. For a budget-limited candidate prefix, the handler should
continue that same saved capture until the runner's decision changes, or let
an operator inspect the unchanged decision. Handler exceptions propagate while the camera
context closes, leaving the runner's already written evidence intact.

Programmatic use from the repository root (after providing commissioned
camera, planner, feedback, Qwen and motion adapters):

```python
from pathlib import Path
from precision_insertion.live_orchestration import run_live_precision_session

def build_handlers(live):
    # Each function takes (runner, decision) and calls the named existing
    # capture/planning/execution/observation boundary. Supply only handlers
    # implemented and commissioned on this AutoDex PC.
    return {
        "capture_fresh_key": capture_and_preflight,
        "reobserve_key_and_preflight": capture_and_preflight,
        "await_lift_observation": lift_and_observe,
        # Add each remaining required action before unattended operation.
    }

result = run_live_precision_session(
    launch_config_path=Path("/absolute/commissioned_session_launch.json"),
    endpoint_catalog_path=Path("/absolute/current_endpoint_catalog.json"),
    run_output_dir=Path("/absolute/new_trial_run"),
    max_xy_retries=2, build_handlers=build_handlers,
)
print(result.to_record())
```

The names `capture_and_preflight` and `lift_and_observe` above are **operator
integration placeholders**, not provided functions. This API connects the
stage order and camera lifetime; it does not claim a packaged hardware
executor or a complete live trial. A passing offline endpoint catalogue and
synthetic plan are not physical-success evidence.
