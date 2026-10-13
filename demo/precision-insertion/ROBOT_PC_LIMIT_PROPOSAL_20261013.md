# Board 9 / cylindrical 20 mm socket: proposed live values

Status: **proposal for operator approval, not an approved robot configuration**.
This file is not loaded by `run_live.py`. It does not assert that insertion,
camera/robot clock alignment, contact stopping, or recovery has passed a
physical test. Keep the current board-9 session and fixed white socket/key.

## Use the latest AutoDex values where they exist

| Item | Value | Basis |
| --- | ---: | --- |
| FR3 `arm_speed_scale`, `traj_speed` | `1.0`, `1.0` | Stock `FrankaExecutor` defaults; the precision core binds the stock speed-profile planner. |
| FR3 `joint_vmax`, `ctrl_dt` | `1.2 rad/s`, `0.02 s` | Stock `FrankaExecutor` defaults. Do not add a separate reduced speed. |
| Insertion contact stop | `abs(wrist-Z wrench) > 12 N` | Stock FR3 place threshold, used by `FrankaForceStopInsertionAdapter`; on contact it stops and follows the saved axial path upward before failure recovery. |
| Socket radial clearance | `20 mm` | Task geometry: key radius 15 mm, bore radius 35 mm. This is **radial clearance**, not insertion depth. |
| Preinsert clearance, verification stroke | `30 mm`, `20 mm` | Task geometry. Full 50 mm seating is outside this trial. |
| Axial planning waypoint | `5 mm` | Current demo planning bound, at most 5 mm. |
| Failed-trial XY update | weight `1.0`, at most `10 mm` per trial | Goal contract; update only the next attempt from saved multiview observation. |

## Proposed additional limits

These thresholds are for the precision demo's evidence binding and measured
motion boundaries. They are provisional because there is no approved site
limit record or measured held-motion trace. Their names match the current
dataclasses and JSON fields.

| Group | Field(s) | Proposed value | Reason |
| --- | --- | ---: | --- |
| `feedback` | `max_arm_hand_skew_s` | `0.10 s` | Read-only probe measured about 0.032 s. |
| `feedback` | `max_sample_age_s`, `max_arm_update_wait_s` | `1.0 s`, `0.20 s` | Fresh state without requiring an exact packet instant. |
| `feedback` | `max_hand_command_error_raw`, `max_arm_velocity_rad_s` | `30 raw`, `0.05 rad/s` | Probe saw 14 raw and 0.00090 rad/s while stationary; 0.05 matches the completion-check stop bound. |
| `sampling` | `period_s`, `max_samples` | `0.01 s`, `2048` | About 20 s retained at the requested sampling rate. |
| `sampling` | `max_bracket_span_s`, `max_key_state_skew_s` | `10.0 s`, `5.0 s` | Actual board-9 socket request bound was 7.582 s; the midpoint can be 3.791 s from either end. Robot must remain stationary throughout the entire bounded request. |
| Raw camera checkpoints | `max_capture_skew_s`, `max_joint_frame_skew_s` | `10.0 s`, `5.0 s` | Admit the measured bounded-request interval and its midpoint-to-end time; use only with same-request frame IDs and an exposure-bracketing stationary feedback buffer. |
| `sampling` | `max_arm_hold_drift_rad`, `max_hand_hold_drift_raw` | `0.01 rad`, `30 raw` | Admit small stationary variation while checking all samples in the request interval. |
| `sampling` | `max_sampling_hand_command_error_raw`, `max_sampling_arm_velocity_rad_s`, `warmup_timeout_s` | `200 raw`, `2.0 rad/s`, `2.0 s` | The sampler records normal motion; exposure lookup still enforces the stationary limits. 2.0 rad/s exceeds stock 1.2 rad/s command cap. |
| `PathAuditLimits` | `max_joint_step_rad`, `max_wrist_step_m`, `max_wrist_rotation_deg` | `0.12 rad`, `0.015 m`, `8 deg` | Existing exploratory full-chain sampling settings; stock planner collision checks still apply. |
| `PathAuditLimits` | `goal_position_tolerance_m`, `goal_rotation_tolerance_deg` | `0.005 m`, `5 deg` | Existing exploratory goal settings. |
| `PathAuditLimits` | `axial_lateral_tolerance_m`, `axial_rotation_tolerance_deg` | `0.005 m`, `5 deg` | 5 mm path lateral error plus 80 mm key length at 5 deg tilt is about 12 mm, below the nominal 20 mm radial clearance; actual perception/grasp error remains unmeasured. |
| `PathAuditLimits` | `minimum_hand_clearance_m` | `0 m` | Require non-penetration with the modeled mesh; do not add an arbitrary stand-off distance. |
| `LiftExecutionLimits` for lift/transfer/insertion end checks | `max_start_age_s`, `max_end_state_age_s`, `max_execution_feedback_gap_s` | `1.0 s` each | Permit normal post-command feedback while requiring a new sample. |
| `LiftExecutionLimits` | `max_start_joint_error_rad`, `max_end_joint_error_rad` | `0.02 rad` each | Match the completion-checked stock follower's terminal joint bound. |
| `LiftExecutionLimits` | `max_hand_drift_raw`, `max_arm_hand_skew_s`, `max_hand_command_error_raw`, `max_arm_velocity_rad_s` | `50 raw`, `0.10 s`, `30 raw`, `0.05 rad/s` | Consistent with the feedback and measured-hold proposal. |
| `LiftExecutionLimits` | `max_execution_duration_s` | `90 s` | A generous command deadline; no reduction to stock trajectory speed. |
| `held_motion_monitor` | `max_force_norm_n`, `max_torque_norm_nm`, `max_hand_tracking_error_raw`, `max_state_gap_s`, `minimum_samples` | `20 N`, `3 Nm`, `50 raw`, `0.5 s`, `3` | Noncontact lift/transfer stop thresholds. Probe's stationary wrench was about 1.5 N and 1.0 Nm; these are starting values, not measured dynamic limits. |

The installed ParaDex daemon source and matching installed binary have a
default `200 ms` streaming command-expiry watchdog. The live command line did
not override it. No real timeout, disconnect, E-stop, or dynamic contact test
was conducted by this inspection. `LiftExecutionLimits` currently also asks
for a separately reviewed watchdog record (`max_watchdog_command_age_s`,
`max_commissioning_record_age_s`) before built-in lift/transfer; none exists.
Do not create a record claiming those tests passed. Aligning this built-in
noncontact path with the stock AutoDex follower still requires a code change
and review. The stage factory and its remaining site-specific camera, pickup,
recovery and interlock parameters must also be supplied before a full live run.

Evidence: `task_geometry.json` in the staged 20 mm fixture; the read-only
`feedback_only_probe_post_home_20261013.json`; and the board-9
`strict_camera_only_static_interval_20261013_143002` camera manifests under
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392/AutoDex/precision_insertion/cylindrical`.
