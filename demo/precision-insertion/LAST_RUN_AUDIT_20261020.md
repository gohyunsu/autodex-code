# Last run and downstream code audit (2026-10-20)

This review is read-only. It compares the active board-9, white cylinder,
20 mm route in this checkout with the local `/home/robot/AutoDex` main at
`448732d` (2026-09-30) and its shared ParaDex command sender.
No camera, daemon, FR3 controller, or hand was started for this audit.

## Where the last run stopped

Run `20261017_145731_580842` has 45 timing events in
`/tmp/precision-insertion-timing/20261017_145731_580842.timing/events.jsonl`.
It completed clear view (4.113 s), camera transport open (0.304 s), key
renderer warmup (3.570 s), camera stream open (15.853 s), and Qwen model load
(9.003 s). `key.model_init` started at pipeline time 39.198 s; it has no end
event. There is no key capture, candidate plan, pickup, lift, transfer, or
insertion event, and no completed timing summary or episode. The operator
requested shutdown and the process was terminated during that wait.

The console printed stock `InitOrchestrator`'s `sending init (live) ... to 5
PCs` line but not its `init dispatched` line. In both this checkout and stock
AutoDex, `InitOrchestrator.init_object()` calls ParaDex
`CommandSender.send_command("init", wait=False)`. That sender starts one
REQ/REP thread per capture PC and joins **all** of them. `wait=False` skips
remote work completion, not the command acknowledgement. The stock sender's
default 60,000 ms × 3 retries can therefore hold a session for about 180 s
if one PC does not acknowledge. The sender prints timeout/error messages but
does not raise, and the stock orchestrator previously suppressed those prints.
The saved capture-PC logs do not show a new `init` at this run's time, although
their camera logs continued. `manage_robot_pc_services.sh check` only proved
listening ports, not that an `init` RPC would be answered. The record supports
an unresponsive command channel; it cannot identify the exact PC or its
underlying network/daemon fault. The operator stop is the terminal outcome.

This checkout now passes 5,000 ms and two retries to the stock sender. Its
parallel fan-out bounds the command ACK wait to about 10 s, while the same
distributed FoundPose implementation and result wait remain in use. A failed
PC's printed message is surfaced as one warning with its PC name; successful
acknowledgements stay quiet. ParaDex still returns no success/failure value,
so a failed ACK can lead to the existing 120 s payload collection timeout.
That is a remaining diagnostic limitation, not evidence of a camera or robot
failure in the saved run.

## Stock comparison after the completed stages

| Stage | Current stock AutoDex | Active precision insertion route | Decision |
| --- | --- | --- | --- |
| Init and key pose | Five capture PCs receive `init`/`run`; local SAM/FoundPose; robot PC selects/refines multi-view pose | Same orchestrator and v8 assets, with frame identity capture | Keep distributed processing; shorten stalled ACK wait and show only failures. |
| Grasp plan and motion | v8 candidate, cuRobo FR3 plan, stock `RealExecutor` approach/squeeze/lift | Same planning and follower algorithms, with a URDF-path input for the right-angle adapter; adds exact held-key/Inspire vs socket/table sampled audit | Keep socket geometry checks; accelerate exact-pair work with an AABB lower bound. |
| Lift verdict | `auto_label_charuco` checks whether the board is uncovered after lift | Three selected Qwen views compare initial and after-lift frames; response controls the next stage | This task-specific visual label remains; earlier full-size three-view inference was 358.147 s. The saved 768×576 probe was 52.064 s, but the shorter live route has not completed. |
| After a positive lift | Stock proceeds to its place/reposition sequence | Reconstruct nominal v8 key-in-hand hypothesis from the selected candidate, screen the measured Inspire hold, replan transfer and axial paths from measured joints | No completed live timing. The relation is a hypothesis, not a measured held-key pose. The repeated path is different because the actual lift endpoint and hand joints are inputs. |
| Transfer and preinsert | Stock places; it has no socket preinsert stage | Stock FR3 follower with measured arrival, fresh camera exposure, three-view Qwen preinsert label | No completed live timing. No absolute 120 s photo-age veto remains on the active cylinder path. |
| Axial, final, retry | Stock has no 20 mm insertion or XY diagnosis | Stock FR3 12 N force stop, immediate reverse-path ascent on contact, then release/clear-view/final Qwen and saved pre-stroke XY diagnostic on judged failure | No completed live timing. Contact and measured terminal pose are physical outcomes; no separate contact-commissioning gate is used by this stock adapter. |

The active state machine does not drive transfer after a missing/failed lift
label, and it does not claim a commanded 20 mm stroke is physical seating.
The remaining measured endpoint, collision, contact, camera identity, and
ordered-exposure checks protect the next physical command or the meaning of
its task label. Earlier demo-only absolute image-age, hand-drift, and extra
wrench thresholds were removed in the prior review; see
`FAILURE_GATE_AUDIT_20261017.md`. Historical external-controller and square
routes still exist but are not called by this board-9 cylinder binding.

## Measured latency and geometry replay

The last completed grasp/lift run `20261017_142332_429150` spent 108.057 s
in its 804-sample held geometry audit (105–108 s in the two later saved runs).
It spent 358.147 s in three serial full-resolution lift Qwen calls. It did
not finish the lift label and never measured transfer, preinsert, axial, final,
or XY stages. No values for those stages are inferred here.

The exact saved 495 lift, 95 transfer, and 214 descent joint samples were
replayed offline with the Oct-12 right-angle FR3/Inspire URDF and `yourdfpy`
CPU FK. CUDA was unavailable to PyTorch in this audit environment despite
`nvidia-smi` listing the GPU, so this is a CPU FK performance comparison,
not a new cuRobo runtime measurement. The old full exact-pair loop and the
new AABB branch used the same 804 CPU-FK poses and frozen socket/table meshes.

| Geometry replay | Exact Coal pair calls | Geometry time | Collision failures | Minimum-distance pairs |
| --- | ---: | ---: | ---: | ---: |
| Previous exact loop | 22,511 | 99.889 s | 0 | 28 |
| AABB lower-bound skip | 9,925 | 23.502 s | 0 | 28 |

Every one of the 28 minimum distances and the complete failure list matched
**exactly** between the two CPU-FK replays. The final 1 nm conservative
comparison adjustment took 24.241 s and the same 9,925 exact calls. Against
the original saved cuRobo-FK report, pair names and zero failures match; the
largest minimum-distance difference is 2.1965e-7 m (CPU vs GPU FK precision).
The AABB encloses each transformed moving mesh; a sample is skipped only if
its lower-bound gap exceeds the required clearance and cannot improve the
already recorded exact minimum. Near geometry still receives the original
Coal collision, surface-distance, and solid-occupancy checks.

Validation: `pytest -q demo/precision-insertion/tests` gave 1032 passed,
4 skipped before the new command-warning test was added; targeted tests for
that warning, the geometry audit, and session wiring gave 27 passed. No live
run was made. The current code does not resolve the operator-observed choppy
FR3 approach; stock follower behavior was previously restored after that
trial.
