# Last run and downstream code audit (2026-10-20)

This document audits the earlier 2026-10-17 stopped session and the code as
it stood before the later 2026-10-20 trial. For the latest live trial, the
384×288 setting, and its pending positive-scene validation, see
[TRIAL_20261020.md](TRIAL_20261020.md).

This saved-evidence review compares the active board-9, white cylinder,
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
| Lift verdict | `auto_label_charuco` checks whether the board is uncovered after lift | Five capture-PC Qwen workers compare same-camera initial/after-lift pairs; first three consistent replies can finish the label | This task-specific visual label remains; the saved distributed positive-scene replay took 1.17–1.64 s warm, but live robot latency is unmeasured. |
| After a positive lift | Stock proceeds to its place/reposition sequence | Reconstruct nominal v8 key-in-hand hypothesis from the selected candidate, screen the measured Inspire hold, replan transfer and axial paths from measured joints | No completed live timing. The relation is a hypothesis, not a measured held-key pose. The repeated path is different because the actual lift endpoint and hand joints are inputs. |
| Transfer and preinsert | Stock places; it has no socket preinsert stage | Stock FR3 follower with measured arrival, fresh camera exposure, five-worker Qwen preinsert label with early agreement | No completed live timing. No absolute 120 s photo-age veto remains on the active cylinder path. |
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

Validation after the original audit was 1033 passed, 4 skipped. No live
run was made. The current code does not resolve the operator-observed choppy
FR3 approach; stock follower behavior was previously restored after that
trial.

## Follow-up: active path after a positive lift

The 2026-10-17 run stopped in `key.model_init`, so no physical transfer or
insertion timing exists. The previous completed lift reached the label but
was stopped during its serial Qwen calls. The following is a code-path audit,
not a claim that transfer or insertion has completed on hardware.

| Active step | Stock AutoDex equivalent | Extra work and decision |
| --- | --- | --- |
| Lift image and label | `run_auto.py` saves `label_at_lift` and calls `auto_label_charuco` | The white key needs a paired initial/after-lift raw visual label; five independent Qwen workers are task-specific. The first three consistent replies can finish the label, which still requires two supporting views. |
| Positive-lift handoff | Stock uses the selected v8 grasp and proceeds to place from the reached wrist | `nominal_postlift.py` reuses the selected v8 wrist relation, takes measured FR3/Inspire joints, screens the held endpoint, and replans transfer plus socket-axis path. Re-selection and source binding are small compared with planning and prevent a different grasp relation from being used. |
| Held path audit | Stock checks its own planning/trajectory and endpoint | This socket task checks sampled key/hand versus socket/table geometry. The exact AABB skip preserves collision results and cut the saved 804-sample replay from 99.889 s to 24.241 s; live time is unmeasured. |
| Preinsert | No socket-preinsert stage in stock | Fresh same-camera after-lift/preinsert image pairs and measured arrival precede insertion. The old absolute photo-age veto was removed; only capture order and feedback identity remain. |
| Axial and final | Stock has no insertion stage | Stock 12 N force stop and immediate ascent on contact, then release, clear view, initial/final paired Qwen label, and saved XY diagnosis on a judged miss. These are task semantics; no downstream live times have been recorded. |

`verify_lift_checkpoint` was being called after saving the report, during
nominal post-lift planning, and again while validating the saved plan. It
re-read and decoded the same selected camera PNGs each time. The first strict
check now stores its result in the existing session-local evidence cache;
subsequent calls hash the small report file and reuse the validated record.
Offline replay still performs full byte and pixel verification. A changed
report invalidates the cache. This removes redundant live NAS reads without
changing any visual verdict or motion gate. A direct six-image read/hash/decode
of the saved camera pairs took 11.703 s cold and 0.237 s on each of two warm
repeats on this PC; this is a component benchmark, not a measured full
checkpoint latency or a guaranteed live saving.

The old 120/180-second visual age gates, post-lift hand-command drift abort,
and separate torque/force cutoffs were already removed as described in
`FAILURE_GATE_AUDIT_20261017.md`. Current measured-start, collision,
controller completion, force-stop retreat, and exposure-order checks protect
the next physical action or the meaning of its visual label. The audit found
no remaining active downstream timeout or extra contact threshold whose
removal would be justified by the saved run.

### Saved-image latency probes

The same three real before/after camera pairs from run
`20261017_142332_429150` were replayed with the production Qwen3-VL-2B
backend, three independent per-camera prompts, and a 128-token cap. Model
load is excluded from the inference sums. All six responses parsed as `held`
with the matching camera ID. That single positive-labeled scene did not establish
miss/slip or final inside-socket accuracy at the smaller size. This paragraph
describes the earlier 768×576 runtime; the current 384×288 setting and
negative-scene replay are in [TRIAL_20261020.md](TRIAL_20261020.md).

| VLM input per image | 24122734 | 25305463 | 25322651 | Three-call sum |
| --- | ---: | ---: | ---: | ---: |
| 768×576 | 17.572 s | 19.721 s | 18.120 s | 55.413 s |
| 512×384 | 8.575 s | 8.302 s | 8.467 s | 25.344 s |

The former full-frame live run took 358.147 s for three calls. The 768×576
and 512×384 rows are read-only replays and should not be added to an unmeasured
motor/camera/network duration. A smaller global input would also affect the
final socket judgment, which this evidence does not validate.

The exact 40 saved after-lift PNG arrays (20 undistorted and 20 source
originals; 125,829,120 pixels) were re-encoded to separate temporary
directories on this NAS with the production writer. For three runs each,
4 workers took 20.032, 39.338, and 15.363 s; 8 workers took 16.893, 18.684,
and 15.529 s. Medians were 20.032 and 16.893 s. The default is now 8 workers;
the operator can still set `PRECISION_CAPTURE_IO_WORKERS=4` if NAS contention
reverses that result. Storage and camera-load variability mean this is not a
guaranteed per-episode saving.

### Capture-PC VLM route

The distributed route is implemented in this checkout. Each capture PC has
an isolated Qwen3-VL runtime and a local 4.25 GB model copy. The robot PC
sends the original same-camera prompt and ordered PNG inputs over one
persistent SSH process per capture PC. It receives only text and timing;
FoundPose and the 20-camera acquisition remain on their existing paths.

| Capture PC | Fixed VLM camera |
| --- | --- |
| capture1 | 24122734 |
| capture2 | 25305463 |
| capture3 | 25322638 |
| capture5 | 25322649 |
| capture6 | 25305461 |

The same five cameras are selected for lift, temporal preinsert, insertion,
released final, and XY/axis diagnostics. Classification uses five concurrent
requests; a decisive agreement from the first three completed replies lets
the checkpoint proceed while the other requests drain. Conflicting or
unobservable replies wait for additional completed views. XY and original
pixel-coordinate axis observations use all five in parallel because their
metric resolver needs the individual evidence. Classification inputs were
768×576 at the time of this replay. They are now 384×288; native coordinate
grounding keeps original undistorted pixels.
Saved per-camera replies retain the prompt, source frame IDs/times and
original evidence, and checkpoint replay uses the exact early-completion
camera set.

A read-only replay of the real 2026-10-17 before/after lift images loaded
all five capture-PC models concurrently in 9.21 s. Three warm repetitions
returned the same `held` consensus in 1.636, 1.193 and 1.170 s wall time;
each request used two 768×576 images. This measures a saved positive scene,
not live camera transport, robot motion, other checkpoint accuracy, or
concurrent FoundPose GPU contention. The workers' JSONL log also records
per-view CUDA peak and inference duration. The current static `--check`
passes without opening cameras or moving the robot.

### FR3 speed profile in this checkout

The stock FR3 follower uses the minimum distance between the current
hand/palm/finger link origins and the transformed target mesh surface. Its
joint state comes from the live arm; the hand shape comes from the planned or
last commanded hand joints. A mesh query runs asynchronously; the last
completed distance is used at the next 50 Hz control tick. If mesh/FK is
unavailable, the follower uses its wrist-to-object-reference fallback. The
free-hand scale is 1.0 at or beyond 30 cm, linear from 30 to 15 cm, and
stock 0.4 within 15 cm. Every held-object move has the same stock 0.4 cap.

This demo now configures 0.5 for the near-object and held-object cap. The
30/15 cm thresholds, joint velocity maximum, 50 Hz reference follower,
velocity smoothing and 4 rad/s² commanded acceleration limit remain as in
the stock executor. This raises the cap by 25%; it does not prove a 25%
reduction in physical motion duration. The follower still stops its stream
at trajectory segment boundaries, so visible pauses can remain. No live trial
had been completed when this paragraph was written. The subsequent live trace
and follower update are documented in [TRIAL_20261020.md](TRIAL_20261020.md).

Final local validation: `bash demo/precision-insertion/run_robot_pc.sh --check`
passed as a static file check, and
`/home/robot/anaconda3/envs/planner/bin/python -m pytest -q
demo/precision-insertion/tests` reported 1042 passed, 4 skipped. Neither
command connects to a camera or moves the robot.
