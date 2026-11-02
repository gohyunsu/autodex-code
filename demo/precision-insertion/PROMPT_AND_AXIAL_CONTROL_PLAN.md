# Cylinder insertion: prompt and axial-control contract

This document records the stage semantics and the current runtime. On the
active ZeroDex profile the VLM reads **raw, undistorted capture4 camera
images**. The historical AutoDex profile follows the same evidence rule. CAD
overlays are saved for humans, never sent as visual evidence.
All precision keys and sockets are white. Segmentation prompts must identify
the loose key and fixed socket by shape and location, not a different color.
The runtime rejects stale color words before a SAM request.

## 1. Segmentation before pickup

- SAM3 text prompt for the loose key: `white plastic cylindrical peg`.
- Use the same prompt for every camera and every cylinder gap. Save the exact
  prompt, image hashes, camera IDs, selected mask and confidence per view.
- SAM3 receives a **noun phrase**, not a task instruction or JSON request.
  The session-frozen socket projection/region is checked separately so a
  white socket mask cannot silently become a key pose. FoundPose then fits
  the key CAD to admitted key masks. The segmentation mask alone is not a
  6D key-pose measurement.
- The socket-only session capture needs a separate tested noun phrase, e.g.
  `white plastic cylindrical socket`; this is *not* validated by the key
  probe. Capture the socket without the key on the board.

On 2026-10-13 the active SAM and Qwen object prompts were checked for stale
object colors. They describe both physical objects as white; the other color
words in this demo refer to diagnostic overlays or gray image padding. A
saved live-camera socket-only SAM3 replay using the exact prompt
`white plastic cylindrical socket` produced a mask in 19 of 20 views and no
mask in one. This checks the prompt against this scene, not the mask's full
object coverage or metric pose accuracy.

## 2. Current model checkpoints and exact per-camera evidence

Send one camera's image(s) per request to the default local Qwen backend;
aggregate parsed votes across cameras afterward. Label the image order in
the prompt. Require JSON only, check the field set and enum values, and save
the model's raw answer, parsed answer, camera ID, source image hashes and
prompt version. The short strings below are the **model-facing core**;
transport code can add image dimensions and JSON schema constraints.

| Stage | Images, in order | Model-facing question | Parsed output | Downstream use |
| --- | --- | --- | --- | --- |
| Lift | One undistorted `after_lift` crop centered by measured FR3/Inspire FK, independently for each camera | `Is the small solid white cylindrical peg visibly held by the robot's black fingers in this image? The robot also has white finger shells. Reply only YES or NO.` | One `YES`/`NO` token per view; deterministic aggregation requires at least two `YES` cameras for `held`, treats one `YES` as `unobservable`, and does not run the ambiguous fingertip recrop | Grasp outcome. The initial frame remains saved for provenance, but is not a v5 inference image. |
| Transfer / preinsert | `after_lift`, `preinsert` from the **same camera** | `Is the same peg still held above the socket, with a roughly compatible axis? Judge visible pixels only.` | Same JSON keys; `class` is exactly one of `coarse_match`, `gross_misalignment`, `slip_or_miss`, `unobservable` | Detect lost key or gross arrival error; not a metric XY estimate. |
| Final released state | One socket-centered `final_clear_view` crop per selected camera, after opening the hand and clearing the view | `If the solid key protrudes from inside the circular socket rim, answer INSIDE. If the rim is empty, answer OUTSIDE; if hidden, answer UNCLEAR.` | One class token per view, followed by deterministic camera majority | Visual task label; the initial tabletop frames remain in the evidence bundle but are not sent to the current final classifier. This is **not** a claim of measured 20 mm depth. |
| Same-trial alignment | One dynamically projected crop of the already saved **raw preinsert** image from each ranked socket-pose-inlier camera | Qwen: `Find the WHITE CYLINDRICAL KEY directly held by the BLACK ROBOT HAND... Return the tight bounding box...`; SAM3: `white cylinder` | Qwen box, SAM mask, distal half-ellipse center and visible shaft direction. Calibrated image geometry produces socket-frame XY/tilt; the VLM never outputs robot-frame millimetres. | Move, recapture and repeat inside the same held-key attempt. The same observation is retained for next-trial diagnosis if insertion later fails. |

For closed-set JSON requests, the implementation retains `evidence_views`
because the observer validates it. Use the sole camera's ID for a decisive
per-camera result; use `[]` when unobservable. Lift v5 and the final
socket-crop classifier instead use short class tokens per camera and save the
deterministic aggregate separately. In particular, the active
[`observe_preinsert_temporal_views`](precision_insertion/observer.py) sends
each camera's ordered raw `after_lift`/`preinsert` pair. The saved checkpoint
binds both source manifests, camera/frame IDs, timestamps, PNG hashes, prompt
and raw answer, and replays parsing. The historical
`observe_preinsert_raw_views` remains a single-frame diagnostic only. The
released-state source pair is wired in
[`observe_released_insertion_visual`](precision_insertion/observer.py), while
the current model request uses only the final socket crop. Shortening prompts
must preserve these semantics and parser tests, not merely replace strings.

For the transfer JSON request, the formatter constructs the **literal model
input** like this for each camera, substituting the stage's image names,
question, field and allowed values from the table:

```text
Camera {camera_id}. Image 1: {before}. Image 2: {after}.
{one-sentence question from the table}
Choose exactly one {field}: {comma-separated allowed values}.
Return JSON only with exactly {field}, evidence_views, evidence.
evidence_views = ["{camera_id}"] for a visible conclusion, otherwise [].
evidence = one short phrase about visible pixels.
```

The alignment request uses a single projected crop and the exact normalized
Qwen box schema shown in the table. The crop origin, scale, source-image hash
and undistorted calibration are saved. Qwen coordinates are restored to
native camera pixels before SAM3 and calibrated geometry. The default
endpoint route fuses visible end centers across eligible cameras and measured
motion over successive holds. The optional partial-shape route jointly fits
socket-frame `(x, y, tilt_x, tilt_y)` to the mask-derived end center and shaft
direction; cylinder yaw is intentionally absent. Missing shape geometry
falls back to endpoint/temporal evidence and does not end the episode.

The resulting correction is first used **inside the same held-key attempt**:
one wrist increment is capped at 10 mm, the robot recaptures at the new hold,
and the loop continues until the configured convergence condition or no new
metric correction is available. A single-view planar estimate may be used
only as one active-observation probe; it is returned unless a later true
multiview observation confirms it. If insertion later fails, the same saved
preinsert evidence can update the next physical attempt as
`offset_next = offset_current + 1.0 * measured_XY_error`, again with a 10 mm
per-update cap and no cumulative cap. An unavailable observation is recorded
as unavailable; it is never converted to a zero-error measurement.

## 3. Socket-axis stroke and Jacobian control

The reference path is geometric and independent of the inverse-kinematics
solver. With the session-frozen socket pose `T_robot_socket`, define
`a = R_robot_socket @ insertion_direction_socket`. Let `s` progress from
0 at preinsert to `preinsert_clearance + 0.020 m`; the desired wrist pose
holds orientation fixed and sets `p(s) = p_preinsert + s*a`. The key remains
rigidly attached to the grasp frame through this stroke.

The appropriate local solver is a **damped Jacobian continuation** from the
measured Franka joint state: at short consecutive `s` targets, compute the
6D pose error, solve a damped least-squares update with the 6x7 Franka
Jacobian, seed each step with the last accepted joint state, and check the
sampled joint/whole-hand/key/socket path. Retiming and joint limits are
separate from geometric path accuracy. The external guarded controller
tracks this axial reference with measured joint/force feedback and stops on
its commissioned contact limit. A Jacobian is a kinematic method, **not** a
guarantee of exact physical tool travel or safe contact by itself.

AutoDex's existing
[`plan_jacobian_vertical_stroke`](../../autodex/planner/jacobian_stroke.py)
is hard-coded to pure world-Z and rejects lateral target displacement;
passing a tilted socket-axis target to it is incorrect. Adapt its
continuation/collision-validation machinery to a general measured axis, or
add a sibling axis-stroke planner without changing stock AutoDex behavior.
The demo-local [`axis_jacobian.py`](precision_insertion/axis_jacobian.py)
implements a sibling continuation on the frozen socket axis. Its short
waypoints seed from the preceding accepted joints, and it reuses stock
AutoDex numerical Jacobian, joint limits, collision rollout and retiming.
[`preflight.py`](precision_insertion/preflight.py) calls it after the
non-contact transfer and passes its dense path to the sampled held-key/hand
audit. A successful plan is still only a kinematic reference.
The current insertion audit already measures progress and lateral drift
relative to `insertion_axis_robot` in
[`path_audit.py`](precision_insertion/path_audit.py). The robot-side
`follow_guarded_insertion` adapter still has to execute the reference and
record actual feedback; the offline planner cannot substitute for it.
