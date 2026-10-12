# Cylinder insertion: prompt and axial-control contract

This document distinguishes the proposed short English prompts from the
currently wired runtime. The VLM reads **raw, undistorted AutoDex camera
images**. CAD overlays are saved for humans, never sent as visual evidence.
The cylinder key and socket in the real-camera NAS shots are white; the old
square-demo text `blue precision key ... red socket` is not a cylinder prompt.

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

## 2. VLM checkpoints: exact per-camera input and proposed short text

Send one camera's image(s) per request to the default local Qwen backend;
aggregate parsed votes across cameras afterward. Label the image order in
the prompt. Require JSON only, check the field set and enum values, and save
the model's raw answer, parsed answer, camera ID, source image hashes and
prompt version. The short strings below are the **model-facing core**;
transport code can add image dimensions and JSON schema constraints.

| Stage | Images, in order | Model-facing question | JSON fields | Downstream use |
| --- | --- | --- | --- | --- |
| Lift | `before_grasp` (initial tabletop capture), `after_lift` from the **same camera** | `Is the same white peg visibly held after the lift? Do not infer from closed fingers or occlusion.` | `{"label":"held|miss|slip|unobservable","evidence":"..."}` | Grasp outcome; if not held, do not start transfer. |
| Transfer / preinsert | `after_lift`, `preinsert` from the **same camera** | `Is the same peg still held above the socket, with a roughly compatible axis? Judge visible pixels only.` | `{"label":"coarse_match|gross_misalignment|slip_or_miss|unobservable","evidence":"..."}` | Detect lost key or gross arrival error; not a metric XY estimate. |
| Final released state | `initial_tabletop`, `final_clear_view` from the **same camera**, after opening the hand and clearing the view | `Is the peg visibly inside the socket now? Disappearance or an empty hand is not proof.` | `{"label":"inside_socket|outside_socket|unobservable","evidence":"..."}` | Visual task label; this is **not** a claim of measured 20 mm depth. |
| Failed trial, next-trial XY | The already saved **raw preinsert** image from each camera; no post-failure recapture | `Mark the visible insertion-tip center and two separated points on the peg axis. Use image pixels; return null for hidden features.` | `{"tip_px":[x,y]|null,"axis_line_px":[[x1,y1],[x2,y2]]|null}` | Calibrated multi-view geometry computes socket-frame millimetres. The VLM never outputs robot-frame millimetres. |

For label requests, the implementation may retain `evidence_views` because
the current observer validates it; the JSON snippets above omit that
transport field only to show the question concisely. In particular, the
current [`observe_preinsert_raw_views`](precision_insertion/observer.py)
sends **one** preinsert image, so the transfer row is a planned change:
capture/persist after-lift and preinsert using matching camera IDs and call a
two-image version of the same label contract. The current released-state
pair is already wired in [`observe_released_insertion_visual`](precision_insertion/observer.py).
The existing prompts are longer; shortening them must preserve these
semantics and parser tests, not merely replace string literals.

The pixel-grounding request should use the full undistorted image, or a crop
with its exact origin and scale saved. Validate coordinates against that
image. Convert crop pixels back to original camera pixels **before**
calibrated triangulation. Fuse camera rays/shaft lines with reprojection
residuals, then compute the cylinder's XY displacement in the frozen socket
frame. The saved result affects the **next** physical attempt only:
`offset_next = offset_current + 1.0 * measured_XY_error`, with the existing
per-attempt 10 mm step cap and no cumulative cap. A visually unobservable
key cannot yield a measured tip center; fallback behavior, if used, must be
logged distinctly from VLM-derived metric correction.

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
The current insertion audit already measures progress and lateral drift
relative to `insertion_axis_robot` in
[`path_audit.py`](precision_insertion/path_audit.py). The robot-side
`follow_guarded_insertion` adapter still has to execute the reference and
record actual feedback; the offline planner cannot substitute for it.
