# Capture4 held-key observability review (2026-10-29)

This is a read-only replay of three saved FR3/Inspire pre-insertion states.
The robot and hand were off. It evaluates a possible observation action for
the existing same-trial XY estimator; it does **not** establish that a real
key face was detected at that action, that the key stayed rigidly held, or
that a corrected physical insertion succeeded.

## Should two-camera visibility be an offline grasp filter?

It should be an **online ranking signal**, not a mandatory offline rejection.
The final camera/socket geometry is measured at session start, and the actual
key pose after squeeze can differ from its nominal grasp. More fundamentally,
the distal face of a vertical key points down. In these three saved holds,
the dot product between its outward normal and the direction to each of the
four camera centers is **negative** (about -0.54 to -0.89). No grasp change
alone can make that face directly visible in the same vertical insertion
pose. Requiring two direct face views there would reject the two human
confirmed 5 mm insertions with grasp `table/0/20195`.

Seeing the shaft in two images is a weaker condition. Its visible side can
support an axis/known-length hypothesis, but the stored masks of the two
successful 5 mm episodes frequently selected the white socket or robot hand.
Small mask perturbations also changed the inferred XY substantially. Two
nearby cameras are not enough: `25305462` and `25322642` are separated by
only about 4 degrees at the socket. Prefer cameras that show the **same
physical key feature** with useful parallax. Do not count two positive
object-presence labels as two metric end-center observations.

A fresh replay of the **current** Qwen-box → SAM3 implementation on the two
human-confirmed `20195` insertions found no metric XY point in either case.
The socket-overlap exclusion correctly rejected the `25305462` and
`25322642` boxes (95–99% overlap with the frozen socket projection). The
remaining `25322639` and `26053248` requests returned no key SAM mask.
Both observations therefore abstained instead of issuing the large false
corrections that an earlier single-view shaft-pixel hypothesis suggested.
The replay used the saved original images, current code, local Qwen3-VL-2B
and SAM3; it sent no robot command. Exact per-view prompts, boxes, exclusion
fractions and reasons are in `replay_current_sam_xy_20195_20261029.{py,json,log}`
under the development review directory.

## Alternative observation action tested offline

For each frozen session, the probe takes the 80 mm key CAD tip
`p_tip = T_robot_key [0, 0, 0.08, 1]` and the distal face normal
`n = R_robot_key [0, 0, 1]`. It samples a 40 mm raise along the measured
socket axis, then a temporary tilt about the nominal key origin. For each
camera it evaluates `n · normalize(C_camera - p_tip)` at the **moved** tip.
Positive values put the camera in front of the face. Candidate pairs are
ranked by the smaller face dot product multiplied by the sine of their ray
separation angle. This ranking has no hand visibility information by itself.

The saved-state planner then checks the complete raise, tilt, untilt and
return from measured joints, followed by the existing whole-hand 20 mm
endpoint screen and sampled socket-axis descent. A separate CAD ray test
checks the face center and four points 5 mm away from it for hand or socket
occlusion. The cases below all have a predicted unoccluded face center in
both listed cameras; this is a **CAD prediction**, not a captured image.

| Saved attempt | Gap | Nominal grasp | Raise / tilt / socket azimuth | Predicted camera pair | Full observation path, 20 mm endpoint, axial plan |
| --- | ---: | --- | --- | --- | --- |
| `5e08757b` | 5 mm | `table/0/20195` | 40 mm / 90° / 15° | `25305462`, `26053248` | pass / pass / pass |
| `4e26ec99` | 5 mm | `table/0/20195` | 40 mm / 90° / 15° | `25305462`, `26053248` | pass / pass / pass |
| `30e9cd0b` | 10 mm | `table/0/20267` | 40 mm / 75° / 60° | `25322639`, `25322642` | pass / pass / pass |

The tested 75°/15° tilt for `4e26ec99` hit the socket while tilting;
the 90°/15° alternative passed. The most visible predicted 90°/30°
direction for `30e9cd0b` had no FR3 tilt endpoint plan; the 75°/60°
alternative passed. Thus choosing a fixed tilt direction from camera
geometry alone is also inadequate. Rank by image geometry, then test arm
reachability and held geometry before using an observation action.

The probe scripts and exact numerical results are under
`~/precision_insertion_project/development/zerodex_capture4/reviews/`:
`probe_face_visibility_20261029`,
`probe_tilt_observation_return_axial_20261029`, and
`probe_tilt_face_occlusion_20261029` (`.py` and `.json`; the planning probe
also has a `.log`). These files and the current-code Qwen/SAM replay have
SHA256-matched copies under
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/tilt_observation_20261029/`.
The final outcome Qwen replay in the same review folder
reproduced the six available human reviewed released-key labels, five 5 mm
and one 10 mm, with zero response parse failures. This is an in-sample
visual-label result, not a measured insertion success rate.

## Pipeline decision

Keep the current nominal pre-insertion capture and measured XY correction.
Store a camera visibility score per candidate after socket calibration, but
do not reject an insertion-compatible grasp solely because the vertical
key face is hidden. When the normal observation has no metric end center,
an additional **optional** camera-informed observation action is justified
only after its real tilted images show that the same physical face is
detectable in two separated views. The observation must then return to the
insertion pose before the measured socket-axis descent. The offline probe
does not activate this extra robot motion in the live pipeline.

The next decisive check needs a held-key tilted snapshot in the actual rig:
measure per-view face-center detection, parallax, hand occlusion and
reprojection consistency, then compare the inferred correction with the
released-key outcome. This will determine whether the extra motion earns
its latency or whether a different camera/hold arrangement is needed.
