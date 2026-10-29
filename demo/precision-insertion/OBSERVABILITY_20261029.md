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

An additional depth render used the **saved measured FR3/Inspire joints** and
nominal held-key transform from `5e08757b`, with the existing AutoDex visual
meshes and calibrated capture4 cameras. This distinguishes a projected key
from one hidden behind the robot; it does not measure the key after squeeze:

| Camera | Predicted key pixels visible after robot occlusion | Current saved-image Qwen/SAM3 result |
| --- | ---: | --- |
| `25305462` | 1,389 / 44% of isolated key projection | Qwen box matched the socket |
| `25322639` | 0 / 0% | No key mask |
| `25322642` | 1,348 / 41% | Qwen box matched the socket |
| `26053248` | 6,610 / 63% | Qwen box selected the key region; SAM3 returned no mask |

On the same raw preinsert snapshot, 36 combinations of three ROIs and four
short English SAM3 prompts produced seven masks. Their overlap with the
nominal rendered *visible key* ranged from 0 to 9.4% of selected pixels;
visual inspection showed the masks primarily on the socket or arm. Changing
the prompt or ROI alone therefore did not recover two usable key-end views
for this grasp. The render is a nominal CAD comparison, and its overlap is
not segmentation ground truth. The local probe files are
`probe_key_visibility_render_20261029.py`,
`probe_key_visibility_5e08757b_20261029.json`, and
`sam_prompt_roi_probe_20261029/results.json` under the development review
directory. These measurements strengthen the distinction between geometric
camera coverage and actual *identifiable* key-face observations; they add no
grasp rejection rule or live inference step.

### Focused crop and geometric SAM3 prompt follow-up

On saved `5e08757b`, a 200-pixel-wide crop centered on the *nominal* shaft
improved Qwen's coarse key boxes in the two similar side views. Unlike the
runtime lossless crop, this exploratory input was resampled into a 512-pixel
canvas with an explicitly recorded inverse pixel mapping. The box comparison
used the nominal rendered visible-key mask, **not** an annotated physical key.
In those focused crops, text-only SAM3 still returned no mask in eight
trials. The installed `Sam3Processor.add_geometric_prompt` accepts a positive
box. Passing the Qwen box to that stock API produced masks visually covering
the held key in `25305462`, `25322642` and `26053248`. The first view's mask
covered 73% of the rendered nominal visible-key pixels at 83% precision;
the other two covered about 48% at 55–57% precision. These figures describe
agreement with a nominal projection, not actual segmentation accuracy.

The existing half-ellipse end fitter did **not** turn those three masks into
a consistent key-face center. Its fitted point differed from the nominal
tip projection by 14.7, 9.7 and 73.0 pixels respectively; the three-view
fit left 14.8–21.7 pixel reprojection residuals. A separate direct-Qwen
point prompt also returned a point in `25322639`, where the nominal key is
fully hidden by the hand. Its four per-view points differed from the nominal
tip by 34–89 pixels. These are not independent measurements of the actual
key pose after squeeze.

For comparison, a fixed-height XY least-squares fit to the two side-view
mask displacements relative to the nominal rendered key yielded roughly
(-0.3, -8.8) mm. The saved 5 mm-gap attempt was human-labelled as inserted.
That mismatch does not establish the true physical offset, but rules out
using these projected-mask shifts as an automatic correction without a
physical key-center reference. The two side cameras also have only about
4 degrees of parallax. In earlier `20096` episodes the existing text-prompt
route obtained three consistent inlier cameras; replacing it globally with
the box prompt lost one fitted end and moved another by 37.7 pixels. Keep
the production estimator unchanged until a method produces consistent
physical-face observations across both grasp families.

Reproducible local-only probes and outputs are in the development review
directory: `qwen_tip_crop_probe_20261029.{py,json}`,
`sam_focused_qwen_probe_20261029/results.json`,
`qwen_focused_tip_point_probe_20261029.{py,json}`,
`probe_sam_box_prompt_20261029.py`,
`sam_box_prompt_20261029/{results,tip_evaluation}.json`, and
`sam_box_prior_metric_20261029.json`, plus
`probe_mask_shift_xy_20261029.{py,json}`. No camera or robot command was sent.

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
For the three saved holds, a 40 mm straight raise changes the distal-face
view dots but leaves every one negative (the largest is -0.51). The two
human-confirmed 5 mm insertions did not execute an observation raise, so
there is no live timing or detection gain to support enabling it by default.
As of 2026-10-30 the ZeroDex profile uses zero raise by default; 40 mm is
still available as an explicit observation experiment.
The grasp catalog and trial preflight currently do not reject candidates by
camera visibility. A future candidate ranking could score visibility after
socket calibration, but should not reject an insertion-compatible grasp
solely because the vertical key face is hidden. This ranking is not yet
implemented. The live XY controller accepts a metric correction with at least
two inlier cameras; the SAM half-ellipse route uses three consistent views
for its early completion and robust fit, then processes the fourth view when
needed. Thus two cameras containing key pixels do not themselves authorize
an XY move. When the normal observation has no metric end center,
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

### Saved-view stopping and Qwen batch cost (2026-10-30)

Six saved same-trial observations have the current adaptive 3+1 camera record:
four successive snapshots from `0c787032` obtained metric XY from the first
three preferred cameras and did not run the fourth SAM view; the two
human-confirmed `20195` insertions, `5e08757b` and `4e26ec99`, did not obtain
metric XY from those three and processed the fourth. These are six snapshots
from three episodes, not six independent attempts. In this set, none had a
valid two-camera metric fit in the preferred group that a two-view early stop
would have saved. Counting two masks as a metric pair would have been wrong
for `5e08757b`: its two admitted views were the nearly coincident
`25305462`/`25322642` pair.

On four warm repetitions of one archived four-view Qwen box request, the
three-view GPU batch took a median 0.457 s and the four-view batch 0.530 s;
the first three answers were identical. The extra fourth Qwen question costs
about 0.073 s on this saved crop. The current one-batch four-view prefetch is
reasonable while the fourth view remains a common SAM fallback. This is an
isolated saved-image Qwen measurement, not end-to-end trial latency. Reproducer
and raw durations are `probe_qwen_3v4_batch_20261030.{py,json}` in the local
ZeroDex development review directory, with SHA256-matched copies under
`/mnt/paradex2/hyunsu/precision_insertion/development/zerodex_capture4/reviews/qwen_batch_timing_20261030/`.
No camera or robot command
was issued. The ranking decision remains unchanged: retain all insertion-
compatible grasps, then evaluate same-feature observability in the measured
scene rather than adding a two-camera grasp rejection.

## Reusable observation geometry (2026-10-30)

`precision_insertion/held_face_observation.py` now computes a nominal held
key/wrist raise and tilt, ranks face-facing camera pairs by the weaker face
view times ray parallax, and converts a measured tilted face centre back to
the insertion pose's socket-frame XY translation. If the actual key differs
from the nominal key by translation `d` before tilting, a rigid grasp gives
`p_observed = p_nominal_tilted + R_tilt d`. Thus the proposed correction is
`-R_socket_xy^T R_tilt^T (p_observed - p_nominal_tilted)`. The calculation
does not treat a projected CAD point as a camera measurement and does not
assume that one face-facing camera is unoccluded by the hand.

Synthetic calibrated two-view pixels recover the imposed XY translation in
the unit test. With the three archived capture4 calibrations and pre-insertion
key poses, the new face-direction ranking reproduces the prior offline top
view geometry for the two 5 mm holds (90° tilt, 30° socket azimuth). This
ranking is **not** a motion plan: the 10 mm hold's highest camera score had
no FR3 endpoint IK in the earlier planning probe, while a lower-ranked
75°/60° alternative passed. Neither the new geometry helper nor a tilted
VLM observation is active in the live controller. Actual tilted photographs
must establish that the same physical face centre can be found in separated
views before this inverse calculation is used for an XY motion.
