# Precision insertion: extra-gate audit (2026-10-11; live-path update 2026-10-26)

This inventory separates user task requirements, unchanged AutoDex v8
acceptance, and project-specific experimental gates. A numerical value in an
example is **not** a calibrated or professor-approved acceptance threshold.
The current live CLI admits **cylinder only**; square remains an offline
planning mode without XY error correction.
The October 11 one-millimetre, repeated-socket, and physical-held-relation
defaults have been superseded. The active cylinder action uses
`assess_saved_preinsert_xy_diagnostic`, a single socket capture is permitted,
and `runtime_assembly` selects the nominal v8 grasp relation. The older
`xy_retry.py` cardinal-choice path remains an offline/legacy utility; it is
not selected by the live action map. See
[PIPELINE_CODE_AUDIT_20261026.md](PIPELINE_CODE_AUDIT_20261026.md) for the
current source and saved-run review.

## Current next-trial XY update

For a failed cylinder insertion with a usable multi-view observation, the
VLM marks the key tip and visible projected axis. Calibrated cameras recover
the key axis in socket coordinates. If its lateral offsets at the rim and at
20 mm depth are `e_rim` and `e_depth`, a translation-only least-squares target
is `d = -(e_rim + e_depth)/2`. This `d` is a continuous metric vector, **not**
one of four cardinal choices and not a pixel or per-axis rounding. The optional
SAM half-ellipse path instead triangulates a visible end-face proxy from the
same saved preinsert frames; both paths feed the same next-trial ledger.
The applied update is

`step = d * min(1, 0.010 m / ||d||)`;
`offset_next = offset_current + xy_update_weight * step`.

The default weight is 1.0. Thus a measured 0.6 mm error yields a 0.6 mm
update, a measured 2.4 mm error yields a 2.4 mm update, and a measured 12 mm
error yields a 10 mm update in the measured direction. The complete prior
offset is retained for the *next physical pickup* of the same grasp and
tabletop pose; it is not applied during the
failed attempt. A fresh failed attempt can add another update. There is no
default retry-count or action-count cap. A caller-supplied total-offset
budget exists only in the older 1 mm utility, not in this live route.
Multi-view abstention, inability to replan the next pickup, changed
grasp/tabletop pose, or robot/controller stop can still interrupt this
sequence. Translation-only updates cannot remove key-axis tilt or grasp slip.

| Gate / value | Origin and current effect | Can next-trial XY correction fix the underlying error? |
| --- | --- | --- |
| 20 mm key insertion, key/socket/whole-hand noncollision at the aligned endpoint | User-requested task gate. `endpoint.py` screens the exact CAD endpoint; current catalogues also require original v8 full-key grasp-simulation evidence. | No. XY can change lateral alignment but cannot remove an inherently colliding grasp. |
| v8 scene collision, squeeze contact, MuJoCo gravity stability | Original AutoDex `src/grasp_generation/sim_filter/run_sim_filter.py`; the demo verifies the source and result in `candidates.py` and `reset_candidates.py`. | No. A failed pickup/unstable grasp is a different problem. |
| Historical 25 mm cylindrical grip-proxy proposal region | Historical project asset in `scripts/precision_insertion/build_cylindrical_assets.py`; **removed from new proposal generation**. `build_full_length_cylinder_proposal.py` provides a distinct 80 mm/64-side proposal object and both staging scripts now require its full-length scenes. The original 25 mm files and their candidates remain historical evidence. Full-key collision/MuJoCo and 20 mm socket screens still apply. | No. This controlled offline search-space coverage, not socket XY alignment. |
| Nominal hand/socket clearance margin | Project endpoint screen now accepts an explicit **0 mm** margin for CAD noncollision alone; a positive value adds a project-specific screen. Earlier historical catalogs used 0.2 mm; the most recent pre-change cylinder catalogs used 0.001 mm. Neither is a measured physical tolerance. The new collision-only catalogs in `COLLISION_ONLY_CATALOGS.md` use 0 mm and still reject every detected hand/socket collision. | A lateral offset can affect clearance, but cannot excuse a hand/socket collision at the intended endpoint. Catalogs are bound to the exact implementation and must be rebuilt after a code change. |
| `grasp_error ≤ 0.2`, mean contact-distance error `≤ 10 mm` | Historical project numerical screen in `scripts/precision_insertion/filter_contact_safe_grasps.py`; **not** the stock v8 MuJoCo pass and not a gate in the current cylinder endpoint catalogue. | Not directly. It judges proposal quality, not socket XY alignment. |
| Square handle-only contact, 2 mm edge distance | Historical square contact-policy experiment; current unrestricted square promotion records the policy for analysis rather than rejecting contacts with it. The new cylinder proposal likewise covers the whole 80 mm surface. | No. This changes where the hand grips. |
| Reset key-in-hand drift/tilt, e.g. 5 mm/10° or 3 mm/8° | These were our **uncommissioned diagnostic examples**, not original v8 criteria. As of this change, no drift/tilt gate is applied by default in `reset_candidates.py`, `repose_policy.py`, or `install_cylinder_reset_seeds.py`; supplying both limits explicitly enables an optional screen. The measured trajectory values remain recorded. | Insertion XY correction cannot repair a reorientation grasp that slips or changes orientation. Evaluate repose landing separately. |
| At least one socket capture; repeated-pose agreement only when multiple captures are requested | Project `live_session_start.py`, `perception_evidence.py`, and `calibration.py` admit one frozen multi-view socket measurement. Example mask, pose-quality, frame-skew and silhouette-refinement thresholds are project parameters, not stock v8 criteria or commissioned tolerances. A one-capture session has no artificial 1 mm/1° repeatability comparison. | No. Bad socket pose or timestamp cannot be fixed reliably by adding an XY command learned against that same bad reference. |
| Key tabletop class, fresh exposure/state timing, catalogue provenance | Project `trial_preflight.py`, `live_key_trial.py`, and `candidates.py`. Correct pose class and source binding are needed to select the v8 grasp and avoid stale CAD. Time/pose-error tolerances are caller-supplied, not fixed project numbers. | No. Wrong pose/candidate identification changes the grasp and transfer plan. |
| Dense-path checks, goal/axial tolerances, sampled key/hand/socket clearance | Project `path_audit.py` around unchanged cuRobo planning. Eight `PathAuditLimits` values come from a robot-PC config; no numerical values are endorsed by this repository. Axial waypoint spacing is capped at 5 mm in `trial_preflight.py`. These checks may reject a nominally endpoint-clear candidate. | Sometimes a new XY target changes path feasibility; XY does not replace collision checking. |
| Held key/hand relation after lift | Active cylinder `runtime_assembly.py` selects `nominal` from the stock v8 candidate as an **unmeasured hypothesis** and replans from measured post-lift joints; no physical-grasp-calibration record is required. The older physical route is not selected for cylinder live mode. | XY can compensate a repeatable lateral grasp bias after a failed insertion, but the nominal relation does not measure hidden key orientation or guarantee repeatability. |
| Metric VLM grounding: ≥2 views, parallax, reprojection, cylinder-axis tilt/sweep, 95% XY uncertainty and predicted-improvement tests | Project `grounded_alignment.py`. Geometry-degeneracy checks are necessary for metric triangulation; numerical limits are caller-provided. The previous arbitrary 3-view default was reduced to 2. A single-view planar hypothesis or exploratory step is recorded with a distinct diagnostic status when metric grounding is unavailable. | They test whether the *estimate* is usable, not the insertion mechanics. More attempts do not guarantee a bad visual estimate improves. |
| Per-failure XY increment ≤10 mm; cumulative update weight 1.0 | `grounded_alignment.py` bounds each measured increment, and `session_runner.py` applies `p[n+1] = p[n] + weight × bounded_residual[n]` to the **next physical attempt** only. No default cumulative offset bound or retry-count cap is applied. | This primarily fixes repeatable lateral translation, not yaw, tilt, slip, or a wrong socket pose. |
| Fresh next-attempt path and contact checks | The cumulative wrist correction is screened and replanned for a newly observed key; an infeasible pickup, held path, or insertion still cannot run. The older `xy_retry.py` utility's caller-supplied total-offset budget is not a live failure gate. | A corrected wrist goal still needs a valid current scene and collision-free path. |
| Repose release: 12 cm first; original XY then ±3 cm sites; default release-key clearance 5 mm | Project `repose_target_selector.py` and `live_repose_handlers.py` heuristics. These are additional search choices, not stock reset success criteria or user-approved geometry. The selected site still needs a socket-aware full-chain preflight and observed landing. | No. This is recovery/reorientation, not insertion-axis alignment. |
| `vlm_only` task labels by default; manual-label mode available | Project `run_live.py`, `labeling.py`, and `outcome.py`. `vlm_only` positive insertion is an **experimental visual label**, not independent proof that the key penetrated 20 mm. A controller abort remains a separate motion stop. | An incorrect label can corrupt retry logic; repeated XY updates do not validate the label. Compare against depth/force or manual annotations when evaluating. |
| Optional session action-step and retry-count caps | Project `run_live.py`. Both caps are absent by default; a caller may request either explicitly. No hidden 100-step loop stop truncates repeated XY accumulation. The older same-attempt XY utility accepts a caller-supplied total-offset bound, but the active saved-preinsert diagnostic route does not use it. | No; these only limit experiment duration. |

The active cylinder route uses an **unmeasured nominal** held-key relation.
Its physical accuracy and the visual-grounding limits still need evaluation
against independently labeled camera data. The 25 mm proposal restriction and mandatory positive CAD
clearance margin have been removed for new offline runs. The 112 new
full-length seeds passing unchanged v8 and at least one 20 mm socket screen
were appended to the canonical v8 candidate tree without replacing the
historical 12; the per-gap full-pool catalogues were regenerated and
preflighted after this candidate-set change.
No uncommissioned example value should be presented as a measured acceptance
threshold. Conversely, dropping key/socket or hand/socket collision checks,
the original v8 simulation checks, or the robot's guarded-contact stop is not
equivalent to studying XY error correction.
