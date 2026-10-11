# Offline asset/use status (2026-10-11)

This page answers **which files the current demo can use for offline
development**. It deliberately does not grade live camera timing, robot
controllers, force control, or physical insertion. `audit_assets()` reports
file presence only; it does not mean a mode has a complete candidate pool.

Here **available** means a local file at the expected path; **pending** means
a candidate file exists only in a handoff or lacks independent qualification;
**missing** means no canonical file/pool is present at the expected runtime
path. These labels are about offline inputs, not a claim of robot readiness.

The three quoted NAS handoffs were read back on 2026-10-11. Every
manifest-listed payload file matched its recorded byte count and SHA256:
asset/FoundPose 529 files (9,708,884,390 bytes), runtime addendum 1,744 files
(5,693,404,373 bytes), and perception evaluation 1,311 files (134,276,114
bytes). There were no missing or changed listed files. The runtime addendum
also contains 55 **unlisted** `.pyc` files under `__pycache__`; all were
created after its manifest and are not counted as handoff payload. The NAS
mount is read-only here, so this audit did not alter the handoffs. Payload
integrity alone does not install those assets into the local runtime.

| Asset or mode | Present and usable now for offline work | Not available at the current canonical runtime path / qualification |
| --- | --- | --- |
| Object geometry | Metric key/socket `object_processing` meshes, exact socket collision meshes, task geometry, tabletop poses and FR3/Inspire URDF for all configured square and cylinder gaps | None of these CAD files is a measured socket pose. |
| Perception model code/weights | NAS [runtime addendum](/mnt/paradex2/hyunsu/autodex_precision_insertion_runtime_addendum_20261010_652ac909) contains SAM3 and DINOv2 weights and source snapshots; YOLOE, FoundationPose and MV-GoTrack are optional alternatives, not extra requirements for the current SAM3/FoundPose path | NAS presence does not install a weight at a recipient process's expected cache/path. |
| FoundPose representations | NAS [asset handoff](/mnt/paradex2/hyunsu/autodex_precision_insertion_handoff_20261010_652ac909/pending_foundpose) contains 12 candidate `repre.pth` files for offline replay. This workstation has five independently generated canonical PTHs: square 1.5 mm key, square unified socket, cylinder key, cylinder 15 mm socket and cylinder 20 mm socket | Square 0.3/0.5/1.0 mm keys and cylinder 1/3/5/10 mm sockets have **no canonical PTH on this workstation**. A pending NAS PTH is not automatically a canonical installed asset. None of the candidate/replay results is independent metric pose accuracy evidence. |
| Square 1.5 mm grasp pool | Current local complete v8 catalog has seven nominal 20 mm endpoint-eligible grasps, all on tabletop pose `004` | Poses `000`–`003` have no eligible grasp in that finite pool. Other square gaps have one raw v8 grasp directory each, but no current complete endpoint-eligible catalog. |
| Cylinder grasp pool | Current local [catalogs](CYLINDER_ENDPOINT_CATALOGS.md) for radial gaps 1/3/5/10/15/20 mm each contain 12 source-verified nominally eligible grasps (11 pose `000`, one pose `001`). For the 15 mm gap, candidate `table/0/194` also passed one hypothetical pickup→lift→transfer→20 mm sampled-plan replay and its saved trajectory verifier. | Older `endpoint_catalog_staged*` files are stale against the current `candidates.py` hash. The refreshed catalogs are local; this workstation's NAS mount is read-only. The other 11 cylinder-15 candidates and other gaps have not passed that full-chain replay. |
| Reset/reorient | Proposal scenes and curated **staging** handoffs exist for offline study | Canonical v8 reset candidate directories for these precision objects contain **zero** complete grasp seeds here. A staged seed is not an installed repose trajectory. |

For a minimal file-backed prototype, choose **square 1.5 mm pose `004`** or
**cylinder 15/20 mm pose `000` or `001`** if perception files must already be
at canonical local paths. Cylinder 1/3/5/10 mm can still run saved-pose
geometry/planner diagnostics without their socket PTH. A PTH is required
only when that object is actually passed through FoundPose, not for a
synthetic known-pose endpoint screen. The catalog eligibility numbers are
offline filters, not measured grasp or insertion success.

An **asset inventory** needs only the matching CAD/object IDs, source files,
and their provenance. A candidate qualified for the proposed offline grasp
pool additionally needs its stock v8 grasp-simulation evidence and the
whole-hand/key 20 mm endpoint screen. Run complete
pickup→lift→transfer→20 mm sampled planning for a *selected hypothetical
scene* when studying that trajectory; this is not a prerequisite for saying
the mesh or candidate file exists. Do not add live timestamp, force,
VLM-confidence or physical-outcome gates to a file-only asset inventory.
Those belong to later execution/evaluation. Conversely, do not call a raw
candidate, a staged reset seed or a
`repre.pth` from another object a usable canonical asset just to make an
inventory appear complete.

The **code** now has a demo-local live action loop and stage handlers for
pickup, lift, transfer, first guarded insertion, one grounded XY retry,
held-key return, transfer-recovery observation and directed repose. The
`run_auto.py` CLI itself still only opens an evidence ledger; the new
[`run_live.py`](run_live.py) CLI drives the live loop through an explicit
AutoDex-PC integration module and independently commissioned controllers.
An optional replayable depth-admission path exists, but no real
held-out depth audit or reviewed calibration is installed. These are runtime
integration/measurement gaps, not missing 3D meshes.
