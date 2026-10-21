# Cylinder radial-gap readiness on the robot PC (2026-10-22)

This is an asset and static-code audit of the **separate**
`/home/robot/autodex-precision-insertion` checkout and its staged shared-data
root. It is not a camera or robot test of the other physical sockets. The
mode name's `gap_mm` is the **radial clearance between the 15 mm-radius key
and the socket bore**, not the commanded insertion depth. Every mode still
uses the same 20 mm axial verification target.

Recipient root:
`/home/robot/shared_data/hyunsu/tmp/precision_insertion_recipient_stage_20261013_b7e2392`.
The source handoffs are under `/home/robot/shared_data/hyunsu/`.

| Radial gap | Exact socket object ID | CAD outer wall radius | FoundPose source | Eligible / screened v8 candidates |
| ---: | --- | ---: | --- | ---: |
| 1 mm | `precision_socket_cylinder_gap_01mm` | 21 mm | 2026-10-11 generated seven-object handoff | 97 / 112 |
| 3 mm | `precision_socket_cylinder_gap_03mm` | 23 mm | same | 97 / 112 |
| 5 mm | `precision_socket_cylinder_gap_05mm` | 25 mm | same | 100 / 112 |
| 10 mm | `precision_socket_cylinder_gap_10mm` | 30 mm | same | 100 / 112 |
| 15 mm | `precision_socket_cylinder_gap_15mm` | 35 mm | 2026-10-11 four-object FoundPose handoff | 100 / 112 |
| 20 mm | `precision_socket_cylinder_gap_20mm` | 40 mm | previously staged model | 98 / 112 |

Each runtime catalog is at `AutoDex/precision_insertion/cylindrical/recipient_gap{gap}_endpoint_catalog_20261022.json`
under the recipient root. All six report zero scan errors, passed the static
launcher `--check`, and passed full source-file verification for both tabletop
pose classes (`000`, `001`). The verification used the same
`PRECISION_FR3_INSPIRE_URDF` pointing to
`fr3_inspire_oct12_mount.urdf` that the launcher exports. Without that
environment variable, the verifier selects a different default URDF and
correctly reports an endpoint-source mismatch.

The geometry, exact static collision mesh, fixture/task transform, key
mesh, two v8 tabletop poses and two scenes exist for all six modes. The
recipient has 112 full-length v8 candidate directories per mode (the shared
key pool) and two directed `reset_12` seeds; the source-machine handoff's
124-row catalogs include 12 historical candidates that are not in this
recipient pool. Its `audit_only` JSONs contain `/home/hyunsu` paths and an
older implementation hash, so they are **not** runtime catalogs. Each
recipient catalog is newly screened with the current exact key/socket/hand
CAD and `--min-hand-clearance-mm 0`. The `endpoint_pass` count is only a
centered 20 mm CAD endpoint result; it does not establish a reachable
pickup/transfer path or actual insertion.

The 1/3/5/10 mm generated FoundPose handoff and the separate 15 mm archive
were installed into the pre-existing recipient object directories only
after checking the manifest-listed `repre.pth` SHA-256. A leftover
`GENERATION_REQUIRED.json` marker was removed only after its object's
validated representation and templates were installed. No model from a
different socket was copied or renamed as a substitute. Generation and
successful reload on the source PC are **not** independent metric pose
validation on this robot PC. The active board-supported body fit still runs
the stock FoundPose capture for compatible evidence; each exact socket
representation must therefore be present. The fit reads the wall radius
from that mode's raw mesh and can use the same measured-board procedure.
The installed representation hashes, in gap order 1/3/5/10/15/20, are
`baa4433e7b5a874dea467cfcfef4126e0d5c1d68ba8ba18f25ab0e18da2be07c`,
`f244d88e50f8bc9881844f03dd3d716e2d18ec929f7c4522bf3e6407d7455e74`,
`27eb484804f1f97e71a676e0e795d4ba48391b1895b280820555e61457d486f5`,
`5c6adebe0e9d127897ea0acd1cb846faa016b472ffb3a9e8e96cac34b5de3985`,
`aa5894bc7637b672388c1c7f073acd490c6335e37ed7f183ee90e3624b40861f`,
and `b63a88d900eb04223985b26e0c8aeda4f155a3443555675e818f768f834ceb0e`.
The archived template metadata retains source-machine paths as provenance;
active model loading resolves each representation from the recipient object
directory and does not follow those paths.

The launcher accepts `PRECISION_GAP_MM=1|3|5|10|15|20` and defaults to 20.
`robot_pc_stages:configure` accepts the same six cylinder modes on board 9;
its common FR3, Inspire, camera and distributed Qwen settings are unchanged.
`--check` is a file-only preflight and uses neither camera nor robot. For
example:

```bash
cd /home/robot/autodex-precision-insertion
PRECISION_GAP_MM=15 bash demo/precision-insertion/run_robot_pc.sh --check
```

A physical change of socket invalidates the previous frozen socket pose and
20 mm image evidence. The matching physical fixture must be identified,
placed flat on the board and captured afresh without the key; then the key
is placed and Enter continues the session. No 1/3/5/10/15 mm physical
socket capture, FoundPose metric-pose comparison, grasp, contact or insertion
result is claimed by this preparation. In particular, the narrow gaps need
their own real pose/axis and insertion evidence before reporting task
success; a static `--check` is not that evidence.
