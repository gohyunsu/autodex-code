# Precision-insertion assets

This directory stores the canonical binary STL inputs for the unified socket
and the four experiment keys. Source STL coordinates are millimetres; generated
AutoDex assets are metres.

| AutoDex object | Nominal per-side gap | Source |
|---|---:|---|
| `precision_key_1p5mm` | 1.5 mm | `plug_gap_1p5.stl` |
| `precision_key_1p0mm` | 1.0 mm | `plug_gap_1p0.stl` |
| `precision_key_0p5mm` | 0.5 mm | `plug_gap_0p5.stl` |
| `precision_key_0p3mm` | 0.3 mm | `plug_gap_0p3.stl` |

Contact policy: the hand may touch only the four lateral faces and rear face of
the 39 x 33 x 45 mm handle. The shaft, tip, bevel, and the socket-facing handle
shoulder are forbidden because contact there obstructs insertion. Accepted
contacts also stay at least 2 mm from handle edges.

`autodex_camera_profile.json` preserves the existing AutoDex acquisition
contract: capture PCs 1/2/3/5/6, hardware-triggered FLIR video, the local
timestamp camera, and the active ParaDex network profile. Camera serials are
resolved from `paradex/system/current/pc.json`; an explicit `--calib_dir` must
cover that complete active set. ZeroDex cameras are not used by this task.

The shared socket STL is also built as the independent perception object
`precision_socket_unified`. Its raw mesh keeps the source STL frame exactly,
so a pose estimated from that mesh is a socket-frame pose rather than a pose
of an undocumented recentered model. The keyed bore must stay visible in the
segmentation mask because it is the feature that resolves insertion yaw; the
outer body alone is close to yaw-symmetric.

The socket object's exact concave mesh is valid as a static fixture collision
mesh. It intentionally has no convex-hull `coacd` asset: a hull would close the
bore and make every insertion collide. It is not a BODex grasp target and has
no tabletop scenes or grasp candidates.

The builder creates one handle-only BODex proposal proxy per key. The visible
proxy geometry is identical, but its gravity centre and mass come from the
corresponding full key. Proxies are not runtime objects and cannot be used for
perception, collision checking, or task success. Every proposed grasp must be
rechecked independently on every complete key mesh on which it will run.

`bodex_handle_proxy_objects_all.txt` is the reproducible four-proxy input list.
The 0.3 mm proxy is also used to propose the current common grasp because that
single wrist/finger configuration passed all four full-key simulation checks.

Do not hand-edit generated OBJ/JSON/NPY files. Rebuild them with
`scripts/precision_insertion/build_assets.py`.
