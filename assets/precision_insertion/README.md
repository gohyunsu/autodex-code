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
shoulder are forbidden because contact there obstructs insertion.

Do not hand-edit generated OBJ/JSON/NPY files. Rebuild them with
`scripts/precision_insertion/build_assets.py`.
