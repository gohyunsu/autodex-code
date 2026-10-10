# Fixed cylindrical key and six sockets

**구성:** 모따기 없는 원기둥 키 1개와 원형 소켓 6개. 각 부품은 STL,
P2S용 `.gcode.3mf`, 치수 표시 PNG로 제공되며 개별 출력합니다.
`all_parts_overview.png`는 7개 형상을 한눈에 비교하는 이미지입니다.

All dimensions below are nominal CAD dimensions in **millimetres**. This is
a new circular family; the earlier keyed-square parts are incompatible.

## Geometry

- **One key:** smooth, straight cylinder, radius **15**, height **80**. Flat
  ends and a constant diameter throughout. No handle step, shoulder, groove,
  bevel/chamfer, fillet, or depth stop.
- **Six sockets:** each has a circular blind bore of depth **50**, measured
  vertically from the top rim at `z=55` to the top of the base at `z=5`.
  The socket's outer cylinder rises **50** above the base (`z=5..55`).
- **Each socket's outer radius** is its inner radius plus **5**. The straight
  bore, rim, and floor have no chamfer or funnel.
- **Each socket's fixed base** is a solid circular disk of radius **60** and
  height **5** (`z=0..5`). It closes the bore. Overall socket height is **55**.
- The **20 mm task insertion goal** is an experimental outcome threshold only.
  It does not add a CAD feature or change the key, bore, or floor dimensions.

`clearance` means the **radial, one-sided nominal gap** between the centered
key and the straight socket wall, not the total difference in diameters.

| Socket | Radial clearance | Inner radius | Inner diameter | Outer radius | Outer diameter | Wall thickness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gap_01 | 1 | 16 | 32 | 21 | 42 | 5 |
| gap_03 | 3 | 18 | 36 | 23 | 46 | 5 |
| gap_05 | 5 | 20 | 40 | 25 | 50 | 5 |
| gap_10 | 10 | 25 | 50 | 30 | 60 | 5 |
| gap_15 | 15 | 30 | 60 | 35 | 70 | 5 |
| gap_20 | 20 | 35 | 70 | 40 | 80 | 5 |

Every socket uses the same key. The easiest (largest-gap) socket is `gap_20`.
The base has no bolt holes; secure its circular bottom with an external clamp
or fixture without covering the bore or robot approach area.

## Files

Each STL and each `.gcode.3mf` is **one part printed separately**. STL is the
editable geometric model; `.gcode.3mf` contains the P2S sliced toolpath.
Each PNG shows a geometry preview matching its STL plus a dimensioned centre section.
Print times are Bambu Studio estimates, not measured printer times.

| Part | STL | P2S sliced job | Image | Est. time |
| --- | --- | --- | --- | ---: |
| Fixed key | `stl/key_r15_h80.stl` | `P2S_0p4_GenericPLA_key_r15_h80.gcode.3mf` | `renders/key_r15_h80.png` | 44m 56s |
| 1 mm socket | `stl/socket_gap_01_r16.stl` | `P2S_0p4_GenericPLA_socket_gap_01_r16.gcode.3mf` | `renders/socket_gap_01_r16.png` | 1h 35m 53s |
| 3 mm socket | `stl/socket_gap_03_r18.stl` | `P2S_0p4_GenericPLA_socket_gap_03_r18.gcode.3mf` | `renders/socket_gap_03_r18.png` | 1h 41m 15s |
| 5 mm socket | `stl/socket_gap_05_r20.stl` | `P2S_0p4_GenericPLA_socket_gap_05_r20.gcode.3mf` | `renders/socket_gap_05_r20.png` | 1h 44m 59s |
| 10 mm socket | `stl/socket_gap_10_r25.stl` | `P2S_0p4_GenericPLA_socket_gap_10_r25.gcode.3mf` | `renders/socket_gap_10_r25.png` | 1h 57m 11s |
| 15 mm socket | `stl/socket_gap_15_r30.stl` | `P2S_0p4_GenericPLA_socket_gap_15_r30.gcode.3mf` | `renders/socket_gap_15_r30.png` | 2h 6m 32s |
| 20 mm socket | `stl/socket_gap_20_r35.stl` | `P2S_0p4_GenericPLA_socket_gap_20_r35.gcode.3mf` | `renders/socket_gap_20_r35.png` | 2h 19m 19s |

The STL cylinders are approximated with 256 planar segments. The largest
radial polygon error is below 0.005 mm; the CAD deliberately contains no
bevel. When printed, actual dimensions depend on extrusion and shrinkage.

## Slice settings and use

The G-code jobs use the installed **Bambu Lab P2S 0.4 mm nozzle** machine
profile, **0.20 mm Standard @BBL P2S** process, **Generic PLA @BBL P2S**
filament profile, **Textured PEI Plate**, four wall loops, and 30% sparse
infill. The generated G-code uses 220 °C nozzle and 45 °C bed settings.
The key stands on a flat end; each socket rests on its circular base.
The jobs have no manually added supports or geometric brim. Bambu Studio's
process profile may apply its own automatic adhesion settings.

The reported filament is PLA+. Check the actual spool's temperature range,
installed nozzle, and plate before printing. If any differ from the profiles
above, import the STL into Bambu Studio and reslice for the real setup. Measure
the printed key diameter and bore diameters with calipers; use those measured
gaps in experiment records. Hand-fit the 20 mm gap first, then narrow gaps.
The embedded material-weight estimate is zero because the installed slicer
profile supplies no usable density value; this does not alter the toolpath.

`verify_output.py` checks every STL for closed, consistently oriented mesh
edges, expected dimensions and volume, and checks every `.gcode.3mf` ZIP,
embedded G-code, machine profile, and print height. All seven files passed.

Regenerate geometry with `py generate.py`; regenerate P2S toolpaths with
`powershell -File slice_p2s.ps1 -Force`; render with `python render_parts.py`;
verify with `python verify_output.py` from this directory.
