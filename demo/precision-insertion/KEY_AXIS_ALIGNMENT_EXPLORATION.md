# Cylinder key-axis alignment from the saved preinsert views

This is a read-only replay of the same saved undistorted `preinsert` images,
Qwen boxes, SAM3 masks, calibrated cameras, frozen socket pose, and
half-ellipse tip centres used by the 2026-10-24 XY animation. It does not
connect to a camera or robot and does not change an episode correction.

## Task-relevant state

The cylindrical key has five visually relevant pose degrees of freedom:

- insertion-tip centre `p = (x, y, z)` in the frozen socket frame;
- directed insertion axis `a`, a unit vector with two degrees of freedom.

Rotation about `a` is unobservable for a textureless circular cylinder and is
irrelevant to a circular socket. The desired insertion direction in the
current task geometry is socket-frame `-Z`, not world `Z`.

## Saved-data methods compared

Every image line is anchored at the already selected SAM half-ellipse face
centre. The planned key projection selects only which direction is the
insertion end; it does not set the observed line angle.

| Method | Image observation | Interpretation |
| --- | --- | --- |
| Minimum-area rectangle | Long edge of the largest SAM3 key component | Uses the full visible shaft and many mask pixels. |
| Partial-face ellipse | Minor-axis direction of the free ellipse fitted to the distal partial contour | Independent end-face diagnostic, but sensitive to a clipped or occluded arc. |
| Trimmed mask PCA | Principal component of the middle 70% of mask pixels | Biased when the hand truncates one side or the component is asymmetric. |
| Bilateral strip midline | Midpoint between lateral mask quantiles in axial strips | Sensitive to which axial range remains visible. |

For calibrated view `i`, image line `l_i` back-projects to a 3-D plane through
the camera. With `T_camera_socket = [R_i | t_i]` and undistorted intrinsics
`K_i`, its socket-frame plane normal is

```text
n_i = normalize(R_i^T K_i^T l_i).
```

The observed cylinder direction is the unit vector that best intersects all
planes:

```text
a = argmin ||N a||^2  subject to ||a|| = 1,
N = [n_1^T; ...; n_m^T].
```

It is the right singular vector of `N` with the smallest singular value. Its
sign is selected toward socket-frame `-Z`. This is the same projective-line
geometry already implemented by `grounded_alignment._line_axis`; unlike the
old Qwen axis-coordinate probe, it does not ask a VLM to draw a metric line.

## Replay result

| Saved episode | Usable calibrated views | Method | Tilt from socket `-Z` | Median / max line error | Leave-one-view-out axis change |
| --- | ---: | --- | ---: | ---: | ---: |
| `20261022_103429_752726` | 5 | minimum rectangle | **3.72 deg** | **1.62 / 7.44 px** | **0.21 deg median, 1.98 deg max** |
| | 5 | partial-face ellipse | 1.11 deg | 1.03 / 28.55 px | 1.14 deg median, 2.87 deg max |
| | 5 | trimmed PCA | 14.06 deg | 8.26 / 100.42 px | 26.10 deg median, 74.64 deg max |
| | 5 | strip midline | 13.06 deg | 5.53 / 64.35 px | 5.27 deg median, 37.41 deg max |
| `20261022_105828_126084` | 2 of the 3 saved image views have preserved calibration in the review manifest | minimum rectangle | **15.60 deg** | **1.63 / 2.50 px** | not observable with only two views |
| | 2 | partial-face ellipse | 16.90 deg | 1.64 / 2.56 px | not observable with only two views |

The five-view minimum-rectangle result is the most stable under removal of a
camera. In the second episode, the shaft and face methods independently agree
within 1.3 degrees on a large tilt. This supports using the full-shaft line as
the primary estimate and the face ellipse only as a diagnostic. PCA and strip
midlines are rejected for this mask/occlusion pattern.

The two retained primary estimates are:

| Episode | Tip centre in socket frame | Observed axis | Minimal rotation vector to `-Z`, socket frame |
| --- | --- | --- | --- |
| `103429` | `(16.80, 4.56, 83.03) mm` | `(0.0605, 0.0236, -0.9979)` | `(-1.35, +3.47, 0.00) deg` |
| `105828` | `(31.23, 9.53, 93.82) mm` | `(0.2672, -0.0304, -0.9632)` | `(+1.76, +15.50, 0.00) deg` |

These are image-derived estimates without physical pose ground truth. The
comparison ranks methods on multiview consistency; it does not establish
millimetre or degree accuracy.

## Recommended next-trial correction

Retain the current policy that a failed attempt only updates the **next**
attempt. Replace the translation-only state with one socket-frame rigid
correction.

1. At `preinsert`, run the existing Qwen ROI and SAM3 mask once per selected
   view. Reuse each mask for both the half-ellipse tip centre and its
   minimum-rectangle shaft line; no additional VLM call is required.
2. Robustly triangulate the tip `p` as today. Back-project the shaft lines and
   obtain `a` with pair-seeded consensus followed by an all-inlier SVD fit.
3. Set the target axis `a* = (0,0,-1)` and same-height target tip
   `p* = (0,0,p_z)`. Let `R_delta` be the minimum rotation from `a` to `a*`.
   Cylinder axial yaw is left unchanged.
4. Build a correction about the observed tip, not the socket origin:

   ```text
   t_delta = p* - R_delta p
   T_delta = [R_delta, t_delta; 0 0 0 1].
   ```

   This maps both `p -> p*` and `a -> a*`; it therefore corrects XY and tilt
   without introducing a spurious tip displacement from wrist rotation.
5. With translation gain `g_xy` and orientation gain `g_R`, use
   `p*_g = p + g_xy(p*-p)` and
   `R_delta = Exp(g_R Log(R_min))`. Defaults can remain `1.0`. Accumulate on
   `SE(3)`, not by adding Euler angles:

   ```text
   T_correction,next = T_delta T_correction,current.
   ```

6. Apply this accumulated transform to the next trial's socket-frame
   preinsert hand target, then rerun the existing cuRobo transfer and 20 mm
   key/hand endpoint collision screen. The insertion stroke remains along the
   measured socket axis.

The initial implementation should compute and record this transform without
changing motion. Once the overlay and saved replay are accepted, wire it to
the next-trial correction state. It should not add a new conservative task
failure condition: with two or more visible calibrated shaft lines use the
geometric estimate; otherwise preserve the prior commanded axis and record
that the orientation residual came from the prior rather than observation.

## Reproduction and outputs

```bash
python demo/precision-insertion/compare_saved_key_axis_methods.py \
  --evidence-root /mnt/paradex2/hyunsu/tmp/20261024_key_face_center_methods \
  --review-root /mnt/paradex2/hyunsu/tmp/20261023_prev \
  --output /home/hyunsu/shared_data/AutoDex/precision_insertion/presentation_assets/261024/05_multiview_key_face/orientation_exploration
```

The output contains `axis_method_comparison.json` and one camera-grid overlay
per replayed episode. Overlay colours are red (minimum rectangle), yellow
(partial-face ellipse), blue (strip midline), and green (tip centre).

## Methodological context

- Gummeson and Oskarsson, *Relative pose from cylinder silhouettes*, ACCV
  2024, supports using cylinder silhouette lines when point correspondences do
  not exist: <https://openaccess.thecvf.com/content/ACCV2024/html/Gummeson_Relative_pose_from_cylinder_silhouettes_ACCV_2024_paper.html>.
- CFVS uses coarse pose estimation followed by visual refinement for 6-DoF
  peg-in-hole alignment: <https://doi.org/10.1109/ICRA48891.2023.10160525>.
- Haugaard et al. use multi-camera peg/hole estimates for continuous visual
  servoing under calibration and grasp uncertainty:
  <https://arxiv.org/abs/2011.06399>.

