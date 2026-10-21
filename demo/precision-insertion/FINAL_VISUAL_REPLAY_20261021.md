# 2026-10-21 released-key visual false negative

The board-9, white 20 mm socket run `20261021_111900_186891` grasped,
lifted, transferred, performed the nominal axial stroke, opened Inspire,
and returned to clear view. The controller recorded no safety abort. The
operator subsequently confirmed that the white key was properly positioned
inside the socket. This is a physical placement observation, not an
independent measurement that the key penetrated exactly 20 mm.

The original episode `attempt_f39d3f2468944e199496c2469bcd286f` remains
immutable and records `insertion_success=false`: three Qwen views called the
key `outside_socket` from whole-frame 384×288 pairs. Full-resolution final
frames visibly show the narrow white key standing in the cup. This incorrect
label triggered the optional saved-preinsert XY diagnostic. That diagnostic
sent full-resolution images to the capture-PC Qwen workers; capture1 ran out
of GPU memory. Its GPU also hosted two init daemons (stock 5006/5007 and the
precision adapter 15006/15007), using about 10.8 GiB before Qwen. No XY
correction was saved. The robot was already at clear view when inference
failed.

The final classifier now crops the final raw image around the socket hull
projected and saved during the initial key observation, then applies the
existing 384×288 resize. The raw full image, camera/frame IDs, timestamps,
and source hashes remain intact. The crop coordinates and RGB source-pixel
hash are saved in the checkpoint and recomputed during verification. The
English prompt remains neutral. A prompt that suggested a narrow rod was
rejected after it falsely called an empty-socket capture `inside_socket` in
three views. With the original prompt and crop, two of three tested successful
views identified the inserted key, while all three empty-socket controls
avoided `inside_socket`. The released-key aggregate now needs a majority of
configured cameras: two of three or three of five. If the first three
disagree, it waits for a deciding view instead of prematurely failing. Lift
and preinsert voting are unchanged.
Force and collision limits are unchanged. If a valid projection is
unavailable, classification uses the full frame.

The optional XY diagnostic now records a capture-PC Qwen runtime failure as
`vlm_backend_unavailable` and takes the existing released-key recovery path.
It cannot turn an OOM into a measured XY correction.

The saved trial and empty-socket control were replayed read-only against the
same Qwen3-VL-2B model. In the live five-camera selection, three views voted
`inside_socket` and two voted `outside_socket`; the configured-camera majority
returned `inside_socket` and checkpoint replay passed. In the empty-socket
control, all three probed views avoided false `inside_socket` but returned
`outside_socket` instead of the more precise `unobservable`. This remaining
absence-versus-visible-outside limitation needs a separate real outside-key
case before changing the failure diagnosis. The durable posthoc replay and operator observation
are under `attempt_f39d3f2468944e199496c2469bcd286f/posthoc_visual_review_20261021/`.
The original automatic label remains unchanged. The replay is a visual task
label, not a metric depth measurement.

Verification after the final code change: 61 focused tests passed;
the full precision-insertion suite passed 1078 tests with 4 skipped;
`run_robot_pc.sh --check` passed without connecting to cameras or Franka.
